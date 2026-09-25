"""browser_controller.py v3 — управление браузером.

Цепочка клика v3:
  метка data-agent-id → scroll_into_view → click(trial=True) — проверка, что элемент
  видим, стабилен (анимация tui-expand закончилась), активен и НЕ перекрыт →
  плавное движение мыши + mousedown/mouseup → при неудаче JS-фоллбэк (PointerEvent + el.click()).

Главные исправления относительно v2:
- клик по стабильной метке снимка, а не по `.nth(index)` другого селектора
  (индексы парсера и контроллера расходились);
- JS-фоллбэк реально срабатывает: v2 переходил на него только при исключении
  mouse.*, а перекрытый элемент исключений не даёт — клик молча уходил в оверлей;
- масштаб 75% — через device_scale_factor и увеличенный viewport: CSS-zoom на
  body ломал координаты Playwright внутри iframe (проверено: клик по bbox
  «Завершить» попадал в пустую область, а locator.click() падал по таймауту);
- ожидание стабилизации DOM (MutationObserver) вместо фиксированных sleep;
- постоянный профиль браузера (логин переживает перезапуск).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Frame,
    Locator,
    Page,
    Playwright,
    async_playwright,
)

from config import (
    ALLOW_MAIN_FRAME,
    BROWSER_CHANNEL,
    CAPTCHA_URL_KEYWORDS,
    CLICK_TIMEOUT_MS,
    FRAME_IGNORE_KEYWORDS,
    FRAME_KEYWORDS,
    FRAME_LOAD_WAIT,
    HEADLESS,
    MOUSE_MOVE_STEPS,
    PAGE_ZOOM_FACTOR,
    SETTLE_QUIET_MS,
    SETTLE_TIMEOUT_MS,
    SLOW_MO,
    TARGET_URL,
    TYPE_DELAY_MS,
    USER_AGENT,
    USER_DATA_DIR,
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
)
from dom_parser import INTERACTIVE_SELECTOR
from models import ElementKind, ParsedElement, normalize_text

logger = logging.getLogger("twork.browser")


@dataclass
class ActionOutcome:
    """Результат действия в браузере."""
    ok: bool
    stale: bool = False     # метка исчезла: DOM перерисован, нужен новый снимок
    method: str = ""        # mouse / js / type / wheel
    detail: str = ""


def is_connection_lost(exc: BaseException) -> bool:
    """Связь с браузером потеряна (закрыт, упал или завершён драйвер) — шаги повторять бессмысленно."""
    text = str(exc)
    return any(marker in text for marker in (
        "Connection closed", "has been closed", "Browser closed", "browser has disconnected",
    ))


def _short(exc: BaseException) -> str:
    return str(exc).strip().splitlines()[0][:160] if str(exc).strip() else exc.__class__.__name__


def _compact(value: str) -> str:
    """Для сравнения введённого значения: без пробелов и регистра (маски «1 000»)."""
    return re.sub(r"\s+", "", value or "").lower()


# Ждём, пока DOM «успокоится»: нет мутаций quietMs подряд (свои метки не в счёт)
_JS_SETTLE = """
({ quietMs, timeoutMs }) => new Promise((resolve) => {
    let quietTimer = null;
    const finish = (reason) => {
        observer.disconnect(); clearTimeout(quietTimer); clearTimeout(hardTimer); resolve(reason);
    };
    const observer = new MutationObserver((mutations) => {
        const relevant = mutations.some((m) => !(m.type === 'attributes'
            && m.attributeName && m.attributeName.startsWith('data-agent')));
        if (!relevant) return;
        clearTimeout(quietTimer);
        quietTimer = setTimeout(() => finish('quiet'), quietMs);
    });
    observer.observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
    quietTimer = setTimeout(() => finish('quiet'), quietMs);
    const hardTimer = setTimeout(() => finish('timeout'), timeoutMs);
})
"""

# JS-клик: полноценная последовательность событий. v2 создавал 'pointerdown'
# через new MouseEvent (без pointerId/pointerType) и не вызывал el.click(),
# поэтому label не активировал свой radio.
_JS_CLICK = """
el => {
    const r = el.getBoundingClientRect();
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    const base = { bubbles: true, cancelable: true, composed: true, view: window,
                   clientX: x, clientY: y, button: 0 };
    const ptr = { ...base, pointerId: 1, pointerType: 'mouse', isPrimary: true };
    el.dispatchEvent(new PointerEvent('pointerover', ptr));
    el.dispatchEvent(new MouseEvent('mouseover', base));
    el.dispatchEvent(new PointerEvent('pointerdown', { ...ptr, buttons: 1 }));
    el.dispatchEvent(new MouseEvent('mousedown', { ...base, buttons: 1 }));
    if (typeof el.focus === 'function') el.focus({ preventScroll: true });
    el.dispatchEvent(new PointerEvent('pointerup', ptr));
    el.dispatchEvent(new MouseEvent('mouseup', base));
    if (typeof el.click === 'function') el.click();
    else el.dispatchEvent(new MouseEvent('click', base));
    return true;
}
"""

# Кто перекрывает центр элемента (для логов и истории)
_JS_BLOCKER = """
el => {
    const r = el.getBoundingClientRect();
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) return 'вне видимой области';
    let hit = document.elementFromPoint(x, y);
    while (hit && hit.shadowRoot) {
        const inner = hit.shadowRoot.elementFromPoint(x, y);
        if (!inner || inner === hit) break;
        hit = inner;
    }
    if (!hit || el === hit || el.contains(hit) || hit.contains(el)) return '';
    const cls = (hit.getAttribute('class') || '').split(/\\s+/).filter(Boolean).slice(0, 2).join('.');
    const txt = (hit.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 60);
    return hit.tagName.toLowerCase() + (cls ? '.' + cls : '') + (txt ? ' «' + txt + '»' : '');
}
"""

_JS_READ_VALUE = """
e => (typeof e.value === 'string') ? e.value : (e.innerText || '')
"""


class BrowserController:
    """Обёртка Playwright для T-Work.

    Ключевые особенности:
    - ignore_https_errors=True (сертификаты Минцифры; надёжнее — установить их корневой сертификат)
    - масштаб PAGE_ZOOM через device_scale_factor (координаты кликов остаются точными)
    - клик только по метке текущего снимка, с проверкой кликабельности и JS-фоллбэком
    """

    def __init__(
        self,
        *,
        on_context: Optional[Callable[[BrowserContext], Awaitable[None]]] = None,
        headless: Optional[bool] = None,
    ) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser:    Optional[Browser] = None
        self._context:    Optional[BrowserContext] = None
        self._page:       Optional[Page] = None
        self._last_frame_warning = 0.0
        # хук после создания контекста: маршруты (тесты, блокировка аналитики), куки и т.п.
        self._on_context = on_context
        # None — как в .env (HEADLESS); режим записи принудительно открывает окно
        self._headless = HEADLESS if headless is None else headless

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    async def start(self) -> None:
        zoom = PAGE_ZOOM_FACTOR
        # «Отдалить» страницу = больше CSS-пикселей в том же окне
        viewport = {"width": round(VIEWPORT_WIDTH / zoom), "height": round(VIEWPORT_HEIGHT / zoom)}
        logger.info(
            "Старт Playwright: headless=%s url=%s viewport=%s zoom=%.2f profile=%s",
            self._headless, TARGET_URL, viewport, zoom, USER_DATA_DIR or "—",
        )
        self._playwright = await async_playwright().start()
        launch_opts: dict = {
            "headless": self._headless,
            "slow_mo": SLOW_MO,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        }
        if BROWSER_CHANNEL:
            launch_opts["channel"] = BROWSER_CHANNEL
        context_opts: dict = {
            "viewport": viewport,
            "ignore_https_errors": True,  # КРИТИЧНО: российские SSL
            "locale": "ru-RU",
            "timezone_id": "Europe/Moscow",
        }
        if zoom != 1.0:
            context_opts["device_scale_factor"] = zoom
        if USER_AGENT:
            context_opts["user_agent"] = USER_AGENT

        chromium = self._playwright.chromium
        if USER_DATA_DIR:
            # Постоянный профиль: куки и логин сохраняются между запусками
            self._context = await chromium.launch_persistent_context(
                USER_DATA_DIR, **launch_opts, **context_opts,
            )
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        else:
            self._browser = await chromium.launch(**launch_opts)
            if self._headless and not USER_AGENT:
                # headless-UA содержит «HeadlessChrome»; берём реальную версию браузера
                context_opts["user_agent"] = await self._headful_user_agent()
            self._context = await self._browser.new_context(**context_opts)
            self._page = await self._context.new_page()

        if self._on_context is not None:
            await self._on_context(self._context)
        logger.info("Переход на %s", TARGET_URL)
        await self._page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60_000)
        logger.info("Браузер готов")

    async def _headful_user_agent(self) -> Optional[str]:
        assert self._browser is not None
        probe = await self._browser.new_page()
        try:
            ua: str = await probe.evaluate("() => navigator.userAgent")
        finally:
            await probe.close()
        return ua.replace("HeadlessChrome", "Chrome")

    async def stop(self) -> None:
        logger.info("Остановка браузера")
        for closer in (
            self._context.close if self._context else None,
            self._browser.close if self._browser else None,
            self._playwright.stop if self._playwright else None,
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception as exc:  # noqa: BLE001 — закрытие best effort: по Ctrl+C драйвер
                # Playwright получает сигнал вместе с Python и завершается первым, а его
                # «Connection closed» — обычный Exception, не playwright.Error
                logger.debug("Ошибка при остановке: %s", _short(exc))

    async def __aenter__(self) -> "BrowserController":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    def is_closed(self) -> bool:
        return self._page is None or self._page.is_closed()

    # ------------------------------------------------------------------
    # Фреймы
    # ------------------------------------------------------------------

    def _frame_score(self, url: str) -> int:
        """Чем раньше ключевое слово в FRAME_KEYWORDS, тем выше приоритет фрейма."""
        url = url.lower()
        if not url or url.startswith("about:") or any(ign in url for ign in FRAME_IGNORE_KEYWORDS):
            return 0
        for i, keyword in enumerate(FRAME_KEYWORDS):
            if keyword in url:
                return len(FRAME_KEYWORDS) - i
        return 0

    async def find_target_frame(self) -> Optional[Frame]:
        """Найти видимый целевой фрейм с заданием.

        v2 возвращал ПЕРВЫЙ фрейм с 'task' в URL — в том числе отсоединённый
        или нулевого размера. Здесь: приоритет ключевых слов, is_detached(),
        реальный размер iframe на странице.
        """
        page = self.page
        best: Optional[Frame] = None
        best_score = 0
        for frame in page.frames:
            if frame is page.main_frame or frame.is_detached():
                continue
            score = self._frame_score(frame.url)
            if score <= best_score:
                continue
            if not await self._frame_is_visible(frame):
                continue
            best, best_score = frame, score

        if best is None and ALLOW_MAIN_FRAME:
            return page.main_frame
        if best is None and time.monotonic() - self._last_frame_warning > 30:
            self._last_frame_warning = time.monotonic()
            logger.warning("Целевой фрейм не найден. Доступные: %s", [f.url for f in page.frames])
        return best

    def get_target_frame(self) -> Optional[Frame]:
        """Синхронный вариант (совместимость с v2): без проверки размера фрейма."""
        page = self.page
        candidates = [
            (self._frame_score(f.url), f) for f in page.frames
            if f is not page.main_frame and not f.is_detached()
        ]
        candidates = [c for c in candidates if c[0] > 0]
        return max(candidates, key=lambda c: c[0])[1] if candidates else None

    @staticmethod
    async def _frame_is_visible(frame: Frame) -> bool:
        try:
            handle = await frame.frame_element()
            box = await handle.bounding_box()
        except PlaywrightError:
            return False
        return bool(box and box["width"] >= 50 and box["height"] >= 50)

    async def page_has_captcha(self) -> bool:
        """Капча — видимый iframe капчи на странице (а не слово «капча» в тексте, как в v2)."""
        for frame in self.page.frames:
            url = frame.url.lower()
            if frame is self.page.main_frame or not any(k in url for k in CAPTCHA_URL_KEYWORDS):
                continue
            try:
                box = await (await frame.frame_element()).bounding_box()
            except PlaywrightError:
                continue
            # невидимый бейдж reCAPTCHA (256×60) не считаем, чекбокс/челлендж — считаем
            if box and box["width"] >= 100 and box["height"] >= 70:
                return True
        return False

    # ------------------------------------------------------------------
    # Действия по элементам снимка
    # ------------------------------------------------------------------

    async def click_element(
        self, frame: Frame, el: ParsedElement, *, prefer_toggle: bool = False,
    ) -> ActionOutcome:
        """Кликнуть элемент снимка по его метке.

        prefer_toggle — для «open»: кликаем отдельную стрелку-раскрывашку, если она есть
        (клик по тексту строки во многих деревьях ВЫБИРАЕТ узел, а не раскрывает его).
        Для выбираемой папки click идёт по её собственному radio/checkbox.
        """
        selectors: list[str] = []
        if prefer_toggle and el.has_toggle:
            selectors.append(f'[data-agent-toggle="{el.uid}"]')
        if not prefer_toggle and el.has_select and el.kind == ElementKind.FOLDER:
            selectors.append(f'[data-agent-select="{el.uid}"]')
        selectors.append(f'[data-agent-id="{el.uid}"]')

        for selector in selectors:
            locator = frame.locator(selector)
            try:
                count = await locator.count()
            except PlaywrightError as exc:
                return ActionOutcome(ok=False, stale=True, detail=f"фрейм недоступен: {_short(exc)}")
            if count:
                return await self._click_locator(locator.first, el.label())
        return ActionOutcome(ok=False, stale=True, detail="элемент исчез из DOM (перерисовка)")

    async def type_into(self, frame: Frame, el: ParsedElement, text: str) -> ActionOutcome:
        """Ввести текст (или выбрать пункт нативного select) и проверить значение."""
        locator = frame.locator(f'[data-agent-id="{el.uid}"]')
        try:
            if not await locator.count():
                return ActionOutcome(ok=False, stale=True, detail="поле исчезло из DOM")
            locator = locator.first
            if el.input_type == "select":
                try:
                    await locator.select_option(label=text, timeout=3_000)
                except PlaywrightError:
                    await locator.select_option(value=text, timeout=3_000)
                actual = await locator.evaluate(
                    "e => ((e.selectedOptions[0] || {}).textContent || '').trim()"
                )
            else:
                focus = await self._click_locator(locator, el.label())
                if not focus.ok:
                    return focus
                await locator.fill("", timeout=3_000)
                await locator.press_sequentially(
                    text, delay=TYPE_DELAY_MS, timeout=max(5_000, len(text) * (TYPE_DELAY_MS + 50)),
                )
                actual = await locator.evaluate(_JS_READ_VALUE)
                if _compact(actual) != _compact(text):
                    # маска/автоформатирование съели символы — вводим значение целиком
                    await locator.fill(text, timeout=3_000)
                    actual = await locator.evaluate(_JS_READ_VALUE)
        except PlaywrightError as exc:
            return ActionOutcome(ok=False, detail=f"ошибка ввода: {_short(exc)}")
        matches = _compact(actual) == _compact(text)
        return ActionOutcome(
            ok=matches or bool(actual.strip()), method="type",
            detail=f"в поле: «{actual[:80]}»" + ("" if matches else " (отличается от введённого)"),
        )

    async def scroll(
        self, frame: Frame, el: Optional[ParsedElement], direction: str = "down",
    ) -> ActionOutcome:
        """Прокрутка колесом мыши над списком: срабатывают и виртуальный скролл
        (cdk-virtual-scroll-viewport), и ленивые подгрузки по scroll-событию."""
        page = self.page
        box = None
        try:
            if el is not None:
                locator = frame.locator(f'[data-agent-id="{el.uid}"]')
                if await locator.count():
                    box = await locator.first.bounding_box(timeout=1_000)
            if box is None and frame is not page.main_frame:
                box = await (await frame.frame_element()).bounding_box()
        except PlaywrightError:
            box = None
        vp = page.viewport_size or {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
        if box:
            x = min(max(box["x"] + box["width"] / 2, 5), vp["width"] - 5)
            y = min(max(box["y"] + box["height"] / 2, 5), vp["height"] - 5)
        else:
            x, y = vp["width"] / 2, vp["height"] / 2
        delta = vp["height"] * 0.6 * (-1 if direction == "up" else 1)
        try:
            await page.mouse.move(x, y, steps=MOUSE_MOVE_STEPS)
            await page.mouse.wheel(0, delta)
        except PlaywrightError as exc:
            return ActionOutcome(ok=False, detail=f"прокрутка не удалась: {_short(exc)}")
        return ActionOutcome(ok=True, method="wheel", detail=f"прокрутка {direction}")

    async def click_by_text(
        self,
        frame: Frame,
        texts: tuple[str, ...],
        *,
        skip_disabled: bool = True,
        deny: tuple[str, ...] = (),
    ) -> bool:
        """Фоллбэк: найти КНОПКУ по тексту и кликнуть. Возвращает True при успехе.

        В v2 при len(text) > 3 сравнение было подстрокой по ЛЮБОМУ элементу
        (get_by_text): «начать» совпадало с «Начать заново», «хорошо» — с
        вариантом ответа «Хорошо», «завершить» — с «Завершить смену».
        Теперь: только role=button, сначала точное совпадение, затем начало фразы,
        плюс список запрещённых подстрок.
        """
        for text in texts:
            patterns = (
                re.compile(rf"^\s*{re.escape(text)}\s*$", re.IGNORECASE),
                re.compile(rf"^\s*{re.escape(text)}(?:\s|$)", re.IGNORECASE),
            )
            for pattern in patterns:
                buttons = frame.get_by_role("button", name=pattern)
                try:
                    count = await buttons.count()
                except PlaywrightError:
                    continue
                for i in range(count):
                    candidate = buttons.nth(i)
                    try:
                        if not await candidate.is_visible():
                            continue
                        if skip_disabled and not await candidate.is_enabled(timeout=1_000):
                            logger.debug("Кнопка %r заблокирована", text)
                            continue
                        name = normalize_text(await candidate.inner_text(timeout=1_000))
                    except PlaywrightError:
                        continue
                    if any(d in name for d in deny):
                        logger.info("Кнопка «%s» пропущена (запрещённая подстрока)", name)
                        continue
                    outcome = await self._click_locator(candidate, name)
                    if outcome.ok:
                        logger.info("Клик по тексту «%s»", name)
                        return True
        return False

    # ------------------------------------------------------------------
    # Ожидания и скриншоты
    # ------------------------------------------------------------------

    async def wait_settle(
        self, frame: Frame, *, quiet_ms: int = SETTLE_QUIET_MS, timeout_ms: int = SETTLE_TIMEOUT_MS,
    ) -> str:
        """Дождаться, пока DOM фрейма перестанет меняться (анимации раскрытия,
        change detection Angular, подгрузка детей). Возвращает причину выхода."""
        try:
            return await frame.evaluate(_JS_SETTLE, {"quietMs": quiet_ms, "timeoutMs": timeout_ms})
        except PlaywrightError as exc:
            # фрейм перезагрузился во время ожидания — это тоже «изменение»
            logger.debug("wait_settle: %s", _short(exc))
            await asyncio.sleep(quiet_ms / 1000)
            return "navigated"

    async def screenshot_b64(self, frame: Frame, mode: str) -> Optional[str]:
        """Скриншот для vision-модели: главная картинка задания или весь фрейм (JPEG, base64)."""
        try:
            if mode == "image":
                locator = frame.locator("[data-agent-img]")
                if not await locator.count():
                    return None
                data = await locator.first.screenshot(
                    type="jpeg", quality=80, timeout=5_000, animations="disabled",
                )
            elif mode == "frame" and frame is not self.page.main_frame:
                handle = await frame.frame_element()
                data = await handle.screenshot(type="jpeg", quality=70, timeout=5_000)
            elif mode == "frame":
                data = await self.page.screenshot(type="jpeg", quality=70, timeout=5_000)
            else:
                return None
        except PlaywrightError as exc:
            logger.debug("Скриншот не получен: %s", _short(exc))
            return None
        return base64.b64encode(data).decode("ascii")

    # ------------------------------------------------------------------
    # Клик: проверка кликабельности → мышь → JS-фоллбэк
    # ------------------------------------------------------------------

    async def _click_locator(self, locator: Locator, what: str) -> ActionOutcome:
        try:
            await locator.scroll_into_view_if_needed(timeout=3_000)
        except PlaywrightError as exc:
            logger.debug("Прокрутка не удалась: %s", _short(exc))

        try:
            # trial=True: все проверки Playwright (attached, visible, stable, enabled,
            # «точку клика не перекрывает другой элемент») без самого клика
            await locator.click(trial=True, timeout=CLICK_TIMEOUT_MS)
        except PlaywrightError as exc:
            blocker = await self._describe_blocker(locator)
            logger.warning("«%s» не готов к клику (%s) — JS-фоллбэк", what, blocker or _short(exc))
            if await self._js_click(locator):
                detail = f"js-клик (перекрыт: {blocker})" if blocker else "js-клик"
                return ActionOutcome(ok=True, method="js", detail=detail)
            return ActionOutcome(ok=False, detail=f"не удалось кликнуть: {blocker or _short(exc)}")

        try:
            box = await locator.bounding_box(timeout=1_000)
        except PlaywrightError:
            box = None
        if not box or box["width"] < 1 or box["height"] < 1:
            ok = await self._js_click(locator)
            return ActionOutcome(ok=ok, method="js", detail="js-клик (нет bounding box)")

        x, y = self._aim(box)
        try:
            await self._human_click(x, y)
        except PlaywrightError as exc:
            logger.warning("Физический клик не удался: %s — JS-фоллбэк", _short(exc))
            ok = await self._js_click(locator)
            return ActionOutcome(ok=ok, method="js", detail="js-клик после ошибки мыши")
        logger.debug("Физический клик «%s»: (%.1f, %.1f)", what, x, y)
        return ActionOutcome(ok=True, method="mouse")

    @staticmethod
    def _aim(box: dict) -> tuple[float, float]:
        """Центр элемента с небольшим разбросом (не выходит за внутренние 15%)."""
        jx = min(box["width"] * 0.15, 4.0)
        jy = min(box["height"] * 0.15, 3.0)
        return (
            box["x"] + box["width"] / 2 + random.uniform(-jx, jx),
            box["y"] + box["height"] / 2 + random.uniform(-jy, jy),
        )

    async def _human_click(self, x: float, y: float) -> None:
        """Плавное движение мыши + mousedown/mouseup с человеческими паузами."""
        mouse = self.page.mouse
        await mouse.move(x, y, steps=MOUSE_MOVE_STEPS)
        await asyncio.sleep(random.uniform(0.04, 0.09))
        await mouse.down()
        await asyncio.sleep(random.uniform(0.05, 0.12))
        await mouse.up()

    @staticmethod
    async def _js_click(locator: Locator) -> bool:
        try:
            await locator.evaluate(_JS_CLICK)
            logger.debug("JS-клик выполнен")
            return True
        except PlaywrightError as exc:
            logger.error("JS-клик тоже не удался: %s", _short(exc))
            return False

    @staticmethod
    async def _describe_blocker(locator: Locator) -> str:
        try:
            return str(await locator.evaluate(_JS_BLOCKER))
        except PlaywrightError:
            return ""

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    async def reload_and_wait(self) -> None:
        await self.page.reload(wait_until="domcontentloaded", timeout=60_000)
        await asyncio.sleep(FRAME_LOAD_WAIT)

    @property
    def page(self) -> Page:
        assert self._page is not None, "BrowserController не запущен"
        return self._page

    @property
    def context(self) -> BrowserContext:
        assert self._context is not None, "BrowserController не запущен"
        return self._context

    @property
    def interactive_selector(self) -> str:
        return INTERACTIVE_SELECTOR
