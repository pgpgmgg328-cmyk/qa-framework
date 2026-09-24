"""browser_controller.py v2 — управление браузером: точные клики + JS-фоллбэк."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Frame,
    Locator,
    Page,
    Playwright,
    async_playwright,
)

from config import (
    ACTION_WAIT,
    FINISH_BUTTON_TEXTS,
    FRAME_IGNORE_KEYWORDS,
    FRAME_KEYWORDS,
    FRAME_LOAD_WAIT,
    HEADLESS,
    MOUSE_MOVE_STEPS,
    PAGE_ZOOM,
    SLOW_MO,
    START_BUTTON_TEXTS,
    SUBMIT_WAIT,
    TARGET_URL,
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
)

logger = logging.getLogger("twork.browser")

# CSS-селектор всех интерактивных элементов
_INTERACTIVE_SELECTOR = ", ".join([
    "button",
    "a[href]",
    "[role='option']",
    "[role='menuitem']",
    "[role='treeitem']",
    "[role='combobox']",
    "tui-select",
    ".t-select",
    "tui-radio-labeled",
    "tui-checkbox-labeled",
    "label.t-item",
    ".tui-tree-item__content",
    ".tui-tree-item",
    "div[class*='child__header']",
    "button[class*='expand']",
    "[class*='tree-item']",
    "[class*='category-item']",
    "[class*='list-item']",
    "input:not([type='hidden'])",
    "textarea",
    "[contenteditable='true']",
])


class BrowserController:
    """Обёртка Playwright для T-Work.

    Ключевые особенности:
    - ignore_https_errors=True (сертификаты Минцифры)
    - zoom 75% (кнопки «завершить» влезают в экран)
    - Цепочка клика: scroll → bounding_box → mouse.move → mousedown/up → JS-фоллбэк
    """

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser:   Optional[Browser]     = None
        self._context:   Optional[BrowserContext] = None
        self._page:      Optional[Page]         = None

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    async def start(self) -> None:
        logger.info("Старт Playwright: headless=%s url=%s", HEADLESS, TARGET_URL)
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=HEADLESS,
            slow_mo=SLOW_MO,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            ignore_https_errors=True,  # КРИТИЧНО: российские SSL
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()

        logger.info("Переход на %s", TARGET_URL)
        await self._page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60_000)

        logger.info("Ожидаем %.1f сек прогрузки фрейма…", FRAME_LOAD_WAIT)
        await asyncio.sleep(FRAME_LOAD_WAIT)
        await self._inject_zoom()
        logger.info("Браузер готов")

    async def stop(self) -> None:
        logger.info("Остановка браузера")
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def __aenter__(self) -> "BrowserController":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Фреймы
    # ------------------------------------------------------------------

    def get_target_frame(self) -> Optional[Frame]:
        """Найти целевой фрейм с заданием.

        Правило: URL содержит klecks-operator или task,
        не captcha / about:blank, не main_frame.
        """
        assert self._page is not None
        main = self._page.main_frame

        for frame in self._page.frames:
            if frame is main:
                continue
            url = frame.url.lower()
            if any(ign in url for ign in FRAME_IGNORE_KEYWORDS):
                continue
            if url in ("about:blank", "", "about:srcdoc"):
                continue
            if any(kw in url for kw in FRAME_KEYWORDS):
                logger.debug("Целевой фрейм: %s", frame.url)
                return frame

        logger.warning(
            "Целевой фрейм не найден. Доступные: %s",
            [f.url for f in self._page.frames],
        )
        return None

    # ------------------------------------------------------------------
    # Точный клик по индексу
    # ------------------------------------------------------------------

    async def click_by_index(self, frame: Frame, index: int) -> None:
        """Надёжная цепочка клика:
        scroll → bounding_box → mouse.move → mousedown+mouseup → JS fallback.
        """
        logger.info("Клик по index=%d", index)
        locator = frame.locator(_INTERACTIVE_SELECTOR).nth(index)

        # 1. Прокрутка к элементу
        try:
            await locator.scroll_into_view_if_needed(timeout=5_000)
        except Exception as exc:
            logger.debug("Прокрутка не удалась: %s", exc)

        # 2. Получаем точные координаты
        bbox = await locator.bounding_box()

        if bbox and bbox["width"] > 0 and bbox["height"] > 0:
            cx = bbox["x"] + bbox["width"]  / 2
            cy = bbox["y"] + bbox["height"] / 2
            await self._physical_click(cx, cy, locator)
        else:
            # Элемент не виден — сразу JS
            logger.warning("bounding_box пустой, прямой JS-клик")
            await self._js_click(locator)

        await asyncio.sleep(ACTION_WAIT)

    async def type_by_index(self, frame: Frame, index: int, text: str) -> None:
        """Ввести текст в элемент по индексу."""
        logger.info("Текст index=%d: %r", index, text[:40])
        locator = frame.locator(_INTERACTIVE_SELECTOR).nth(index)
        try:
            await locator.scroll_into_view_if_needed(timeout=5_000)
            await locator.click(timeout=5_000)
            await locator.fill("")          # стереть старое значение
            await locator.type(text, delay=40)
        except Exception as exc:
            logger.error("Ошибка type_by_index(%d): %s", index, exc)
        await asyncio.sleep(ACTION_WAIT)

    async def click_by_text(
        self,
        frame: Frame,
        texts: tuple[str, ...],
        *,
        skip_disabled: bool = True,
    ) -> bool:
        """Найти кнопку по одному из texts и кликнуть. Возвращает True при успехе."""
        for text in texts:
            text_lower = text.lower()
            try:
                # Попытка 1: role=button
                loc = frame.get_by_role("button", name=text, exact=False)
                cnt = await loc.count()
                if cnt == 0:
                    # Попытка 2: любой элемент с таким текстом
                    loc = frame.get_by_text(text, exact=False)
                    cnt = await loc.count()
                if cnt == 0:
                    continue

                for i in range(cnt):
                    candidate = loc.nth(i)

                    if skip_disabled:
                        disabled = await candidate.get_attribute("disabled")
                        aria_dis = await candidate.get_attribute("aria-disabled")
                        cls      = (await candidate.get_attribute("class")) or ""
                        if (
                            disabled is not None
                            or aria_dis == "true"
                            or "disabled" in cls.lower()
                        ):
                            logger.debug("Кнопка %r заблокирована", text)
                            continue

                    # --- ИСПРАВЛЕННЫЙ БЛОК НАЧАЛО ---
                    # Очищаем текст от лишних пробелов по краям
                    el_text = (await candidate.inner_text() or "").strip().lower()
                    
                    # Если искомое слово короткое (например "ок", "ok"), требуем ТОЧНОГО совпадения,
                    # чтобы не кликать по случайным словам вроде "пОКазать" или "стрОКа".
                    if len(text_lower) <= 3:
                        if text_lower != el_text:
                            continue
                    # Для длинных слов оставляем гибкое вхождение подстроки
                    else:
                        if text_lower not in el_text:
                            continue
                    # --- ИСПРАВЛЕННЫЙ БЛОК КОНЕЦ ---

                    bbox = await candidate.bounding_box()
                    if bbox and bbox["width"] > 0:
                        cx = bbox["x"] + bbox["width"]  / 2
                        cy = bbox["y"] + bbox["height"] / 2
                        await self._physical_click(cx, cy, candidate)
                    else:
                        await self._js_click(candidate)

                    logger.info("Клик по тексту %r", text)
                    await asyncio.sleep(ACTION_WAIT)
                    return True

            except Exception as exc:
                logger.debug("Не удалось нажать %r: %s", text, exc)

        return False

    # ------------------------------------------------------------------
    # Физический клик (основной метод)
    # ------------------------------------------------------------------

    async def _physical_click(self, cx: float, cy: float, locator: Locator) -> None:
        """Плавное движение мыши + mousedown/mouseup.

        Если физический клик не сработал — JS фоллбэк.
        """
        assert self._page is not None
        page = self._page

        try:
            # Плавное перемещение мыши в центр элемента
            await page.mouse.move(cx, cy, steps=MOUSE_MOVE_STEPS)
            await asyncio.sleep(0.05)

            # Физический mousedown → пауза → mouseup
            await page.mouse.down()
            await asyncio.sleep(0.08)
            await page.mouse.up()

            logger.debug("Физический клик: (%.1f, %.1f)", cx, cy)

        except Exception as exc:
            logger.warning("Физический клик слетел: %s. JS fallback.", exc)
            await self._js_click(locator)

    # ------------------------------------------------------------------
    # JS фоллбэк (для перекрытых / нереагирующих)
    # ------------------------------------------------------------------

    @staticmethod
    async def _js_click(locator: Locator) -> None:
        """Безопасный JS-клик через dispatchEvent."""
        try:
            await locator.evaluate(
                """
                el => {
                    ['pointerdown','mousedown','pointerup','mouseup','click']
                        .forEach(type => {
                            el.dispatchEvent(new MouseEvent(type, {
                                bubbles: true, cancelable: true, view: window
                            }));
                        });
                }
                """
            )
            logger.debug("JS dispatchEvent клик выполнен")
        except Exception as exc:
            logger.error("JS клик тоже слетел: %s", exc)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    async def _inject_zoom(self) -> None:
        """Уменьшить масштаб страницы до PAGE_ZOOM."""
        assert self._page is not None
        try:
            await self._page.evaluate(
                f"() => {{ document.body.style.zoom = '{PAGE_ZOOM}'; }}"
            )
            logger.debug("Zoom установлен: %s", PAGE_ZOOM)
        except Exception as exc:
            logger.warning("Не удалось установить зум: %s", exc)

    async def reload_and_wait(self) -> None:
        assert self._page is not None
        await self._page.reload(wait_until="domcontentloaded", timeout=60_000)
        await asyncio.sleep(FRAME_LOAD_WAIT)
        await self._inject_zoom()

    @property
    def page(self) -> Page:
        assert self._page is not None
        return self._page

    @property
    def interactive_selector(self) -> str:
        return _INTERACTIVE_SELECTOR
