"""agent.py v4 — главный цикл агента.

Агент решает задания любого вида, пока платформа не вернёт его на список заказов.
Шаблонов под виды заданий нет: страница задания целиком (текст, таблицы, ссылки, фото,
аудио, поля) уходит модели, а модель сама решает, как выполнить задание.

Шаг агента:
  1. Всплывающие окна сайта (новости T-Work) поверх фрейма — закрыть.
  2. Найти видимый фрейм задания; капча — ждать ручного решения.
  3. Снимок DomParser (reader-view, элементы размечены метками data-agent-id).
  4. Список заказов («Приступить») после выполненных заданий — остановка.
  5. Новое задание (TaskIdentity) → сброс памяти, знания о виде задания.
  6. Диалоги: «Выйти из задания?» → «остаться»; инструкция → прочитать, сохранить,
     закрыть; «Тренировка … Начать» → начать.
  7. Инструкция и подсказки «?» вида задания — прочитать один раз за запуск.
  8. Аудио — расшифровка в фоне и воспроизведение; фото — скачивание для модели.
  9. Решение LLM (знания, страница, фото, аудио, поиск, план, история, неверные ответы).
 10. Выполнение по метке; web — поиск в отдельной вкладке; submit — дослушать аудио,
     нажать и дождаться: следующее задание или «Неверный ответ» (тогда урок + исправление).
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlsplit

from playwright.async_api import Error as PlaywrightError, Frame

from browser_controller import BrowserController, is_connection_lost
from config import (
    ACTION_WAIT,
    AUDIO_PLAY_TO_END,
    BATCH_ACTIONS,
    CAPTCHA_TIMEOUT,
    DIALOG_CLOSE_TEXTS,
    EXIT_CANCEL_TEXTS,
    FINISH_BUTTON_TEXTS,
    FINISH_DENY_SUBSTRINGS,
    FRAME_LOAD_WAIT,
    INSTRUCTION_WAIT,
    LLM_HISTORY_SIZE,
    LLM_TEMPERATURE,
    LLM_VISION,
    MAX_BATCH_ACTIONS,
    MAX_IDLE_SECONDS,
    MAX_STEPS,
    MAX_STEPS_PER_TASK,
    MAX_WEB_PER_TASK,
    ORDERS_BUTTON_TEXTS,
    READ_INSTRUCTIONS,
    READ_TOOLTIPS,
    START_BUTTON_TEXTS,
    STOP_ON_ORDERS_LIST,
    SUBMIT_WAIT,
    TASK_URL_KEYWORDS,
    WEB_RESEARCH,
)
from dom_parser import DomParser, _plain_text, is_denied_button
from knowledge import KnowledgeBase, PoolKnowledge
from media import MediaManager
from models import (
    FLAG_DISABLED,
    FLAG_IN_DIALOG,
    FLAG_IN_POPUP,
    FLAG_OCCLUDED,
    FLAG_SELECTABLE,
    FLAG_SELECTED,
    PAGE_CHANGING_ACTIONS,
    ActionType,
    DecisionContext,
    ElementKind,
    FolderState,
    LLMDecision,
    PageState,
    ParsedElement,
    normalize_text,
)
from openrouter_connector import LLMConnector
from research import WebResearch
from task_memory import TaskMemory

logger = logging.getLogger("twork.agent")


class StepResult(str, Enum):
    ACTED = "acted"   # шаг с действием — расходует MAX_STEPS
    IDLE = "idle"     # ожидание (нет фрейма, капча, загрузка) — не расходует
    STOP = "stop"


# Какие типы элементов допустимы для действия (в v2 автокоррекция могла
# «исправить» open на OPTION, а type — на BUTTON)
_ACTION_KINDS: dict[ActionType, set[ElementKind]] = {
    ActionType.OPEN:   {ElementKind.FOLDER, ElementKind.DROPDOWN},
    ActionType.CLICK:  {ElementKind.OPTION, ElementKind.BUTTON, ElementKind.FOLDER,
                        ElementKind.DROPDOWN, ElementKind.OTHER},
    ActionType.TYPE:   {ElementKind.INPUT},
    ActionType.SUBMIT: {ElementKind.BUTTON},
    ActionType.SCROLL: set(ElementKind),
}
# Мягкий фоллбэк: open по строке, которую парсер не распознал как папку
_FALLBACK_KINDS: dict[ActionType, set[ElementKind]] = {
    ActionType.OPEN: {ElementKind.OPTION, ElementKind.OTHER},
}
_MATCH_THRESHOLD = 0.85
_WRONG_RE = re.compile(r"(неверн|неправильн|ошибк[аи] в ответе|incorrect|wrong)", re.IGNORECASE)
_INSTRUCTION_RE = re.compile(r"инструкц", re.IGNORECASE)

# Всплывающее окно на ГЛАВНОЙ странице сайта (новости, объявления) поверх фрейма задания
_JS_PAGE_POPUP = r"""
() => {
    const SEL = '[role="dialog"], [aria-modal="true"], dialog[open], tui-dialog, [class*="modal" i]';
    const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim();
    const visible = (el) => {
        const r = el.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) return false;
        return el.checkVisibility ? el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }) : true;
    };
    document.querySelectorAll('[data-agent-popup]').forEach((b) => b.removeAttribute('data-agent-popup'));
    const frames = Array.from(document.querySelectorAll('iframe'));
    for (const d of document.querySelectorAll(SEL)) {
        let box = d.getBoundingClientRect();
        for (const ch of d.children) {
            const r = ch.getBoundingClientRect();
            if (r.width * r.height > box.width * box.height) box = r;
        }
        if (box.width < 150 || box.height < 100 || !visible(d)) continue;
        if (frames.some((f) => d.contains(f))) continue;         // это оболочка самого задания
        const buttons = [];
        d.querySelectorAll('button, [role="button"], a[href]').forEach((b, i) => {
            if (!visible(b)) return;
            b.setAttribute('data-agent-popup', String(i));
            buttons.push({ i, text: norm(b.innerText || b.getAttribute('aria-label') || b.title || ''),
                           aria: norm((b.getAttribute('aria-label') || '') + ' ' + (b.getAttribute('class') || '')) });
        });
        if (buttons.length) return { text: norm(d.innerText).slice(0, 300), buttons };   // подложка без кнопок — дальше
    }
    return null;
}
"""

# Подсказки «?» у вариантов ответа: иконки-триггеры внутри строк формы
_JS_TOOLTIP_TRIGGERS = r"""
() => {
    const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim();
    const SEL = 'tui-tooltip, [tuitooltip], [tuihint], [data-tooltip], [class*="tooltip" i]';
    document.querySelectorAll('[data-agent-tip]').forEach((x) => x.removeAttribute('data-agent-tip'));
    const out = [], rows = new Set();
    let n = 0;
    for (const t of document.querySelectorAll(SEL)) {
        const r = t.getBoundingClientRect();
        if (r.width < 4 || r.height < 4 || r.width > 48 || r.height > 48) continue;
        const row = t.closest('[data-agent-id]');
        if (!row || rows.has(row)) continue;
        if (t.checkVisibility && !t.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) continue;
        rows.add(row);
        t.setAttribute('data-agent-tip', String(n));
        out.push({ n, uid: row.getAttribute('data-agent-id') });
        n++;
        if (n >= 16) break;
    }
    return out;
}
"""

_JS_HINT_TEXT = r"""
() => {
    const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim();
    const out = [];
    for (const h of document.querySelectorAll('tui-hint, [role="tooltip"], [class*="tooltip" i][class*="content" i]')) {
        const r = h.getBoundingClientRect();
        if (r.width < 4 || r.height < 4) continue;
        const t = norm(h.innerText);
        if (t) out.push(t);
    }
    return out.join(' ').slice(0, 800);
}
"""


def _similarity(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.8 + 0.15 * min(len(a), len(b)) / max(len(a), len(b))
    return difflib.SequenceMatcher(None, a, b).ratio()


def _clean_target_text(text: str) -> str:
    """LLM иногда копирует строку целиком: «[3] [OPTION] «Смартфоны» ✓ВЫБРАН»."""
    t = re.sub(r"^\s*\[\d+\]\s*", "", text or "")
    t = re.sub(r"^\s*\[[^\]]*\]\s*", "", t)
    quoted = re.search(r"«([^»]+)»", t)
    if quoted:
        t = quoted.group(1)
    for flag in (FLAG_SELECTED, FLAG_DISABLED, FLAG_OCCLUDED, FLAG_SELECTABLE, FLAG_IN_DIALOG, FLAG_IN_POPUP):
        t = t.replace(flag, "")
    t = re.sub(r"⟨[^⟩]*⟩", "", t)
    t = re.sub(r"\s→\s\S+$", "", t)
    return t.strip()


def _short(exc: BaseException) -> str:
    return str(exc).strip().splitlines()[0][:160] if str(exc).strip() else exc.__class__.__name__


def _label_matches(label: str, texts: tuple[str, ...]) -> bool:
    label = normalize_text(label)
    return any(label == t or label.startswith(t + " ") for t in texts)


@dataclass(frozen=True)
class TaskIdentity:
    """Что считать «тем же заданием».

    Один хэш не годится: фото карусели догружаются (набор адресов растёт), счётчик
    символов в поле меняет цифры, а в заданиях «оцените фото» текст одинаков у всех
    заданий — различаются только фото."""
    content: str
    loose: str
    media: frozenset[str]
    form: str = ""

    @classmethod
    def of(cls, state: PageState) -> "TaskIdentity":
        return cls(state.content_hash, state.loose_hash, frozenset(state.media_srcs), state.form_hash)

    def same_task(self, other: "TaskIdentity", *, submitted: bool) -> bool:
        media_related = (not self.media or not other.media or bool(self.media & other.media))
        if not media_related:
            return False
        if self.content == other.content:
            # форма изменилась без отправки — это наш же выбор открыл/скрыл поле
            return self.form == other.form or not submitted
        return self.loose == other.loose and not submitted   # только цифры: таймер, счётчик


class Agent:
    """Автономный агент для платформы T-Work v4."""

    def __init__(
        self,
        *,
        browser: Optional[BrowserController] = None,
        llm: Optional[LLMConnector] = None,
        knowledge: Optional[KnowledgeBase] = None,
    ) -> None:
        # зависимости можно подменить (тесты, другой провайдер LLM)
        self._browser = browser or BrowserController()
        self._llm = llm or LLMConnector()
        self._memory = TaskMemory()
        self._identity: Optional[TaskIdentity] = None
        self._task_identifier: str = ""
        self._submitted = False             # после нажатия «Завершить» новое задание ожидаемо
        self._tasks_done = 0
        self._wrong_total = 0
        self._seen_task = False             # агент уже был в задании (для остановки на списке заказов)
        self._captcha_suppressed_until = 0.0
        self._last_idle_log = 0.0
        self._popup_attempts: dict[str, int] = {}
        self._media = MediaManager(self._browser, getattr(self._llm, "transcribe", None))
        self._web = WebResearch(self._browser)
        self._knowledge = knowledge or KnowledgeBase()
        self._pool: Optional[PoolKnowledge] = None
        self._frame_shot: Optional[str] = None
        self._frame_shot_task = ""
        # учёт токенов: у LLMConnector есть usage (сценарные «LLM» тестов — без него)
        usage = getattr(self._llm, "usage", None)
        self._task_usage_start = usage.snapshot() if usage is not None else None

    # ------------------------------------------------------------------
    # Главная точка входа
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Запустить агента: работает до возврата на список заказов (или лимитов)."""
        started = time.monotonic()
        async with self._browser:
            logger.info("=" * 60)
            logger.info("АГЕНТ v4 ЗАПУЩЕН. Работаю, пока платформа не вернёт на список заказов.")
            logger.info("Шагов с действием максимум: %d, на одно задание: %d", MAX_STEPS, MAX_STEPS_PER_TASK)
            logger.info("=" * 60)
            acted = 0
            idle_since: Optional[float] = None
            while acted < MAX_STEPS:
                if self._browser.is_closed():
                    logger.warning("Окно браузера закрыто — остановка")
                    break
                try:
                    result = await self._step()
                except PlaywrightError as exc:
                    if self._browser.is_closed() or is_connection_lost(exc):
                        logger.warning("Браузер закрыт — остановка")
                        break
                    # «Execution context was destroyed» и т.п.: фрейм перезагрузился посреди шага
                    logger.warning("Ошибка Playwright в шаге (%s) — повтор на свежем снимке", _short(exc))
                    await asyncio.sleep(1.0)
                    continue
                except Exception as exc:  # noqa: BLE001 — один сбойный шаг не должен ронять агента
                    if is_connection_lost(exc):
                        logger.warning("Связь с браузером потеряна — остановка")
                        break
                    logger.exception("Непредвиденная ошибка в шаге: %s", exc)
                    await asyncio.sleep(2.0)
                    continue

                if result == StepResult.STOP:
                    break
                if result == StepResult.ACTED:
                    acted += 1
                    idle_since = None
                else:
                    idle_since = idle_since or time.monotonic()
                    if time.monotonic() - idle_since > MAX_IDLE_SECONDS:
                        logger.error("Нет активности %.0f с — остановка", MAX_IDLE_SECONDS)
                        break
                await asyncio.sleep(0.2)
            else:
                logger.warning("ДОСТИГНУТ ЛИМИТ ШАГОВ (%d)", MAX_STEPS)
            await self._web.close()
            self._media.close()
            self._log_task_usage()
            minutes = (time.monotonic() - started) / 60
            logger.info(
                "ИТОГ: отправлено заданий=%d, из них платформа признала неверными=%d, "
                "шагов с действием=%d, время %.1f мин",
                self._tasks_done, self._wrong_total, acted, minutes,
            )
            usage = getattr(self._llm, "usage", None)
            if usage is not None and usage.calls:
                per_task = ""
                if self._tasks_done:
                    per_task = (f"; в среднем на задание: вход {usage.prompt // self._tasks_done}, "
                                f"выход {usage.completion // self._tasks_done}")
                logger.info("ТОКЕНЫ за запуск: %s%s", usage.render(), per_task)

    # ------------------------------------------------------------------
    # Один шаг агента
    # ------------------------------------------------------------------

    async def _step(self) -> StepResult:
        # 1. Окно новостей сайта поверх задания
        if await self._dismiss_page_popup():
            return StepResult.IDLE

        # 2. Фрейм задания
        frame = await self._browser.find_target_frame()
        if frame is None:
            self._log_idle("Фрейм задания не найден — жду (если нужен вход в аккаунт, войдите в окне "
                           "браузера и откройте заказ кнопкой «Приступить»)")
            await asyncio.sleep(FRAME_LOAD_WAIT)
            return StepResult.IDLE

        if time.monotonic() > self._captcha_suppressed_until and await self._browser.page_has_captcha():
            await self._wait_captcha_solved()
            return StepResult.IDLE

        # 3. Снимок
        state = await DomParser(frame).parse()

        # 4. Список заказов
        if self._is_orders_list(state):
            return await self._on_orders_list()

        self._refresh_task_context(frame, state)
        self._memory.verify(state)          # фактический результат прошлого действия → в историю

        if state.has_captcha and time.monotonic() > self._captcha_suppressed_until:
            await self._wait_captcha_solved()
            return StepResult.IDLE

        # 5. Диалоги поверх задания
        handled = await self._handle_dialog(frame, state)
        if handled is not None:
            return handled

        # 6. Загрузка на всю страницу: ждём, но не бесконечно
        if state.loading and self._memory.loading_waits < 5:
            self._memory.loading_waits += 1
            logger.info("Идёт загрузка — жду стабилизации DOM")
            await self._browser.wait_settle(frame)
            await asyncio.sleep(0.5)
            return StepResult.IDLE
        if not state.loading:
            self._memory.loading_waits = 0

        # 7. Локальный флоу «Начать / ОК» (заставки и диалоги тренировки)
        if await self._maybe_click_start(frame, state):
            return StepResult.ACTED

        # 8. Нет активных элементов — ждём
        if not any(not e.is_disabled for e in state.visible_elements):
            self._log_idle("Активных элементов нет — жду загрузки/следующего задания")
            await asyncio.sleep(max(ACTION_WAIT, 1.0))
            return StepResult.IDLE
        self._seen_task = True

        # 9. Знания о виде задания: инструкция и подсказки «?» (один раз за запуск)
        if await self._maybe_open_instruction(frame, state):
            return StepResult.IDLE
        await self._maybe_read_tooltips(frame, state)

        # 10. Аудио: расшифровка в фоне, воспроизведение
        if state.audios:
            self._media.start_transcription(frame, state)
            await self._media.ensure_playing(frame, state)

        # 11. Бюджет шагов на задание
        self._memory.steps += 1
        if self._memory.steps > MAX_STEPS_PER_TASK:
            return await self._handle_budget_exhausted(frame, state)

        # 12. Решение LLM
        logger.info(
            "%s задание %s · шаг %d/%d · «%s» %s",
            "-" * 10, state.task_identifier[:8], self._memory.steps, MAX_STEPS_PER_TASK,
            state.task_preview[:50], "-" * 10,
        )
        context = await self._build_context(frame, state)
        decision = await self._llm.decide(state, context)
        if decision.plan:
            self._memory.plan = decision.plan

        # 13. Сверка целей и выполнение пакета действий
        await self._run_batch(frame, decision, state)
        await self._browser.wait_settle(frame)
        return StepResult.ACTED

    # ------------------------------------------------------------------
    # Список заказов и окна сайта
    # ------------------------------------------------------------------

    @staticmethod
    def _is_orders_list(state: PageState) -> bool:
        """Список заказов: фрейм не на странице задания и есть кнопки «Приступить»."""
        url = state.frame_url.lower()
        if any(keyword in url for keyword in TASK_URL_KEYWORDS):
            return False
        return any(
            e.kind == ElementKind.BUTTON and _label_matches(e.text, ORDERS_BUTTON_TEXTS)
            for e in state.elements
        )

    async def _on_orders_list(self) -> StepResult:
        if self._seen_task and STOP_ON_ORDERS_LIST:
            logger.info("✅ Платформа вернула на список заказов — заказ выполнен, агент завершает работу")
            return StepResult.STOP
        self._log_idle("Открыт список заказов. Выберите заказ и нажмите «Приступить» — агент начнёт решать "
                       "задания и остановится, когда платформа вернёт сюда")
        await asyncio.sleep(FRAME_LOAD_WAIT)
        return StepResult.IDLE

    async def _dismiss_page_popup(self) -> bool:
        """Новости/объявления сайта (например, «Одноразовые пароли для TWork») открываются
        поверх фрейма задания и перехватывают клики. Закрываем кнопкой «Закрыть/Далее/OK»."""
        page = self._browser.page
        try:
            popup = await page.main_frame.evaluate(_JS_PAGE_POPUP)
        except PlaywrightError:
            return False
        if not popup:
            return False
        text = str(popup.get("text") or "")
        key = normalize_text(text)[:120]
        if self._popup_attempts.get(key, 0) >= 3:
            return False
        buttons = popup.get("buttons") or []

        def pick(texts: tuple[str, ...]) -> Optional[dict]:
            for b in buttons:
                label = str(b.get("text") or "")
                if label and _label_matches(label, texts) and not any(d in normalize_text(label)
                                                                      for d in FINISH_DENY_SUBSTRINGS):
                    return b
            return None

        button = pick(DIALOG_CLOSE_TEXTS) or pick(EXIT_CANCEL_TEXTS)
        if button is None:
            button = next((b for b in buttons if not b.get("text")
                           and re.search(r"(закрыть|close|cross)", str(b.get("aria") or ""), re.IGNORECASE)), None)
        if button is None:
            return False
        self._popup_attempts[key] = self._popup_attempts.get(key, 0) + 1
        logger.info("Закрываю всплывающее окно сайта «%s» кнопкой «%s»", text[:60], button.get("text") or "×")
        locator = page.main_frame.locator(f'[data-agent-popup="{button["i"]}"]')
        try:
            if await locator.count():
                await self._browser.click_locator(locator.first, str(button.get("text") or "закрыть"))
                await asyncio.sleep(0.6)
                return True
        except PlaywrightError as exc:
            logger.debug("Окно сайта не закрылось: %s", _short(exc))
        return False

    # ------------------------------------------------------------------
    # Память и знания задания
    # ------------------------------------------------------------------

    def _refresh_task_context(self, frame: Frame, state: PageState) -> None:
        """Сбрасывать память ТОЛЬКО при смене задания (см. TaskIdentity)."""
        identity = TaskIdentity.of(state)
        if self._identity is None or not self._identity.same_task(identity, submitted=self._submitted):
            if self._identity is not None:
                self._log_task_usage()
            logger.info(
                "СМЕНА ЗАДАНИЯ: %s → %s «%s»",
                self._task_identifier[:8] or "—", state.task_identifier[:8], state.task_preview[:60],
            )
            self._memory.reset(state.task_identifier)
            self._media.forget_task()
            self._frame_shot = None
            pool = self._knowledge.for_state(state)
            if pool is not None and pool is not self._pool:
                known = "есть инструкция" if pool.has_instruction else "инструкции пока нет"
                logger.info("Вид задания: «%s» (%s, уроков: %d)", pool.title, known, len(pool.lessons))
            self._pool = pool
        self._identity = identity
        self._task_identifier = state.task_identifier
        self._submitted = False

    def _log_task_usage(self) -> None:
        """Расход токенов на прошлое задание (по данным API: вход, из кэша, выход)."""
        usage = getattr(self._llm, "usage", None)
        if usage is None or self._task_usage_start is None:
            return
        spent = usage.since(self._task_usage_start)
        self._task_usage_start = usage.snapshot()
        if spent.calls:
            logger.info("Токены на задание %s: %s", self._task_identifier[:8] or "—", spent.render())

    async def _build_context(self, frame: Frame, state: PageState) -> DecisionContext:
        mem = self._memory
        steps_left = MAX_STEPS_PER_TASK - mem.steps
        notes: list[str] = []
        if steps_left <= 5:
            notes.append(
                f"Осталось шагов на это задание: {steps_left}. Если ответ уже заполнен — submit; "
                "иначе заверши заполнение лучшими доступными вариантами."
            )
        if mem.consecutive_skips >= 2:
            notes.append(f"Ты пропустил {mem.consecutive_skips} шага подряд — выбери конкретное действие.")
        if mem.invalid_targets:
            notes.append("Прошлое действие ссылалось на несуществующий элемент: бери номер и текст "
                         "ТОЛЬКО из текущей страницы.")
        if mem.repeated_forbidden:
            notes.append("Ты повторил действие из НЕ ПОВТОРЯТЬ — выбери другой элемент или другое действие.")
        notes += mem.batch_notes
        mem.batch_notes = []
        if mem.web_queries and sum(mem.web_queries.values()) >= MAX_WEB_PER_TASK:
            notes.append("Лимит поисковых запросов на задание исчерпан — отвечай по уже найденным данным.")

        # При зацикливании слегка «встряхиваем» детерминированную модель
        temperature = None
        if mem.consecutive_skips >= 3 or mem.repeated_forbidden >= 1:
            temperature = max(LLM_TEMPERATURE, 0.4)

        images, image_notes = [], []
        image_b64 = None
        if LLM_VISION in ("auto", "image") and state.images:
            images, image_notes = await self._media.vision_images(frame, state)
        if LLM_VISION == "frame" or (LLM_VISION in ("auto", "image") and not state.images and state.image_src):
            image_b64 = await self._frame_screenshot(frame, state)

        transcripts = await self._media.transcripts(state) if state.audios else []
        research = [
            r.render(i + 1, full=i >= len(mem.web_results) - 2) for i, r in enumerate(mem.web_results)
        ]
        return DecisionContext(
            history=mem.history_lines(LLM_HISTORY_SIZE),
            forbidden=mem.forbidden_lines(),
            notes=notes,
            step_in_task=mem.steps,
            steps_left=steps_left,
            image_b64=image_b64,
            temperature=temperature,
            images=images,
            image_notes=image_notes,
            transcripts=transcripts,
            knowledge=self._knowledge.prompt_text(self._pool),
            research=research,
            plan=mem.plan,
            feedback=list(mem.feedback),
            wrong_answers=list(mem.wrong_answers),
        )

    async def _frame_screenshot(self, frame: Frame, state: PageState) -> Optional[str]:
        """Скриншот фрейма (LLM_VISION=frame или картинка без <img>, например canvas)."""
        if LLM_VISION == "frame" or self._frame_shot_task != state.task_identifier:
            self._frame_shot = await self._browser.screenshot_b64(frame, "frame")
            self._frame_shot_task = state.task_identifier
        return self._frame_shot

    # ------------------------------------------------------------------
    # Диалоги поверх задания
    # ------------------------------------------------------------------

    async def _handle_dialog(self, frame: Frame, state: PageState) -> Optional[StepResult]:
        """Диалоги, которые агент закрывает сам. None — диалога нет или решает LLM."""
        if not state.dialog_open:
            return None
        dialog = [e for e in state.visible_elements if e.container == "dialog"]
        buttons = [e for e in dialog if e.kind == ElementKind.BUTTON and not e.is_disabled]
        text = _plain_text(state.dialog_lines)

        # «Выйти из задания?» — агент задания не бросает: «Нет, остаться»
        denied = [e for e in state.elements if e.container == "dialog" and is_denied_button(e)]
        if denied:
            cancel = next((b for b in buttons if _label_matches(b.text, EXIT_CANCEL_TEXTS)), None)
            if cancel is not None:
                logger.warning("Открыт диалог «%s» — отвечаю «%s»", text[:60], cancel.label())
                await self._browser.click_element(frame, cancel)
                await self._browser.wait_settle(frame)
                return StepResult.ACTED

        # Инструкция к заданию: дождаться загрузки, прочитать, сохранить, закрыть.
        # Заставка «Тренировка … изучите инструкцию … [Начать]» — не инструкция: её
        # закрывает локальный флоу кнопкой «Начать».
        start_button = any(_label_matches(b.text, START_BUTTON_TEXTS) for b in buttons)
        size = len(re.sub(r"\s+", "", text))
        instruction = (state.dialog_loading or state.dialog_frames > 0 or size > 700
                       or (bool(_INSTRUCTION_RE.search(text[:120])) and not start_button))
        if instruction:
            key = normalize_text(text)[:80]
            if self._memory.dialog_attempts[key] < 3:
                self._memory.dialog_attempts[key] += 1
                return await self._read_instruction_dialog(frame, state)
        return None

    async def _read_instruction_dialog(self, frame: Frame, state: PageState) -> StepResult:
        deadline = time.monotonic() + INSTRUCTION_WAIT
        if state.dialog_loading:
            logger.info("Открыта инструкция — жду загрузки (до %.0f с)", INSTRUCTION_WAIT)
        while state.dialog_loading and time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            state = await DomParser(frame).parse(quiet=True)
            if not state.dialog_open:
                return StepResult.IDLE
        text = _plain_text(state.dialog_lines)
        if state.dialog_frames:
            text = "\n".join(filter(None, [text, await self._read_child_frames(frame)]))
        failed = any("ошибка загрузки" in n.lower() for n in state.notice_texts())
        title = (text.strip().splitlines() or ["инструкция"])[0][:80]
        body = text
        if len(re.sub(r"\s+", "", body)) < 200 and not failed:
            extra = await self._read_instruction_tab(frame, state)
            if extra:
                body = f"{text}\n{extra}"
        if self._pool is not None:
            self._pool.instruction_attempted = True
            if len(re.sub(r"\s+", "", body)) >= 200:
                # вторая и следующие страницы (кнопка «Далее») дописываются, а не заменяют первую
                self._knowledge.save_instruction(self._pool, body, f"диалог «{title}»",
                                                 append=self._memory.instruction_pages > 0)
                self._memory.instruction_pages += 1
                self._memory.add(ActionType.CLICK, None, note="(авто)",
                                 result=f"📘 прочитана инструкция «{title[:60]}» — она в разделе ЗНАНИЯ")
        if failed:
            logger.warning("Инструкция не загрузилась (сообщение платформы) — продолжаю без неё")
        elif len(re.sub(r"\s+", "", body)) < 200:
            logger.warning("Инструкция открыта, но текста в ней не найдено (%d симв.)", len(body))
        await self._close_dialog(frame, state)
        return StepResult.ACTED

    async def _read_child_frames(self, frame: Frame) -> str:
        """Текст документов во вложенных iframe (инструкция бывает отдельной страницей)."""
        texts: list[str] = []
        for child in frame.child_frames:
            try:
                element = await child.frame_element()
                inside = await element.evaluate(
                    "(f) => !!f.closest('[role=\"dialog\"], [aria-modal=\"true\"], dialog[open], tui-dialog')"
                )
                if not inside:
                    continue
                await child.wait_for_load_state("load", timeout=10_000)
                texts.append(await child.evaluate("() => document.body ? document.body.innerText : ''"))
            except PlaywrightError as exc:
                logger.debug("Вложенный документ не прочитан: %s", _short(exc))
        return "\n".join(t for t in texts if t and t.strip())

    async def _read_instruction_tab(self, frame: Frame, state: PageState) -> str:
        """Инструкция без текста в диалоге: кнопка «открыть в новой вкладке» → читаем вкладку."""
        close_like = DIALOG_CLOSE_TEXTS + START_BUTTON_TEXTS
        candidates = [
            e for e in state.visible_elements
            if e.container == "dialog" and e.kind in (ElementKind.BUTTON, ElementKind.OTHER)
            and not _label_matches(e.text, close_like) and _INSTRUCTION_RE.search(e.text or "")
        ]
        if not candidates:
            return ""
        context = self._browser.context
        before = set(context.pages)
        await self._browser.click_element(frame, candidates[0])
        await asyncio.sleep(2.0)
        new_pages = [p for p in context.pages if p not in before]
        text = ""
        for page in new_pages:
            try:
                await page.wait_for_load_state("load", timeout=15_000)
                text = await page.evaluate("() => document.body ? document.body.innerText : ''")
                logger.info("Инструкция прочитана из вкладки %s (%d симв.)", page.url[:80], len(text))
            except PlaywrightError as exc:
                logger.info("Вкладку с инструкцией прочитать не удалось (%s)", _short(exc))
            finally:
                try:
                    await page.close()
                except PlaywrightError:
                    pass
        try:
            await self._browser.page.bring_to_front()
        except PlaywrightError:
            pass
        return text.strip()

    async def _close_dialog(self, frame: Frame, state: PageState) -> None:
        fresh = await DomParser(frame).parse(quiet=True)
        buttons = [e for e in fresh.visible_elements
                   if e.container == "dialog" and e.kind == ElementKind.BUTTON and not e.is_disabled]
        button = next((b for b in buttons if _label_matches(b.text, DIALOG_CLOSE_TEXTS)), None) \
            or next((b for b in buttons if _label_matches(b.text, START_BUTTON_TEXTS)), None)
        if button is not None:
            logger.info("Закрываю диалог кнопкой «%s»", button.label())
            await self._browser.click_element(frame, button)
        else:
            logger.info("Закрываю диалог клавишей Escape")
            try:
                await self._browser.page.keyboard.press("Escape")
            except PlaywrightError:
                pass
        await self._browser.wait_settle(frame)

    # ------------------------------------------------------------------
    # Инструкция и подсказки вида задания
    # ------------------------------------------------------------------

    async def _maybe_open_instruction(self, frame: Frame, state: PageState) -> bool:
        """Открыть «Подробную инструкцию» один раз для вида задания без сохранённой инструкции."""
        pool = self._pool
        if not READ_INSTRUCTIONS or pool is None or pool.has_instruction or pool.instruction_attempted:
            return False
        if state.dialog_open or state.loading:
            return False
        link = next((
            e for e in state.visible_elements
            if e.kind in (ElementKind.BUTTON, ElementKind.OTHER) and not e.occluded and not e.is_disabled
            and e.container == "" and _INSTRUCTION_RE.search(e.text or "") and len(e.text) <= 60
        ), None)
        pool.instruction_attempted = True
        if link is None:
            return False
        logger.info("📘 Открываю «%s», чтобы прочитать правила задания", link.label())
        context = self._browser.context
        before = set(context.pages)
        outcome = await self._browser.click_element(frame, link)
        if not outcome.ok:
            return False
        await self._browser.wait_settle(frame)
        await asyncio.sleep(0.5)
        # инструкция открылась в новой вкладке, а не диалогом
        for page in [p for p in context.pages if p not in before]:
            try:
                await page.wait_for_load_state("load", timeout=15_000)
                text = await page.evaluate("() => document.body ? document.body.innerText : ''")
                self._knowledge.save_instruction(pool, text, f"вкладка {page.url[:80]}")
            except PlaywrightError as exc:
                logger.info("Вкладку с инструкцией прочитать не удалось (%s)", _short(exc))
            finally:
                try:
                    await page.close()
                except PlaywrightError:
                    pass
            await self._browser.page.bring_to_front()
        return True

    async def _maybe_read_tooltips(self, frame: Frame, state: PageState) -> None:
        """Прочитать подсказки «?» у вариантов ответа (наведение мыши) — один раз для вида."""
        pool = self._pool
        if not READ_TOOLTIPS or pool is None or pool.tooltips_attempted or state.dialog_open or state.loading:
            return
        pool.tooltips_attempted = True
        try:
            triggers = await frame.evaluate(_JS_TOOLTIP_TRIGGERS)
        except PlaywrightError:
            return
        if not triggers:
            return
        by_uid = {e.uid: e for e in state.elements}
        tips: dict[str, str] = {}
        page = self._browser.page
        started = time.monotonic()
        for trig in triggers:
            if time.monotonic() - started > 25:
                break
            row = by_uid.get(str(trig.get("uid")))
            if row is None or row.aux:
                continue
            locator = frame.locator(f'[data-agent-tip="{trig["n"]}"]')
            try:
                await locator.scroll_into_view_if_needed(timeout=2_000)
                box = await locator.bounding_box(timeout=1_000)
                if not box:
                    continue
                await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=4)
                await asyncio.sleep(0.9)
                text = str(await frame.evaluate(_JS_HINT_TEXT) or "").strip()
                await page.mouse.move(5, 5, steps=2)
                await asyncio.sleep(0.2)
            except PlaywrightError:
                continue
            if text and normalize_text(text) != normalize_text(row.label()):
                tips[row.label()] = text
        if tips:
            self._knowledge.save_tooltips(pool, tips)

    # ------------------------------------------------------------------
    # Сверка цели (автокоррекция v3)
    # ------------------------------------------------------------------

    def _resolve_target(self, decision: LLMDecision, state: PageState) -> Optional[ParsedElement]:
        """Найти элемент, который имела в виду LLM.

        Индекс, подтверждённый текстом, приоритетнее; поиск по тексту учитывает тип
        элемента, а при дублях берёт ближайший к названному индексу и не запрещённый.
        Служебные элементы (плеер, карусель) и кнопки стоп-списка не выбираются никогда.
        """
        action = decision.action
        if action in (ActionType.SKIP, ActionType.WEB):
            return None
        kinds = _ACTION_KINDS[action]
        fallback = _FALLBACK_KINDS.get(action, set())
        by_index = state.by_index(decision.target_index)
        if by_index is not None and (is_denied_button(by_index) or by_index.aux):
            logger.warning("LLM указала недоступный элемент «%s» — игнорирую", by_index.label())
            by_index = None
        text = _clean_target_text(decision.target_text or "")

        def label(e: ParsedElement) -> str:
            return e.text or e.placeholder or e.caption

        # 1. Индекс и текст согласованы
        if by_index is not None and by_index.kind in kinds and (
            not text or _similarity(label(by_index), text) >= 0.9
        ):
            return by_index

        # 2. Поиск по тексту среди элементов подходящего типа
        if text:
            for allowed in (kinds, fallback):
                scored = [
                    (_similarity(label(e), text), e)
                    for e in state.elements if e.kind in allowed and not is_denied_button(e) and not e.aux
                ]
                good = [(s, e) for s, e in scored if s >= _MATCH_THRESHOLD]
                if not good:
                    continue
                best = max(s for s, _ in good)
                top = [e for s, e in good if s >= best - 1e-6]
                ref = decision.target_index if decision.target_index is not None else 0
                top.sort(key=lambda e: (self._memory.is_forbidden(action, e.key), abs(e.index - ref)))
                chosen = top[0]
                if len(top) > 1:
                    logger.info(
                        "Автокоррекция: %d элементов с текстом «%s», выбран [%d] ⟨%s⟩",
                        len(top), text, chosen.index, " › ".join(chosen.path) or "корень",
                    )
                if by_index is None or chosen.index != by_index.index:
                    logger.info(
                        "Автокоррекция: idx %s → %d (по тексту «%s», сходство %.2f)",
                        decision.target_index, chosen.index, text, best,
                    )
                return chosen

        # 3. Текст не найден (перефразирован) — доверяем индексу, если тип подходит
        if by_index is not None and (by_index.kind in kinds or by_index.kind in fallback):
            if text:
                logger.warning(
                    "Текст «%s» не найден — использую индекс %d («%s»)", text, by_index.index, by_index.label(),
                )
            return by_index

        if action not in (ActionType.SUBMIT, ActionType.SCROLL):
            logger.warning("Цель не найдена: idx=%s text=%r", decision.target_index, decision.target_text)
        return None

    # ------------------------------------------------------------------
    # Пакет действий
    # ------------------------------------------------------------------

    @staticmethod
    def _order_batch(steps: list[LLMDecision]) -> list[LLMDecision]:
        """Порядок пакета: submit — только последним; после open/web/scroll действия
        отбрасываются (страница меняется, нужен новый взгляд модели); лимит длины."""
        if not BATCH_ACTIONS:
            return steps[:1]
        body = [s for s in steps if s.action not in (ActionType.SKIP, ActionType.SUBMIT)]
        submit = next((s for s in steps if s.action == ActionType.SUBMIT), None)
        if not body and submit is None:
            return steps[:1]                              # только skip
        ordered: list[LLMDecision] = []
        for step in body:
            ordered.append(step)
            if step.action in PAGE_CHANGING_ACTIONS:
                return ordered[:MAX_BATCH_ACTIONS]
        if submit is not None and len(ordered) < MAX_BATCH_ACTIONS:
            ordered.append(submit)                        # не влез в лимит — отправит следующим шагом
        return ordered[:MAX_BATCH_ACTIONS]

    @staticmethod
    def _step_label(step: LLMDecision) -> str:
        what = step.target_text or step.type_text or step.query or ""
        return f"{step.action.value}" + (f" «{what[:40]}»" if what else "")

    async def _run_batch(self, frame: Frame, decision: LLMDecision, state: PageState) -> None:
        """Выполнить действия по очереди. Перед каждым следующим: новый снимок, проверка
        эффекта предыдущего и «ничего неожиданного» (задание то же, не открылся диалог, нет
        новых ошибок, не появились и не исчезли поля). Иначе пакет останавливается, и модель
        на следующем шаге видит новое состояние и заметку, что осталось не выполненным."""
        mem = self._memory
        all_steps = decision.steps()
        steps = self._order_batch(all_steps)
        dropped = [s for s in all_steps if s.action != ActionType.SKIP and not any(s is t for t in steps)]
        if dropped:
            mem.batch_notes.append(
                "Не выполнено (после open/web/scroll нужен новый взгляд на страницу, submit — только "
                "последним): " + ", ".join(self._step_label(s) for s in dropped)
            )

        # цели — по тому снимку, который видела модель (номера из него)
        planned: list[tuple[LLMDecision, Optional[ParsedElement]]] = []
        for i, step in enumerate(steps):
            target = self._resolve_target(step, state)
            if i > 0 and target is None and step.action in (ActionType.CLICK, ActionType.OPEN, ActionType.TYPE):
                note = f"Действие {self._step_label(step)} не выполнено: элемента [{step.target_index}] нет на странице."
                rest = steps[i + 1:]
                if rest:
                    note += " Следующие действия пакета тоже не выполнены: " + ", ".join(self._step_label(s) for s in rest)
                mem.batch_notes.append(note)
                break
            planned.append((step, target))
        if len(planned) > 1:
            logger.info("Пакет из %d действий: %s", len(planned), " → ".join(self._step_label(s) for s, _ in planned))

        current = state
        last_status, last_target = "", None
        for i, (step, target) in enumerate(planned):
            if i > 0:
                await self._browser.wait_settle(frame)
                fresh = await DomParser(frame).parse(quiet=True)
                mem.verify(fresh)
                reason = self._batch_break_reason(current, fresh, last_status, last_target)
                if reason is None and target is not None:
                    remapped = fresh.by_key(target.key)
                    if remapped is None:
                        reason = f"элемент «{target.label()}» исчез со страницы"
                    target = remapped
                if reason is not None:
                    rest = ", ".join(self._step_label(s) for s, _ in planned[i:])
                    logger.info("Пакет остановлен: %s. Не выполнено: %s", reason, rest)
                    mem.batch_notes.append(f"Пакет действий остановлен: {reason}. Не выполнено: {rest}. "
                                           "Посмотри на страницу заново.")
                    return
                current = fresh
            last_status = await self._execute(frame, step, target, current)
            last_target = target
            if last_status in ("done", "fail"):
                return

    def _batch_break_reason(
        self, before: PageState, after: PageState, status: str, target: Optional[ParsedElement],
    ) -> Optional[str]:
        """Почему нельзя продолжать пакет после очередного действия (None — можно)."""
        if status == "continue":
            result = self._memory.history[-1].result if self._memory.history else ""
            toggled_off = (target is not None and target.is_selected and target.choice_type == "checkbox"
                           and result.startswith("⚠ выбор снят"))
            if not (result.startswith("✓") or toggled_off):
                return f"«{target.label() if target else '?'}» → {result}"
        if not TaskIdentity.of(before).same_task(TaskIdentity.of(after), submitted=False):
            return "задание сменилось"
        if after.dialog_open and not before.dialog_open:
            return "открылся диалог"
        fresh_errors = set(after.notice_texts("error", "warning")) - set(before.notice_texts("error", "warning"))
        if fresh_errors:
            return "сообщение платформы: " + " | ".join(sorted(fresh_errors))[:200]

        def form_keys(state: PageState) -> dict[str, str]:
            return {e.key: e.label() for e in state.elements
                    if not e.aux and e.container not in ("popup", "dialog", "toast")}

        # Новое поле/вариант может требовать ответа — модель должна его увидеть до отправки.
        # Исчезнувшие элементы новых обязанностей не создают (если исчезла цель следующего
        # действия — пакет остановится при поиске цели).
        was, now = form_keys(before), form_keys(after)
        added = [now[k] for k in now if k not in was]
        if added:
            return "на странице появились " + ", ".join(f"«{x}»" for x in added[:3])
        return None

    # ------------------------------------------------------------------
    # Выполнение действия
    # ------------------------------------------------------------------

    async def _execute(
        self,
        frame: Frame,
        decision: LLMDecision,
        target: Optional[ParsedElement],
        state: PageState,
    ) -> str:
        """Выполнить одно действие. Возвращает исход для пакета действий:
        continue — выполнено, эффект проверяется следующим снимком; noop — делать нечего
        (вариант уже выбран); done — действие завершает пакет (submit, web, open, scroll,
        skip); fail — не выполнено."""
        action = decision.action
        mem = self._memory
        logger.info(
            "ВЫПОЛНЯЕМ: %s → %s conf=%.2f",
            action.value,
            f"[{target.index}] «{target.label()}»" if target else (decision.query or "—"),
            decision.confidence,
        )

        if action == ActionType.SKIP:
            mem.consecutive_skips += 1
            mem.add(action, None, result="ожидание")
            await asyncio.sleep(ACTION_WAIT * 2)
            return "done"
        mem.consecutive_skips = 0

        if action == ActionType.WEB:
            await self._do_web(decision.query or decision.type_text or decision.target_text or "")
            return "done"

        if action == ActionType.SUBMIT:
            await self._do_submit(frame, state, target)
            return "done"

        if action == ActionType.SCROLL:
            direction = decision.scroll_direction or "down"
            key = target.key if target else f"__page__|{direction}"
            if mem.is_forbidden(action, key):
                mem.repeated_forbidden += 1
                mem.add(action, target, result="⛔ прокрутка уже ничего не меняла")
                return "fail"
            outcome = await self._browser.scroll(frame, target, direction)
            if outcome.ok:
                mem.expect(action, target, state, note=direction, key=key)
            else:
                mem.add(action, target, result=f"✗ {outcome.detail}")
            return "done"

        if target is None:
            mem.invalid_targets += 1
            mem.add(action, None, result=(
                f"✗ элемент не найден (номер={decision.target_index}, текст=«{decision.target_text or ''}»)"
            ))
            return "fail"
        mem.invalid_targets = 0

        if is_denied_button(target):
            mem.add(action, target, result="⛔ кнопка из стоп-списка — агент её не нажимает")
            return "fail"
        if target.is_disabled:
            mem.add(action, target, result="✗ элемент неактивен — нажать нельзя")
            mem.mark_no_effect(action, target.key)
            return "fail"
        if mem.is_forbidden(action, target.key):
            mem.repeated_forbidden += 1
            mem.add(action, target, result="⛔ уже пробовали без эффекта — выбери другое действие")
            return "fail"
        mem.repeated_forbidden = 0

        # Ссылка на внешний сайт: переход увёл бы фрейм задания со страницы (задание
        # потерялось бы) — открываем её во вкладке поиска и показываем модели как web
        if action == ActionType.CLICK and target.href and self._is_external(target.href, frame):
            await self._do_web(target.href, label=target.label())
            return "done"

        if action == ActionType.OPEN:
            if (target.kind in (ElementKind.FOLDER, ElementKind.DROPDOWN)
                    and target.folder_state == FolderState.OPEN
                    and mem.insist[(action, target.key)] == 0):
                # Первый раз не кликаем (клик свернул бы папку), а подсказываем. Если модель
                # настаивает — эвристика состояния могла ошибиться, выполняем.
                mem.insist[(action, target.key)] += 1
                mem.add(action, target, result="уже раскрыта — её содержимое ниже, с бо́льшим отступом")
                return "done"
            mem.open_attempts[target.key] += 1
            outcome = await self._browser.click_element(frame, target, prefer_toggle=True)
        elif action == ActionType.CLICK:
            if (target.is_selected and target.kind in (ElementKind.OPTION, ElementKind.FOLDER)
                    and target.choice_type != "checkbox" and mem.insist[(action, target.key)] == 0):
                # Checkbox можно снять сознательно, поэтому для него клик выполняется.
                mem.insist[(action, target.key)] += 1
                mem.add(action, target, result="уже ✓ВЫБРАН — повторный клик снял бы выбор; если всё готово — submit")
                return "noop"
            if target.kind == ElementKind.BUTTON and self._is_finish_button(target):
                await self._do_submit(frame, state, target)
                return "done"
            outcome = await self._browser.click_element(frame, target)
        elif action == ActionType.TYPE:
            if decision.type_text is None:
                mem.add(action, target, result="✗ пустой type_text")
                return "fail"
            outcome = await self._browser.type_into(frame, target, decision.type_text)
        else:
            logger.error("Неизвестное действие: %s", action.value)
            return "fail"

        if outcome.stale:
            mem.add(action, target, result="⚠ элемент перерисовался до клика — повтор на свежем снимке")
            return "fail"
        if not outcome.ok:
            mem.add(action, target, result=f"✗ {outcome.detail}")
            mem.mark_no_effect(action, target.key)
            return "fail"
        note = f"«{decision.type_text[:80]}»" if action == ActionType.TYPE and decision.type_text else ""
        if outcome.method == "js":
            note = (note + " (js-клик)").strip()
        mem.expect(action, target, state, typed=decision.type_text, note=note)
        # open меняет страницу (раскрытый список/ветка) — после него нужен новый взгляд модели
        return "done" if action == ActionType.OPEN else "continue"

    @staticmethod
    def _is_external(href: str, frame: Frame) -> bool:
        try:
            return urlsplit(href).netloc.lower() != urlsplit(frame.url).netloc.lower()
        except ValueError:
            return True

    async def _do_web(self, query: str, *, label: str = "") -> None:
        mem = self._memory
        query = (query or "").strip()
        if not WEB_RESEARCH:
            mem.add(ActionType.WEB, None, note=f"«{query[:80]}»",
                    result="✗ поиск в интернете отключён (WEB_RESEARCH=false)")
            return
        if not query:
            mem.add(ActionType.WEB, None, result="✗ пустой query — укажи запрос или адрес")
            return
        key = normalize_text(query)
        if sum(mem.web_queries.values()) >= MAX_WEB_PER_TASK:
            mem.add(ActionType.WEB, None, note=f"«{query[:80]}»", result="⛔ лимит запросов на задание исчерпан")
            return
        if mem.web_queries[key] >= 2:
            mem.repeated_forbidden += 1
            mem.add(ActionType.WEB, None, note=f"«{query[:80]}»",
                    result="⛔ этот запрос уже выполнялся — его результаты в разделе ВЕБ-ПОИСК")
            return
        mem.web_queries[key] += 1
        result = await self._web.run(query)
        mem.web_results.append(result)
        note = f"«{label or query[:100]}»"
        if result.error:
            mem.add(ActionType.WEB, None, note=note, result=f"✗ {result.error}")
        else:
            mem.add(ActionType.WEB, None, note=note,
                    result=f"✓ прочитано: {result.url[:160]} (результат #{len(mem.web_results)} в ВЕБ-ПОИСК)")

    # ------------------------------------------------------------------
    # Отправка ответа
    # ------------------------------------------------------------------

    @staticmethod
    def _is_finish_button(el: ParsedElement, *, strict: bool = True) -> bool:
        label = normalize_text(el.text)
        if not label or any(deny in label for deny in FINISH_DENY_SUBSTRINGS):
            return False
        if not strict:
            return True
        return any(label == t or label.startswith(t + " ") for t in FINISH_BUTTON_TEXTS)

    def _find_finish_button(self, state: PageState) -> Optional[ParsedElement]:
        buttons = [e for e in state.elements
                   if e.kind == ElementKind.BUTTON and not e.aux and self._is_finish_button(e)]
        if not buttons:
            return None

        def rank(e: ParsedElement) -> tuple:
            label = normalize_text(e.text)
            priority = next(
                (i for i, t in enumerate(FINISH_BUTTON_TEXTS) if label == t or label.startswith(t + " ")), 99,
            )
            return (e.is_disabled, e.container != "dialog", priority, e.index)

        return min(buttons, key=rank)

    @staticmethod
    def _describe_answer(state: PageState) -> str:
        """Текущий ответ в форме: отмеченные варианты, значения полей и списков."""
        parts: list[str] = []
        for e in state.visible_elements:
            if e.container in ("dialog", "popup", "toast"):
                continue
            if e.is_selected and e.kind in (ElementKind.OPTION, ElementKind.FOLDER):
                parts.append(f"«{e.label()}»")
            elif e.kind in (ElementKind.INPUT, ElementKind.DROPDOWN) and e.value:
                parts.append(f"{e.text or e.placeholder or e.caption or 'поле'} = «{e.value[:120]}»")
        return ", ".join(parts) or "(ничего не выбрано)"

    async def _do_submit(self, frame: Frame, state: PageState, target: Optional[ParsedElement]) -> None:
        """Нажать кнопку отправки и дождаться исхода: следующее задание или ответ платформы.

        v2 сбрасывал память сразу после клика, даже если форма не прошла валидацию.
        v4: «Неверный ответ» (тренировка) — отмечается в памяти, превращается в урок
        для этого вида заданий, модель исправляет ответ на следующем шаге.
        """
        mem = self._memory
        # «Прослушайте звонок до конца»: запись доигрывается ДО нажатия
        if AUDIO_PLAY_TO_END and any(not a.ended for a in state.audios):
            fresh = await DomParser(frame).parse(quiet=True)
            await self._media.wait_finished(frame, fresh)
            state = await DomParser(frame).parse(quiet=True)
            target = state.by_key(target.key) if target is not None else None

        answer = self._describe_answer(state)
        if target is not None and target.kind == ElementKind.BUTTON and self._is_finish_button(target, strict=False):
            button: Optional[ParsedElement] = target
        else:
            button = self._find_finish_button(state)

        errors_before = set(state.notice_texts("error", "warning"))
        if button is None:
            logger.info("SUBMIT: кнопка в снимке не найдена — поиск по тексту")
            if not await self._browser.click_by_text(frame, FINISH_BUTTON_TEXTS, deny=FINISH_DENY_SUBSTRINGS):
                mem.add(ActionType.SUBMIT, None, result="✗ кнопка отправки не найдена")
                mem.submit_failures += 1
                return
        elif button.is_disabled:
            mem.add(ActionType.SUBMIT, button,
                    result="✗ кнопка неактивна — ответ ещё не заполнен (проверь ✓ВЫБРАН и поля)")
            mem.submit_failures += 1
            return
        else:
            outcome = await self._browser.click_element(frame, button)
            if not outcome.ok:
                mem.add(ActionType.SUBMIT, button, result=f"✗ {outcome.detail}")
                mem.submit_failures += 1
                return

        mem.submits += 1
        logger.info("SUBMIT: ответ %s", answer)
        changed, after = await self._wait_task_change(errors_before)
        if changed:
            self._submitted = True
            self._tasks_done += 1
            logger.info("✅ ЗАДАНИЕ ОТПРАВЛЕНО (всего: %d)", self._tasks_done)
            mem.add(ActionType.SUBMIT, button, result="✓ отправлено, задание сменилось")
            return

        mem.submit_failures += 1
        errors = [t for t in (after.notice_texts("error", "warning") if after else []) if t not in errors_before]
        hints = after.notice_texts("hint") if after else []
        if any(_WRONG_RE.search(t) for t in errors):
            self._wrong_total += 1
            mem.wrong_answers.append(answer)
            mem.feedback = errors + [f"Подсказка платформы: {h}" for h in hints]
            logger.warning("❌ Платформа: неверный ответ (%s)%s", answer,
                           f"; подсказка: {hints[0][:160]}" if hints else "")
            if self._pool is not None:
                lesson = f"Задание «{state.task_preview[:70]}»: ответ {answer} — неверно."
                if hints:
                    lesson += " Подсказка платформы: " + " ".join(hints)[:600]
                self._knowledge.add_lesson(self._pool, lesson)
            mem.add(ActionType.SUBMIT, button, result="✗ платформа: НЕВЕРНЫЙ ОТВЕТ — прочитай подсказку и исправь ответ")
            return

        result = "✗ задание не сменилось"
        if errors:
            result += "; сообщения: " + " | ".join(errors)
            mem.feedback = errors
        dialog = [e.label() for e in (after.visible_elements if after else []) if e.container == "dialog"][:4]
        if dialog:
            result += "; открыт диалог: " + ", ".join(f"«{d}»" for d in dialog)
        logger.warning("SUBMIT: %s", result)
        mem.add(ActionType.SUBMIT, button, result=result)

    async def _wait_task_change(self, errors_before: set[str]) -> tuple[bool, Optional[PageState]]:
        """Поллинг до SUBMIT_WAIT: сменилось задание (True) или платформа ответила
        сообщением об ошибке на том же задании (False, снимок с сообщением)."""
        identity = self._identity
        started = time.monotonic()
        deadline = started + SUBMIT_WAIT
        hard_deadline = started + SUBMIT_WAIT + 30        # «вечный» лоадер не вешает агента
        last: Optional[PageState] = None
        missing = 0
        while time.monotonic() < min(deadline, hard_deadline):
            await asyncio.sleep(0.6)
            frame = await self._browser.find_target_frame()
            if frame is None:
                missing += 1
                if missing >= 2:          # фрейм задания ушёл — задание принято
                    return True, None
                continue
            try:
                last = await DomParser(frame).parse(quiet=True)
            except PlaywrightError:
                continue                  # фрейм перезагружается
            if last.loading:
                deadline = max(deadline, time.monotonic() + 1.0)   # идёт отправка — ждём дольше
                continue
            if self._is_orders_list(last):
                return True, last
            if identity is None or not identity.same_task(TaskIdentity.of(last), submitted=True):
                return True, last
            fresh = [t for t in last.notice_texts("error", "warning") if t not in errors_before]
            if fresh:
                await asyncio.sleep(0.8)  # подсказка обычно появляется вместе с «Неверный ответ»
                try:
                    last = await DomParser(frame).parse(quiet=True)
                except PlaywrightError:
                    pass
                return False, last
        return False, last

    # ------------------------------------------------------------------
    # Локальный флоу, бюджет, капча
    # ------------------------------------------------------------------

    async def _maybe_click_start(self, frame: Frame, state: PageState) -> bool:
        """«Начать/ОК/Понятно» — без LLM, но только на заставке или в диалоге.

        v2 искал эти слова на КАЖДОМ шаге до LLM, подстрокой и по любому элементу:
        вариант ответа «Хорошо» или кнопка «Далее» до ответа нажимались бесконечно.
        """
        visible = state.visible_elements
        enabled_buttons = [
            e for e in visible
            if e.kind == ElementKind.BUTTON and not e.is_disabled and not e.occluded and not is_denied_button(e)
        ]
        dialog_buttons = [b for b in enabled_buttons if b.container == "dialog"]
        working = [
            e for e in visible
            if e.kind in (ElementKind.OPTION, ElementKind.FOLDER, ElementKind.INPUT, ElementKind.DROPDOWN)
            and not e.is_disabled and not e.occluded and e.container not in ("dialog", "popup")
        ]
        if working and not dialog_buttons:
            return False

        def start_like(b: ParsedElement) -> bool:
            return _label_matches(b.text, START_BUTTON_TEXTS)

        pool = dialog_buttons or enabled_buttons
        if not dialog_buttons and any(not start_like(b) for b in pool):
            return False       # на экране есть и другие кнопки (ответы «Да/Нет»?) — решает LLM
        button: Optional[ParsedElement] = None
        for text in START_BUTTON_TEXTS:
            button = next(
                (b for b in pool if normalize_text(b.text) == text or normalize_text(b.text).startswith(text + " ")),
                None,
            )
            if button is not None:
                break
        if button is None:
            return False
        if self._memory.start_clicks[state.state_hash] >= 2:
            logger.warning("Кнопка «%s» уже нажималась без эффекта — решение передаю LLM", button.label())
            return False

        self._memory.start_clicks[state.state_hash] += 1
        logger.info("Локальный флоу: нажимаю «%s»", button.label())
        outcome = await self._browser.click_element(frame, button)
        self._memory.add(ActionType.CLICK, button, note="(авто)",
                         result="нажата" if outcome.ok else f"✗ {outcome.detail}")
        await self._browser.wait_settle(frame)
        return outcome.ok

    async def _handle_budget_exhausted(self, frame: Frame, state: PageState) -> StepResult:
        selected = TaskMemory.selected_options(state)
        if selected and self._memory.submit_failures == 0:
            logger.warning(
                "Бюджет задания исчерпан — отправляю текущий выбор: %s",
                ", ".join(f"«{e.label()}»" for e in selected),
            )
            await self._do_submit(frame, state, None)
            return StepResult.ACTED
        logger.error(
            "Бюджет задания (%d шагов) исчерпан, ответ не найден. Нужна помощь человека: "
            "жду смены задания до %.0f с", MAX_STEPS_PER_TASK, MAX_IDLE_SECONDS,
        )
        identity = self._identity
        deadline = time.monotonic() + MAX_IDLE_SECONDS
        while time.monotonic() < deadline:
            await asyncio.sleep(5.0)
            if self._browser.is_closed():
                return StepResult.STOP
            current = await self._browser.find_target_frame()
            if current is None:
                continue
            try:
                snapshot = await DomParser(current).parse(quiet=True)
            except PlaywrightError:
                continue
            if identity is None or not identity.same_task(TaskIdentity.of(snapshot), submitted=True):
                logger.info("Задание сменилось — продолжаю")
                return StepResult.IDLE
        return StepResult.STOP

    async def _wait_captcha_solved(self, poll: float = 5.0) -> None:
        """Ждём ручного решения капчи. v2 ждал бесконечно (while True) и проверял
        слово «капча» в тексте страницы — задание про капчу вешало агента навсегда."""
        logger.warning("КАПЧА! Решите её вручную в окне браузера (жду до %.0f с)…", CAPTCHA_TIMEOUT)
        deadline = time.monotonic() + CAPTCHA_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(poll)
            if self._browser.is_closed():
                return
            if await self._browser.page_has_captcha():
                continue
            frame = await self._browser.find_target_frame()
            if frame is None:
                return                    # фрейм грузится — основной цикл подождёт сам
            try:
                state = await DomParser(frame).parse(quiet=True)
            except PlaywrightError:
                continue
            if not state.has_captcha:
                logger.info("Капча решена, продолжаем")
                return
        logger.error("Капча не решена за %.0f с — 5 минут не реагирую на её признаки", CAPTCHA_TIMEOUT)
        self._captcha_suppressed_until = time.monotonic() + 300

    def _log_idle(self, message: str) -> None:
        if time.monotonic() - self._last_idle_log > 30:
            self._last_idle_log = time.monotonic()
            logger.warning(message)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(Agent().run())
