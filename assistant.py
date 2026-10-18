"""assistant.py — режим помощника (python main.py --assist).

Задания решает человек. В текстовых заданиях (без фото, аудио и полей для ввода) модель
подсказывает: вариант, который она выбрала бы, обводится рамкой с меткой «ИИ», в панели в углу
окна — ответы и короткое объяснение. Помощник ничего не нажимает и не отправляет. Задания с фото,
записью или поиском (поле для ссылки) — на человеке: модель для них не вызывается.

Окно инструкции, которое человек открыл сам, помощник читает в базу знаний (как агент) —
подсказки дальше учитывают правила проекта.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Optional

from playwright.async_api import Error as PlaywrightError, Frame

from agent import Agent, TaskIdentity, _INSTRUCTION_RE, _is_task_url
from browser_controller import BrowserController, is_connection_lost
from config import ASSIST_MODEL, GEMINI_API_KEY, GEMINI_MODEL, MAX_STEPS_PER_TASK
from dom_parser import ASSIST_PANEL_ID, DomParser, _plain_text
from knowledge import KnowledgeBase
from models import ActionType, DecisionContext, ElementKind, LLMDecision, PageState
from openrouter_connector import LLMConnector

logger = logging.getLogger("twork.assistant")

# Подсветка выбранных моделью вариантов и панель подсказки (только вид: ответы не меняются)
_JS_DRAW = r"""
(args) => {
    const PANEL = args.panelId;
    if (!document.getElementById('ai-assist-style')) {
        const st = document.createElement('style');
        st.id = 'ai-assist-style';
        st.textContent = `
          [data-ai-hint] { outline: 3px solid #16a34a !important; outline-offset: 3px; border-radius: 6px; }
          [data-ai-hint]::after { content: "  ← ИИ"; color: #16a34a; font-weight: 700; }
          #${PANEL} { position: fixed; top: 12px; right: 12px; width: 380px; max-height: 55vh; overflow: auto;
            z-index: 2147483647; background: rgba(255,255,255,.97); color: #111; border: 2px solid #16a34a;
            border-radius: 10px; padding: 10px 12px; font: 13px/1.45 Arial, sans-serif;
            box-shadow: 0 4px 18px rgba(0,0,0,.18); }
          #${PANEL}.ai-muted { border-color: #9ca3af; }
          #${PANEL}.ai-small > .ai-body { display: none; }
          #${PANEL} .ai-head { font-weight: 700; margin-bottom: 6px; cursor: pointer; }
          #${PANEL} .ai-line { margin: 3px 0; white-space: pre-wrap; }`;
        document.head.appendChild(st);
    }
    document.querySelectorAll('[data-ai-hint]').forEach((e) => e.removeAttribute('data-ai-hint'));
    for (const uid of args.hints) {
        const el = document.querySelector(`[data-agent-id="${uid}"]`);
        if (el) el.setAttribute('data-ai-hint', '1');
    }
    let panel = document.getElementById(PANEL);
    if (!panel) {
        panel = document.createElement('div');
        panel.id = PANEL;
        const head = document.createElement('div');
        head.className = 'ai-head';
        head.title = 'Свернуть / развернуть';
        head.addEventListener('click', () => panel.classList.toggle('ai-small'));
        const body = document.createElement('div');
        body.className = 'ai-body';
        panel.append(head, body);
        document.body.appendChild(panel);
    }
    if (panel.dataset.sig === args.sig) return true;           // то же содержимое — не перерисовываем
    panel.dataset.sig = args.sig;
    panel.classList.toggle('ai-muted', !!args.muted);
    panel.querySelector('.ai-head').textContent = args.title;
    const body = panel.querySelector('.ai-body');
    body.replaceChildren(...args.lines.map((t) => {
        const d = document.createElement('div'); d.className = 'ai-line'; d.textContent = t; return d;
    }));
    return true;
}
"""

_JS_CLEAR = r"""
(panelId) => {
    document.querySelectorAll('[data-ai-hint]').forEach((e) => e.removeAttribute('data-ai-hint'));
    const p = document.getElementById(panelId);
    if (p) p.remove();
}
"""

_CHOICE_ACTIONS = (ActionType.CLICK,)


def not_text_reason(state: PageState) -> Optional[str]:
    """Почему задание не текстовое (подсказки нет), или None."""
    if state.images:
        return "на странице фото"
    if state.audios:
        return "в задании запись"
    fields = [e for e in state.visible_elements
              if e.kind == ElementKind.INPUT and e.container == "" and not e.is_disabled]
    if fields:
        return "ответ — в поле для ввода (поиск, ссылка, текст)"
    if not any(e.kind in (ElementKind.OPTION, ElementKind.DROPDOWN) and e.container == ""
               for e in state.visible_elements):
        return "вариантов ответа нет"
    return None


class Assistant:
    """Подсказки в текстовых заданиях; сам ничего не нажимает."""

    def __init__(self, *, browser: Optional[BrowserController] = None, llm: Optional[LLMConnector] = None,
                 knowledge: Optional[KnowledgeBase] = None, interval: float = 1.0) -> None:
        self._browser = browser or BrowserController(headless=False)
        # ключ Google AI Studio — напрямую в Gemini (баланс OpenRouter не тратится), иначе OpenRouter
        self._llm = llm or (LLMConnector(provider="gemini") if GEMINI_API_KEY else LLMConnector())
        self._model = GEMINI_MODEL if getattr(self._llm, "provider", "") == "gemini" else ASSIST_MODEL
        self._knowledge = knowledge or KnowledgeBase()
        self._interval = interval
        self._identity: Optional[TaskIdentity] = None
        self._form: frozenset[str] = frozenset()
        self._pending: Optional[asyncio.Task] = None
        self._hint: Optional[tuple[list[str], LLMDecision]] = None   # ключи элементов, решение модели
        self._asked = 0
        self._briefing: dict[str, asyncio.Task] = {}   # вид задания → выжимка инструкции в работе
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        logger.info("=" * 60)
        logger.info("РЕЖИМ ПОМОЩНИКА. Решаете вы; в текстовых заданиях модель (%s) подсвечивает ответ «← ИИ» "
                    "и объясняет его в панели справа вверху. Помощник ничего не нажимает. Задания с фото, "
                    "записью и поиском — ваши, подсказки там нет.", self._model)
        logger.info("Закончить: закройте окно браузера или нажмите Ctrl+C в терминале.")
        logger.info("=" * 60)
        async with self._browser:
            while not self._stop.is_set():
                if self._browser.is_closed():
                    logger.info("Окно браузера закрыто — помощник завершает работу")
                    break
                try:
                    await self.tick()
                except Exception as exc:  # noqa: BLE001 — сбой одного такта не прерывает помощника
                    if self._browser.is_closed() or is_connection_lost(exc):
                        break
                    logger.debug("Такт помощника пропущен: %s", exc)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
                except asyncio.TimeoutError:
                    pass
            if self._pending is not None:
                self._pending.cancel()
        logger.info("Помощник: подсказок за запуск — %d", self._asked)

    # ------------------------------------------------------------------

    async def tick(self) -> str:
        """Один такт: что на странице и что показать. Возвращает состояние (для лога и тестов)."""
        browser = self._browser
        active = await browser.visible_page()
        if active is not None and active is not browser.page:
            browser.adopt_page(active)
        frame = await browser.find_target_frame()
        if frame is None:
            return "нет задания"
        state = await DomParser(frame).parse(quiet=True)
        if (Agent._is_orders_list(state) or Agent._is_order_done(state)
                or (frame.parent_frame is None and not _is_task_url(state.frame_url))):
            await self._clear(frame)
            self._forget()
            return "не страница задания"

        if state.dialog_open:
            self._read_instruction(state)
            return "открыто окно"

        identity = TaskIdentity.of(state)
        form = frozenset(e.key for e in state.visible_elements
                         if e.kind in (ElementKind.OPTION, ElementKind.DROPDOWN) and e.container == "")
        # «не отправлено»: ваши клики и открывшиеся вопросы — то же задание; новое задание — новый текст
        if self._identity is None or not self._identity.same_task(identity, submitted=False):
            self._forget()
        self._identity = identity

        reason = not_text_reason(state)
        if reason:
            self._form = form
            title, lines = self._how_to(state, reason)
            await self._draw(frame, state, [], title, lines, muted=True)
            return "не текстовое"

        if form != self._form:                               # новое задание или ответ открыл вопросы
            self._form = form
            self._hint = None
            if self._pending is not None:
                self._pending.cancel()
            self._pending = asyncio.create_task(self._ask(state))
        if self._pending is not None and self._pending.done():
            task, self._pending = self._pending, None
            try:
                self._hint = task.result()
            except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001 — модель недоступна и т.п.
                logger.warning("Подсказка не получена: %s", exc)
                self._hint = ([], LLMDecision.skip(f"подсказка не получена: {exc}"))
        if self._hint is None:
            await self._draw(frame, state, [], "ИИ думает…", ["Смотрю задание."], muted=True)
            return "думает"
        keys, decision = self._hint
        uids = [el.uid for el in (state.by_key(k) for k in keys) if el is not None]
        await self._draw(frame, state, uids, *self._panel(state, keys, decision))
        return "подсказка"

    async def _ask(self, state: PageState) -> tuple[list[str], LLMDecision]:
        pool = self._knowledge.for_state(state)
        context = DecisionContext(knowledge=self._knowledge.prompt_text(pool), model=self._model,
                                  step_in_task=1, steps_left=MAX_STEPS_PER_TASK)
        started = time.monotonic()
        decision = await self._llm.decide(state, context)
        self._asked += 1
        keys: list[str] = []
        for step in decision.steps():
            if step.action not in _CHOICE_ACTIONS:
                continue
            el = self._target(state, step)
            if el is not None and el.kind in (ElementKind.OPTION, ElementKind.DROPDOWN) and el.key not in keys:
                keys.append(el.key)
        logger.info("Подсказка за %.1f с: %s", time.monotonic() - started,
                    ", ".join(f"«{state.by_key(k).label()}»" for k in keys) or "—")
        return keys, decision

    @staticmethod
    def _target(state: PageState, step: LLMDecision):
        """Элемент, который выбрала модель: по номеру, а если текст не совпал — по тексту."""
        i = step.target_index
        el = state.elements[i] if i is not None and 0 <= i < len(state.elements) else None
        if el is not None and step.target_text and el.label() != step.target_text and el.text != step.target_text:
            el = next((e for e in state.elements if e.text == step.target_text), el)
        return el

    def _panel(self, state: PageState, keys: list[str], decision: LLMDecision) -> tuple[str, list[str]]:
        labels = [state.by_key(k).label() for k in keys if state.by_key(k) is not None]
        if not labels:
            return "ИИ: без подсказки", [decision.reasoning or decision.observation or "Модель не выбрала ответ."]
        lines = [f"Ответ: {'; '.join(f'«{x}»' for x in labels)}"]
        why = decision.reasoning or decision.plan or decision.observation
        if why:
            lines.append(f"Почему: {why}")
        lines.append(f"Уверенность: {round(decision.confidence * 100)}%"
                     + (" — проверьте сами" if decision.confidence < 0.7 else ""))
        return f"ИИ ({self._model.split('/')[-1]})", lines

    def _how_to(self, state: PageState, reason: str) -> tuple[str, list[str]]:
        """Задание с фото или поиском — решает человек; панель — выжимка инструкции «как решать»
        (делается один раз на вид заданий и хранится в knowledge/)."""
        head = f"Ответ — ваш: {reason}."
        pool = self._knowledge.for_state(state)
        if pool is not None and pool.brief:
            return "Задание ваше — как решать", [head, *pool.brief.splitlines()]
        if pool is None or not pool.has_instruction:
            return "Задание ваше", [head, "Откройте «Инструкцию» проекта — прочитаю правила и коротко покажу, "
                                          "как решать."]
        task = self._briefing.get(pool.key)
        brief = getattr(self._llm, "brief_instruction", None)
        if task is None and brief is not None:
            self._briefing[pool.key] = asyncio.create_task(brief(pool.instruction, self._model))
        elif task is not None and task.done():
            text = None if task.cancelled() or task.exception() else task.result()
            if text:
                self._knowledge.save_brief(pool, text)
                logger.info("📘 Выжимка инструкции «%s» сохранена", pool.title[:60])
                return "Задание ваше — как решать", [head, *pool.brief.splitlines()]
            # не повторяем каждую секунду: до следующего запуска помощника — без выжимки
            return "Задание ваше", [head, "Выжимку инструкции сделать не удалось (модель не ответила)."]
        return "Задание ваше", [head, "Читаю инструкцию проекта — сейчас покажу, как решать."]

    def _read_instruction(self, state: PageState) -> None:
        """Человек открыл инструкцию — в базу знаний (один раз на вид заданий)."""
        text = _plain_text(state.dialog_lines)
        pool = self._knowledge.for_state(state)
        if pool is None or pool.has_instruction or not _INSTRUCTION_RE.search(text[:120]):
            return
        if len(re.sub(r"\s+", "", text)) >= 200:
            self._knowledge.save_instruction(pool, text, "диалог «Инструкция» (помощник)")

    def _forget(self) -> None:
        self._identity, self._form, self._hint = None, frozenset(), None
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None

    @staticmethod
    async def _draw(frame: Frame, state: PageState, uids: list[str], title: str, lines: list[str], *,
                    muted: bool = False) -> None:
        sig = f"{title}|{'|'.join(lines)}|{muted}"
        try:
            await frame.evaluate(_JS_DRAW, {"panelId": ASSIST_PANEL_ID, "hints": uids, "title": title,
                                            "lines": lines, "muted": muted, "sig": sig})
        except PlaywrightError as exc:
            logger.debug("Подсказка не нарисована: %s", exc)

    @staticmethod
    async def _clear(frame: Frame) -> None:
        try:
            await frame.evaluate(_JS_CLEAR, ASSIST_PANEL_ID)
        except PlaywrightError:
            pass


async def assist() -> None:
    """Точка входа режима помощника (python main.py --assist)."""
    await Assistant().run()
