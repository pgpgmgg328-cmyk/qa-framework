"""agent.py v2 — главный цикл агента с защитой от циклов и автокоррекцией индексов."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from playwright.async_api import Frame

from browser_controller import BrowserController
from config import (
    FINISH_BUTTON_TEXTS,
    FRAME_LOAD_WAIT,
    MAX_STEPS,
    START_BUTTON_TEXTS,
    SUBMIT_WAIT,
)
from dom_parser import DomParser
from models import ActionType, ElementKind, FolderState, LLMDecision, PageState, ParsedElement
from openrouter_connector import LLMConnector

logger = logging.getLogger("twork.agent")


class Agent:
    """Автономный агент для платформы T-Work v2.

    Цикл на каждом шаге:
    1.  Получить целевой фрейм или подождать.
    2.  Распарсить DOM через DomParser.
    3.  Обновить task_identifier; если сменился — сбросить _banned.
    4.  Проверить капчу; если есть — поллинг до решения.
    5.  Локальный флоу: кнопка «Приступить/ОК».
    6.  LLM-решение с автокоррекцией target_index.
    7.  Выполнить действие.
    8.  При submit: нажать финальную кнопку и сбросить память.
    """

    def __init__(self) -> None:
        self._browser = BrowserController()
        self._llm     = LLMConnector()

        # Чёрный список индексов — не кликать повторно
        # Сбрасывается ТОЛЬКО при смене сигнатуры задания
        self._banned: set[int] = set()

        # Сигнатура текущего задания
        self._task_identifier: str = ""

        # Счётчик повторных open-действий для одной папки
        # если > MAX_OPEN_RETRIES — папка заблокирована
        self._open_counters: dict[int, int] = {}
        self._MAX_OPEN_RETRIES = 3

    # ------------------------------------------------------------------
    # Главная точка входа
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Запустить агента."""
        async with self._browser:
            logger.info("=" * 60)
            logger.info("АГЕНТ v2 ЗАПУЩЕН. Макс шагов: %d", MAX_STEPS)
            logger.info("=" * 60)

            for step in range(1, MAX_STEPS + 1):
                logger.info("\n%s ШАГ %d/%d %s", "-" * 20, step, MAX_STEPS, "-" * 20)
                try:
                    await self._step()
                except Exception as exc:
                    logger.exception("Непредвиденная ошибка в шаге %d: %s", step, exc)
                    await asyncio.sleep(2)

                await asyncio.sleep(0.3)

            logger.warning("ДОСТИГНУТ ЛИМИТ ШАГОВ (%d)", MAX_STEPS)

    # ------------------------------------------------------------------
    # Один шаг агента
    # ------------------------------------------------------------------

    async def _step(self) -> None:

        # 1. Получить целевой фрейм
        frame = self._browser.get_target_frame()
        if frame is None:
            logger.warning("Целевой фрейм не найден, ожидаем…")
            await asyncio.sleep(FRAME_LOAD_WAIT)
            return

        # 2. Распарсить DOM
        parser = DomParser(frame)
        state: PageState = await parser.parse()

        # 3. Обновить сигнатуру задания
        self._refresh_task_context(state)

        # 4. Проверка капчи
        if state.has_captcha:
            logger.warning("КАПЧА! Ожидаю ручного ввода…")
            await self._wait_captcha_solved(frame)
            return

        # 5. Локальный флоу: «Приступить / ОК»
        clicked = await self._browser.click_by_text(
            frame, START_BUTTON_TEXTS, skip_disabled=True
        )
        if clicked:
            logger.info("Локальный флоу: стартовая кнопка найдена и нажата")
            return

        # 6. Нет элементов — подождать
        active = [e for e in state.elements if not e.is_disabled]
        if not active:
            logger.warning("Активных элементов нет, пропускаем")
            return

        # 7. LLM-решение
        decision: LLMDecision = await self._llm.decide(state, self._banned)

        # 8. Автокоррекция target_index по target_text
        decision = self._autocorrect(decision, state.elements)

        # 9. Выполнить
        await self._execute(frame, decision, state)

    # ------------------------------------------------------------------
    # Сигнатура задания / сброс памяти
    # ------------------------------------------------------------------

    def _refresh_task_context(self, state: PageState) -> None:
        """Cбрасывать _banned ТОЛЬКО при смене задания.

        Раскрытие папки (древесные переходы) не меняют
        task_identifier и не сбрасывают память!
        """
        new_id = state.task_identifier
        if new_id and new_id != self._task_identifier:
            logger.info(
                "СМЕНА ЗАДАНИЯ: %r → %r",
                self._task_identifier[:60],
                new_id[:60],
            )
            self._task_identifier = new_id
            self._banned.clear()
            self._open_counters.clear()
            logger.info("Сброс _banned и _open_counters")
        else:
            logger.debug(
                "Задание не изменилось. _banned=%d, id=%r",
                len(self._banned), (new_id or "")[:40],
            )

    # ------------------------------------------------------------------
    # Автокоррекция индекса
    # ------------------------------------------------------------------

    def _autocorrect(
        self,
        decision: LLMDecision,
        elements: list[ParsedElement],
    ) -> LLMDecision:
        """Eсли LLM передал target_text — ищем по тексту в DOM
        и заменяем галлюцинированный target_index на реальный."""
        target_text = (decision.target_text or "").strip()
        if not target_text:
            # LLM не дала target_text — нечего корректировать
            return decision

        target_lower = target_text.lower()
        active = [e for e in elements if not e.is_disabled]

        # Приоритет 1: точное совпадение
        for el in active:
            if el.text.strip().lower() == target_lower:
                return self._apply_correction(decision, el)

        # Приоритет 2: частичное вхождение
        for el in active:
            if target_lower in el.text.strip().lower():
                logger.debug("Автокоррекция (partial): %r → idx=%d", target_text, el.index)
                return self._apply_correction(decision, el)

        logger.warning(
            "Автокоррекция: target_text=%r не найден, оставляем idx=%s",
            target_text, decision.target_index,
        )
        return decision

    @staticmethod
    def _apply_correction(decision: LLMDecision, el: ParsedElement) -> LLMDecision:
        if el.index == decision.target_index:
            return decision  # и так совпадает
        logger.info(
            "Автокоррекция: idx %s → %d (по target_text=%r)",
            decision.target_index, el.index, el.text,
        )
        return decision.model_copy(update={"target_index": el.index})

    # ------------------------------------------------------------------
    # Выполнение действия
    # ------------------------------------------------------------------

    async def _execute(
        self,
        frame: Frame,
        decision: LLMDecision,
        state: PageState,
    ) -> None:
        action = decision.action
        idx    = decision.target_index

        logger.info(
            "ВЫПОЛНЯЕМ: action=%s idx=%s text=%r conf=%.2f",
            action.value, idx,
            (decision.target_text or "")[:50],
            decision.confidence,
        )

        # --- skip ---
        if action == ActionType.SKIP:
            logger.info("Действие SKIP: пропускаем шаг")
            return

        # --- submit ---
        if action == ActionType.SUBMIT:
            await self._do_submit(frame)
            return

        # Остальные действия требуют target_index
        if idx is None:
            logger.warning("Для %s необходим target_index", action.value)
            return

        # Проверяем, что элемент не заблокирован
        el = self._find_element(state.elements, idx)
        if el and el.is_disabled:
            logger.warning("idx=%d (%r) заблокирован, пропускаем", idx, el.text)
            self._banned.add(idx)
            return

        # --- open: развернуть папку ---
        if action == ActionType.OPEN:
            await self._do_open(frame, idx, el)
            return

        # --- click: выбрать опцию ---
        if action == ActionType.CLICK:
            if idx in self._banned:
                logger.warning("idx=%d в _banned, пропускаем")
                return
            # Защита: не кликаем уже выбранный элемент
            if el and el.is_selected:
                logger.info("idx=%d (%r) уже выбран, переходим к submit", idx, el.text)
                await self._do_submit(frame)
                return
            await self._browser.click_by_index(frame, idx)
            self._banned.add(idx)
            return

        # --- type: ввод текста ---
        if action == ActionType.TYPE:
            if not decision.type_text:
                logger.warning("type_text пусто, пропускаем")
                return
            await self._browser.type_by_index(frame, idx, decision.type_text)
            return

        logger.error("Неизвестное действие: %s", action.value)

    # ------------------------------------------------------------------
    # Частные методы execute
    # ------------------------------------------------------------------

    async def _do_open(self, frame: Frame, idx: int, el: Optional[ParsedElement]) -> None:
        """Развернуть папку с защитой от бесконечных повторов."""

        # Если папка уже открыта — не кликаем
        if el and el.folder_state == FolderState.OPEN:
            logger.info("OPEN: папка idx=%d (%r) уже раскрыта, пропускаем", idx, el.text)
            self._banned.add(idx)
            return

        # Счётчик попыток развернуть эту папку
        self._open_counters[idx] = self._open_counters.get(idx, 0) + 1
        if self._open_counters[idx] > self._MAX_OPEN_RETRIES:
            logger.warning(
                "OPEN: папка idx=%d не раскрывается за %d попытки, блокируем",
                idx, self._MAX_OPEN_RETRIES,
            )
            self._banned.add(idx)
            return

        logger.info(
            "OPEN: развертываем папку idx=%d (%r), попытка %d/%d",
            idx, el.text if el else "?",
            self._open_counters[idx], self._MAX_OPEN_RETRIES,
        )
        await self._browser.click_by_index(frame, idx)
        # Не добавляем в _banned: возможно, папку понадобится развернуть ещё раз

    async def _do_submit(self, frame: Frame) -> None:
        """Отправить форму: нажать кнопку submit, подождать, сбросить память."""
        logger.info("SUBMIT: ищем финальную кнопку…")
        found = await self._browser.click_by_text(
            frame, FINISH_BUTTON_TEXTS, skip_disabled=True
        )
        if found:
            logger.info("SUBMIT: кнопка нажата. Ожидаем %.1fс…", SUBMIT_WAIT)
            await asyncio.sleep(SUBMIT_WAIT)
            # Полный сброс памяти после успешной отправки
            self._banned.clear()
            self._open_counters.clear()
            self._task_identifier = ""
            logger.info("SUBMIT: память сброшена, ждём новое задание")
        else:
            logger.warning("SUBMIT: кнопка не найдена (все заблокированы?)")

    # ------------------------------------------------------------------
    # Капча: поллинг
    # ------------------------------------------------------------------

    async def _wait_captcha_solved(
        self, frame: Frame, poll: float = 5.0
    ) -> None:
        logger.info("Поллинг капчи каждые %.0fс…", poll)
        while True:
            await asyncio.sleep(poll)
            parser = DomParser(frame)
            s = await parser.parse()
            if not s.has_captcha:
                logger.info("Капча решена, продолжаем")
                break

    # ------------------------------------------------------------------
    # Утилита
    # ------------------------------------------------------------------

    @staticmethod
    def _find_element(
        elements: list[ParsedElement], index: int
    ) -> Optional[ParsedElement]:
        for el in elements:
            if el.index == index:
                return el
        return None


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    asyncio.run(Agent().run())
