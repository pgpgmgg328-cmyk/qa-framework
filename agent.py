"""agent.py v3 — главный цикл агента.

Шаг агента:
  1. Найти видимый целевой фрейм (иначе — ожидание, шаг не расходуется).
  2. Капча в отдельном iframe → ожидание ручного решения (с таймаутом).
  3. Атомарный снимок DomParser: элементы размечены метками data-agent-id.
  4. Смена отпечатка задания → сброс памяти. Проверка эффекта прошлого действия.
  5. Загрузка/спиннер → ждём стабилизации DOM.
  6. Локальный флоу «Приступить/ОК» — только на заставке или в диалоге.
  7. Решение LLM (история, запреты, бюджет, картинка задания).
  8. Сверка target_index с target_text (с учётом типа элемента и дублей).
  9. Выполнение по метке + проверка submit по смене задания.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
import time
from enum import Enum
from typing import Optional

from playwright.async_api import Error as PlaywrightError, Frame

from browser_controller import BrowserController, is_connection_lost
from config import (
    ACTION_WAIT,
    CAPTCHA_TIMEOUT,
    FINISH_BUTTON_TEXTS,
    FINISH_DENY_SUBSTRINGS,
    FRAME_LOAD_WAIT,
    LLM_HISTORY_SIZE,
    LLM_TEMPERATURE,
    LLM_VISION,
    MAX_IDLE_SECONDS,
    MAX_STEPS,
    MAX_STEPS_PER_TASK,
    START_BUTTON_TEXTS,
    SUBMIT_WAIT,
)
from dom_parser import DomParser, is_denied_button
from models import (
    FLAG_DISABLED,
    FLAG_IN_DIALOG,
    FLAG_IN_POPUP,
    FLAG_OCCLUDED,
    FLAG_SELECTABLE,
    FLAG_SELECTED,
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
    return t.strip()


def _short(exc: BaseException) -> str:
    return str(exc).strip().splitlines()[0][:160] if str(exc).strip() else exc.__class__.__name__


class Agent:
    """Автономный агент для платформы T-Work v3."""

    def __init__(
        self,
        *,
        browser: Optional[BrowserController] = None,
        llm: Optional[LLMConnector] = None,
    ) -> None:
        # зависимости можно подменить (тесты, другой провайдер LLM)
        self._browser = browser or BrowserController()
        self._llm = llm or LLMConnector()
        self._memory = TaskMemory()
        self._task_identifier: str = ""
        self._task_basis: tuple[str, str] = ("", "")
        self._submitted = False             # отправка подтверждена — следующая смена отпечатка ожидаема
        self._tasks_done = 0
        self._captcha_suppressed_until = 0.0
        self._last_idle_log = 0.0

    # ------------------------------------------------------------------
    # Главная точка входа
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Запустить агента."""
        async with self._browser:
            logger.info("=" * 60)
            logger.info(
                "АГЕНТ v3 ЗАПУЩЕН. Шагов с действием: %d, на одно задание: %d",
                MAX_STEPS, MAX_STEPS_PER_TASK,
            )
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
            logger.info("ИТОГ: отправлено заданий=%d, шагов с действием=%d", self._tasks_done, acted)

    # ------------------------------------------------------------------
    # Один шаг агента
    # ------------------------------------------------------------------

    async def _step(self) -> StepResult:
        # 1. Целевой фрейм
        frame = await self._browser.find_target_frame()
        if frame is None:
            self._log_idle("Целевой фрейм не найден — жду (если нужен вход в аккаунт, войдите в окне браузера)")
            await asyncio.sleep(FRAME_LOAD_WAIT)
            return StepResult.IDLE

        # 2. Капча в отдельном iframe страницы
        if time.monotonic() > self._captcha_suppressed_until and await self._browser.page_has_captcha():
            await self._wait_captcha_solved()
            return StepResult.IDLE

        # 3. Атомарный снимок + память задания
        state = await DomParser(frame).parse()
        self._refresh_task_context(state)
        self._memory.verify(state)          # фактический результат прошлого действия → в историю

        if state.has_captcha and time.monotonic() > self._captcha_suppressed_until:
            await self._wait_captcha_solved()
            return StepResult.IDLE

        # 4. Загрузка: ждём, но не бесконечно (ложный «спиннер» не должен вешать агента)
        if state.loading and self._memory.loading_waits < 5:
            self._memory.loading_waits += 1
            logger.info("Идёт загрузка — жду стабилизации DOM")
            await self._browser.wait_settle(frame)
            await asyncio.sleep(0.5)
            return StepResult.IDLE
        if not state.loading:
            self._memory.loading_waits = 0

        # 5. Локальный флоу «Приступить / ОК»
        if await self._maybe_click_start(frame, state):
            return StepResult.ACTED

        # 6. Нет активных элементов — ждём
        if not any(not e.is_disabled for e in state.elements):
            self._log_idle("Активных элементов нет — жду загрузки/следующего задания")
            await asyncio.sleep(max(ACTION_WAIT, 1.0))
            return StepResult.IDLE

        # 7. Бюджет шагов на задание
        self._memory.steps += 1
        if self._memory.steps > MAX_STEPS_PER_TASK:
            return await self._handle_budget_exhausted(frame, state)

        # 8. Решение LLM
        logger.info(
            "%s задание %s · шаг %d/%d · «%s» %s",
            "-" * 10, state.task_identifier[:8], self._memory.steps, MAX_STEPS_PER_TASK,
            state.task_preview[:50], "-" * 10,
        )
        context = await self._build_context(frame, state)
        decision = await self._llm.decide(state, context)

        # 9. Сверка цели и выполнение
        target = self._resolve_target(decision, state)
        await self._execute(frame, decision, target, state)
        await self._browser.wait_settle(frame)
        return StepResult.ACTED

    # ------------------------------------------------------------------
    # Память задания
    # ------------------------------------------------------------------

    def _refresh_task_context(self, state: PageState) -> None:
        """Сбрасывать память ТОЛЬКО при смене задания.

        Отпечаток не включает текст дерева, поэтому раскрытие папок память не сбрасывает.
        """
        new_id = state.task_identifier
        basis = (re.sub(r"\d+", "", normalize_text(state.task_text)), state.image_src)
        if new_id and new_id != self._task_identifier:
            logger.info(
                "СМЕНА ЗАДАНИЯ: %s → %s «%s»",
                self._task_identifier[:8] or "—", new_id[:8], state.task_preview[:60],
            )
            if self._task_identifier and basis == self._task_basis and not self._submitted:
                # Отличаются только цифры, картинка та же, отправки не было — вероятно,
                # в текст задания попал таймер/счётчик. Память сбрасывается, но стоит
                # сузить TASK_TEXT_SELECTORS, иначе защита от зацикливания не работает.
                logger.warning("Отпечаток задания изменился только в цифрах — проверьте, "
                               "не попал ли в текст задания таймер или счётчик")
            self._task_identifier = new_id
            self._memory.reset(new_id)
        self._task_basis = basis
        self._submitted = False

    async def _build_context(self, frame: Frame, state: PageState) -> DecisionContext:
        mem = self._memory
        steps_left = MAX_STEPS_PER_TASK - mem.steps
        notes: list[str] = []
        if steps_left <= 5:
            notes.append(
                f"Осталось шагов на это задание: {steps_left}. Если ответ уже выбран — submit; "
                "иначе выбери лучший доступный вариант."
            )
        if mem.consecutive_skips >= 2:
            notes.append(f"Ты пропустил {mem.consecutive_skips} шага подряд — выбери конкретное действие.")
        if mem.invalid_targets:
            notes.append("Прошлое действие ссылалось на несуществующий элемент: бери номер и текст "
                         "ТОЛЬКО из текущего списка.")
        if mem.repeated_forbidden:
            notes.append("Ты повторил действие из НЕ ПОВТОРЯТЬ — выбери другой элемент или другое действие.")

        # При зацикливании слегка «встряхиваем» детерминированную модель
        temperature = None
        if mem.consecutive_skips >= 3 or mem.repeated_forbidden >= 1:
            temperature = max(LLM_TEMPERATURE, 0.4)

        return DecisionContext(
            history=mem.history_lines(LLM_HISTORY_SIZE),
            forbidden=mem.forbidden_lines(),
            notes=notes,
            step_in_task=mem.steps,
            steps_left=steps_left,
            image_b64=await self._task_image(frame, state),
            temperature=temperature,
        )

    async def _task_image(self, frame: Frame, state: PageState) -> Optional[str]:
        """Картинка задания для vision-модели; для режима image кэшируется на задание."""
        if LLM_VISION == "off":
            return None
        mem = self._memory
        if LLM_VISION == "frame":
            return await self._browser.screenshot_b64(frame, "frame")
        if not mem.image_checked or mem.image_src != state.image_src:
            mem.image_checked = True
            mem.image_src = state.image_src
            mem.image_b64 = await self._browser.screenshot_b64(frame, "image") if state.image_src else None
        return mem.image_b64

    # ------------------------------------------------------------------
    # Сверка цели (автокоррекция v3)
    # ------------------------------------------------------------------

    def _resolve_target(self, decision: LLMDecision, state: PageState) -> Optional[ParsedElement]:
        """Найти элемент, который имела в виду LLM.

        v2: первое совпадение по тексту перетирало индекс ВСЕГДА — при дублях
        («Другое» в каждой папке) выбиралась первая ветка, а частичное совпадение
        цепляло контейнер с текстом всех детей. v3: индекс, подтверждённый текстом,
        приоритетнее; поиск по тексту учитывает тип элемента, а при дублях берёт
        ближайший к названному индексу и не запрещённый.
        """
        action = decision.action
        if action == ActionType.SKIP:
            return None
        kinds = _ACTION_KINDS[action]
        fallback = _FALLBACK_KINDS.get(action, set())
        by_index = state.by_index(decision.target_index)
        if by_index is not None and is_denied_button(by_index):
            logger.warning("LLM указала кнопку из стоп-списка «%s» — игнорирую", by_index.label())
            by_index = None
        text = _clean_target_text(decision.target_text or "")

        # 1. Индекс и текст согласованы
        if by_index is not None and by_index.kind in kinds and (
            not text or _similarity(by_index.text or by_index.placeholder, text) >= 0.9
        ):
            return by_index

        # 2. Поиск по тексту среди элементов подходящего типа
        if text:
            for allowed in (kinds, fallback):
                scored = [
                    (_similarity(e.text or e.placeholder, text), e)
                    for e in state.elements if e.kind in allowed and not is_denied_button(e)
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
    # Выполнение действия
    # ------------------------------------------------------------------

    async def _execute(
        self,
        frame: Frame,
        decision: LLMDecision,
        target: Optional[ParsedElement],
        state: PageState,
    ) -> None:
        action = decision.action
        mem = self._memory
        logger.info(
            "ВЫПОЛНЯЕМ: %s → %s conf=%.2f",
            action.value, f"[{target.index}] «{target.label()}»" if target else "—", decision.confidence,
        )

        if action == ActionType.SKIP:
            mem.consecutive_skips += 1
            mem.add(action, None, result="ожидание")
            await asyncio.sleep(ACTION_WAIT * 2)
            return
        mem.consecutive_skips = 0

        if action == ActionType.SUBMIT:
            await self._do_submit(frame, state, target)
            return

        if action == ActionType.SCROLL:
            direction = decision.scroll_direction or "down"
            key = target.key if target else f"__page__|{direction}"
            if mem.is_forbidden(action, key):
                mem.repeated_forbidden += 1
                mem.add(action, target, result="⛔ прокрутка уже ничего не меняла")
                return
            outcome = await self._browser.scroll(frame, target, direction)
            if outcome.ok:
                mem.expect(action, target, state, note=direction, key=key)
            else:
                mem.add(action, target, result=f"✗ {outcome.detail}")
            return

        if target is None:
            mem.invalid_targets += 1
            mem.add(action, None, result=(
                f"✗ элемент не найден (номер={decision.target_index}, текст=«{decision.target_text or ''}»)"
            ))
            return
        mem.invalid_targets = 0

        if is_denied_button(target):
            mem.add(action, target, result="⛔ кнопка из стоп-списка — агент её не нажимает")
            return
        if target.is_disabled:
            mem.add(action, target, result="✗ элемент неактивен — нажать нельзя")
            mem.mark_no_effect(action, target.key)
            return
        if mem.is_forbidden(action, target.key):
            mem.repeated_forbidden += 1
            mem.add(action, target, result="⛔ уже пробовали без эффекта — выбери другое действие")
            return
        mem.repeated_forbidden = 0

        if action == ActionType.OPEN:
            if (target.kind in (ElementKind.FOLDER, ElementKind.DROPDOWN)
                    and target.folder_state == FolderState.OPEN
                    and mem.insist[(action, target.key)] == 0):
                # Первый раз не кликаем (клик свернул бы папку), а подсказываем. Если модель
                # настаивает — эвристика состояния могла ошибиться, выполняем.
                # v2 в этом случае банил индекс — после сдвига индексов бан попадал на чужой элемент.
                mem.insist[(action, target.key)] += 1
                mem.add(action, target, result="уже раскрыта — её содержимое ниже, с бо́льшим отступом")
                return
            mem.open_attempts[target.key] += 1
            outcome = await self._browser.click_element(frame, target, prefer_toggle=True)
        elif action == ActionType.CLICK:
            if (target.is_selected and target.kind in (ElementKind.OPTION, ElementKind.FOLDER)
                    and target.choice_type != "checkbox" and mem.insist[(action, target.key)] == 0):
                # v2 здесь сразу жал submit — при нескольких вопросах это отправляло неполный ответ.
                # Checkbox можно снять сознательно, поэтому для него клик выполняется.
                mem.insist[(action, target.key)] += 1
                mem.add(action, target, result="уже ✓ВЫБРАН — повторный клик снял бы выбор; если всё готово — submit")
                return
            if target.kind == ElementKind.BUTTON and self._is_finish_button(target):
                await self._do_submit(frame, state, target)
                return
            outcome = await self._browser.click_element(frame, target)
        elif action == ActionType.TYPE:
            if not decision.type_text:
                mem.add(action, target, result="✗ пустой type_text")
                return
            outcome = await self._browser.type_into(frame, target, decision.type_text)
        else:
            logger.error("Неизвестное действие: %s", action.value)
            return

        if outcome.stale:
            mem.add(action, target, result="⚠ элемент перерисовался до клика — повтор на свежем снимке")
            return
        if not outcome.ok:
            mem.add(action, target, result=f"✗ {outcome.detail}")
            mem.mark_no_effect(action, target.key)
            return
        note = f"«{decision.type_text[:60]}»" if action == ActionType.TYPE and decision.type_text else ""
        if outcome.method == "js":
            note = (note + " (js-клик)").strip()
        mem.expect(action, target, state, typed=decision.type_text, note=note)

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
        buttons = [e for e in state.elements if e.kind == ElementKind.BUTTON and self._is_finish_button(e)]
        if not buttons:
            return None

        def rank(e: ParsedElement) -> tuple:
            label = normalize_text(e.text)
            priority = next(
                (i for i, t in enumerate(FINISH_BUTTON_TEXTS) if label == t or label.startswith(t + " ")), 99,
            )
            return (e.is_disabled, e.container != "dialog", priority, e.index)

        return min(buttons, key=rank)

    async def _do_submit(self, frame: Frame, state: PageState, target: Optional[ParsedElement]) -> None:
        """Нажать кнопку отправки и убедиться, что задание действительно сменилось.

        v2 сбрасывал память сразу после клика, даже если форма не прошла
        валидацию — агент начинал задание «с нуля» и зацикливался.
        """
        mem = self._memory
        if target is not None and target.kind == ElementKind.BUTTON and self._is_finish_button(target, strict=False):
            button: Optional[ParsedElement] = target
        else:
            button = self._find_finish_button(state)

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

        changed, after = await self._wait_task_change(state.task_identifier)
        if changed:
            self._submitted = True
            self._tasks_done += 1
            logger.info("✅ ЗАДАНИЕ ОТПРАВЛЕНО (всего: %d)", self._tasks_done)
            mem.add(ActionType.SUBMIT, button, result="✓ отправлено, задание сменилось")
            return

        mem.submit_failures += 1
        result = "✗ задание не сменилось"
        if after is not None and after.alerts:
            result += "; сообщения: " + " | ".join(after.alerts)
        dialog = [e.label() for e in (after.elements if after else []) if e.container == "dialog"][:4]
        if dialog:
            result += "; открыт диалог: " + ", ".join(f"«{d}»" for d in dialog)
        logger.warning("SUBMIT: %s", result)
        mem.add(ActionType.SUBMIT, button, result=result)

    async def _wait_task_change(self, old_id: str) -> tuple[bool, Optional[PageState]]:
        """Поллинг отпечатка задания до SUBMIT_WAIT секунд."""
        deadline = time.monotonic() + SUBMIT_WAIT
        last: Optional[PageState] = None
        missing = 0
        while time.monotonic() < deadline:
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
            if last.task_identifier != old_id:
                return True, last
        return False, last

    # ------------------------------------------------------------------
    # Локальный флоу, бюджет, капча
    # ------------------------------------------------------------------

    async def _maybe_click_start(self, frame: Frame, state: PageState) -> bool:
        """«Приступить/ОК/Понятно» — без LLM, но только на заставке или в диалоге.

        v2 искал эти слова на КАЖДОМ шаге до LLM, подстрокой и по любому элементу:
        вариант ответа «Хорошо», кнопка «Далее» до ответа на вопрос или абзац
        «Чтобы приступить…» нажимались бесконечно, LLM не получала управления.
        """
        enabled_buttons = [
            e for e in state.elements
            if e.kind == ElementKind.BUTTON and not e.is_disabled and not e.occluded and not is_denied_button(e)
        ]
        dialog_buttons = [b for b in enabled_buttons if b.container == "dialog"]
        working = [
            e for e in state.elements
            if e.kind in (ElementKind.OPTION, ElementKind.FOLDER, ElementKind.INPUT, ElementKind.DROPDOWN)
            and not e.is_disabled and not e.occluded and e.container != "dialog"
        ]
        if working and not dialog_buttons:
            return False

        def start_like(b: ParsedElement) -> bool:
            label = normalize_text(b.text)
            return any(label == t or label.startswith(t + " ") for t in START_BUTTON_TEXTS)

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
            if snapshot.task_identifier != self._task_identifier:
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
