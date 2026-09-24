"""openrouter_connector.py v2 — универсальный LLM-клиент через ProxyAPI."""

from __future__ import annotations

import json
import logging
from typing import Optional

import openai

from config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)
from dom_parser import build_elements_prompt
from models import ActionType, LLMDecision, PageState

logger = logging.getLogger("twork.llm")


# ---------------------------------------------------------------------------
# Системный промпт: агент — General Problem Solver
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Ты — автономный AI-агент, задача которого — взаимодействовать с веб-интерфейсом для выполнения заданий.
Ты получаешь текст задания, подсказку и список интерактивных элементов страницы, принимаешь решение и возвращаешь его в JSON.

———
СТРУКТУРА ЭЛЕМЕНТОВ
Каждый элемент описан в формате:
  [INDEX] [TIP] "Текст" ✓Выбран?

ТИПЫ ЧЕРЕЗ [...]:
  [FOLDER ЗАКРЫТА]  — ветка/папка дерева, её нужно РАСКРЫТЬ (действие: open)
  [FOLDER РАСКРЫТА]  — ветка уже развернута, ищи нужный элемент внутри
  [OPTION]            — конечный вариант (выбрать кликом)
  [INPUT]             — поле ввода
  [BUTTON]            — кнопка действия
  ✓Выбран — элемент уже выбран/активен
———
ПРАВИЛА ВЫБОРА ДЕЙСТВИЯ (action)

→ "open"
   Применяй ТОЛЬКО к элементам типа [FOLDER ЗАКРЫТА].
   Цель — развернуть ветку дерева, чтобы увидеть внутренние элементы.
   НИКОГДА не применяй open к [FOLDER РАСКРЫТА] — она уже открыта!

→ "click"
   Применяй к элементам типа [OPTION] или [BUTTON] — чтобы сделать выбор.
   Не кликай элементы, у которых уже есть ✓Выбран — это бессмысленный повтор действия.

→ "submit"
   Применяй только когда задание выполнено полностью.
   Признаки завершения:
     • Нужный элемент имеет ✓Выбран, ИЛИ
     • Форма заполнена и больше нечего выбирать.
   target_index для submit = null.

→ "type"
   Применяй к элементам типа [INPUT]. Укажи type_text.

→ "skip"
   Если недостаточно данных для принятия решения, пропусти шаг.
———
АЛГОРИТМ МЫШЛЕНИЯ

1. Читаю задание и подсказку.
2. Анализирую список элементов:
   • Есть ли уже выбранный (✓Выбран) правильный элемент → submit.
   • Есть ли закрытая папка, ведущая к нужному элементу → open.
   • Есть ли видимый нужный OPTION → click.
3. Передаю target_text — ТОЧНЫЙ видимый текст элемента (копирую дословно из списка).
   Python-скрипт использует target_text для проверки и коррекции target_index.
———
ТРЕБОВАНИЯ К ВЫВОДУ

Отвечай строго JSON без комментариев, markdown-оформления и пояснений:
{
  "reasoning": "<цепочка рассуждений на русском>",
  "action": "click | open | submit | type | skip",
  "target_index": <целое число или null>,
  "target_text": "<точный видимый текст или null>",
  "type_text": "<текст для ввода или null>",
  "confidence": <0.0–1.0>
}
"""


class LLMConnector:
    """Клиент для взаимодействия с LLM через ProxyAPI (OpenAI-совместимый)."""

    def __init__(self) -> None:
        self._client = openai.AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        logger.info("LLM инициализирован: model=%s base=%s", LLM_MODEL, OPENAI_BASE_URL)

    async def decide(
        self,
        state: PageState,
        banned_indices: set[int],
    ) -> LLMDecision:
        """Отправить состояние в LLM и получить решение."""
        user_message = self._build_user_message(state, banned_indices)
        logger.debug("LLM user msg (%d симв.): %s", len(user_message), user_message[:300])

        try:
            response = await self._client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
            )
        except openai.OpenAIError as exc:
            logger.error("LLM API ошибка: %s", exc)
            return LLMDecision(reasoning=str(exc), action=ActionType.SKIP)

        raw = response.choices[0].message.content or "{}"
        logger.debug("LLM raw: %s", raw[:500])
        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Построение user-сообщения
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_message(state: PageState, banned: set[int]) -> str:
        parts: list[str] = []

        parts.append("═══ ЗАДАНИЕ ═══")
        parts.append(state.task_text or "(текст задания не обнаружен)")

        if state.hint_text:
            parts.append(f"\n═══ ПОДСКАЗКА ═══\n{state.hint_text}")

        parts.append("\n═══ ЭЛЕМЕНТЫ СТРАНИЦЫ ═══")
        parts.append(build_elements_prompt(state.elements))

        if banned:
            parts.append(f"\n[ЗАПРЕЩЕНО] Индексы, на которые уже кликали: {sorted(banned)}")
            parts.append("Не выбирай эти индексы!")

        parts.append("\nПрими решение. Отвечай только в JSON.")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Парсинг ответа
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> LLMDecision:
        try:
            data: dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("JSON парсинг LLM: %s", exc)
            return LLMDecision(reasoning=f"JSON error: {exc}", action=ActionType.SKIP)

        action_raw = data.get("action", "skip")
        try:
            action = ActionType(action_raw)
        except ValueError:
            logger.warning("Неизвестное action: %r, заменяем на skip", action_raw)
            action = ActionType.SKIP

        target_index = data.get("target_index")
        if target_index is not None:
            try:
                target_index = int(target_index)
            except (TypeError, ValueError):
                target_index = None

        decision = LLMDecision(
            reasoning=str(data.get("reasoning", "")),
            action=action,
            target_index=target_index,
            target_text=data.get("target_text"),
            type_text=data.get("type_text"),
            confidence=float(data.get("confidence", 1.0)),
        )
        logger.info(
            "LLM решение: action=%s index=%s text=%r conf=%.2f",
            decision.action.value,
            decision.target_index,
            (decision.target_text or "")[:40],
            decision.confidence,
        )
        logger.debug("Рассуждения: %s", decision.reasoning[:200])
        return decision
