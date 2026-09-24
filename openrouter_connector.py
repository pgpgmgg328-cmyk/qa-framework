"""openrouter_connector.py v3 — LLM-клиент (ProxyAPI / OpenRouter, OpenAI-совместимый API).

Изменения относительно v2:
- системный промпт: легенда формата генерируется из models.py (в v2 промпт
  описывал «[FOLDER ЗАКРЫТА]», а строки приходили как «[ПАПКА ЗАКРЫТА]»);
  добавлены стратегия поиска по дереву, правила отправки, анти-галлюцинационные
  правила и пример;
- в промпт передаются история действий с фактическим результатом, запреты по
  ТЕКСТУ (а не по сдвигающимся индексам), сообщения страницы и бюджет шагов;
- Structured Outputs (json_schema, strict) с автоматическим откатом на json_object;
- vision: скриншот картинки задания (в v2 модель классифицировала фото вслепую);
- таймаут/ретраи клиента, одна попытка «ремонта» невалидного JSON, учёт refusal
  и finish_reason=length.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Union

import openai
from pydantic import ValidationError

from config import (
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_STRUCTURED_OUTPUT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_VISION_DETAIL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENROUTER_REFERER,
)
from dom_parser import build_elements_prompt, select_for_prompt
from models import (
    FLAG_DISABLED,
    FLAG_IN_DIALOG,
    FLAG_SELECTABLE,
    FLAG_SELECTED,
    PROMPT_FLAGS,
    PROMPT_LEGEND,
    ActionType,
    DecisionContext,
    LLMDecision,
    PageState,
)

logger = logging.getLogger("twork.llm")


# ---------------------------------------------------------------------------
# Системный промпт v3
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""Ты — автономный агент-оператор платформы заданий T-Work. На каждом шаге ты получаешь снимок \
страницы задания и выбираешь РОВНО ОДНО следующее действие. Python-скрипт выполняет его, проверяет \
фактический результат и присылает новый снимок вместе с историей твоих действий.

━━━ ЧТО ТЫ ПОЛУЧАЕШЬ ━━━
• ЗАДАНИЕ — текст вопроса/инструкции. Если приложено изображение — это картинка из задания \
(например, фото товара); это главный источник фактов об объекте.
• ПОДСКАЗКА и СООБЩЕНИЯ СТРАНИЦЫ — инструкции, ошибки валидации, уведомления.
• ЭЛЕМЕНТЫ — интерактивные элементы, по одному в строке: [N] [ТИП] «текст» флаги
  N — номер элемента в ЭТОМ снимке. После каждого действия номера пересчитываются — не переноси \
номера из истории.
  Отступ (2 пробела на уровень) — вложенность в дереве: строки с бо́льшим отступом под раскрытой \
папкой — её содержимое.
  Типы:
{PROMPT_LEGEND}
  Флаги:
{PROMPT_FLAGS}
• ИСТОРИЯ — твои прошлые действия по этому заданию и их ФАКТИЧЕСКИЙ результат \
(✓ — сработало, ✗ — не сработало, ⚠ — побочный эффект).
• НЕ ПОВТОРЯТЬ — действия, которые уже не дали эффекта; скрипт их не выполнит.

━━━ ДЕЙСТВИЯ ━━━
• open   — раскрыть [FOLDER закрыта] или [DROPDOWN закрыт]. Никогда — для уже раскрытых.
• click  — выбрать [OPTION]; нажать [BUTTON]/[OTHER]; выбрать саму папку — только если у неё есть \
{FLAG_SELECTABLE} и задание требует именно эту общую категорию.
• type   — ввести текст в [INPUT …]; type_text обязателен.
• scroll — прокрутить список, если нужного варианта нет, а в СОСТОЯНИИ сказано, что список \
прокручивается (scroll_direction "down"/"up"; target_index — любой элемент этого списка или null).
• submit — отправить ответ кнопкой «Завершить»/«Отправить»; target_index — номер этой кнопки, \
если она видна, иначе null.
• skip   — пропустить шаг: страница грузится или элементы перекрыты. Не используй skip, если можно \
продвинуться.

━━━ КАК ИСКАТЬ В ДЕРЕВЕ КАТЕГОРИЙ ━━━
1. Сначала пойми объект задания: что это, для чего, из какой области (по тексту и изображению).
2. Подходящий [OPTION] уже виден → выбери его; среди подходящих — самый конкретный.
3. Иначе раскрой ОДНУ закрытую папку — самую вероятную по смыслу. Двигайся от общего к частному.
4. В раскрытой папке нет подходящего → раскрой следующую по вероятности папку. Раскрытые папки \
не сворачивай и не раскрывай повторно.
5. «Другое»/«Прочее»/«Иное» — только если в ПРАВИЛЬНОЙ ветке точно нет более конкретного варианта.
6. Одинаковые названия в разных ветках различай по отступу и ⟨путь: …⟩.

━━━ КОГДА ОТПРАВЛЯТЬ (submit) ━━━
• Только когда выполнены ВСЕ требования задания: нужные варианты имеют {FLAG_SELECTED}, обязательные \
поля заполнены, все вопросы на странице отвечены.
• [OPTION checkbox] или формулировка «выберите все подходящие» → сначала отметь все подходящие.
• Кнопка отправки {FLAG_DISABLED} → ответ ещё не заполнен; найди, чего не хватает.
• Отправка в ИСТОРИИ не удалась → прочитай СООБЩЕНИЯ СТРАНИЦЫ и исправь ответ, прежде чем \
отправлять снова.

━━━ ПРАВИЛА ПРОТИВ ОШИБОК ━━━
• Сначала найди строку с нужным текстом, затем перепиши её номер — не наоборот.
• target_index — только номер из текущего списка ЭЛЕМЕНТЫ.
• target_text — дословная копия текста из той же строки: без номера, [ТИПА], кавычек «» и флагов. \
Скрипт сверяет номер по тексту и исправит его, если номер съехал.
• Не кликай [OPTION] с {FLAG_SELECTED}: повторный клик снимет выбор.
• Не выбирай элементы с {FLAG_DISABLED} и действия из НЕ ПОВТОРЯТЬ.
• Нужного элемента нет в списке → open/scroll, а не click по «похожему». Не выдумывай элементы.
• Есть элементы {FLAG_IN_DIALOG} → сначала разберись с диалогом.

━━━ ФОРМАТ ОТВЕТА ━━━
Только JSON-объект:
{{
  "goal": "что требуется в задании — 1 фраза",
  "observation": "что уже сделано (✓ в списке, история) и какие элементы релевантны: [N] «текст», …",
  "reasoning": "почему именно это действие — 1–3 фразы",
  "action": "open | click | type | scroll | submit | skip",
  "target_index": N или null,
  "target_text": "дословный текст элемента" или null,
  "type_text": "текст для ввода" или null,
  "scroll_direction": "down" | "up" | null,
  "confidence": число 0.0–1.0 — уверенность, что действие ведёт к ПРАВИЛЬНОМУ ответу
}}

━━━ ПРИМЕР ━━━
ЗАДАНИЕ: Выберите категорию товара: «Смартфон Samsung Galaxy A55».
ЭЛЕМЕНТЫ:
[0] [FOLDER закрыта] «Одежда»
[1] [FOLDER раскрыта] «Электроника»
  [2] [OPTION radio] «Ноутбуки»
  [3] [FOLDER закрыта] «Телефоны и связь»
  [4] [OPTION radio] «Другое»
[5] [BUTTON] «Завершить» {FLAG_DISABLED}
Ответ:
{{"goal": "категория для смартфона", "observation": "Электроника раскрыта, ничего не выбрано; \
смартфон относится к [3] «Телефоны и связь» (закрыта)", "reasoning": "Нужная подкатегория внутри [3] — \
раскрываю её; «Другое» преждевременно", "action": "open", "target_index": 3, \
"target_text": "Телефоны и связь", "type_text": null, "scroll_direction": null, "confidence": 0.9}}
"""

# Совместимость с v2
_SYSTEM_PROMPT = SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Structured Outputs: схема ответа (порядок полей = порядок генерации,
# поэтому рассуждения идут ДО выбора действия — это и есть Chain-of-Thought)
# ---------------------------------------------------------------------------

def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


DECISION_JSON_SCHEMA: dict[str, Any] = {
    "name": "agent_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "goal":             {"type": "string"},
            "observation":      {"type": "string"},
            "reasoning":        {"type": "string"},
            "action":           {"type": "string", "enum": [a.value for a in ActionType]},
            "target_index":     _nullable({"type": "integer"}),
            "target_text":      _nullable({"type": "string"}),
            "type_text":        _nullable({"type": "string"}),
            "scroll_direction": _nullable({"type": "string", "enum": ["down", "up"]}),
            "confidence":       {"type": "number"},
        },
        "required": [
            "goal", "observation", "reasoning", "action", "target_index",
            "target_text", "type_text", "scroll_direction", "confidence",
        ],
    },
}


class DecisionParseError(ValueError):
    """Ответ модели не удалось превратить в LLMDecision."""


class LLMConnector:
    """Клиент для взаимодействия с LLM через ProxyAPI / OpenRouter (OpenAI-совместимый)."""

    def __init__(self) -> None:
        headers: Optional[dict[str, str]] = None
        if "openrouter.ai" in OPENAI_BASE_URL:
            # необязательные заголовки OpenRouter (атрибуция приложения)
            headers = {"HTTP-Referer": OPENROUTER_REFERER, "X-Title": "T-Work Agent"}
        self._client = openai.AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=LLM_TIMEOUT,          # v2: дефолт SDK 600 с — зависший запрос блокировал агента
            max_retries=LLM_MAX_RETRIES,  # 429/5xx/обрывы соединения — с экспоненциальной паузой
            default_headers=headers,
        )
        self._structured = LLM_STRUCTURED_OUTPUT
        self._vision_enabled = True
        # reasoning-модели (o-серия, gpt-5) требуют max_completion_tokens и не
        # принимают temperature — параметры подстраиваются по первой ошибке 400
        self._tokens_param = "max_tokens"
        self._send_temperature = True
        logger.info(
            "LLM инициализирован: model=%s base=%s structured=%s",
            LLM_MODEL, OPENAI_BASE_URL, self._structured,
        )

    async def decide(
        self,
        state: PageState,
        context: Union[DecisionContext, set[int], None] = None,
    ) -> LLMDecision:
        """Отправить состояние в LLM и получить решение."""
        if not isinstance(context, DecisionContext):
            context = DecisionContext()   # совместимость с v2: decide(state, banned_indices)

        text = self.build_user_message(state, context)
        logger.debug("LLM user msg (%d симв.):\n%s", len(text), text)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._user_content(text, context.image_b64)},
        ]
        temperature = context.temperature if context.temperature is not None else LLM_TEMPERATURE

        for attempt in (1, 2):
            raw = await self._complete(messages, temperature)
            if raw is None:
                return LLMDecision.skip("ошибка API LLM")
            try:
                decision = self.parse_response(raw)
            except DecisionParseError as exc:
                logger.warning("Ответ LLM не прошёл проверку (попытка %d): %s", attempt, exc)
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        f"Ответ не прошёл проверку: {exc}. Верни исправленный JSON строго по схеме, "
                        "без пояснений."
                    )},
                ]
                continue
            self._log_decision(decision)
            return decision
        return LLMDecision.skip("невалидный ответ LLM дважды")

    # ------------------------------------------------------------------
    # Запрос
    # ------------------------------------------------------------------

    async def _complete(self, messages: list[dict[str, Any]], temperature: float) -> Optional[str]:
        kwargs: dict[str, Any] = {
            "model": LLM_MODEL,
            self._tokens_param: LLM_MAX_TOKENS,
            "messages": messages,
            "response_format": (
                {"type": "json_schema", "json_schema": DECISION_JSON_SCHEMA}
                if self._structured else {"type": "json_object"}
            ),
        }
        if self._send_temperature:
            kwargs["temperature"] = temperature
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as exc:
            message = str(exc).lower()
            if "max_tokens" in message and self._tokens_param == "max_tokens":
                logger.warning("Модель требует max_completion_tokens — переключаюсь")
                self._tokens_param = "max_completion_tokens"
                return await self._complete(messages, temperature)
            if "temperature" in message and self._send_temperature:
                logger.warning("Модель не принимает temperature — отправляю без него")
                self._send_temperature = False
                return await self._complete(messages, temperature)
            if self._structured and ("response_format" in message or "json_schema" in message or "schema" in message):
                logger.warning("Structured Outputs не поддерживаются (%s) — переключаюсь на json_object", exc)
                self._structured = False
                return await self._complete(messages, temperature)
            if self._vision_enabled and "image" in message and self._has_image(messages):
                logger.warning("Модель не принимает изображения (%s) — отправляю без картинки", exc)
                self._vision_enabled = False
                return await self._complete(self._strip_images(messages), temperature)
            logger.error("LLM API ошибка запроса: %s", exc)
            return None
        except openai.OpenAIError as exc:
            logger.error("LLM API ошибка: %s", exc)
            return None

        choice = response.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            logger.warning("Модель отказалась отвечать: %s", refusal)
            return None
        if choice.finish_reason == "length":
            logger.warning("Ответ обрезан по max_tokens=%d — увеличьте LLM_MAX_TOKENS", LLM_MAX_TOKENS)
        usage = getattr(response, "usage", None)
        if usage is not None:
            logger.debug("Токены: prompt=%s completion=%s", usage.prompt_tokens, usage.completion_tokens)
        return choice.message.content or ""

    def _user_content(self, text: str, image_b64: Optional[str]) -> Union[str, list[dict[str, Any]]]:
        if not image_b64 or not self._vision_enabled:
            return text
        return [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}", "detail": LLM_VISION_DETAIL,
            }},
        ]

    @staticmethod
    def _has_image(messages: list[dict[str, Any]]) -> bool:
        return any(isinstance(m.get("content"), list) for m in messages)

    @staticmethod
    def _strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stripped = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
            stripped.append({**message, "content": content})
        return stripped

    # ------------------------------------------------------------------
    # Построение user-сообщения
    # ------------------------------------------------------------------

    @staticmethod
    def build_user_message(state: PageState, context: DecisionContext) -> str:
        parts: list[str] = ["═══ ЗАДАНИЕ ═══"]
        parts.append(state.task_text or "(текст задания не обнаружен — ориентируйся на элементы и изображение)")
        if context.image_b64:
            parts.append("(К сообщению приложено изображение из задания.)")

        if state.hint_text:
            parts += ["", "═══ ПОДСКАЗКА ═══", state.hint_text]
        if state.alerts:
            parts += ["", "═══ СООБЩЕНИЯ СТРАНИЦЫ ═══"] + [f"• {a}" for a in state.alerts]

        status: list[str] = []
        if state.loading:
            status.append("Идёт загрузка (виден спиннер).")
        if state.overlay_text:
            status.append(f"Часть элементов перекрыта: {state.overlay_text}")
        status += state.scroll_hints
        if status:
            parts += ["", "═══ СОСТОЯНИЕ ═══"] + status

        shown, _ = select_for_prompt(state.elements)   # число строк, которые реально увидит модель
        parts += ["", f"═══ ЭЛЕМЕНТЫ ({len(shown)}) ═══", build_elements_prompt(state.elements)]

        if context.history:
            parts += ["", f"═══ ИСТОРИЯ (шаг {context.step_in_task} этого задания) ═══"] + context.history
        if context.forbidden:
            parts += ["", "═══ НЕ ПОВТОРЯТЬ ═══"] + [f"• {f}" for f in context.forbidden]
        if context.notes:
            parts += ["", "═══ ВНИМАНИЕ ═══"] + [f"⚠ {n}" for n in context.notes]

        parts += ["", "Выбери одно действие. Ответ — только JSON по схеме."]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Парсинг ответа
    # ------------------------------------------------------------------

    @staticmethod
    def parse_response(raw: str) -> LLMDecision:
        """JSON → LLMDecision. Терпит ```json-обёртки и «грязные» типы полей."""
        text = (raw or "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise DecisionParseError("в ответе нет JSON-объекта")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise DecisionParseError(f"невалидный JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise DecisionParseError("JSON должен быть объектом")
        try:
            return LLMDecision.model_validate(data)
        except ValidationError as exc:
            errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise DecisionParseError(errors) from exc

    # Совместимость с v2
    _parse_response = parse_response

    @staticmethod
    def _log_decision(decision: LLMDecision) -> None:
        logger.info(
            "LLM решение: action=%s index=%s text=%r conf=%.2f",
            decision.action.value, decision.target_index,
            (decision.target_text or "")[:40], decision.confidence,
        )
        if decision.goal:
            logger.info("  цель: %s", decision.goal[:200])
        if decision.observation:
            logger.info("  наблюдение: %s", decision.observation[:300])
        logger.info("  рассуждение: %s", decision.reasoning[:300])
