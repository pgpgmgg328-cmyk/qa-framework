"""openrouter_connector.py v4 — LLM-клиент (ProxyAPI / OpenRouter, OpenAI-совместимый API).

v4 (экономия токенов без потери точности):
- пакет действий: за один ответ модель выбирает ответы на все видимые вопросы и отправляет —
  вызовов на задание в 2–3 раза меньше; исполнитель проверяет каждое действие;
- системный промпт сокращён; советы по фото, аудио, картам, спискам, дереву и поиску
  приходят в сообщении только для страниц, где они нужны («ПОДСКАЗКИ»);
- порядок сообщения под кэш OpenAI: неизменное внутри задания (знания, подсказки, фото,
  расшифровка) — в начале, меняющееся (страница, история) — в конце; повторные вызовы по
  заданию оплачиваются по цене кэша;
- компактный формат ответа: без поля goal, одно поле value вместо type_text/query/scroll;
- учёт токенов: вызовы, вход (из них из кэша), выход — по заданию и за весь запуск.

v3: Structured Outputs с откатом на json_object, адаптация параметров reasoning-моделей,
одна попытка «ремонта» JSON, учёт refusal и finish_reason=length.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
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
    TRANSCRIBE_LANGUAGE,
    TRANSCRIBE_MODELS,
)
from dom_parser import _plain_text, geo_facts, media_facts, render_page
from models import (
    FLAG_DISABLED,
    FLAG_SELECTED,
    PROMPT_FLAGS,
    PROMPT_LEGEND,
    ActionType,
    DecisionContext,
    ElementKind,
    LLMDecision,
    PageState,
    PlannedAction,
    split_value,
)

logger = logging.getLogger("twork.llm")


# ---------------------------------------------------------------------------
# Системный промпт v4 (неизменный — кэшируется OpenAI во всех вызовах)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""Ты — опытный и очень внимательный оператор платформы разметки T-Work (Клекс). Задания \
бывают разные: сравнение карточек, выбор значения характеристики, оценка фото, прослушивание \
звонков, поиск в интернете, тесты, деревья категорий. Шаблонов нет: реши задание как эксперт — \
прочитай инструкцию и условие, изучи ВСЕ данные, сделай вывод, заполни форму и отправь ответ. \
Главное — ТОЧНОСТЬ: за ошибки снижается рейтинг.

━━━ ЧТО ТЫ ПОЛУЧАЕШЬ ━━━
• ЗНАНИЯ — инструкция к этому виду заданий, пояснения к вариантам, уроки прошлых ошибок, заметки \
пользователя. По этим правилам проверяют ответы: они важнее твоих общих представлений.
• ПОДСКАЗКИ — советы для элементов этой страницы (фото, аудио, карты, списки, дерево, поиск).
• ФОТО — изображения с подписью «ФОТО n» (в коллаже номер написан в углу каждого фото); номера \
совпадают с [ФОТО n] на странице. АУДИО — расшифровка записи.
• СТРАНИЦА — всё содержимое по порядку: заголовки (#), текст, таблицы (| … |), [ФОТО n], [АУДИО n], \
сообщения платформы (‼ ошибка, 💡 подсказка, ⚠ предупреждение) и элементы. Вопрос или подпись поля \
стоит строкой выше элемента.
  Элемент: [N] [ТИП] «текст» флаги. N — номер в ЭТОМ снимке; после действий номера пересчитываются.
{PROMPT_LEGEND}
{PROMPT_FLAGS}
• ВЕБ-ПОИСК — результаты твоих запросов. ПЛАН — твои выводы с прошлого шага. ИСТОРИЯ — действия и их \
фактический результат (✓ сработало, ✗ нет, ⚠ побочный эффект). НЕ ПОВТОРЯТЬ. НЕВЕРНЫЕ ОТВЕТЫ — \
уже отклонены платформой.

━━━ КАК РЕШАТЬ ━━━
1. Пойми, что требуется: все вопросы, группы вариантов и поля формы — ответить нужно на каждый.
2. Примени ЗНАНИЯ и примечания на странице (особые значения, что делать, если данных нет).
3. Изучи все данные и сделай вывод по фактам; не додумывай. Нужны сведения из интернета — action web.
4. В plan запиши вывод с ключевыми фактами (найденные адреса, ID, что совпало) и что осталось сделать \
— план вернётся к тебе на следующем шаге.
5. После «Неверный ответ» или неудачной отправки (✗ в ИСТОРИИ) прочитай сообщения и подсказку платформы, \
исправь ответ (сними неверные отметки, выбери правильные) и отправь снова. Ответы из НЕВЕРНЫХ ОТВЕТОВ \
не повторяй.

━━━ ДЕЙСТВИЯ (поле actions — одно или несколько, выполняются по порядку) ━━━
• click — выбрать [OPTION], нажать [BUTTON]/[OTHER], выбрать пункт открытого списка; повторный click \
по checkbox с {FLAG_SELECTED} снимает отметку.
• open — раскрыть [FOLDER закрыта] или [DROPDOWN закрыт].
• type — ввести value в [INPUT …] (старое значение заменяется). URL и ID копируй дословно.
• scroll — прокрутить список, value "down"/"up" (если нужного пункта не видно, а список прокручивается).
• web — поиск в интернете или открытие страницы: value — запрос или полный адрес (https://…), \
например https://yandex.ru/maps/?text=<запрос> или https://otzovik.com/?search_text=<запрос>.
• submit — отправить ответ («Завершить задание»); target_index — номер кнопки.
• skip — подождать загрузку. Не используй, если можно продвинуться.
Пакет действий:
• включай сразу все действия, цели которых видны СЕЙЧАС: ответы на все вопросы и ввод во все поля;
• submit — последним и только если ответ полностью готов и ты в нём уверен: пакет вместе с уже \
отмеченным ({FLAG_SELECTED}) отвечает на ВСЕ вопросы, обязательные поля заполнены, лишние отметки сняты;
• open, web и scroll — только последним действием: после них нужен новый взгляд на страницу;
• скрипт проверяет каждое действие; при сбое или неожиданном изменении страницы (например, появилось \
новое поле) остальные действия отменяются, и ты получишь новое состояние.
Кнопок выхода из задания в списке нет: задание всегда нужно довести до ответа.

━━━ ПРАВИЛА ТОЧНОСТИ ━━━
• Сначала найди строку с нужным текстом, затем перепиши её номер. target_index — только из текущей \
СТРАНИЦЫ; target_text — дословный текст элемента (без номера, [ТИПА], кавычек и флагов).
• Не кликай [OPTION radio] с {FLAG_SELECTED}. Не выбирай элементы с {FLAG_DISABLED} и действия из \
НЕ ПОВТОРЯТЬ. Нужного элемента нет — open/scroll, а не click по «похожему»; не выдумывай элементы.
• Открыт диалог поверх страницы — сначала разберись с ним.
• confidence — честная уверенность, что ответ ПРАВИЛЬНЫЙ.

━━━ ФОРМАТ ОТВЕТА (только JSON) ━━━
{{"observation": "что требуется и ключевые факты", "plan": "вывод по данным и что осталось", \
"reasoning": "почему эти действия", "actions": [{{"action": "click|open|type|scroll|web|submit|skip", \
"target_index": N или null, "target_text": "дословный текст элемента" или null, "value": "текст для \
type/web/scroll" или null}}], "confidence": 0.0–1.0}}
Пример: вопрос «Отели совпадают?» с [6] «Да», [7] «Нет» и кнопкой [8] «Завершить задание»; названия — \
одно имя в транслитерации, адреса совпадают, точки в 30 м:
{{"observation": "решить, один ли это отель; названия и адреса совпадают, точки в 30 м", "plan": "один \
и тот же отель → «Да», отправить", "reasoning": "совпали название, адрес и геоточка", "actions": \
[{{"action": "click", "target_index": 6, "target_text": "Да", "value": null}}, {{"action": "submit", \
"target_index": 8, "target_text": "Завершить задание", "value": null}}], "confidence": 0.9}}
"""

# Советы, которые нужны только на некоторых страницах: модель получает их в сообщении
# («ПОДСКАЗКИ»), когда на странице есть соответствующие элементы. Так системный промпт
# короче, а на нужных страницах совет не теряется.
HINTS: dict[str, str] = {
    "compare": (
        "Сравнение карточек: сопоставь поле за полем — название (с учётом транслитерации и перевода), "
        "адрес, город, координаты (расстояние между точками посчитано в ВЫЧИСЛЕНО СКРИПТОМ), телефон, "
        "почта, сайт, фото (какие совпадают по номерам). Решающие поля — по инструкции."
    ),
    "photos": (
        "Фото: рассмотри каждое фото по номеру и сопоставь с вариантами ответа и требованиями инструкции; "
        "учитывай, какие фото не загрузились."
    ),
    "audio": (
        "Аудио: по расшифровке определи, кто говорит (робот, живой человек, автоответчик или голосовой "
        "помощник), чем закончился разговор и совпадает ли итог с заявленным результатом."
    ),
    "dropdown": (
        "Списки значений: сначала open, затем click по пункту с пометкой «во всплывающем списке». Значение "
        "ищи в названии, описании и характеристиках; предразметку проверяй, а не принимай на веру."
    ),
    "special": (
        "Особые значения: значения нет в списке — «Другое»; определить нельзя — «Неизвестно / Не указано» "
        "(или как велит инструкция)."
    ),
    "tree": (
        "Дерево категорий: иди от общего к частному — раскрывай за шаг ОДНУ самую вероятную закрытую "
        "папку; строки с бо́льшим отступом — содержимое раскрытой папки; выбирай самый конкретный "
        "подходящий вариант; «Другое/Прочее» — только если в правильной ветке нет точного; одинаковые "
        "названия различай по ⟨путь: …⟩; раскрытые папки не сворачивай."
    ),
    "web": (
        "Поиск в интернете: открывай найденные карточки и сверяй название, адрес, вид деятельности, сайт. "
        "Адреса и ID копируй ДОСЛОВНО из «Адрес страницы» или списка ссылок (ID организации Яндекс Карт — "
        "число в адресе …/maps/org/<название>/<ID>/). Не выдумывай ссылки. После 2–3 разумных запросов "
        "без результата отвечай по правилу задания для случая «не найдено». Если задание запрещает поиск "
        "в интернете — не ищи."
    ),
}
# Задание про поиск: «найти организацию», «через поиск в Яндексе», «URL на Otzovik»
_WEB_WORDS = re.compile(r"(найд|найти|поиск|интернет|яндекс\s*карт|otzovik|отзовик|2гис|google|гугл)", re.IGNORECASE)
# …но если задание прямо запрещает поиск, совет по поиску не нужен (запрет модель прочтёт на странице)
_WEB_FORBIDDEN = re.compile(r"(нельзя|запрещ|не используй)[^.\n]{0,60}(поиск|интернет)", re.IGNORECASE)
_SPECIAL_WORDS = re.compile(r"(другое|неизвестно|не указано)", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^(выберите|выбрать|не выбран|select|choose|укажите)", re.IGNORECASE)


def _is_value_list(e) -> bool:
    """Список значений ответа («Цвет: Выберите значение»), а не меню кнопки («Опции завершения
    задания») и не служебный переключатель."""
    if e.kind != ElementKind.DROPDOWN:
        return e.kind == ElementKind.INPUT and e.input_type == "select"
    return bool(e.caption or e.placeholder or e.value or _PLACEHOLDER.match(e.text or ""))


def page_hints(state: PageState, context: DecisionContext) -> list[str]:
    """Подсказки для этой страницы — по признакам, которые не меняются внутри задания."""
    text = _plain_text(state.reader) + "\n" + " ".join(e.label() for e in state.elements if not e.aux)
    popup = _plain_text(state.popup_lines)
    visible = [e for e in state.elements if not e.aux]
    keys: list[str] = []
    if len(geo_facts(state)) or re.search(r"совпада", text, re.IGNORECASE):
        keys.append("compare")
    if state.images:
        keys.append("photos")
    if state.audios:
        keys.append("audio")
    if any(_is_value_list(e) for e in visible):
        keys.append("dropdown")
        if _SPECIAL_WORDS.search(text) or _SPECIAL_WORDS.search(popup):
            keys.append("special")
    if any(e.kind == ElementKind.FOLDER for e in visible):
        keys.append("tree")
    if context.research or (_WEB_WORDS.search(text) and not _WEB_FORBIDDEN.search(text)):
        keys.append("web")
    return [HINTS[k] for k in keys]


# Совместимость с v2
_SYSTEM_PROMPT = SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Structured Outputs: схема ответа (порядок полей = порядок генерации,
# поэтому рассуждения идут ДО выбора действия — это и есть Chain-of-Thought)
# ---------------------------------------------------------------------------

def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


_ACTION_ITEM: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action":       {"type": "string", "enum": [a.value for a in ActionType]},
        "target_index": _nullable({"type": "integer"}),
        "target_text":  _nullable({"type": "string"}),
        "value":        _nullable({"type": "string"}),
    },
    "required": ["action", "target_index", "target_text", "value"],
}

DECISION_JSON_SCHEMA: dict[str, Any] = {
    "name": "agent_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "observation": {"type": "string"},
            "plan":        {"type": "string"},
            "reasoning":   {"type": "string"},
            "actions":     {"type": "array", "items": _ACTION_ITEM},
            "confidence":  {"type": "number"},
        },
        "required": ["observation", "plan", "reasoning", "actions", "confidence"],
    },
}


@dataclass
class UsageStats:
    """Расход токенов: вызовы модели, вход (из них из кэша OpenAI), выход."""
    calls: int = 0
    prompt: int = 0
    cached: int = 0
    completion: int = 0

    def add(self, usage: Any) -> None:
        self.calls += 1
        self.prompt += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion += int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        self.cached += int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0

    def snapshot(self) -> "UsageStats":
        return UsageStats(self.calls, self.prompt, self.cached, self.completion)

    def since(self, before: "UsageStats") -> "UsageStats":
        return UsageStats(self.calls - before.calls, self.prompt - before.prompt,
                          self.cached - before.cached, self.completion - before.completion)

    def render(self) -> str:
        cached = f" (из кэша {self.cached})" if self.cached else ""
        return f"вызовов модели {self.calls}, токенов: вход {self.prompt}{cached}, выход {self.completion}"


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
        self._dead_transcribers: set[str] = set()
        self.usage = UsageStats()
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

        prefix, main = self.build_user_parts(state, context)
        logger.debug("LLM user msg (%d симв., изображений %d):\n%s\n\n%s",
                     len(prefix) + len(main), len(context.images) + bool(context.image_b64), prefix, main)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._user_content(prefix, main, context)},
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
            self.usage.add(usage)
            details = getattr(usage, "prompt_tokens_details", None)
            logger.debug("Токены: вход=%s (кэш %s) выход=%s", usage.prompt_tokens,
                         getattr(details, "cached_tokens", 0) if details is not None else 0,
                         usage.completion_tokens)
        return choice.message.content or ""

    def _user_content(
        self, prefix: str, main: str, context: DecisionContext,
    ) -> Union[str, list[dict[str, Any]]]:
        """Сообщение для модели. Фото стоят ПОСЛЕ неизменной части (знания, подсказки) и ДО
        меняющейся (страница, история): начало запроса одинаково во всех вызовах по заданию,
        и OpenAI берёт его из кэша по сниженной цене."""
        if not self._vision_enabled or not (context.images or context.image_b64):
            return f"{prefix}\n\n{main}" if prefix else main
        parts: list[dict[str, Any]] = []
        if prefix:
            parts.append({"type": "text", "text": prefix})
        for image in context.images:
            parts.append({"type": "text", "text": f"{image.caption}:"})
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image.b64}", "detail": image.detail or LLM_VISION_DETAIL,
            }})
        parts.append({"type": "text", "text": main})
        if context.image_b64:
            parts.append({"type": "text", "text": "Скриншот страницы задания:"})
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{context.image_b64}", "detail": LLM_VISION_DETAIL,
            }})
        return parts

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
    def build_user_parts(state: PageState, context: DecisionContext) -> tuple[str, str]:
        """(неизменная часть задания, меняющаяся часть). Неизменная — знания, подсказки,
        сведения о фото и расшифровка аудио — идёт первой (кэш OpenAI)."""
        head: list[str] = []
        if context.knowledge:
            head += ["═══ ЗНАНИЯ О ВИДЕ ЗАДАНИЙ ═══", context.knowledge, ""]
        hints = page_hints(state, context)
        if hints:
            head += ["═══ ПОДСКАЗКИ ПО ЭТОЙ СТРАНИЦЕ ═══"] + [f"• {h}" for h in hints] + [""]
        if context.images or context.image_notes:
            head += ["═══ ФОТО ═══"]
            if context.images:
                head.append("Фото задания приложены ниже с подписями: "
                            + "; ".join(i.caption for i in context.images) + ".")
            head += context.image_notes + [""]
        if context.transcripts:
            head += ["═══ АУДИО ═══"] + context.transcripts + [""]
        prefix = "\n".join(head).rstrip()

        parts: list[str] = []
        page_text, _ = render_page(state)
        parts += ["═══ СТРАНИЦА ЗАДАНИЯ ═══", page_text]
        # сообщения, которых нет в тексте страницы (снимок без reader-view)
        extra = [a for a in state.alerts if a not in page_text]
        if extra:
            parts += ["", "═══ СООБЩЕНИЯ СТРАНИЦЫ ═══"] + [f"‼ {a}" for a in extra]
        facts = geo_facts(state) + media_facts(state)
        if facts:
            parts += ["", "═══ ВЫЧИСЛЕНО СКРИПТОМ ═══"] + [f"• {f}" for f in facts]
        if context.image_b64:
            parts += ["", "(В конце сообщения — скриншот страницы задания.)"]

        status: list[str] = []
        if state.loading:
            status.append("Идёт загрузка (виден индикатор на всю страницу).")
        if state.local_loading:
            status.append(f"На странице ещё крутятся индикаторы загрузки ({state.local_loading}) — "
                          "часть фото/данных может догружаться.")
        if state.overlay_text:
            status.append(f"Часть элементов перекрыта: {state.overlay_text}")
        status += state.scroll_hints
        if status:
            parts += ["", "═══ СОСТОЯНИЕ ═══"] + status

        if context.research:
            parts += ["", "═══ ВЕБ-ПОИСК (результаты твоих запросов, последние — полностью) ═══"]
            parts += context.research
        if context.plan:
            parts += ["", "═══ ТВОЙ ПЛАН С ПРОШЛОГО ШАГА ═══", context.plan]
        if context.feedback:
            parts += ["", "═══ ОТВЕТ ПЛАТФОРМЫ НА ОТПРАВКУ ═══"] + [f"‼ {f}" for f in context.feedback]
        if context.wrong_answers:
            parts += ["", "═══ НЕВЕРНЫЕ ОТВЕТЫ (платформа их отклонила — не повторяй) ═══"]
            parts += [f"✗ {w}" for w in context.wrong_answers]
        if context.history:
            parts += ["", f"═══ ИСТОРИЯ (шаг {context.step_in_task} этого задания) ═══"] + context.history
        if context.forbidden:
            parts += ["", "═══ НЕ ПОВТОРЯТЬ ═══"] + [f"• {f}" for f in context.forbidden]
        if context.notes:
            parts += ["", "═══ ВНИМАНИЕ ═══"] + [f"⚠ {n}" for n in context.notes]

        parts += ["", "Выбери действия. Ответ — только JSON по схеме."]
        return prefix, "\n".join(parts)

    @classmethod
    def build_user_message(cls, state: PageState, context: DecisionContext) -> str:
        """Всё сообщение одним текстом (лог, режим записи, тесты)."""
        prefix, main = cls.build_user_parts(state, context)
        return f"{prefix}\n\n{main}" if prefix else main

    # ------------------------------------------------------------------
    # Расшифровка аудио
    # ------------------------------------------------------------------

    async def transcribe(self, data: bytes, filename: str, mime: str) -> Optional[str]:
        """Расшифровать запись (audio.transcriptions). Модели TRANSCRIBE_MODELS пробуются
        по порядку; недоступная модель запоминается и больше не запрашивается."""
        for model in TRANSCRIBE_MODELS:
            if model in self._dead_transcribers:
                continue
            kwargs: dict[str, Any] = {"model": model, "file": (filename, data, mime)}
            if TRANSCRIBE_LANGUAGE:
                kwargs["language"] = TRANSCRIBE_LANGUAGE
            if "diarize" in model:
                kwargs["response_format"] = "diarized_json"
                kwargs["extra_body"] = {"chunking_strategy": "auto"}
            elif model.startswith("whisper"):
                kwargs["response_format"] = "verbose_json"
                kwargs["timestamp_granularities"] = ["segment"]
            else:
                kwargs["response_format"] = "json"
            try:
                result = await self._client.audio.transcriptions.create(**kwargs)
            except (openai.NotFoundError, openai.BadRequestError, openai.PermissionDeniedError) as exc:
                logger.warning("Расшифровка моделью %s недоступна: %s", model, str(exc)[:200])
                self._dead_transcribers.add(model)
                continue
            except openai.OpenAIError as exc:
                logger.warning("Расшифровка моделью %s не удалась: %s", model, str(exc)[:200])
                continue
            text = _format_transcript(result)
            if text:
                return text
        return None

    # ------------------------------------------------------------------
    # Парсинг ответа
    # ------------------------------------------------------------------

    @staticmethod
    def parse_response(raw: str) -> LLMDecision:
        """JSON → LLMDecision. Формат v4 — пакет actions; формат v3 (плоские поля action,
        target_index, …) тоже принимается. Терпит ```json-обёртки и «грязные» типы полей."""
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
        actions = data.pop("actions", None)
        if actions is not None:
            if not isinstance(actions, list) or not all(isinstance(a, dict) for a in actions):
                raise DecisionParseError("actions: нужен список объектов")
            if not actions:
                actions = [{"action": "skip"}]
            first, rest = split_value(actions[0]), actions[1:]
            for key in ("action", "target_index", "target_text", "type_text", "query", "scroll_direction"):
                if key in first:
                    data[key] = first[key]
            try:
                data["next_actions"] = [PlannedAction.from_raw(a) for a in rest]
            except ValidationError as exc:
                errors = "; ".join(f"actions.{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                raise DecisionParseError(errors) from exc
        else:
            data = split_value(data)
        try:
            return LLMDecision.model_validate(data)
        except ValidationError as exc:
            errors = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            raise DecisionParseError(errors) from exc

    # Совместимость с v2
    _parse_response = parse_response

    @staticmethod
    def _log_decision(decision: LLMDecision) -> None:
        steps = decision.steps()
        chain = " → ".join(
            f"{d.action.value}"
            + (f" [{d.target_index}]" if d.target_index is not None else "")
            + (f" «{(d.target_text or '')[:30]}»" if d.target_text else "")
            + (f" «{(d.type_text or d.query or '')[:40]}»" if (d.type_text or d.query) else "")
            for d in steps
        )
        logger.info("LLM решение (%d действ.): %s conf=%.2f", len(steps), chain, decision.confidence)
        if decision.observation:
            logger.info("  наблюдение: %s", decision.observation[:400])
        if decision.plan:
            logger.info("  план: %s", decision.plan[:400])
        if decision.reasoning:
            logger.info("  рассуждение: %s", decision.reasoning[:300])


def _fmt_ts(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "?"
    return f"{total // 60}:{total % 60:02d}"


def _format_transcript(result: Any) -> str:
    """Текст расшифровки; с таймкодами и говорящими, если API их вернул."""
    if isinstance(result, str):
        return result.strip()
    data: dict[str, Any] = {}
    if hasattr(result, "model_dump"):
        try:
            data = result.model_dump()
        except Exception:  # noqa: BLE001 — формат ответа прокси может отличаться
            data = {}
    elif isinstance(result, dict):
        data = result
    segments = data.get("segments") or []
    lines: list[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        who = f"{speaker}: " if speaker else ""
        lines.append(f"[{_fmt_ts(seg.get('start'))}–{_fmt_ts(seg.get('end'))}] {who}{text}")
    if lines:
        return "\n".join(lines)
    return str(data.get("text") or getattr(result, "text", "") or "").strip()
