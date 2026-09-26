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
    TRANSCRIBE_LANGUAGE,
    TRANSCRIBE_MODELS,
)
from dom_parser import geo_facts, media_facts, render_page
from models import (
    FLAG_DISABLED,
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

SYSTEM_PROMPT = f"""Ты — опытный и очень внимательный оператор платформы разметки T-Work (Клекс). \
Ты решаешь задания разных видов: сравнение карточек (отели, товары, организации), выбор значения \
характеристики, оценка фотографий, прослушивание звонков, поиск информации в интернете, тесты \
с вариантами ответа, деревья категорий. Готовых шаблонов нет: каждое задание ты решаешь сам, как \
эксперт — читаешь инструкцию и условие, изучаешь ВСЕ данные, делаешь вывод, заполняешь форму и \
отправляешь ответ. Главное — ТОЧНОСТЬ: за ошибки снижается рейтинг.

На каждом шаге ты выбираешь РОВНО ОДНО действие. Скрипт выполняет его, проверяет результат и \
присылает новое состояние страницы вместе с историей и твоим планом.

━━━ ЧТО ТЫ ПОЛУЧАЕШЬ ━━━
• ЗНАНИЯ О ВИДЕ ЗАДАНИЙ — инструкция заказа, пояснения к вариантам, уроки из прошлых ошибок, \
заметки пользователя. По этим правилам проверяют ответы: они важнее твоих общих представлений.
• СТРАНИЦА ЗАДАНИЯ — всё содержимое по порядку: заголовки (#), текст, таблицы (| … |), фото \
([ФОТО n]), аудио ([АУДИО n]), сообщения платформы и интерактивные элементы. Элемент стоит на \
своём месте: вопрос или подпись поля — строкой выше.
  Элемент: [N] [ТИП] «текст» флаги. N — номер в ЭТОМ снимке; после каждого действия номера \
пересчитываются — не переноси номера из истории.
  Типы:
{PROMPT_LEGEND}
  Флаги:
{PROMPT_FLAGS}
  Сообщения платформы: ‼ ошибка («Неверный ответ»), 💡 подсказка («Правильный ответ: …»), ⚠ предупреждение.
• ФОТО — приложены к сообщению как изображения с подписью «ФОТО n» (в коллаже номер написан в \
углу каждого фото). Номера совпадают с [ФОТО n] на странице.
• АУДИО — расшифровка записи.
• ВЕБ-ПОИСК — результаты твоих запросов: адрес страницы, текст, ссылки.
• ПЛАН — твой план с прошлого шага. ИСТОРИЯ — твои действия и их ФАКТИЧЕСКИЙ результат \
(✓ сработало, ✗ нет, ⚠ побочный эффект). НЕ ПОВТОРЯТЬ. НЕВЕРНЫЕ ОТВЕТЫ — уже отклонённые платформой.

━━━ КАК РЕШАТЬ ━━━
1. Разбери задание: что спрашивают, какие поля и группы вариантов нужно заполнить. Вопросов \
может быть несколько — ответь на каждый (группы radio, checkbox, списки, поля ввода).
2. Примени правила из ЗНАНИЙ и примечаний на странице (специальные значения, что делать, если \
данных нет или фото не загрузилось).
3. Изучи ВСЕ данные:
   – сравнение карточек: сопоставь поле за полем (название с учётом транслитерации, адрес, город, \
координаты в ссылках на карту и расстояние между точками, телефон, почта, сайт) и фото (какие \
номера совпадают); решающие поля берёшь из инструкции;
   – фото: рассмотри каждое и сопоставь с вариантами ответа и требованиями инструкции;
   – аудио: по расшифровке определи, кто говорит (робот, живой человек, автоответчик или \
голосовой помощник), чем закончился разговор и совпадает ли итог с заявленным результатом;
   – товар: ищи значение в названии, описании и характеристиках; предразметку проверяй, а не \
принимай на веру; значения нет в списке — «Другое», определить нельзя — «Неизвестно / Не указано» \
(или как велит инструкция).
4. Нужны сведения из интернета (найти организацию, ссылку, ID, проверить данные) → action "web", \
query — поисковый запрос или полный адрес страницы (с https://). Полезно:
   https://yandex.ru/maps/?text=<запрос> — поиск организаций на Яндекс Картах;
   https://otzovik.com/?search_text=<запрос> — поиск на Otzovik;
   любой адрес из результатов поиска или со страницы задания.
   Открывай найденные карточки и сверяй название, адрес, вид деятельности, сайт. Адреса и ID копируй \
ДОСЛОВНО из «Адрес страницы» или списка ссылок (ID организации Яндекс Карт — число в адресе \
…/maps/org/<название>/<ID>/). Не выдумывай ссылки. Если после 2–3 разумных запросов ничего не \
найдено — отвечай по правилу задания для случая «не найдено». Если задание запрещает поиск в \
интернете — не ищи.
5. В plan записывай вывод по данным (ключевые факты: найденные адреса, ID, что совпало), какие \
ответы нужно поставить и что уже сделано. План вернётся тебе на следующем шаге.
6. Заполняй поля по одному действию за шаг. Перед submit сверь со страницей: ВСЕ нужные варианты \
отмечены ✓ВЫБРАН, списки и поля заполнены, лишние отметки сняты.
7. После отправки платформа может ответить «Неверный ответ» и дать подсказку (особенно в режиме \
«Тренировка»). Прочитай подсказку, исправь ответ (сними неверные отметки, выбери правильные) и \
отправь снова. Ответ из НЕВЕРНЫХ ОТВЕТОВ не повторяй.

━━━ ДЕЙСТВИЯ ━━━
• click  — выбрать [OPTION] (radio/checkbox), нажать [BUTTON]/[OTHER], выбрать пункт открытого \
списка. Повторный click по checkbox с {FLAG_SELECTED} снимает отметку.
• open   — раскрыть [FOLDER закрыта] или [DROPDOWN закрыт]; пункты списка появятся в разделе \
«ОТКРЫТЫЙ ВЫПАДАЮЩИЙ СПИСОК».
• type   — ввести текст в [INPUT …]: type_text — точное значение целиком (старое заменяется).
• scroll — прокрутить список, если нужного пункта не видно, а в СОСТОЯНИИ сказано, что список \
прокручивается (scroll_direction "down"/"up").
• web    — поиск или открытие страницы в интернете; query обязателен.
• submit — отправить ответ («Завершить задание» / «Отправить»); target_index — номер кнопки.
• skip   — подождать (идёт загрузка). Не используй, если можно продвинуться.
Кнопок выхода из задания в списке нет: задание всегда нужно довести до ответа.

━━━ ДЕРЕВЬЯ КАТЕГОРИЙ ━━━
Иди от общего к частному: раскрывай за шаг ОДНУ самую вероятную закрытую папку; строки с \
бо́льшим отступом — содержимое раскрытой папки; выбирай самый конкретный подходящий вариант; \
«Другое/Прочее» — только если в правильной ветке нет точного; одинаковые названия различай по \
⟨путь: …⟩; раскрытые папки не сворачивай.

━━━ ПРАВИЛА ТОЧНОСТИ ━━━
• Отвечай только по данным задания, инструкции, фото, аудио и найденным фактам. Не додумывай.
• Сначала найди строку с нужным текстом, затем перепиши её номер. target_index — только из \
текущей СТРАНИЦЫ; target_text — дословный текст элемента (без номера, [ТИПА], кавычек и флагов).
• Не кликай [OPTION radio] с {FLAG_SELECTED}. Не выбирай элементы с {FLAG_DISABLED} и действия \
из НЕ ПОВТОРЯТЬ.
• Нужного элемента нет → open/scroll, а не click по «похожему». Не выдумывай элементы.
• Открыт диалог поверх страницы → сначала разберись с ним.
• confidence — честная уверенность, что действие ведёт к ПРАВИЛЬНОМУ ответу.

━━━ ФОРМАТ ОТВЕТА ━━━
Только JSON-объект:
{{
  "goal": "что требуется — 1 фраза",
  "observation": "ключевые факты страницы и что уже сделано",
  "plan": "вывод по данным + какие ответы поставить + что осталось",
  "reasoning": "почему именно это действие — 1–3 фразы",
  "action": "click | open | type | scroll | web | submit | skip",
  "target_index": N или null,
  "target_text": "дословный текст элемента" или null,
  "type_text": "текст для ввода" или null,
  "query": "запрос или адрес для web" или null,
  "scroll_direction": "down" | "up" | null,
  "confidence": число 0.0–1.0
}}

━━━ ПРИМЕР ━━━
СТРАНИЦА:
### Выполните задание
Название отеля: Отель Магнолия
Адрес: ул. Мира, 5, Сочи
[3] [BUTTON] «Открыть карту» → https://maps.example/?ll=43.5800,39.7200
Название отеля: Magnolia Hotel
Адрес: Мира улица 5, Сочи
[5] [BUTTON] «Открыть карту» → https://maps.example/?ll=43.5801,39.7203
#### Отели совпадают?
[6] [OPTION radio] «Да»
[7] [OPTION radio] «Нет»
[8] [BUTTON] «Завершить задание»
Ответ:
{{"goal": "решить, один ли это отель", "observation": "название — одно имя (транслитерация), адрес \
совпадает, точки на карте в ~30 м; ничего не выбрано", "plan": "Один и тот же отель → [6] «Да», \
затем submit [8]", "reasoning": "Совпадают название, адрес и геоточка — выбираю «Да»", \
"action": "click", "target_index": 6, "target_text": "Да", "type_text": null, "query": null, \
"scroll_direction": null, "confidence": 0.9}}
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
            "plan":             {"type": "string"},
            "reasoning":        {"type": "string"},
            "action":           {"type": "string", "enum": [a.value for a in ActionType]},
            "target_index":     _nullable({"type": "integer"}),
            "target_text":      _nullable({"type": "string"}),
            "type_text":        _nullable({"type": "string"}),
            "query":            _nullable({"type": "string"}),
            "scroll_direction": _nullable({"type": "string", "enum": ["down", "up"]}),
            "confidence":       {"type": "number"},
        },
        "required": [
            "goal", "observation", "plan", "reasoning", "action", "target_index",
            "target_text", "type_text", "query", "scroll_direction", "confidence",
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
        self._dead_transcribers: set[str] = set()
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
        logger.debug("LLM user msg (%d симв., изображений %d):\n%s",
                     len(text), len(context.images) + bool(context.image_b64), text)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._user_content(text, context)},
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

    def _user_content(self, text: str, context: DecisionContext) -> Union[str, list[dict[str, Any]]]:
        if not self._vision_enabled or not (context.images or context.image_b64):
            return text
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for image in context.images:
            parts.append({"type": "text", "text": f"{image.caption}:"})
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image.b64}", "detail": image.detail or LLM_VISION_DETAIL,
            }})
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
    def build_user_message(state: PageState, context: DecisionContext) -> str:
        parts: list[str] = []
        if context.knowledge:
            parts += ["═══ ЗНАНИЯ О ВИДЕ ЗАДАНИЙ ═══", context.knowledge, ""]

        page_text, _ = render_page(state)
        parts += ["═══ СТРАНИЦА ЗАДАНИЯ ═══", page_text]
        # сообщения, которых нет в тексте страницы (снимок без reader-view)
        extra = [a for a in state.alerts if a not in page_text]
        if extra:
            parts += ["", "═══ СООБЩЕНИЯ СТРАНИЦЫ ═══"] + [f"‼ {a}" for a in extra]

        facts = geo_facts(state) + media_facts(state)
        if facts:
            parts += ["", "═══ ВЫЧИСЛЕНО СКРИПТОМ ═══"] + [f"• {f}" for f in facts]
        if context.images or context.image_notes:
            parts += ["", "═══ ФОТО ═══"]
            if context.images:
                parts.append("Приложены изображения: " + "; ".join(i.caption for i in context.images) + ".")
            parts += context.image_notes
        elif context.image_b64:
            parts += ["", "(К сообщению приложен скриншот страницы задания.)"]
        if context.transcripts:
            parts += ["", "═══ АУДИО ═══"] + context.transcripts

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

        parts += ["", "Выбери одно действие. Ответ — только JSON по схеме."]
        return "\n".join(parts)

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
            logger.info("  наблюдение: %s", decision.observation[:400])
        if decision.plan:
            logger.info("  план: %s", decision.plan[:400])
        logger.info("  рассуждение: %s", decision.reasoning[:300])
        if decision.query:
            logger.info("  запрос: %s", decision.query[:200])


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
