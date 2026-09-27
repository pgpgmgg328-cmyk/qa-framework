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

import base64
import json
import logging
import re
import struct
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import openai
from pydantic import ValidationError

from config import (
    LLM_CHECK_BELOW,
    LLM_CHECK_MODEL,
    LLM_CHECK_PRICE,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PRICE,
    LLM_PRICE_CURRENCY,
    LLM_REASONING_EFFORT,
    LLM_STRUCTURED_OUTPUT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_VISION,
    LLM_VISION_DETAIL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENROUTER_REFERER,
    TRANSCRIBE_LANGUAGE,
    TRANSCRIBE_MODELS,
)
from dom_parser import _plain_text, geo_facts, inspection_task, media_facts, render_page
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
Пиши коротко, без пересказа страницы: observation — 1–2 предложения (при проверке фото — по строке на \
каждое фото), reasoning — одно; в plan — только выводы и факты, которые понадобятся дальше (найденные \
адреса, ID, что уже сделано).
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
    "inspection": (
        "Проверка качества по фото: «сделано качественно» — только если на КАЖДОМ фото ты убедился, что "
        "недостатков нет. В observation пройди все фото по строке: «ФОТО n: что видно (поверхность, ракурс) — "
        "недостатки или „чисто“». Недостатки: грязь, пыль, пятна, разводы, следы рук, остатки скотча или "
        "наклеек, мусор, посторонние предметы — в том числе по краям, в углах и у основания. Проверь места из "
        "уроков: там недостатки встречаются чаще всего. Сверь с инструкцией, есть ли фото всех обязательных "
        "поверхностей: если какой-то нет — ответ по правилу инструкции для этого случая. Не списывай "
        "подозрительное на блики, отражения или царапины, если не уверен."
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
        "Поиск организации: сразу открывай Яндекс Карты — https://yandex.ru/maps/?text=<название> <город> "
        "(без слов «сайт», «website» и без операторов site:, OR — на Картах они не работают); не нашлось — по "
        "домену сайта, по адресу, по названию латиницей и кириллицей. Если подходящих точек несколько — "
        "выбирай по правилу инструкции, а если его нет — ту, у которой больше всего отзывов. Открывай "
        "карточку и сверяй название, адрес, вид деятельности, сайт. Адреса и ID копируй ДОСЛОВНО из «Адрес "
        "страницы» или списка ссылок (ID организации Яндекс Карт — число в адресе …/maps/org/<название>/<ID>/). "
        "Не выдумывай ссылки. Ответ «не найдено» — только если ни Карты, ни поиск по названию, домену и адресу "
        "не дали подходящей организации. Если задание запрещает поиск в интернете — не ищи."
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
        keys.append("inspection" if inspection_task(state) else "photos")
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


# Прайс OpenAI, $ за 1 млн токенов: вход, вход из кэша, выход. У прокси (ProxyAPI) цена своя: её
# можно вписать в .env (LLM_PRICE) — тогда стоимость в логе будет по вашему тарифу, в рублях.
OPENAI_PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-5.1": (1.25, 0.125, 10.00),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.5": (5.00, 0.50, 30.00),
}


def _model_name(model: str) -> str:
    """«openai/gpt-4o-2024-08-06» → «gpt-4o»."""
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model.split("/")[-1].strip().lower())


def parse_price(text: str) -> Optional[tuple[float, float, float]]:
    """«вход,кэш,выход» за 1 млн токенов: «225,22.5,1350» или «0,75; 0,075; 4,5»."""
    parts = text.split(";") if ";" in text else text.split(",")
    try:
        values = [float(x.strip().replace(" ", "").replace(",", ".")) for x in parts]
    except ValueError:
        return None
    if len(values) == 2:                  # без цены кэша — как обычный вход
        values = [values[0], values[0], values[1]]
    return tuple(values) if len(values) == 3 else None  # type: ignore[return-value]


def model_price(model: str) -> Optional[tuple[float, float, float]]:
    """Цена модели для оценки стоимости. Свои цены (LLM_PRICE) — для основной модели и
    модели перепроверки (LLM_CHECK_PRICE); иначе прайс OpenAI для известных моделей."""
    if LLM_PRICE:
        if model == LLM_MODEL:
            return parse_price(LLM_PRICE)
        return parse_price(LLM_CHECK_PRICE) if LLM_CHECK_PRICE and model == LLM_CHECK_MODEL else None
    return OPENAI_PRICES.get(_model_name(model))


def _money(value: float, currency: str) -> str:
    if currency in ("$", "USD", "usd"):
        return f"${value:.4f}" if value < 1 else f"${value:.2f}"
    return f"{value:.2f} {currency}"


@dataclass
class UsageStats:
    """Расход: вызовы модели, токены (вход, из них из кэша OpenAI, выход) и стоимость."""
    calls: int = 0
    prompt: int = 0
    cached: int = 0
    completion: int = 0
    cost: float = 0.0
    unpriced: int = 0                     # вызовы моделей, цена которых неизвестна
    currency: str = "$"

    def add(self, usage: Any, price: Optional[tuple[float, float, float]] = None) -> None:
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        self.calls += 1
        self.prompt += prompt
        self.completion += completion
        self.cached += cached
        if price is None:
            self.unpriced += 1
        else:
            fresh = max(prompt - cached, 0)
            self.cost += (fresh * price[0] + cached * price[1] + completion * price[2]) / 1_000_000

    def snapshot(self) -> "UsageStats":
        return UsageStats(self.calls, self.prompt, self.cached, self.completion, self.cost,
                          self.unpriced, self.currency)

    def since(self, before: "UsageStats") -> "UsageStats":
        return UsageStats(self.calls - before.calls, self.prompt - before.prompt,
                          self.cached - before.cached, self.completion - before.completion,
                          self.cost - before.cost, self.unpriced - before.unpriced, self.currency)

    def render(self) -> str:
        cached = f" (из кэша {self.cached})" if self.cached else ""
        text = f"вызовов модели {self.calls}, токенов: вход {self.prompt}{cached}, выход {self.completion}"
        if self.calls > self.unpriced:
            text += f", ≈ {_money(self.cost, self.currency)}"
            if self.unpriced:
                text += " (без вызовов моделей с неизвестной ценой)"
        return text


@dataclass
class _ModelParams:
    """Параметры запроса, которые принимает модель (подстраиваются по ошибкам 400)."""
    tokens_param: str = "max_tokens"
    temperature: bool = True
    structured: bool = True
    reasoning: list[str] = field(default_factory=list)   # очередь значений reasoning_effort


def _initial_params(model: str) -> _ModelParams:
    """reasoning-модели (gpt-5.x, o-серия): max_completion_tokens и минимальное «обдумывание»
    — иначе лишний запрос с ошибкой на старте и медленные дорогие ответы."""
    params = _ModelParams(structured=LLM_STRUCTURED_OUTPUT)
    effort = LLM_REASONING_EFFORT
    if _model_name(model).startswith(("gpt-5", "o1", "o3", "o4")):
        params.tokens_param = "max_completion_tokens"
        if effort in ("", "auto"):
            params.reasoning = ["none", "minimal", "low"]
        elif effort not in ("off", "default"):
            params.reasoning = [effort]
    elif effort not in ("", "auto", "off", "default"):
        params.reasoning = [effort]
    return params


def _probe_png(size: int = 512) -> str:
    """Серая картинка size×size (PNG, base64) — узнать, во сколько токенов модель ставит фото."""
    raw = b"".join(b"\x00" + b"\x80\x80\x80" * size for _ in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return base64.b64encode(png).decode("ascii")


class ModelUnavailable(RuntimeError):
    """Модели нет у прокси (или ключ не даёт к ней доступа) — работать дальше бессмысленно."""


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
        self._vision_enabled = True
        self._dead_transcribers: set[str] = set()
        self.usage = UsageStats(currency=LLM_PRICE_CURRENCY)
        # параметры запроса по моделям: reasoning-модели (o-серия, gpt-5) требуют
        # max_completion_tokens, не все принимают temperature — подстраиваются по ошибкам 400
        self._params: dict[str, _ModelParams] = {}
        self._last_error: Optional[Exception] = None
        self._last_usage: Any = None
        logger.info(
            "LLM инициализирован: model=%s base=%s%s",
            LLM_MODEL, OPENAI_BASE_URL, f", перепроверка: {LLM_CHECK_MODEL}" if LLM_CHECK_MODEL else "",
        )

    # ------------------------------------------------------------------
    # Проверка модели перед работой
    # ------------------------------------------------------------------

    async def check_model(self) -> None:
        """Модель доступна у прокси, и сколько токенов стоит фото. Модели нет — ModelUnavailable
        (агент не открывает браузер зря); проверка не прошла по другой причине — предупреждение."""
        for model in dict.fromkeys(m for m in (LLM_MODEL, LLM_CHECK_MODEL) if m):
            await self._check_one(model, photo=model == LLM_MODEL and LLM_VISION != "off")

    async def _check_one(self, model: str, *, photo: bool) -> None:
        ask = ("Проверка связи перед работой. Верни JSON по схеме: observation, plan и reasoning — "
               "пустые строки, actions — [{\"action\": \"skip\", \"target_index\": null, "
               "\"target_text\": null, \"value\": null}], confidence — 1.")
        started = time.monotonic()
        if await self._complete([{"role": "user", "content": ask}], LLM_TEMPERATURE, model) is None:
            self._raise_if_unavailable(model)
            logger.warning("Проверка модели %s не прошла (%s) — продолжаю, но вызовы могут не работать",
                           model, str(self._last_error)[:200])
            return
        base = int(getattr(self._last_usage, "prompt_tokens", 0) or 0)
        line = f"Модель {model}: доступна, ответ за {time.monotonic() - started:.1f} с"
        if photo and base:                   # прокси сообщает расход — можно узнать цену фото
            image = [{"type": "text", "text": ask}, {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{_probe_png()}", "detail": LLM_VISION_DETAIL}}]
            if await self._complete([{"role": "user", "content": image}], LLM_TEMPERATURE, model) is not None:
                tokens = int(getattr(self._last_usage, "prompt_tokens", 0) or 0) - base
                price = model_price(model)
                cost = f" (≈ {_money(tokens * price[0] / 1_000_000, self.usage.currency)})" if price else ""
                line += f"; фото 512×512 — {tokens} токенов на вход{cost}"
                if tokens > 1500:        # у gpt-4o — 255, у новых моделей — 400–650
                    logger.warning(
                        "⚠ Модель %s считает фото очень дорого: %d токенов за картинку 512×512 (обычно "
                        "250–650). Задания с фото обойдутся в разы дороже — выберите другую модель "
                        "(LLM_MODEL в .env) или поставьте LLM_VISION_DETAIL=low", model, tokens)
        logger.info(line)

    def _raise_if_unavailable(self, model: str) -> None:
        exc = self._last_error
        message = str(exc).lower()
        missing = isinstance(exc, (openai.NotFoundError, openai.PermissionDeniedError)) or (
            isinstance(exc, openai.BadRequestError) and "model" in message
            and any(w in message for w in ("not found", "does not exist", "unknown", "invalid", "not supported",
                                            "не найден", "не поддерж", "недоступ")))
        if isinstance(exc, openai.AuthenticationError):
            raise ModelUnavailable(f"Ключ OPENAI_API_KEY не подходит к {OPENAI_BASE_URL}: {str(exc)[:200]}")
        if isinstance(exc, openai.APIStatusError) and exc.status_code == 402:
            raise ModelUnavailable(f"Прокси отказал в оплате запроса (недостаточно средств на балансе?): "
                                   f"{str(exc)[:200]}")
        if missing:
            raise ModelUnavailable(
                f"Модель {model} недоступна у {OPENAI_BASE_URL}: {str(exc)[:200]}. Впишите в .env другую "
                "модель (LLM_MODEL=…) из списка моделей в личном кабинете прокси")

    # ------------------------------------------------------------------
    # Решение
    # ------------------------------------------------------------------

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

        decision = await self._ask(LLM_MODEL, messages, temperature)
        if decision is None:
            return LLMDecision.skip("ошибка API LLM или невалидный ответ")
        if self._needs_check(decision):
            logger.info("Модель не уверена в отправляемом ответе (%.2f < %.2f) — перепроверяю моделью %s",
                        decision.confidence, LLM_CHECK_BELOW, LLM_CHECK_MODEL)
            checked = await self._ask(LLM_CHECK_MODEL, messages, temperature)
            if checked is not None:
                decision = checked
        return decision

    @staticmethod
    def _needs_check(decision: LLMDecision) -> bool:
        return (bool(LLM_CHECK_MODEL) and LLM_CHECK_MODEL != LLM_MODEL
                and decision.confidence < LLM_CHECK_BELOW
                and any(step.action == ActionType.SUBMIT for step in decision.steps()))

    async def _ask(self, model: str, messages: list[dict[str, Any]], temperature: float) -> Optional[LLMDecision]:
        """Запрос к модели + одна попытка «ремонта» JSON. None — ошибка API или ответ не разобран."""
        for attempt in (1, 2):
            raw = await self._complete(messages, temperature, model)
            if raw is None:
                return None
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
            self._log_decision(decision, model)
            return decision
        return None

    # ------------------------------------------------------------------
    # Запрос
    # ------------------------------------------------------------------

    async def _complete(
        self, messages: list[dict[str, Any]], temperature: float, model: str = LLM_MODEL,
    ) -> Optional[str]:
        params = self._params.setdefault(model, _initial_params(model))
        kwargs: dict[str, Any] = {
            "model": model,
            params.tokens_param: LLM_MAX_TOKENS,
            "messages": messages,
            "response_format": (
                {"type": "json_schema", "json_schema": DECISION_JSON_SCHEMA}
                if params.structured else {"type": "json_object"}
            ),
        }
        if params.temperature:
            kwargs["temperature"] = temperature
        if params.reasoning:
            kwargs["reasoning_effort"] = params.reasoning[0]
        self._last_error = None
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except openai.BadRequestError as exc:
            message = str(exc).lower()
            if params.reasoning and "reasoning" in message:
                dropped = params.reasoning.pop(0)
                logger.info("Модель %s не принимает reasoning_effort=%s%s", model, dropped,
                            f" — пробую {params.reasoning[0]}" if params.reasoning else " — отправляю без него")
                return await self._complete(messages, temperature, model)
            if "max_tokens" in message and params.tokens_param == "max_tokens":
                logger.warning("Модель %s требует max_completion_tokens — переключаюсь", model)
                params.tokens_param = "max_completion_tokens"
                return await self._complete(messages, temperature, model)
            if "temperature" in message and params.temperature:
                logger.warning("Модель %s не принимает temperature — отправляю без него", model)
                params.temperature = False
                return await self._complete(messages, temperature, model)
            if params.structured and ("response_format" in message or "json_schema" in message or "schema" in message):
                logger.warning("Structured Outputs не поддерживаются (%s) — переключаюсь на json_object", exc)
                params.structured = False
                return await self._complete(messages, temperature, model)
            if self._vision_enabled and "image" in message and self._has_image(messages):
                logger.warning("Модель не принимает изображения (%s) — отправляю без картинки", exc)
                self._vision_enabled = False
                return await self._complete(self._strip_images(messages), temperature, model)
            logger.error("LLM API ошибка запроса: %s", exc)
            self._last_error = exc
            return None
        except openai.OpenAIError as exc:
            logger.error("LLM API ошибка: %s", exc)
            self._last_error = exc
            return None

        choice = response.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            logger.warning("Модель отказалась отвечать: %s", refusal)
            return None
        if choice.finish_reason == "length":
            logger.warning("Ответ обрезан по max_tokens=%d — увеличьте LLM_MAX_TOKENS", LLM_MAX_TOKENS)
        usage = getattr(response, "usage", None)
        self._last_usage = usage
        if usage is not None:
            self.usage.add(usage, model_price(model))
            details = getattr(usage, "prompt_tokens_details", None)
            logger.debug("Токены %s: вход=%s (кэш %s) выход=%s", model, usage.prompt_tokens,
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
    def _log_decision(decision: LLMDecision, model: str = LLM_MODEL) -> None:
        steps = decision.steps()
        chain = " → ".join(
            f"{d.action.value}"
            + (f" [{d.target_index}]" if d.target_index is not None else "")
            + (f" «{(d.target_text or '')[:30]}»" if d.target_text else "")
            + (f" «{(d.type_text or d.query or '')[:40]}»" if (d.type_text or d.query) else "")
            for d in steps
        )
        logger.info("LLM решение%s (%d действ.): %s conf=%.2f", "" if model == LLM_MODEL else f" [{model}]",
                    len(steps), chain, decision.confidence)
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
