"""config.py — централизованная конфигурация агента v4.

Все значения можно переопределить через .env (см. .env.example).
"""

import logging
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from dotenv import dotenv_values, load_dotenv

from adapters import DEFAULT_PLATFORM, get_platform

PROJECT_DIR = Path(__file__).resolve().parent

# Ищем .env рядом с этим файлом
load_dotenv(PROJECT_DIR / ".env")

# Консоль Windows в cp1251/cp866 не умеет печатать ✓, ⛔, ═ — вместо ошибки логгера
# («--- Logging error ---») такие символы заменяются на «?»
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


def _project_path(raw: str) -> str:
    """Относительный путь считается от папки проекта, а не от текущей папки терминала."""
    if not raw:
        return ""
    path = Path(raw).expanduser()
    return str(path if path.is_absolute() else (PROJECT_DIR / path).resolve())


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Список через запятую → кортеж строк в нижнем регистре."""
    raw = os.getenv(name)
    if not raw:
        return default
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


def _env_selectors(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """CSS-селекторы через точку с запятой (внутри селектора бывают запятые), регистр сохраняется."""
    raw = os.getenv(name)
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(";") if item.strip())


def _env_default(name: str, default: str, old_defaults: tuple[str, ...] = ()) -> str:
    """Значение из .env. Значение, которое стояло в прежнем .env.example (его копируют целиком),
    считается неизменённым умолчанием — берётся новое значение по умолчанию."""
    raw = (os.getenv(name) or "").strip()
    return default if not raw or raw in old_defaults else raw


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "да")


def _parse_zoom(raw: str) -> float:
    """«75%» / «0.75» → 0.75. Некорректное значение → 1.0 (без масштабирования)."""
    try:
        value = raw.strip()
        factor = float(value.rstrip("%")) / 100.0 if value.endswith("%") else float(value)
    except (AttributeError, ValueError):
        return 1.0
    return factor if 0.25 <= factor <= 2.0 else 1.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("twork.config")
# клиент OpenAI пишет в INFO каждый HTTP-запрос — в логе агента это шум
for _noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "openai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
# Папка логов: каждый запуск — свой файл (например, logs/agent-20260927-101500.log): по нему
# можно разобрать, что делал агент и сколько потратил. Пусто — без файла
LOG_DIR: str = os.getenv("LOG_DIR", "logs").strip()
LOG_KEEP: int = int(os.getenv("LOG_KEEP", "30"))          # сколько последних файлов хранить


def enable_file_log(kind: str) -> Optional[Path]:
    """Дублировать лог в файл logs/<kind>-<дата-время>.log (старые файлы сверх LOG_KEEP удаляются)."""
    if not LOG_DIR:
        return None
    folder = Path(LOG_DIR)
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent / folder
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if LOG_KEEP > 0:
            previous = sorted(folder.glob(f"{kind}-*.log"))        # имя с датой — старые первыми
            for stale in previous[:max(len(previous) - (LOG_KEEP - 1), 0)]:
                stale.unlink(missing_ok=True)
        path = folder / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.log"
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as exc:
        logger.warning("Файл лога не создан (%s)", exc)
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s — %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(handler)
    return path

# ---------------------------------------------------------------------------
# LLM: OpenRouter (openrouter.ai) — единственный поставщик моделей
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
# ключ со страницы openrouter.ai/keys; OPENAI_API_KEY — прежнее имя (старый .env продолжает работать)
OPENROUTER_API_KEY: str = (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
_PROVIDERS = (("gpt-", "openai"), ("chatgpt-", "openai"), ("o1", "openai"), ("o3", "openai"),
              ("o4", "openai"), ("whisper", "openai"), ("gemini-", "google"), ("claude-", "anthropic"))


def openrouter_model(name: str) -> str:
    """У OpenRouter модели называются «поставщик/модель»: «gpt-5.4-mini» → «openai/gpt-5.4-mini»."""
    name = name.strip()
    if not name or "/" in name:
        return name
    return next((f"{provider}/{name}" for prefix, provider in _PROVIDERS if name.lower().startswith(prefix)), name)


def parse_models(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(openrouter_model(m) for m in text.split(",") if m.strip()))


# Модель или «лестница» моделей через запятую — от дешёвой к сильной. Каждый вид заданий агент
# начинает первой (самой дешёвой) моделью; если в тренировке она ошибается с первого раза чаще,
# чем допускает порог, этот вид переходит к следующей модели. Gemini 3.1 Flash-Lite — в 3 раза
# дешевле gpt-5.4-mini и хорошо понимает фото; gpt-5.4-mini — для того, с чем она не справится
DEFAULT_MODELS = "google/gemini-3.1-flash-lite,openai/gpt-5.4-mini"
LLM_MODELS: tuple[str, ...] = parse_models(os.getenv("LLM_MODEL", "")) or parse_models(DEFAULT_MODELS)
LLM_MODEL: str = LLM_MODELS[0]
# Лестница: переход к следующей модели, когда в тренировке у текущей верно с первого раза меньше
# этой доли ответов (решение — после LADDER_MIN_TASKS заданий на этой модели или раньше, если
# порог уже недостижим). Модель, к которой перешли, решает не меньше LADDER_MIN_TASKS заданий,
# прежде чем агент сравнит её с прежней: по 1–3 заданиям точность не видна
LADDER_MIN_ACCURACY: float = float(os.getenv("LADDER_MIN_ACCURACY", "0.8"))
LADDER_MIN_TASKS: int = int(_env_default("LADDER_MIN_TASKS", "5", ("3",)))   # «3» — из образца до v4.8
# «Обдумывание» перед ответом (reasoning у OpenRouter): auto — выключено (none → minimal → low,
# что примет модель): быстрее и дешевле; low / medium — точнее, но оплачивается как ответ
LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "auto").strip().lower()
# Перепроверка: основная модель не уверена (confidence ниже LLM_CHECK_BELOW) в ответе, который
# отправляется, — решение перепроверяет более сильная модель LLM_CHECK_MODEL. Пусто — выключено
LLM_CHECK_MODEL: str = openrouter_model(os.getenv("LLM_CHECK_MODEL", ""))
LLM_CHECK_BELOW: float = float(os.getenv("LLM_CHECK_BELOW", "0.7"))
# Прокси только для запросов к OpenRouter (браузер с T-Work работает напрямую). Нужен, если
# OpenRouter отвечает 403 «Access denied by security policy»: из России OpenRouter без VPN не
# пускает, а T-Work через VPN может не работать. Пример: socks5://127.0.0.1:10808 — локальный порт
# вашего VPN-клиента (v2rayN, Hiddify, Nekoray…), или http://логин:пароль@адрес:порт
LLM_PROXY: str = os.getenv("LLM_PROXY", "").strip()
# OpenRouter временно недоступен (блокировка, нет связи, нет денег): сколько секунд ждать и
# повторять запрос внутри одного шага; дальше агент возвращается к странице и пробует снова
LLM_OUTAGE_WAIT: float = float(os.getenv("LLM_OUTAGE_WAIT", "300"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
# план + рассуждения + ответ: 1024 токенов иногда не хватало на задания с несколькими полями
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
# Таймаут одного запроса. В v2 действовал дефолт SDK — 600 с: зависший запрос
# «замораживал» агента на 10 минут.
LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))   # 429/5xx/обрывы — ретраи SDK
# Structured Outputs (json_schema strict). Если модель не поддерживает — коннектор сам откатится
# на json_object.
LLM_STRUCTURED_OUTPUT: bool = _env_bool("LLM_STRUCTURED_OUTPUT", True)
# Vision:
#   auto  — все фото задания (скачиваются в исходном качестве, по одному или коллажами
#           с номерами [ФОТО n]); если фото нет, но есть картинка/холст — скриншот фрейма;
#   frame — всегда скриншот всего фрейма задания;
#   off   — без изображений.
# (image — режим v3, «одна главная картинка», оставлен как синоним auto)
LLM_VISION: str = os.getenv("LLM_VISION", "auto").strip().lower()
LLM_VISION_DETAIL: str = os.getenv("LLM_VISION_DETAIL", "high")   # low | high | auto
VISION_MAX_IMAGES: int = int(os.getenv("VISION_MAX_IMAGES", "36"))   # больше — не отправляются
VISION_SINGLE_MAX: int = int(os.getenv("VISION_SINGLE_MAX", "6"))    # до стольких фото — по одному
# проверка качества по фото (клининг, грязь, дефекты): мелкие детали на коллаже не видны — до
# стольких фото отправляются по одному, в хорошем разрешении (дороже, но точнее)
VISION_DETAIL_MAX: int = int(os.getenv("VISION_DETAIL_MAX", "16"))
VISION_IMAGE_SIDE: int = int(os.getenv("VISION_IMAGE_SIDE", "1024"))  # длинная сторона одиночного фото
# ячейка коллажа, px: коллаж 2×2 — 774 px. gpt-4o всё равно уменьшает картинку до 768 px (и берёт
# за неё столько же токенов), а новые модели считают токены по площади — коллаж в 1030 px был
# для них на ~40% дороже без выигрыша в деталях
VISION_CELL: int = int(os.getenv("VISION_CELL", "384"))
LLM_HISTORY_SIZE: int = int(os.getenv("LLM_HISTORY_SIZE", "14"))  # строк истории в промпте

# Аудио: модели расшифровки по порядку. Gemini слушает запись сама (в обычном запросе) и отмечает
# ещё и гудки, автоответчик, голос робота; модели …transcribe и whisper — через audio.transcriptions
# (запасные). «…-diarize» дополнительно размечает говорящих.
AUDIO_TRANSCRIBE: bool = _env_bool("AUDIO_TRANSCRIBE", True)
DEFAULT_TRANSCRIBE = "google/gemini-3.1-flash-lite,openai/gpt-4o-transcribe,openai/whisper-1"
TRANSCRIBE_MODELS: tuple[str, ...] = parse_models(os.getenv("TRANSCRIBE_MODELS", ""))
if TRANSCRIBE_MODELS in ((), ("openai/gpt-4o-transcribe", "openai/whisper-1")):
    # пусто или значение из прежнего .env.example — как по умолчанию: сначала запись слушает Gemini
    TRANSCRIBE_MODELS = parse_models(DEFAULT_TRANSCRIBE)
TRANSCRIBE_LANGUAGE: str = os.getenv("TRANSCRIBE_LANGUAGE", "ru")
# Модель, которая решает задание и умеет слушать (Gemini), получает и саму запись звонка, а не
# только расшифровку: человек или автоответчик — слышно по голосу. До AUDIO_TO_MODEL_MB мегабайт
# на запись; минута звонка — около 2 тыс. токенов (≈ $0.001 у Gemini 3.1 Flash-Lite)
AUDIO_TO_MODEL: bool = _env_bool("AUDIO_TO_MODEL", True)
AUDIO_TO_MODEL_MB: float = float(os.getenv("AUDIO_TO_MODEL_MB", "8"))
# «Прослушайте звонок до конца»: перед отправкой ответа запись доигрывается до конца
AUDIO_PLAY_TO_END: bool = _env_bool("AUDIO_PLAY_TO_END", True)
AUDIO_MAX_WAIT: float = float(os.getenv("AUDIO_MAX_WAIT", "600"))   # макс. ожидание конца записи, с
OPENROUTER_REFERER: str = os.getenv("OPENROUTER_REFERER", "https://localhost/twork-agent")

# ---------------------------------------------------------------------------
# Площадка: twork (по умолчанию) | ozon. Модуль площадки (adapters/) задаёт значения по
# умолчанию для настроек ниже — адрес, фреймы, тексты кнопок, признаки окон; .env их перекрывает.
# ---------------------------------------------------------------------------
PLATFORM_NAME: str = (os.getenv("PLATFORM") or DEFAULT_PLATFORM).strip().lower()
try:
    PLATFORM = get_platform(PLATFORM_NAME)
    _PLATFORM_ERROR = ""
except ValueError as _exc:
    PLATFORM, _PLATFORM_ERROR = get_platform(DEFAULT_PLATFORM), str(_exc)
_site = PLATFORM.setting
_TWORK_URL = "https://twork.tbank.ru"


def _site_list(name: str, twork_default: tuple[str, ...]) -> tuple[str, ...]:
    """Список площадки. Значение T-Work, оставшееся в .env из .env.example (его копируют целиком),
    на другой площадке считается незаданным — берётся значение площадки."""
    value = _env_list(name, _site(name, twork_default))
    if PLATFORM.key != DEFAULT_PLATFORM and name in PLATFORM.settings and value == twork_default:
        return PLATFORM.settings[name]
    return value

# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------
def _is_twork_url(url: str) -> bool:
    """Адрес T-Work в любом виде: с «/» на конце, с путём из адресной строки, прежний t-work.ru."""
    host = (urlsplit(url if "://" in url else "https://" + url).hostname or "").lower()
    return host in ("twork.tbank.ru", "t-work.ru", "www.t-work.ru") or host.endswith(".twork.tbank.ru")


# адрес T-Work из прежнего .env на другой площадке считается незаданным
TARGET_URL: str = _env_default("TARGET_URL", _site("TARGET_URL", _TWORK_URL))
if PLATFORM.key != DEFAULT_PLATFORM and _is_twork_url(TARGET_URL):
    TARGET_URL = _site("TARGET_URL", _TWORK_URL)
VIEWPORT_WIDTH: int = int(os.getenv("VIEWPORT_WIDTH", "1280"))
VIEWPORT_HEIGHT: int = int(os.getenv("VIEWPORT_HEIGHT", "720"))
HEADLESS: bool = _env_bool("HEADLESS", False)
# Windows: свёрнутое окно браузера агент убирает за край экрана (свёрнутый Chrome не рисует
# страницу, и агент в нём работает в разы медленнее); вернуть окно — кнопка на панели задач
WINDOW_GUARD: bool = _env_bool("WINDOW_GUARD", True)
# slow_mo замедляет КАЖДЫЙ вызов Playwright (включая чтение DOM), а не только клики.
# «Человечность» кликов обеспечивают явные паузы в BrowserController, поэтому 0.
SLOW_MO: int = int(os.getenv("SLOW_MO", "0"))
BROWSER_CHANNEL: str = os.getenv("BROWSER_CHANNEL", "")   # "chrome" — установленный Google Chrome
# Постоянный профиль: куки/логин сохраняются между запусками (пустая строка — выкл.).
# Относительный путь («.browser-profile») — внутри папки проекта, откуда бы ни запускали.
USER_DATA_DIR: str = _project_path(os.getenv("USER_DATA_DIR", "").strip())
# Пусто — нативный UA установленного Chromium (жёстко зашитый Chrome/124 в v2
# расходился с реальной версией браузера и заголовками Client Hints).
USER_AGENT: str = os.getenv("USER_AGENT", "")

# Масштаб страницы — только без окна (HEADLESS=true): device_scale_factor + увеличенный
# viewport, а НЕ document.body.style.zoom (CSS-zoom ломает координаты Playwright внутри
# iframe — клики уходят мимо цели). В видимом окне страница показывается как в обычном
# браузере. Уменьшать страницу, чтобы «всё влезло», агенту не нужно: перед кликом он сам
# прокручивает к элементу.
PAGE_ZOOM: str = os.getenv("PAGE_ZOOM", "100%")
PAGE_ZOOM_FACTOR: float = _parse_zoom(PAGE_ZOOM)

# Задержки (секунды)
FRAME_LOAD_WAIT: float = float(os.getenv("FRAME_LOAD_WAIT", "4.0"))   # ждём тяжёлый фрейм
ACTION_WAIT: float = float(os.getenv("ACTION_WAIT", "0.4"))            # мин. пауза после действия
SUBMIT_WAIT: float = float(os.getenv("SUBMIT_WAIT", "6.0"))            # макс. ожидание смены задания
MOUSE_MOVE_STEPS: int = int(os.getenv("MOUSE_MOVE_STEPS", "10"))       # шагов при движении мыши
CLICK_TIMEOUT_MS: int = int(os.getenv("CLICK_TIMEOUT_MS", "2500"))     # проверка кликабельности
TYPE_DELAY_MS: int = int(os.getenv("TYPE_DELAY_MS", "40"))             # задержка между символами
# Ожидание «успокоения» DOM после действия (вместо фиксированного sleep):
SETTLE_QUIET_MS: int = int(os.getenv("SETTLE_QUIET_MS", "350"))        # тишина без мутаций
SETTLE_TIMEOUT_MS: int = int(os.getenv("SETTLE_TIMEOUT_MS", "4000"))   # верхняя граница

# ---------------------------------------------------------------------------
# Фреймы
# ---------------------------------------------------------------------------
# URL целевого фрейма должен содержать одно из этих слов (порядок = приоритет)
FRAME_KEYWORDS: tuple[str, ...] = _site_list("FRAME_KEYWORDS", ("klecks-operator", "task"))
# URL фреймов, которые нужно игнорировать
FRAME_IGNORE_KEYWORDS: tuple[str, ...] = _env_list(
    "FRAME_IGNORE_KEYWORDS", ("captcha", "about:blank", "about:srcdoc"),
)
# Разрешить работу в главном фрейме, если подходящий iframe не найден. В главном фрейме агент
# решает только на страницах заданий (TASK_URL_KEYWORDS) — вход в аккаунт и прочие страницы
# сайта он не трогает
ALLOW_MAIN_FRAME: bool = _env_bool("ALLOW_MAIN_FRAME", _site("ALLOW_MAIN_FRAME", False))
# Признаки фрейма капчи (проверяются на всей странице, а не по тексту body)
CAPTCHA_URL_KEYWORDS: tuple[str, ...] = _env_list(
    "CAPTCHA_URL_KEYWORDS",
    ("captcha", "recaptcha", "hcaptcha", "smartcaptcha", "turnstile", "challenges.cloudflare"),
)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
# Агент работает, пока не вернётся на список заказов (STOP_ON_ORDERS_LIST); MAX_STEPS —
# только страховка от бесконечной работы
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "2000"))                # шагов С ДЕЙСТВИЕМ (ожидание не считается)
MAX_STEPS_PER_TASK: int = int(os.getenv("MAX_STEPS_PER_TASK", "40"))  # бюджет на одно задание (с поиском)
MAX_ELEMENTS: int = int(os.getenv("MAX_ELEMENTS", "200"))           # сколько элементов показывать LLM
READER_MAX_CHARS: int = int(os.getenv("READER_MAX_CHARS", "16000"))   # текст страницы задания в промпте
MAX_IDLE_SECONDS: float = float(os.getenv("MAX_IDLE_SECONDS", "900"))  # сколько ждать фрейм/задание подряд
CAPTCHA_TIMEOUT: float = float(os.getenv("CAPTCHA_TIMEOUT", "600"))  # макс. ожидание ручного решения капчи
MAX_OPEN_RETRIES: int = int(os.getenv("MAX_OPEN_RETRIES", "3"))     # попыток раскрыть одну папку
# Пакет действий: модель за один ответ выбирает ответы на все видимые вопросы и отправляет.
# Каждое действие проверяется; при сбое/неожиданном изменении страницы пакет останавливается.
BATCH_ACTIONS: bool = _env_bool("BATCH_ACTIONS", True)
MAX_BATCH_ACTIONS: int = int(os.getenv("MAX_BATCH_ACTIONS", "12"))
MAX_NO_EFFECT: int = int(os.getenv("MAX_NO_EFFECT", "2"))           # повторов действия без эффекта до запрета

# Тексты стартовых кнопок (локальный флоу «Приступить»). Сравнение — точное
# или по началу фразы («приступить» ≈ «Приступить к заданию»), и только когда
# на экране нет рабочих элементов или открыт диалог.
START_BUTTON_TEXTS: tuple[str, ...] = _env_list(
    "START_BUTTON_TEXTS",
    ("приступить", "начать", "ок", "ok", "хорошо", "понятно", "продолжить", "далее"),
)

# Тексты финальных кнопок (submit) в порядке приоритета
FINISH_BUTTON_TEXTS: tuple[str, ...] = _site_list(
    "FINISH_BUTTON_TEXTS",
    ("завершить", "отправить", "сохранить", "готово", "submit", "finish", "save"),
)
# Стоп-список кнопок: агент их НИКОГДА не нажимает — ни как отправку ответа, ни по
# решению LLM, и не показывает их модели («Завершить смену», «Выйти», «Отправить на
# доработку» …). Подстроки узкие: «смен» задело бы «Сменить категорию», «работу» —
# стартовую «Начать работу».
_DENY_COMMON = ("смену", "смены", "сессию", "сессии", "выйти", "выход", "logout", "аккаунт",
                "завершить работу", "доработк")
# площадка может добавить свои (Ozon: «Пропустить» — пропуски ограничены, агент отвечает сам)
FINISH_DENY_SUBSTRINGS: tuple[str, ...] = _env_list(
    "FINISH_DENY_SUBSTRINGS", _DENY_COMMON + tuple(_site("EXTRA_DENY_SUBSTRINGS", ())),
)
# «Выйти из задания?» — если диалог выхода всё же открылся, агент отвечает «остаться»
EXIT_CANCEL_TEXTS: tuple[str, ...] = _env_list(
    "EXIT_CANCEL_TEXTS", ("нет, остаться", "остаться", "отмена", "отменить", "нет"),
)
# Кнопки, закрывающие диалоги (инструкция, новости, уведомления)
DIALOG_CLOSE_TEXTS: tuple[str, ...] = _env_list(
    "DIALOG_CLOSE_TEXTS", ("закрыть", "понятно", "хорошо", "ок", "ok", "готово", "далее", "продолжить"),
)
# Окна площадки без role="dialog" / aria-modal (CSS-селекторы через «;»): их агент читает и
# закрывает как диалоги задания, а не как новости сайта
DIALOG_SELECTORS: tuple[str, ...] = _env_selectors("DIALOG_SELECTORS", _site("DIALOG_SELECTORS", ()))
# Части страницы, которые модель не видит и не нажимает: шапка сайта, чат (CSS-селекторы через «;»)
PAGE_SKIP_SELECTORS: tuple[str, ...] = _env_selectors("PAGE_SKIP_SELECTORS", _site("PAGE_SKIP_SELECTORS", ()))

# ---------------------------------------------------------------------------
# Список заказов: сюда платформа возвращает после последнего задания заказа
# ---------------------------------------------------------------------------
STOP_ON_ORDERS_LIST: bool = _env_bool("STOP_ON_ORDERS_LIST", True)
# Заказ выполнен (платформа вернула на список) — что дальше. false (по умолчанию): браузер
# остаётся открытым, агент ждёт — откройте следующий заказ, и он продолжит; закрыли окно
# браузера или нажали Ctrl+C — агент завершает работу. true: сразу завершить работу (как в v4).
CLOSE_BROWSER_WHEN_DONE: bool = _env_bool("CLOSE_BROWSER_WHEN_DONE", False)
# Экзамен после тренировки агент начинает, только если в тренировке с первого раза верно не меньше
# этой доли ответов (иначе экзамен, скорее всего, не будет сдан — денег за него не будет, а токены
# уйдут). Решение принимается, когда в тренировке не меньше EXAM_MIN_TASKS заданий. 0 — не проверять.
# 0.7: к концу тренировки агент накапливает разборы ошибок и отвечает точнее, чем в её начале
EXAM_MIN_ACCURACY: float = float(_env_default("EXAM_MIN_ACCURACY", "0.7", ("0.8",)))   # «0.8» — из образца до v4.8
EXAM_MIN_TASKS: int = int(os.getenv("EXAM_MIN_TASKS", "3"))
# Признаки: в адресе фрейма (без домена) нет ни одного из TASK_URL_KEYWORDS и есть кнопки «Приступить»
TASK_URL_KEYWORDS: tuple[str, ...] = _site_list("TASK_URL_KEYWORDS", ("/task",))
# …но не страницы с этими словами в пути (Ozon: /task/<id>/instruction — инструкция во вкладке)
NON_TASK_URL_KEYWORDS: tuple[str, ...] = _site_list("NON_TASK_URL_KEYWORDS", ())
ORDERS_BUTTON_TEXTS: tuple[str, ...] = _site_list("ORDERS_BUTTON_TEXTS", ("приступить",))
# Экран «задачи в проекте закончились» на месте задания — заказ выполнен, как возврат на список
ORDERS_DONE_TEXTS: tuple[str, ...] = _site_list("ORDERS_DONE_TEXTS", ())
# Сообщение платформы о неверном ответе в тренировке, кроме «неверный / неправильный ответ»
# (Ozon пишет «Для отправки ответа необходимо решить все задания»: неполный ответ агент не отправляет,
# значит, это неверный)
WRONG_ANSWER_TEXTS: tuple[str, ...] = _site_list("WRONG_ANSWER_TEXTS", ())
# Уменьшенная копия фото с CDN площадки: «регулярное выражение => замена» для адреса (пусто — оригинал).
# Ozon отдаёт копию шириной 1000 px по …/wc1000/… — в 10–40 раз меньше оригинала
IMAGE_URL_REWRITE: str = os.getenv("IMAGE_URL_REWRITE", _site("IMAGE_URL_REWRITE", "")).strip()
# Вид задания (инструкция, разборы ошибок, статистика тренировки) — по адресу страницы, если он
# совпадает с этим регулярным выражением (Ozon: /task/<id> — проект). Пусто — по структуре формы
POOL_URL_RE: str = os.getenv("POOL_URL_RE", _site("POOL_URL_RE", "")).strip()

# ---------------------------------------------------------------------------
# Инструкции и база знаний (папка knowledge/ — её можно читать и дополнять вручную)
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR: str = _project_path(os.getenv("KNOWLEDGE_DIR", "knowledge"))
READ_INSTRUCTIONS: bool = _env_bool("READ_INSTRUCTIONS", True)   # открыть «Подробную инструкцию» один раз
INSTRUCTION_WAIT: float = float(os.getenv("INSTRUCTION_WAIT", "25"))  # ждать загрузку инструкции, с
READ_TOOLTIPS: bool = _env_bool("READ_TOOLTIPS", True)           # прочитать подсказки «?» у вариантов
# Знания о виде заданий в каждом запросе, символов: инструкция — целиком (её не вытесняют уроки),
# на разборы ошибок из тренировки остаётся не меньше KNOWLEDGE_LESSON_CHARS
KNOWLEDGE_PROMPT_CHARS: int = int(os.getenv("KNOWLEDGE_PROMPT_CHARS", "14000"))
KNOWLEDGE_LESSON_CHARS: int = int(os.getenv("KNOWLEDGE_LESSON_CHARS", "3500"))

# ---------------------------------------------------------------------------
# Поиск в интернете (отдельная вкладка того же окна)
# ---------------------------------------------------------------------------
WEB_RESEARCH: bool = _env_bool("WEB_RESEARCH", True)
SEARCH_URL: str = os.getenv("SEARCH_URL", "https://yandex.ru/search/?text={query}")
WEB_TEXT_LIMIT: int = int(os.getenv("WEB_TEXT_LIMIT", "6000"))    # символов текста страницы в промпте
WEB_LINKS_LIMIT: int = int(os.getenv("WEB_LINKS_LIMIT", "25"))
WEB_TIMEOUT: float = float(os.getenv("WEB_TIMEOUT", "30"))
MAX_WEB_PER_TASK: int = int(os.getenv("MAX_WEB_PER_TASK", "12"))  # запросов на одно задание


def _missing_key_hint() -> str:
    """Почему ключ не прочитан: частые ошибки с файлом .env (особенно в Windows)."""
    win = sys.platform == "win32"
    env, example = PROJECT_DIR / ".env", PROJECT_DIR / ".env.example"
    try:
        values = dotenv_values(example) if example.is_file() else {}
        key_in_example = bool((values.get("OPENROUTER_API_KEY") or values.get("OPENAI_API_KEY") or "").strip())
    except (OSError, UnicodeDecodeError):
        key_in_example = False
    if env.is_file():
        where = " (сейчас ключ вписан в .env.example — этот файл агент не читает)" if key_in_example else ""
        return f"впишите ключ OpenRouter в файл {env} после OPENROUTER_API_KEY={where}"
    for name in (".env.txt", "env", "env.txt"):
        if (PROJECT_DIR / name).is_file():
            return (f"файла .env нет, но есть «{name}» — Блокнот сохранил его под другим именем. "
                    f"Выполните в папке агента: {'ren' if win else 'mv'} {name} .env")
    if key_in_example:
        return ("ключ вписан в .env.example, а агент читает только файл .env. Выполните в папке агента: "
                f"{'ren' if win else 'mv'} .env.example .env")
    return (f"файла .env нет в папке {PROJECT_DIR}. Выполните: {'copy' if win else 'cp'} .env.example .env "
            "и впишите ключ в .env")


def validate_config(*, require_llm: bool = True) -> None:
    """Проверить критичные параметры до запуска браузера (fail fast).

    require_llm=False — для режима записи: там LLM не вызывается и ключ не нужен."""
    problems: list[str] = []
    if _PLATFORM_ERROR:
        problems.append(_PLATFORM_ERROR)
    if _site("ALLOW_MAIN_FRAME", False) and not ALLOW_MAIN_FRAME and not FRAME_KEYWORDS:
        problems.append(f"ALLOW_MAIN_FRAME=false: на площадке {PLATFORM.title} задания открываются в самой "
                        "странице, без фрейма — агент их никогда не найдёт. Удалите строку ALLOW_MAIN_FRAME из .env")
    if require_llm and not OPENROUTER_API_KEY:
        problems.append("ключ OpenRouter не задан (OPENROUTER_API_KEY) — ключ создаётся на openrouter.ai/keys; "
                        + _missing_key_hint())
    if LLM_VISION not in ("off", "auto", "image", "frame"):
        problems.append(f"LLM_VISION={LLM_VISION!r}: допустимо auto | frame | off")
    if LLM_PROXY:
        scheme = LLM_PROXY.split("://", 1)[0].lower() if "://" in LLM_PROXY else ""
        if scheme not in ("http", "https", "socks5", "socks5h"):
            problems.append(f"LLM_PROXY={LLM_PROXY!r}: нужен адрес вида socks5://127.0.0.1:10808 или "
                            "http://адрес:порт")
        elif scheme.startswith("socks") and importlib.util.find_spec("socksio") is None:
            python = r".venv\Scripts\python" if sys.platform == "win32" else ".venv/bin/python"
            problems.append(f"LLM_PROXY с socks5 требует пакет socksio: выполните в папке агента "
                            f"{python} -m pip install -r requirements.txt")
    if "{query}" not in SEARCH_URL:
        problems.append(f"SEARCH_URL={SEARCH_URL!r}: в адресе нужен шаблон {{query}}")
    if PAGE_ZOOM_FACTOR == 1.0 and PAGE_ZOOM.strip() not in ("100%", "1", "1.0"):
        logger.warning("PAGE_ZOOM=%r не распознан — масштабирование отключено", PAGE_ZOOM)
    # значения из .env версии v3, мешающие v4 (агент теперь работает до списка заказов)
    if require_llm and MAX_STEPS < 300:
        logger.warning("MAX_STEPS=%d: агент остановится после %d действий, даже если задания не кончились. "
                       "Для v4 уберите строку MAX_STEPS из .env (по умолчанию 2000)", MAX_STEPS, MAX_STEPS)
    # настройки ProxyAPI из старого .env: агент работает только через OpenRouter
    base = os.getenv("OPENAI_BASE_URL", "")
    if require_llm and base and "openrouter.ai" not in base:
        logger.warning("OPENAI_BASE_URL=%s больше не используется: агент работает только через OpenRouter. "
                       "Удалите эту строку из .env и впишите ключ OpenRouter в OPENROUTER_API_KEY", base)
    if require_llm and any(os.getenv(k) for k in ("LLM_PRICE", "LLM_CHECK_PRICE", "LLM_PRICE_CURRENCY")):
        logger.warning("LLM_PRICE / LLM_PRICE_CURRENCY больше не нужны: точную стоимость в долларах агент "
                       "получает от OpenRouter. Удалите эти строки из .env")
    if require_llm and LLM_VISION != "off" and LLM_VISION_DETAIL == "low":
        logger.warning("LLM_VISION_DETAIL=low: фото уходят модели уменьшенными до 512 px — мелкие детали "
                       "(грязь, надписи, номера на коллажах) теряются. Рекомендуется high")
    if problems:
        raise ValueError("; ".join(problems))


logger.debug(
    "Config v4 загружен: площадка=%s model=%s url=%s headless=%s zoom=%.2f",
    PLATFORM.key, LLM_MODEL, TARGET_URL, HEADLESS, PAGE_ZOOM_FACTOR,
)
