"""config.py — централизованная конфигурация агента v4.

Все значения можно переопределить через .env (см. .env.example).
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, load_dotenv

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
# LLM / ProxyAPI / OpenRouter
# ---------------------------------------------------------------------------
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
# gpt-5.4-mini: в ~3 раза дешевле gpt-4o на входе, повторяющаяся часть запроса — за 10% цены.
# При запуске агент проверяет модель и пишет, сколько токенов стоит фото (см. README, «Выбор модели»)
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5.4-mini")
# Reasoning-модели (gpt-5.x, o-серия): сколько «думать» перед ответом. auto — минимум, который
# принимает модель (none → minimal → low): быстрее и дешевле; low / medium — точнее, но дороже
LLM_REASONING_EFFORT: str = os.getenv("LLM_REASONING_EFFORT", "auto").strip().lower()
# Перепроверка: основная модель не уверена (confidence ниже LLM_CHECK_BELOW) в ответе, который
# отправляется, — решение перепроверяет более сильная модель LLM_CHECK_MODEL. Пусто — выключено
LLM_CHECK_MODEL: str = os.getenv("LLM_CHECK_MODEL", "").strip()
LLM_CHECK_BELOW: float = float(os.getenv("LLM_CHECK_BELOW", "0.7"))
# Цена за 1 млн токенов «вход,вход из кэша,выход» — для стоимости в логе (например, цены ProxyAPI
# в рублях из личного кабинета). Пусто — прайс OpenAI для известных моделей, в долларах
LLM_PRICE: str = os.getenv("LLM_PRICE", "").strip()
LLM_CHECK_PRICE: str = os.getenv("LLM_CHECK_PRICE", "").strip()
LLM_PRICE_CURRENCY: str = os.getenv("LLM_PRICE_CURRENCY", "₽" if LLM_PRICE else "$").strip()
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
# план + рассуждения + ответ: 1024 токенов иногда не хватало на задания с несколькими полями
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
# Таймаут одного запроса. В v2 действовал дефолт SDK — 600 с: зависший запрос
# «замораживал» агента на 10 минут.
LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))   # 429/5xx/обрывы — ретраи SDK
# Structured Outputs (json_schema strict). Если прокси/модель не поддерживает —
# коннектор сам откатится на json_object.
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
VISION_IMAGE_SIDE: int = int(os.getenv("VISION_IMAGE_SIDE", "1024"))  # длинная сторона одиночного фото
# ячейка коллажа, px: коллаж 2×2 — 774 px. gpt-4o всё равно уменьшает картинку до 768 px (и берёт
# за неё столько же токенов), а новые модели считают токены по площади — коллаж в 1030 px был
# для них на ~40% дороже без выигрыша в деталях
VISION_CELL: int = int(os.getenv("VISION_CELL", "384"))
LLM_HISTORY_SIZE: int = int(os.getenv("LLM_HISTORY_SIZE", "14"))  # строк истории в промпте

# Аудио: расшифровка через audio.transcriptions того же API (ProxyAPI/OpenAI).
# Модели пробуются по порядку; «…-diarize» дополнительно размечает говорящих.
AUDIO_TRANSCRIBE: bool = _env_bool("AUDIO_TRANSCRIBE", True)
TRANSCRIBE_MODELS: tuple[str, ...] = tuple(
    m.strip() for m in os.getenv("TRANSCRIBE_MODELS", "gpt-4o-transcribe,whisper-1").split(",") if m.strip()
)
TRANSCRIBE_LANGUAGE: str = os.getenv("TRANSCRIBE_LANGUAGE", "ru")
# «Прослушайте звонок до конца»: перед отправкой ответа запись доигрывается до конца
AUDIO_PLAY_TO_END: bool = _env_bool("AUDIO_PLAY_TO_END", True)
AUDIO_MAX_WAIT: float = float(os.getenv("AUDIO_MAX_WAIT", "600"))   # макс. ожидание конца записи, с
OPENROUTER_REFERER: str = os.getenv("OPENROUTER_REFERER", "https://localhost/twork-agent")

# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------
TARGET_URL: str = os.getenv("TARGET_URL", "https://twork.tbank.ru")
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
FRAME_KEYWORDS: tuple[str, ...] = _env_list("FRAME_KEYWORDS", ("klecks-operator", "task"))
# URL фреймов, которые нужно игнорировать
FRAME_IGNORE_KEYWORDS: tuple[str, ...] = _env_list(
    "FRAME_IGNORE_KEYWORDS", ("captcha", "about:blank", "about:srcdoc"),
)
# Разрешить работу в главном фрейме, если подходящий iframe не найден
ALLOW_MAIN_FRAME: bool = _env_bool("ALLOW_MAIN_FRAME", False)
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
FINISH_BUTTON_TEXTS: tuple[str, ...] = _env_list(
    "FINISH_BUTTON_TEXTS",
    ("завершить", "отправить", "сохранить", "готово", "submit", "finish", "save"),
)
# Стоп-список кнопок: агент их НИКОГДА не нажимает — ни как отправку ответа, ни по
# решению LLM, и не показывает их модели («Завершить смену», «Выйти», «Отправить на
# доработку» …). Подстроки узкие: «смен» задело бы «Сменить категорию», «работу» —
# стартовую «Начать работу».
FINISH_DENY_SUBSTRINGS: tuple[str, ...] = _env_list(
    "FINISH_DENY_SUBSTRINGS",
    ("смену", "смены", "сессию", "сессии", "выйти", "выход", "logout", "аккаунт",
     "завершить работу", "доработк"),
)
# «Выйти из задания?» — если диалог выхода всё же открылся, агент отвечает «остаться»
EXIT_CANCEL_TEXTS: tuple[str, ...] = _env_list(
    "EXIT_CANCEL_TEXTS", ("нет, остаться", "остаться", "отмена", "отменить", "нет"),
)
# Кнопки, закрывающие диалоги (инструкция, новости, уведомления)
DIALOG_CLOSE_TEXTS: tuple[str, ...] = _env_list(
    "DIALOG_CLOSE_TEXTS", ("закрыть", "понятно", "хорошо", "ок", "ok", "готово", "далее", "продолжить"),
)

# ---------------------------------------------------------------------------
# Список заказов: сюда платформа возвращает после последнего задания заказа
# ---------------------------------------------------------------------------
STOP_ON_ORDERS_LIST: bool = _env_bool("STOP_ON_ORDERS_LIST", True)
# Заказ выполнен (платформа вернула на список) — что дальше. false (по умолчанию): браузер
# остаётся открытым, агент ждёт — откройте следующий заказ, и он продолжит; закрыли окно
# браузера или нажали Ctrl+C — агент завершает работу. true: сразу завершить работу (как в v4).
CLOSE_BROWSER_WHEN_DONE: bool = _env_bool("CLOSE_BROWSER_WHEN_DONE", False)
# Признаки: в URL фрейма нет ни одного из TASK_URL_KEYWORDS и есть кнопки «Приступить»
TASK_URL_KEYWORDS: tuple[str, ...] = _env_list("TASK_URL_KEYWORDS", ("/task",))
ORDERS_BUTTON_TEXTS: tuple[str, ...] = _env_list("ORDERS_BUTTON_TEXTS", ("приступить",))

# ---------------------------------------------------------------------------
# Инструкции и база знаний (папка knowledge/ — её можно читать и дополнять вручную)
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR: str = _project_path(os.getenv("KNOWLEDGE_DIR", "knowledge"))
READ_INSTRUCTIONS: bool = _env_bool("READ_INSTRUCTIONS", True)   # открыть «Подробную инструкцию» один раз
INSTRUCTION_WAIT: float = float(os.getenv("INSTRUCTION_WAIT", "25"))  # ждать загрузку инструкции, с
READ_TOOLTIPS: bool = _env_bool("READ_TOOLTIPS", True)           # прочитать подсказки «?» у вариантов
KNOWLEDGE_PROMPT_CHARS: int = int(os.getenv("KNOWLEDGE_PROMPT_CHARS", "9000"))

# ---------------------------------------------------------------------------
# Поиск в интернете (отдельная вкладка того же окна)
# ---------------------------------------------------------------------------
WEB_RESEARCH: bool = _env_bool("WEB_RESEARCH", True)
SEARCH_URL: str = os.getenv("SEARCH_URL", "https://yandex.ru/search/?text={query}")
WEB_TEXT_LIMIT: int = int(os.getenv("WEB_TEXT_LIMIT", "6000"))    # символов текста страницы в промпте
WEB_LINKS_LIMIT: int = int(os.getenv("WEB_LINKS_LIMIT", "40"))
WEB_TIMEOUT: float = float(os.getenv("WEB_TIMEOUT", "30"))
MAX_WEB_PER_TASK: int = int(os.getenv("MAX_WEB_PER_TASK", "12"))  # запросов на одно задание


def _missing_key_hint() -> str:
    """Почему ключ не прочитан: частые ошибки с файлом .env (особенно в Windows)."""
    win = sys.platform == "win32"
    env, example = PROJECT_DIR / ".env", PROJECT_DIR / ".env.example"
    try:
        key_in_example = example.is_file() and bool((dotenv_values(example).get("OPENAI_API_KEY") or "").strip())
    except (OSError, UnicodeDecodeError):
        key_in_example = False
    if env.is_file():
        where = " (сейчас ключ вписан в .env.example — этот файл агент не читает)" if key_in_example else ""
        return f"впишите ключ в файл {env} после OPENAI_API_KEY={where}"
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
    if require_llm and not OPENAI_API_KEY:
        problems.append("OPENAI_API_KEY пуст — все запросы к LLM завершатся 401; " + _missing_key_hint())
    if LLM_VISION not in ("off", "auto", "image", "frame"):
        problems.append(f"LLM_VISION={LLM_VISION!r}: допустимо auto | frame | off")
    if "{query}" not in SEARCH_URL:
        problems.append(f"SEARCH_URL={SEARCH_URL!r}: в адресе нужен шаблон {{query}}")
    if PAGE_ZOOM_FACTOR == 1.0 and PAGE_ZOOM.strip() not in ("100%", "1", "1.0"):
        logger.warning("PAGE_ZOOM=%r не распознан — масштабирование отключено", PAGE_ZOOM)
    # значения из .env версии v3, мешающие v4 (агент теперь работает до списка заказов)
    if require_llm and MAX_STEPS < 300:
        logger.warning("MAX_STEPS=%d: агент остановится после %d действий, даже если задания не кончились. "
                       "Для v4 уберите строку MAX_STEPS из .env (по умолчанию 2000)", MAX_STEPS, MAX_STEPS)
    if require_llm and LLM_VISION != "off" and LLM_VISION_DETAIL == "low":
        logger.warning("LLM_VISION_DETAIL=low: фото уходят модели уменьшенными до 512 px — мелкие детали "
                       "(грязь, надписи, номера на коллажах) теряются. Рекомендуется high")
    if problems:
        raise ValueError("; ".join(problems))


logger.debug(
    "Config v4 загружен: model=%s url=%s headless=%s zoom=%.2f",
    LLM_MODEL, TARGET_URL, HEADLESS, PAGE_ZOOM_FACTOR,
)
