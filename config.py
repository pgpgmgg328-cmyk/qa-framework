"""config.py — централизованная конфигурация агента v3.

Все значения можно переопределить через .env (см. .env.example).
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

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

# ---------------------------------------------------------------------------
# LLM / ProxyAPI / OpenRouter
# ---------------------------------------------------------------------------
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
# Таймаут одного запроса. В v2 действовал дефолт SDK — 600 с: зависший запрос
# «замораживал» агента на 10 минут.
LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))   # 429/5xx/обрывы — ретраи SDK
# Structured Outputs (json_schema strict). Если прокси/модель не поддерживает —
# коннектор сам откатится на json_object.
LLM_STRUCTURED_OUTPUT: bool = _env_bool("LLM_STRUCTURED_OUTPUT", True)
# Vision: off — не отправлять картинку; image — скриншот главной картинки задания;
# frame — скриншот всего фрейма задания.
LLM_VISION: str = os.getenv("LLM_VISION", "image").strip().lower()
LLM_VISION_DETAIL: str = os.getenv("LLM_VISION_DETAIL", "low")    # low | high | auto
LLM_HISTORY_SIZE: int = int(os.getenv("LLM_HISTORY_SIZE", "12"))  # строк истории в промпте
OPENROUTER_REFERER: str = os.getenv("OPENROUTER_REFERER", "https://localhost/twork-agent")

# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------
TARGET_URL: str = os.getenv("TARGET_URL", "https://twork.tbank.ru")
VIEWPORT_WIDTH: int = int(os.getenv("VIEWPORT_WIDTH", "1280"))
VIEWPORT_HEIGHT: int = int(os.getenv("VIEWPORT_HEIGHT", "720"))
HEADLESS: bool = _env_bool("HEADLESS", False)
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

# Масштаб страницы. В v3 реализован через device_scale_factor + увеличенный
# viewport, а НЕ через document.body.style.zoom: CSS-zoom ломает координаты
# Playwright внутри iframe — клики уходят мимо цели.
PAGE_ZOOM: str = os.getenv("PAGE_ZOOM", "75%")
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
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "80"))                  # шагов С ДЕЙСТВИЕМ (ожидание не считается)
MAX_STEPS_PER_TASK: int = int(os.getenv("MAX_STEPS_PER_TASK", "30"))  # бюджет на одно задание
MAX_ELEMENTS: int = int(os.getenv("MAX_ELEMENTS", "200"))           # сколько элементов показывать LLM
MAX_IDLE_SECONDS: float = float(os.getenv("MAX_IDLE_SECONDS", "900"))  # сколько ждать фрейм/задание подряд
CAPTCHA_TIMEOUT: float = float(os.getenv("CAPTCHA_TIMEOUT", "600"))  # макс. ожидание ручного решения капчи
MAX_OPEN_RETRIES: int = int(os.getenv("MAX_OPEN_RETRIES", "3"))     # попыток раскрыть одну папку
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


def validate_config(*, require_llm: bool = True) -> None:
    """Проверить критичные параметры до запуска браузера (fail fast).

    require_llm=False — для режима записи: там LLM не вызывается и ключ не нужен."""
    problems: list[str] = []
    if require_llm and not OPENAI_API_KEY:
        problems.append("OPENAI_API_KEY пуст — все запросы к LLM завершатся 401")
    if LLM_VISION not in ("off", "image", "frame"):
        problems.append(f"LLM_VISION={LLM_VISION!r}: допустимо off | image | frame")
    if PAGE_ZOOM_FACTOR == 1.0 and PAGE_ZOOM.strip() not in ("100%", "1", "1.0"):
        logger.warning("PAGE_ZOOM=%r не распознан — масштабирование отключено", PAGE_ZOOM)
    if problems:
        raise ValueError("; ".join(problems))


logger.debug(
    "Config v3 загружен: model=%s url=%s headless=%s zoom=%.2f",
    LLM_MODEL, TARGET_URL, HEADLESS, PAGE_ZOOM_FACTOR,
)
