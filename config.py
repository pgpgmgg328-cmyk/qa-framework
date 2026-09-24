"""config.py — централизованная конфигурация агента v2."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Ищем .env рядом с этим файлом
load_dotenv(Path(__file__).parent / ".env")

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
# LLM / ProxyAPI
# ---------------------------------------------------------------------------
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ---------------------------------------------------------------------------
# Browser
# ---------------------------------------------------------------------------
TARGET_URL: str = os.getenv("TARGET_URL", "https://t-work.ru")
VIEWPORT_WIDTH: int = int(os.getenv("VIEWPORT_WIDTH", "1280"))
VIEWPORT_HEIGHT: int = int(os.getenv("VIEWPORT_HEIGHT", "720"))
HEADLESS: bool = os.getenv("HEADLESS", "false").lower() == "true"
SLOW_MO: int = int(os.getenv("SLOW_MO", "80"))  # ms между действиями

# Масштаб страницы: 75% чтобы все кнопки влезали в экран
PAGE_ZOOM: str = os.getenv("PAGE_ZOOM", "75%")

# Задержки (секунды)
FRAME_LOAD_WAIT: float = float(os.getenv("FRAME_LOAD_WAIT", "4.0"))   # ждём тяжёлый фрейм
ACTION_WAIT: float = float(os.getenv("ACTION_WAIT", "1.2"))            # пауза после клика
SUBMIT_WAIT: float = float(os.getenv("SUBMIT_WAIT", "4.0"))            # пауза после submit
MOUSE_MOVE_STEPS: int = int(os.getenv("MOUSE_MOVE_STEPS", "10"))       # шагов при движении мыши

# ---------------------------------------------------------------------------
# Фреймы
# ---------------------------------------------------------------------------
# URL целевого фрейма должен содержать одно из этих слов
FRAME_KEYWORDS: tuple[str, ...] = ("klecks-operator", "task")
# URL фреймов, которые нужно игнорировать
FRAME_IGNORE_KEYWORDS: tuple[str, ...] = ("captcha", "about:blank", "about:srcdoc")

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "80"))       # лимит шагов безопасности
MAX_ELEMENTS: int = int(os.getenv("MAX_ELEMENTS", "200")) # макс. элементов в парсере

# Тексты стартовых кнопок (локальный флоу «Приступить»)
START_BUTTON_TEXTS: tuple[str, ...] = (
    "приступить", "ок", "хорошо", "начать", "продолжить", "далее", "ok",
)

# Тексты финальных кнопок (submit)
FINISH_BUTTON_TEXTS: tuple[str, ...] = (
    "завершить", "сохранить", "отправить", "submit", "finish", "save", "готово",
)

logger.debug("Config v2 загружен: model=%s url=%s headless=%s", LLM_MODEL, TARGET_URL, HEADLESS)
