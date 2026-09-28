"""Общая настройка тестов: окружение задаётся ДО импорта config.py."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.update({
    "HEADLESS": "true",
    "OPENROUTER_API_KEY": "test-key",
    "LLM_MODEL": "openai/gpt-5.4-mini",     # одна модель; лестницу задают тесты лестницы
    "LLM_OUTAGE_WAIT": "0",                 # недоступность OpenRouter — без ожидания (тесты ожидания сами)
    "TARGET_URL": "https://t-work.test/index.html?task=task_tree",
    "LLM_VISION": "off",
    "PAGE_ZOOM": "75%",            # проверяем клики именно с масштабированием
    "USER_DATA_DIR": "",
    "SLOW_MO": "0",
    "MOUSE_MOVE_STEPS": "3",
    "SETTLE_QUIET_MS": "150",
    "SETTLE_TIMEOUT_MS": "1500",
    "SUBMIT_WAIT": "3",
    "FRAME_LOAD_WAIT": "0.3",
    "ACTION_WAIT": "0.1",
    "MAX_IDLE_SECONDS": "15",
    # тесты заказа заканчиваются вместе с заказом; ожидание следующего — отдельный тест
    "CLOSE_BROWSER_WHEN_DONE": "true",
})
