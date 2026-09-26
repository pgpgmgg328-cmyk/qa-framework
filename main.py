"""main.py — точка входа агента v3.

  python main.py            — агент решает задания сам
  python main.py --record   — режим записи: вы решаете задания вручную, агент
                              сохраняет экраны и ваши действия (ключ API не нужен)
"""

import argparse
import asyncio
import logging
import sys

from config import validate_config

logger = logging.getLogger("twork.main")


async def main() -> None:
    from agent import Agent  # импорт после проверки конфигурации

    await Agent().run()


# Частые ошибки запуска → понятная подсказка вместо трассировки
_STARTUP_HINTS = (
    ("Executable doesn't exist",
     "Браузер для Playwright не установлен. Выполните: python -m playwright install chromium "
     "(или впишите в .env строку BROWSER_CHANNEL=chrome, чтобы использовать установленный Google Chrome)."),
    ("distribution 'chrome' is not found",
     "BROWSER_CHANNEL=chrome, но Google Chrome не найден. Установите Chrome или уберите эту строку из .env."),
    ("ProcessSingleton",
     "Профиль браузера уже открыт другим окном агента или записи. Закройте его и запустите снова."),
    ("user data directory is already in use",
     "Профиль браузера уже открыт другим окном агента или записи. Закройте его и запустите снова."),
)


def startup_hint(exc: BaseException) -> "str | None":
    text = str(exc)
    for marker, hint in _STARTUP_HINTS:
        if marker in text:
            return hint
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Агент T-Work v3")
    parser.add_argument(
        "--record", action="store_true",
        help="режим записи: агент ничего не нажимает, сохраняет экраны и ваши действия",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        validate_config(require_llm=not args.record)
    except ValueError as exc:
        logger.error("Конфигурация некорректна: %s", exc)
        sys.exit(2)
    try:
        if args.record:
            from recorder import record

            asyncio.run(record())
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run превращает Ctrl+C в KeyboardInterrupt снаружи корутины,
        # поэтому ловить его внутри main(), как в v2, бесполезно
        logger.info("Остановлен пользователем (Ctrl+C)")
    except Exception as exc:  # noqa: BLE001 — известные ошибки запуска объясняем по-человечески
        hint = startup_hint(exc)
        if hint is None:
            raise
        logger.error(hint)
        sys.exit(1)
