"""main.py — точка входа агента v4.

  python main.py            — агент решает задания сам
  python main.py --record   — режим записи: вы решаете задания вручную, агент
                              сохраняет экраны и ваши действия (ключ API не нужен)
  python main.py --record --url https://profit.ozon.ru
                            — то же на другой площадке: запись нужна, чтобы
                              изучить её интерфейс и научить агента работать с ней
"""

import argparse
import asyncio
import logging
import sys

from config import enable_file_log, validate_config

logger = logging.getLogger("twork.main")


async def main() -> None:
    from agent import Agent  # импорт после проверки конфигурации
    from config import KNOWLEDGE_DIR, PROJECT_DIR
    from knowledge import import_previous

    asyncio.get_running_loop().set_exception_handler(_quiet_after_close)
    import_previous(KNOWLEDGE_DIR, PROJECT_DIR)     # новая версия в новой папке — база знаний прежней
    await Agent().run()


def _quiet_after_close(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Браузер закрыт (Ctrl+C, окно закрыто) — недоделанные операции Playwright падают с
    TargetClosedError; это не ошибка агента, трассировку в лог не пишем."""
    if type(context.get("exception")).__name__ == "TargetClosedError":
        return
    loop.default_exception_handler(context)


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
    parser = argparse.ArgumentParser(description="Агент T-Work v4")
    parser.add_argument(
        "--record", action="store_true",
        help="режим записи: агент ничего не нажимает, сохраняет экраны и ваши действия",
    )
    parser.add_argument(
        "--url", default=None,
        help="с --record: открыть другую площадку вместо TARGET_URL (например, https://profit.ozon.ru)",
    )
    args = parser.parse_args()
    if args.url and not args.record:
        parser.error("--url работает только вместе с --record. Решать задания на другой площадке: "
                     "PLATFORM=ozon в .env (адрес — TARGET_URL)")
    if args.url and not args.url.startswith(("http://", "https://")):
        parser.error("--url: нужен полный адрес, например https://profit.ozon.ru")
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        validate_config(require_llm=not args.record)
    except ValueError as exc:
        logger.error("Конфигурация некорректна: %s", exc)
        sys.exit(2)
    log_path = enable_file_log("record" if args.record else "agent")
    if log_path is not None:
        logger.info("Лог работы пишется в файл %s", log_path)
    try:
        if args.record:
            from recorder import record

            asyncio.run(record(args.url))
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run превращает Ctrl+C в KeyboardInterrupt снаружи корутины,
        # поэтому ловить его внутри main(), как в v2, бесполезно
        logger.info("Остановлен пользователем (Ctrl+C)")
    except Exception as exc:  # noqa: BLE001 — известные ошибки запуска объясняем по-человечески
        if type(exc).__name__ == "ModelUnavailable":
            logger.error("%s", exc)
            sys.exit(2)
        hint = startup_hint(exc)
        if hint is None:
            raise
        logger.error(hint)
        sys.exit(1)
