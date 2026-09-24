"""main.py — точка входа агента v3."""

import asyncio
import logging
import sys

from config import validate_config

logger = logging.getLogger("twork.main")


async def main() -> None:
    from agent import Agent  # импорт после проверки конфигурации

    await Agent().run()


if __name__ == "__main__":
    try:
        validate_config()
    except ValueError as exc:
        logger.error("Конфигурация некорректна: %s", exc)
        sys.exit(2)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run превращает Ctrl+C в KeyboardInterrupt снаружи корутины,
        # поэтому ловить его внутри main(), как в v2, бесполезно
        logger.info("Остановлен пользователем (Ctrl+C)")
