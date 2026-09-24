"""main.py — точка входа агента v2."""

import asyncio
import logging

from agent import Agent

logger = logging.getLogger("twork.main")


async def main() -> None:
    agent = Agent()
    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("Остановлен пользователем (Ctrl+C)")
    except Exception as exc:
        logger.exception("Непредвиденная ошибка: %s", exc)
        raise


if __name__ == "__main__":
    asyncio.run(main())
