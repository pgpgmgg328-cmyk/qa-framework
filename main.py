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
