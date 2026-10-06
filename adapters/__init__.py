"""adapters — площадки заданий и то, чем они отличаются от T-Work.

Площадка выбирается строкой PLATFORM= в .env (по умолчанию twork). Модуль площадки задаёт
только значения по умолчанию для настроек config.py (тексты кнопок, фреймы, адреса, признаки
окон); любую из них по-прежнему можно переопределить в .env. Код агента общий.
"""

from __future__ import annotations

from typing import Any, ClassVar

DEFAULT_PLATFORM = "twork"


class PlatformAdapter:
    """Площадка: ключ для PLATFORM=, название для лога и значения настроек по умолчанию."""

    key: ClassVar[str] = ""
    title: ClassVar[str] = ""
    # имя настройки config.py → значение по умолчанию на этой площадке (тип — как у настройки)
    settings: ClassVar[dict[str, Any]] = {}

    @classmethod
    def setting(cls, name: str, default: Any) -> Any:
        return cls.settings.get(name, default)


def platforms() -> dict[str, type[PlatformAdapter]]:
    from adapters.ozon_profit import OzonProfitAdapter
    from adapters.twork import TWorkAdapter

    return {a.key: a for a in (TWorkAdapter, OzonProfitAdapter)}


def get_platform(name: str) -> type[PlatformAdapter]:
    """Площадка по ключу из .env; неизвестный ключ — ValueError со списком допустимых."""
    known = platforms()
    key = (name or DEFAULT_PLATFORM).strip().lower()
    if key not in known:
        raise ValueError(f"PLATFORM={name!r}: допустимо {' | '.join(known)}")
    return known[key]
