"""T-Work (twork.tbank.ru) — площадка по умолчанию: её значения записаны прямо в config.py."""

from adapters import PlatformAdapter


class TWorkAdapter(PlatformAdapter):
    key = "twork"
    title = "T-Work"
    settings = {}
