"""Помощники тестов: раздача фикстур через context.route без HTTP-сервера.

Страница-обёртка — https://t-work.test/index.html?task=<имя>, внутри неё кросс-доменный
iframe https://klecks-operator.test/task/<имя>.html — как на T-Work.

Задания со структурой T-Work (flex_task.html): https://t-work.test/workspace.html?scenario=…
→ iframe https://klecks-operator.test/klecks/task?scenario=… ; после последнего задания
фрейм уходит на список заказов /klecks/orders.html. Вложения задания (фото, аудио) —
/klecks/api/task/get-attachment/<id>, только с кукой сессии (как авторизация платформы).
Поисковые сайты (yandex.ru, otzovik.com) подменены страницами из fixtures/web.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import struct
import wave
import zlib
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine
from urllib.parse import parse_qs, urlsplit

from playwright.async_api import BrowserContext, Route

FIXTURES = Path(__file__).resolve().parent / "fixtures"
WEB = FIXTURES / "web"

WRAPPER = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>T-Work</title></head>
<body style="margin:0"><header style="height:60px">T-Work · шапка сайта</header>
<iframe src="{src}" style="width:1000px;height:700px;border:0"></iframe>{popup}</body></html>"""

# Окно новостей сайта поверх фрейма задания (как «Одноразовые пароли для TWork»)
NEWS_POPUP = """<div role="dialog" aria-modal="true" id="news" style="position:fixed;inset:0;display:flex;
align-items:center;justify-content:center;background:rgba(0,0,0,.5)"><div style="background:#fff;width:500px;
padding:20px"><h2>Одноразовые пароли</h2><p>Рекомендуем создать токен.</p><button type="button"
onclick="document.getElementById('news').remove(); window.__newsClosed = true">Далее</button></div></div>"""

# На T-Work фрейм klecks-operator.tbank.ru и страница twork.tbank.ru — один сайт (tbank.ru), и куки
# уходят как обычно. Здесь домены разные, поэтому кука — SameSite=None (как «сторонняя»).
SESSION_COOKIE = {"name": "session", "value": "ok", "domain": "klecks-operator.test", "path": "/",
                  "sameSite": "None", "secure": True}


def main_url(task: str) -> str:
    return f"https://t-work.test/index.html?task={task}"


def fixture_url(task: str) -> str:
    return f"https://klecks-operator.test/task/{task}.html"


def workspace_url(scenario: str, *, popup: bool = False, autoinstruction: bool = False,
                  start: bool = False, instr: str = "") -> str:
    """start=True — задание открывается заставкой «Тренировка … [Начать]» (под ней идёт загрузка).
    instr=late|pdf|json|stub — инструкция во вкладке: текст догружается скриптом / PDF / JSON с HTML /
    PDF, который браузер получил лишь началом."""
    extra = ("&popup=1" if popup else "") + ("&autoinstruction=1" if autoinstruction else "") + ("&start=1" if start else "")
    extra += f"&instr={instr}" if instr else ""
    return f"https://t-work.test/workspace.html?scenario={scenario}{extra}"


INSTRUCTION_HTML = (
    "<h2>Проверка наличия фото</h2><p>Для каждой поверхности банкомата должны быть фото: лицевая часть, "
    "обе боковые стороны, верхняя панель.</p><p>Если хотя бы одной стороны нет — выбирайте «Фото присутствуют "
    "частично» и отмечайте поверхность, по которой недостаточно фото.</p><ul><li>Чёрные фото считаются "
    "отсутствующими.</li><li>Размытое фото, на котором нельзя различить поверхность, считается отсутствующим.</li></ul>")

# страница инструкции во вкладке (klecks/instruction/<id>): как её показывает T-Work — неизвестно,
# поэтому три варианта, при которых текста на странице сразу (или вообще) нет
INSTRUCTION_PAGES = {
    "late": ("<!doctype html><html><body><div id='root'>Загрузка…</div><script>setTimeout(() => "
             "{ document.getElementById('root').innerHTML = " + repr(INSTRUCTION_HTML) + "; }, 3000)</script></body></html>"),
    "pdf": ("<!doctype html><html><body><canvas width='600' height='800'></canvas><script>"
            "fetch('/klecks/api/instruction-file/3a1e.pdf').then(r => r.arrayBuffer())</script></body></html>"),
    "json": ("<!doctype html><html><body><canvas width='600' height='800'></canvas><script>"
             "fetch('/klecks/api/instruction/3a1e').then(r => r.json())</script></body></html>"),
    # как встроенный просмотрщик PDF в Chrome на T-Work: браузер получает лишь начало файла
    # (запрос с Range — 536 байт), целиком документ отдаётся только обычным запросом
    "stub": ("<!doctype html><html><body><canvas width='600' height='800'></canvas><script>"
             "fetch('/klecks/api/instruction-file/3a1e.pdf', {headers: {Range: 'bytes=0-535'}})"
             ".then(r => r.arrayBuffer())</script></body></html>"),
}
# PDF инструкции: больше 2 КБ и с концом файла %%EOF — как настоящий (меньше — это обрывок или заглушка)
PDF_BYTES = (b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\n2 0 obj << /Length 3000 >>\nstream\n" + b"0" * 3000
             + b"\nendstream\nendobj\ntrailer << /Root 1 0 R >>\n%%EOF\n")


def png_bytes(seed: str, width: int = 320, height: int = 240) -> bytes:
    """Однотонный PNG — цвет зависит от seed (фото отличаются друг от друга)."""
    rgb = hashlib.md5(seed.encode()).digest()[:3]
    raw = b"".join(b"\x00" + rgb * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def wav_bytes(seconds: float = 1.6, rate: int = 8000) -> bytes:
    """Короткий тон 440 Гц — «запись звонка»."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate))
        )
        w.writeframes(frames)
    return buf.getvalue()


_WAV = wav_bytes()


async def install_routes(context: BrowserContext) -> None:
    await context.add_cookies([SESSION_COOKIE])

    async def html(route: Route, body: str) -> None:
        await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=body)

    async def handle(route: Route) -> None:
        url = urlsplit(route.request.url)
        query = parse_qs(url.query)
        host, path = url.hostname or "", url.path
        if host == "t-work.test":
            if path == "/workspace.html" and query.get("orders"):
                await html(route, WRAPPER.format(src="https://klecks-operator.test/klecks/orders.html", popup=""))
            elif path == "/workspace.html":
                scenario = query.get("scenario", ["hotels"])[0]
                src = f"https://klecks-operator.test/klecks/task?scenario={scenario}"
                if query.get("autoinstruction"):
                    src += "&autoinstruction=1"
                if query.get("start"):
                    src += "&start=1"
                if query.get("instr"):
                    src += f"&instr={query['instr'][0]}"
                await html(route, WRAPPER.format(src=src, popup=NEWS_POPUP if query.get("popup") else ""))
            else:
                task = query.get("task", ["task_tree"])[0]
                await html(route, WRAPPER.format(src=fixture_url(task), popup=""))
        elif host == "klecks-operator.test":
            if path == "/klecks/task":
                await html(route, (FIXTURES / "flex_task.html").read_text(encoding="utf-8"))
            elif path == "/klecks/orders.html":
                await html(route, (FIXTURES / "orders.html").read_text(encoding="utf-8"))
            elif path.startswith("/klecks/instruction/"):
                await html(route, INSTRUCTION_PAGES[query.get("kind", ["late"])[0]])
            elif path.startswith("/klecks/api/instruction-file/"):
                if route.request.headers.get("range"):
                    await route.fulfill(status=206, content_type="application/pdf", body=PDF_BYTES[:536],
                                        headers={"Content-Range": f"bytes 0-535/{len(PDF_BYTES)}"})
                else:
                    await route.fulfill(status=200, content_type="application/pdf", body=PDF_BYTES)
            elif path.startswith("/klecks/api/instruction/"):
                await route.fulfill(status=200, content_type="application/json",
                                    body=json.dumps({"id": "3a1e", "title": "Инструкция", "content": INSTRUCTION_HTML}))
            elif path.startswith("/klecks/api/task/get-attachment/"):
                headers = await route.request.all_headers()       # headers без cookie, all_headers — с ней
                if "session=ok" not in (headers.get("cookie") or ""):
                    await route.fulfill(status=401, body="unauthorized")
                    return
                ident = path.rsplit("/", 1)[-1]
                if ident.startswith("call"):
                    await route.fulfill(status=200, content_type="audio/wav", body=_WAV)
                else:
                    await route.fulfill(status=200, content_type="image/png", body=png_bytes(ident))
            else:
                await html(route, (FIXTURES / Path(path).name).read_text(encoding="utf-8"))
        elif host == "cdn.hotels.test":
            await route.fulfill(status=200, content_type="image/png", body=png_bytes(path),
                                headers={"Access-Control-Allow-Origin": "*"})
        elif host == "yandex.ru" and path.startswith("/maps/org/"):
            await html(route, (WEB / "yandex_maps_org.html").read_text(encoding="utf-8"))
        elif host == "yandex.ru" and path.startswith("/maps"):
            await html(route, (WEB / "yandex_maps_search.html").read_text(encoding="utf-8"))
        elif host == "yandex.ru" and path.startswith("/search"):
            await html(route, (WEB / "search.html").read_text(encoding="utf-8"))
        elif host == "otzovik.com":
            await html(route, (WEB / "otzovik_search.html").read_text(encoding="utf-8"))
        else:
            await route.abort()

    await context.route("**/*", handle)


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


RouteInstaller = Callable[[BrowserContext], Awaitable[None]]
