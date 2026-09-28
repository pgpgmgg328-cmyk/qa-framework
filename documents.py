"""documents.py — текст инструкции, которую платформа показывает не текстом страницы.

Инструкция к заказу бывает PDF (в диалоге или во вкладке — просмотрщик PDF или холст, текста на
странице нет) или догружается скриптом уже после загрузки страницы. Поэтому:
- DocumentCatcher, пока агент открывает инструкцию, запоминает ответы сервера с документами:
  PDF и JSON по адресам с «instruction» (в нём бывает HTML/текст инструкции);
- page_text ждёт, пока на вкладке появится текст, и читает все её фреймы;
- describe_page — что на странице вместо текста (для лога, если прочитать не удалось).
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from playwright.async_api import BrowserContext, Error as PlaywrightError, Page, Response

logger = logging.getLogger("twork.documents")

_MAX_BYTES = 20 * 1024 * 1024
_MIN_TEXT = 200                     # символов без пробелов — «в инструкции есть текст»


def meaningful(text: str) -> bool:
    return len(re.sub(r"\s+", "", text or "")) >= _MIN_TEXT


@dataclass
class CaughtDocument:
    url: str
    mime: str
    data: bytes

    @property
    def is_pdf(self) -> bool:
        return "pdf" in self.mime or self.data[:5] == b"%PDF-"


class DocumentCatcher:
    """Запоминает документы, которые браузер скачивает, пока открыта инструкция."""

    def __init__(self, context: BrowserContext) -> None:
        self._context = context
        self.documents: list[CaughtDocument] = []
        self._tasks: set[asyncio.Task] = set()
        context.on("response", self._on_response)

    def _on_response(self, response: Response) -> None:
        mime = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        url = response.url
        wanted = "pdf" in mime or url.lower().split("?")[0].endswith(".pdf") or (
            "json" in mime and "instruct" in url.lower())
        if wanted and response.ok:
            task = asyncio.ensure_future(self._grab(response, mime))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _grab(self, response: Response, mime: str) -> None:
        try:
            data = await response.body()
        except PlaywrightError as exc:
            logger.debug("Документ %s не получен: %s", response.url[:100], exc)
            return
        if data and len(data) <= _MAX_BYTES and len(self.documents) < 8:
            self.documents.append(CaughtDocument(response.url, mime, data))

    async def settle(self, timeout: float = 5.0) -> None:
        """Дождаться документов, которые ещё скачиваются."""
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=timeout)

    def close(self) -> None:
        try:
            self._context.remove_listener("response", self._on_response)
        except (PlaywrightError, KeyError, ValueError):
            pass

    def pdfs(self) -> list[CaughtDocument]:
        seen: set[bytes] = set()
        out = []
        for doc in self.documents:
            if doc.is_pdf and doc.data[:2048] not in seen:
                seen.add(doc.data[:2048])
                out.append(doc)
        return out

    def json_text(self) -> str:
        """Текст инструкции из JSON-ответов сервера: длинные строки, HTML → текст."""
        parts: list[str] = []
        for doc in self.documents:
            if doc.is_pdf:
                continue
            try:
                payload = json.loads(doc.data.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            for value in _strings(payload):
                text = html_to_text(value)
                if meaningful(text) and text not in parts:
                    parts.append(text)
        return "\n\n".join(parts)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def html_to_text(value: str) -> str:
    """HTML или markdown из JSON → читаемый текст (абзацы и пункты списков — с новой строки)."""
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", value)
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|h[1-6]|li|tr|ul|ol|table)>", "\n", text)
    text = re.sub(r"(?i)<li\b[^>]*>", "\n• ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t ]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


_JS_TEXT = "() => document.body ? document.body.innerText : ''"


async def page_text(page: Page, wait: float, give_up: Optional[Callable[[], bool]] = None) -> str:
    """Текст вкладки со всеми фреймами. Страница догружает инструкцию скриптом — ждём, пока текст
    появится и перестанет меняться (не дольше wait секунд). give_up() — текста не будет (страница
    уже скачала документ с инструкцией): ждать не дольше 3 секунд."""
    started = time.monotonic()
    deadline = started + wait
    last = ""
    while True:
        texts = []
        for frame in page.frames:
            try:
                texts.append(await frame.evaluate(_JS_TEXT))
            except PlaywrightError:
                continue
        text = "\n".join(t.strip() for t in texts if t and t.strip())
        if meaningful(text) and text == last:
            return text
        now = time.monotonic()
        if now >= deadline or (not meaningful(text) and give_up is not None and now - started >= 3 and give_up()):
            return text
        last = text
        await asyncio.sleep(1.0)


_JS_DESCRIBE = r"""() => ({
  type: document.contentType || '',
  canvas: document.querySelectorAll('canvas').length,
  img: document.querySelectorAll('img').length,
  embeds: [...document.querySelectorAll('embed, object, iframe')]
    .map(e => (e.tagName.toLowerCase() + ':' + (e.getAttribute('type') || '') + ':' +
               (e.getAttribute('src') || e.getAttribute('data') || '')).slice(0, 120)),
  chars: document.body ? document.body.innerText.length : 0,
})"""


async def describe_page(page: Page) -> str:
    """Что на странице вместо текста — для лога (чтобы понять формат инструкции)."""
    try:
        info = await page.evaluate(_JS_DESCRIBE)
    except PlaywrightError as exc:
        return f"не удалось осмотреть страницу ({exc})"
    frames = [f.url[:100] for f in page.frames[1:]]
    return (f"тип {info.get('type')}, символов {info.get('chars')}, холстов {info.get('canvas')}, "
            f"картинок {info.get('img')}, встроенных {info.get('embeds')}, фреймов {frames}")


def describe_documents(catcher: Optional[DocumentCatcher]) -> str:
    if catcher is None or not catcher.documents:
        return "документов не скачивалось"
    return "; ".join(f"{d.mime or '?'} {len(d.data)} байт {d.url[:80]}" for d in catcher.documents[:5])
