"""research.py — поиск в интернете для заданий вида «найдите организацию / ссылку».

Модель просит action=web с query: поисковый запрос или адрес. Агент открывает его
в ОТДЕЛЬНОЙ вкладке того же окна (вкладка задания не трогается), ждёт загрузки,
читает страницу — адрес, заголовок, текст, ссылки — и возвращает модели сводку.
Адреса и ID модель копирует дословно из этой сводки (в строке «Адрес страницы»
и в списке ссылок). После чтения на передний план возвращается вкладка задания.

Если сайт показал капчу («Я не робот»), агент просит решить её в окне и ждёт.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus, urlsplit, urlunsplit

from playwright.async_api import Error as PlaywrightError, Page

from config import CAPTCHA_TIMEOUT, SEARCH_URL, WEB_LINKS_LIMIT, WEB_TEXT_LIMIT, WEB_TIMEOUT

logger = logging.getLogger("twork.research")

# Адрес — только явный (http://, https://, www.): «example.ru» без схемы — это поисковый
# запрос (так ищут организацию по сайту), а открыть сайт модель может полным адресом
_URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)

# Страница: адрес, заголовок, видимый текст и ссылки (без меню/скрытых элементов)
_JS_READ_PAGE = r"""
({ textLimit, linksLimit }) => {
    const norm = (s) => String(s || '').replace(/[ \t ]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
    const body = document.body;
    let text = body ? body.innerText || '' : '';
    text = norm(text);
    const links = [], seen = new Set();
    for (const a of document.querySelectorAll('a[href]')) {
        const href = String(a.href || '');
        if (!/^https?:/i.test(href) || seen.has(href)) continue;
        const r = a.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) continue;
        const label = norm(a.innerText || a.getAttribute('aria-label') || a.title || '').slice(0, 140);
        if (!label) continue;
        seen.add(href);
        links.push({ text: label, href });
        if (links.length >= linksLimit * 3) break;
    }
    const captcha = /showcaptcha|captcha|smartcaptcha/i.test(location.href)
        || !!document.querySelector('iframe[src*="captcha" i], .CheckboxCaptcha, #checkbox-captcha-form, form[action*="captcha" i]')
        || /подтвердите, что запросы отправляли вы|я не робот|unusual traffic/i.test(text.slice(0, 1500));
    return { url: location.href, title: document.title || '', text: text.slice(0, textLimit * 2),
             links, captcha };
}
"""

# Ссылки, которые почти всегда шум (меню, реклама, авторизация)
_NOISE_LINK_RE = re.compile(
    r"(passport\.|/login|/auth|/signin|/signup|/register|/support|/legal|/policy|/privacy|/terms|"
    r"/ads|adfox|yabs\.|/cookie|mailto:|javascript:)",
    re.IGNORECASE,
)


@dataclass
class WebResult:
    """Что модель увидит о прочитанной странице."""
    query: str
    url: str = ""
    title: str = ""
    text: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)
    error: str = ""

    def render(self, number: int, *, full: bool) -> str:
        head = f"#{number} web «{self.query}»"
        if self.error:
            return f"{head} → ✗ {self.error}"
        lines = [f"{head}", f"   Адрес страницы: {self.url}"]
        clean = _without_query(self.url)
        if clean != self.url:
            lines.append(f"   Адрес без параметров: {clean}")
        if self.title:
            lines.append(f"   Заголовок: {self.title}")
        if not full:
            return "\n".join(lines)
        if self.text:
            lines.append("   Текст страницы:")
            lines += ["   | " + ln for ln in self.text.splitlines() if ln.strip()]
        if self.links:
            lines.append("   Ссылки на странице (текст → адрес):")
            lines += [f"   - {t} → {h}" for t, h in self.links]
        return "\n".join(lines)


def _without_query(url: str) -> str:
    try:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except ValueError:
        return url


def to_url(query: str) -> str:
    """Адрес (http://, https://, www.) → как есть, остальное → поисковый запрос SEARCH_URL."""
    q = (query or "").strip()
    if _URL_RE.match(q):
        return q if q.lower().startswith("http") else f"https://{q}"
    return SEARCH_URL.format(query=quote_plus(q))


class WebResearch:
    """Вкладка поиска: одна на всё время работы, переиспользуется."""

    def __init__(self, browser) -> None:
        self._browser = browser
        self._page: Optional[Page] = None

    async def run(self, query: str) -> WebResult:
        url = to_url(query)
        result = WebResult(query=query)
        page = await self._ensure_page()
        logger.info("WEB: открываю %s", url)
        started = time.monotonic()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=WEB_TIMEOUT * 1000)
        except PlaywrightError as exc:
            if "ERR_ABORTED" not in str(exc):             # загрузка-скачивание — не ошибка страницы
                result.error = f"страница не открылась: {str(exc).splitlines()[0][:160]}"
        if not result.error:
            await self._settle(page)
            data = await self._read(page)
            if data.get("captcha"):
                data = await self._wait_captcha(page, data)
            result.url = str(data.get("url") or url)
            result.title = str(data.get("title") or "")[:200]
            result.text = _trim_text(str(data.get("text") or ""), WEB_TEXT_LIMIT)
            result.links = _pick_links(data.get("links") or [], result.url)
            if data.get("captcha"):
                result.error = "сайт требует пройти капчу — не удалось прочитать страницу"
        await self._back_to_task()
        logger.info("WEB: %s за %.1f с — %s", "✗ " + result.error if result.error else "прочитано",
                    time.monotonic() - started, result.url or url)
        return result

    async def close(self) -> None:
        if self._page is not None and not self._page.is_closed():
            try:
                await self._page.close()
            except PlaywrightError:
                pass
        self._page = None

    # ------------------------------------------------------------------

    async def _ensure_page(self) -> Page:
        if self._page is None or self._page.is_closed():
            self._page = await self._browser.context.new_page()
        else:
            try:
                await self._page.bring_to_front()
            except PlaywrightError:
                pass
        return self._page

    @staticmethod
    async def _settle(page: Page) -> None:
        """SPA (Яндекс Карты) дорисовывают результаты после load — ждём ещё немного."""
        try:
            await page.wait_for_load_state("load", timeout=min(WEB_TIMEOUT, 15) * 1000)
        except PlaywrightError:
            pass
        try:
            await page.wait_for_load_state("networkidle", timeout=5_000)
        except PlaywrightError:
            pass
        await asyncio.sleep(1.0)

    @staticmethod
    async def _read(page: Page) -> dict:
        try:
            return await page.evaluate(_JS_READ_PAGE, {"textLimit": WEB_TEXT_LIMIT, "linksLimit": WEB_LINKS_LIMIT})
        except PlaywrightError as exc:
            return {"url": page.url, "title": "", "text": "", "links": [], "error": str(exc)}

    async def _wait_captcha(self, page: Page, data: dict) -> dict:
        logger.warning("WEB: сайт показал капчу — решите её во вкладке поиска (жду до %.0f с)", CAPTCHA_TIMEOUT)
        deadline = time.monotonic() + CAPTCHA_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(3.0)
            if page.is_closed():
                break
            data = await self._read(page)
            if not data.get("captcha"):
                await self._settle(page)
                return await self._read(page)
        return data

    async def _back_to_task(self) -> None:
        try:
            await self._browser.page.bring_to_front()
        except PlaywrightError:
            pass


def _trim_text(text: str, limit: int) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    out, used = [], 0
    for ln in lines:
        if not ln:
            continue
        if used + len(ln) > limit:
            out.append("…")
            break
        out.append(ln)
        used += len(ln) + 1
    return "\n".join(out)


def _pick_links(raw: list[dict], page_url: str) -> list[tuple[str, str]]:
    """Полезные ссылки первыми: карточки организаций, результаты поиска; шум — прочь."""
    host = urlsplit(page_url).netloc
    scored: list[tuple[int, int, str, str]] = []
    for i, link in enumerate(raw):
        href, text = str(link.get("href") or ""), str(link.get("text") or "")
        if not href or _NOISE_LINK_RE.search(href):
            continue
        score = 0
        if re.search(r"/(org|firm|company|reviews?|review|otzyv|catalog|place)/", href):
            score -= 3
        if urlsplit(href).netloc != host:
            score -= 1                              # внешний результат поиска
        if len(text) < 3:
            score += 2
        scored.append((score, i, text, href))
    scored.sort()
    picked = sorted(scored[:WEB_LINKS_LIMIT], key=lambda s: s[1])
    return [(t, h) for _, _, t, h in picked]
