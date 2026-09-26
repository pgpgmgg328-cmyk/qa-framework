"""recorder.py — режим записи: агент ничего не нажимает, а сохраняет то, что видит,
и то, что делает человек.

Запуск: python main.py --record

Пока вы вручную проходите задания в открывшемся окне, на каждом новом экране
сохраняется папка recordings/<сессия>/<NNN>_<где>_<задание>/:
  frame.html   — HTML фрейма задания (с метками data-agent-id, которые расставил парсер)
  screen.jpg   — скриншот окна
  llm_view.txt — ровно то, что агент отправил бы модели на этом экране
  state.json   — разобранный снимок: элементы, типы, состояния, текст задания
  meta.json    — адреса страницы и фреймов, время
  actions.json — ваши действия на этом экране: клики, выбор, ввод текста, Enter
Отдельно: navigation.jsonl (переходы, в т.ч. в других вкладках — поиск в интернете)
и external/*.jpg (скриншоты внешних страниц). В конце всё упаковывается в zip-части
до 24 МБ — их удобно загрузить на GitHub через браузер.

Эти данные — демонстрации «экран → правильное действие человека» для каждого
типа задания: по ним пишутся правила для агента и проверяется парсер.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from playwright.async_api import BrowserContext, Error as PlaywrightError, Frame, Page

from browser_controller import BrowserController, is_connection_lost
from config import PROJECT_DIR
from dom_parser import DomParser
from models import DecisionContext, PageState
from openrouter_connector import LLMConnector

logger = logging.getLogger("twork.recorder")

_PART_LIMIT_BYTES = 24 * 1024 * 1024      # веб-загрузка GitHub принимает файлы до 25 МБ
_MAX_EXTERNAL_SHOTS = 300
_MAX_SNAPSHOTS = 3000

# Слушатели действий человека. Ставятся в КАЖДЫЙ документ контекста (включая iframe
# и новые вкладки) через add_init_script. Фаза capture — событие фиксируется до того,
# как приложение перерисует DOM, поэтому метка data-agent-id ещё указывает на элемент
# снимка, который человек видел.
_JS_RECORDER_INIT = r"""
(() => {
    if (window.__agentRec) return;
    const events = [];
    window.__agentRec = { events };
    const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim();
    const attr = (el, n) => (el && el.getAttribute ? el.getAttribute(n) : null);
    const describe = (node) => {
        const el = node && node.nodeType === 1 ? node : (node && node.parentElement);
        if (!el) return {};
        const stamped = el.closest ? el.closest('[data-agent-id]') : null;
        const main = stamped
            || (el.closest && el.closest('button, a, label, input, select, textarea, [role]'))
            || el;
        const cls = String(attr(main, 'class') || '').split(/\s+/).filter(Boolean).slice(0, 3).join('.');
        const text = norm(main.innerText || attr(main, 'aria-label') || attr(main, 'title')
                          || attr(main, 'placeholder') || '').slice(0, 160);
        return {
            uid: stamped ? attr(stamped, 'data-agent-id') : null,
            tag: String(main.localName || '').toLowerCase(),
            cls, role: attr(main, 'role'), text,
            name: attr(el, 'name') || attr(el, 'id') || attr(el, 'placeholder') || null,
            href: main.href ? String(main.href).slice(0, 500) : null,
        };
    };
    const valueOf = (el) => {
        if (!el) return null;
        if (el.type === 'password') return el.value ? '***' : '';
        if (typeof el.value === 'string') return el.value.slice(0, 1000);
        if (el.isContentEditable) return norm(el.innerText).slice(0, 1000);
        return null;
    };
    const push = (type, target, extra) => {
        try {
            const ev = Object.assign({ type, t: Date.now(), url: location.href }, describe(target), extra || {});
            ev.key = [ev.uid, ev.tag, ev.name].join('|');
            const same = ev.uid || [ev.tag, ev.name, ev.text].join('|');
            const last = events[events.length - 1];
            // клик по подписи варианта браузер сам повторяет на его radio — это один клик
            if (type === 'click' && last && last.type === 'click' && last.same === same && ev.t - last.t < 150) return;
            ev.same = same;
            // набор текста: храним только последнее значение поля
            if (type === 'input' && last && last.type === 'input' && last.key === ev.key) events[events.length - 1] = ev;
            else events.push(ev);
            if (events.length > 1000) events.shift();
        } catch (e) { /* запись не должна мешать странице */ }
    };
    const real = (e) => (e.composedPath && e.composedPath()[0]) || e.target;   // цель внутри shadow DOM
    document.addEventListener('click', (e) => push('click', real(e), { x: Math.round(e.clientX), y: Math.round(e.clientY) }), true);
    document.addEventListener('change', (e) => {
        const t = real(e);
        push('change', t, { value: valueOf(t), checked: typeof t.checked === 'boolean' ? t.checked : null });
    }, true);
    document.addEventListener('input', (e) => push('input', real(e), { value: valueOf(real(e)) }), true);
    document.addEventListener('keydown', (e) => { if (e.key === 'Enter') push('enter', real(e), { value: valueOf(real(e)) }); }, true);
    document.addEventListener('submit', (e) => push('submit', real(e)), true);
})();
"""

_JS_PULL_EVENTS = "() => (window.__agentRec ? window.__agentRec.events.splice(0) : [])"


@dataclass
class _Snapshot:
    folder: Path
    state: PageState
    signature: str
    frame_url: str
    meta: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)


def pack_session(session_dir: Path, limit_bytes: int = _PART_LIMIT_BYTES) -> list[Path]:
    """Упаковать сессию в zip-части не больше limit_bytes (HTML сжимается ~в 10 раз)."""
    files = sorted(p for p in session_dir.rglob("*") if p.is_file())
    parts: list[Path] = []
    archive: Optional[zipfile.ZipFile] = None
    for path in files:
        if archive is None:
            part = session_dir.parent / f"{session_dir.name}-part{len(parts) + 1}.zip"
            archive = zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9)
            parts.append(part)
        archive.write(path, arcname=str(path.relative_to(session_dir.parent)))
        if archive.fp is not None and archive.fp.tell() > limit_bytes:
            archive.close()
            archive = None
    if archive is not None:
        archive.close()
    return parts


class Recorder:
    """Наблюдатель: снимает каждый новый экран и действия человека, сам ничего не нажимает."""

    def __init__(
        self,
        *,
        browser: Optional[BrowserController] = None,
        out_dir: Optional[Path] = None,
        interval: float = 1.0,
    ) -> None:
        self._browser = browser or BrowserController(headless=False)   # записывать можно только в видимом окне
        self._root = out_dir or (PROJECT_DIR / "recordings")
        self._interval = interval
        self._session_dir = self._root / datetime.now().strftime("%Y%m%d-%H%M%S")
        self._current: Optional[_Snapshot] = None
        self._orphan_events: list[dict[str, Any]] = []
        self._summary: list[dict[str, Any]] = []
        self._frames_seen: dict[str, int] = {}
        self._stop = asyncio.Event()
        self._background: set[asyncio.Task] = set()
        self._watched_pages: set[int] = set()
        self._external_shots = 0
        self._nav_count = 0
        self.snapshots = 0
        self.parts: list[Path] = []

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    async def run(self) -> list[Path]:
        self._session_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 60)
        logger.info("РЕЖИМ ЗАПИСИ. Агент ничего не нажимает — проходите задания сами.")
        logger.info("Папка записи: %s", self._session_dir)
        logger.info("Закончить: закройте окно браузера или нажмите Ctrl+C в терминале.")
        logger.info("=" * 60)
        try:
            async with self._browser:
                await self._setup(self._browser.context)
                try:
                    await self._loop()
                finally:
                    await self._finish_live()
        finally:
            self._finish_files()
            self.parts = pack_session(self._session_dir)
            logger.info("Записано экранов: %d, переходов: %d", self.snapshots, self._nav_count)
            for part in self.parts:
                logger.info("Архив для отправки: %s (%.1f МБ)", part, part.stat().st_size / 1024 / 1024)
        return self.parts

    async def _setup(self, context: BrowserContext) -> None:
        # будущие документы (переходы, новые вкладки, iframe) получат слушатели сами,
        # в уже открытые — внедряем вручную
        await context.add_init_script(_JS_RECORDER_INIT)
        for page in context.pages:
            self._watch_page(page)
            for frame in page.frames:
                try:
                    await frame.evaluate(_JS_RECORDER_INIT)
                except PlaywrightError:
                    pass
        context.on("page", self._on_new_page)

    async def _loop(self) -> None:
        while not self._stop.is_set() and self.snapshots < _MAX_SNAPSHOTS:
            if self._browser.is_closed():
                logger.info("Окно браузера закрыто — завершаю запись")
                break
            try:
                await self._tick()
            except Exception as exc:  # noqa: BLE001 — сбой одного такта не должен прерывать запись
                if self._browser.is_closed() or is_connection_lost(exc):
                    logger.info("Связь с браузером потеряна — завершаю запись")
                    break
                logger.debug("Такт записи пропущен: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    async def _finish_live(self) -> None:
        """То, что требует живого браузера: последние действия человека."""
        for task in list(self._background):
            task.cancel()
        try:
            await self._collect_events()
        except Exception:  # noqa: BLE001 — браузер мог уже закрыться
            pass

    def _finish_files(self) -> None:
        self._close_current()
        if self._orphan_events:
            self._write_json(self._session_dir / "actions_without_screen.json", self._orphan_events)
        self._write_json(self._session_dir / "session.json", {
            "started": self._session_dir.name,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "snapshots": self._summary,
            "frames_seen": self._frames_seen,
            "navigations": self._nav_count,
        })

    # ------------------------------------------------------------------
    # Такт записи
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        await self._collect_events()
        frame, where = await self._pick_frame()
        state = await DomParser(frame).parse(quiet=True)
        signature = self._signature(state, frame, where)
        if self._current is not None and signature == self._current.signature:
            return
        # экран меняется — ждём, пока он «успокоится», и снимаем заново
        await self._browser.wait_settle(frame, timeout_ms=2000)
        state = await DomParser(frame).parse(quiet=True)
        signature = self._signature(state, frame, where)
        if self._current is not None and signature == self._current.signature:
            return
        await self._collect_events()      # действия до смены экрана — к старому снимку
        self._close_current()
        await self._save_snapshot(frame, where, state, signature)

    async def _pick_frame(self) -> tuple[Frame, str]:
        """Фрейм задания по FRAME_KEYWORDS; если его нет — крупнейший видимый iframe
        (задания другого вида могут жить по другому адресу); иначе главная страница."""
        frame = await self._browser.find_target_frame()
        if frame is not None:
            return frame, "task"
        page = self._browser.page
        best: Optional[Frame] = None
        best_area = 0.0
        for candidate in page.frames:
            if candidate is page.main_frame or candidate.is_detached():
                continue
            try:
                box = await (await candidate.frame_element()).bounding_box()
            except PlaywrightError:
                continue
            if box and box["width"] * box["height"] > best_area:
                best, best_area = candidate, box["width"] * box["height"]
        viewport = await self._browser.viewport()
        if best is not None and best_area > 0.25 * viewport["width"] * viewport["height"]:
            return best, "iframe"
        return page.main_frame, "main"

    @staticmethod
    def _signature(state: PageState, frame: Frame, where: str) -> str:
        """Смена экрана: элементы и их состояния. Значения полей не входят — иначе
        набор текста порождал бы новый снимок каждую секунду."""
        h = hashlib.sha1(f"{where}|{frame.url}|{state.task_identifier}\n".encode("utf-8"))
        for el in state.elements:
            h.update(
                f"{el.key}|{el.kind.value}|{el.folder_state.value}|"
                f"{int(el.is_selected)}|{int(el.is_disabled)}\n".encode("utf-8")
            )
        for alert in state.alerts:
            h.update(alert.encode("utf-8"))
        return h.hexdigest()[:16]

    async def _save_snapshot(self, frame: Frame, where: str, state: PageState, signature: str) -> None:
        self.snapshots += 1
        folder = self._session_dir / f"{self.snapshots:03d}_{where}_{state.task_identifier[:8]}"
        folder.mkdir(parents=True, exist_ok=True)
        page = self._browser.page

        try:
            html = await frame.content()
        except PlaywrightError as exc:
            html = f"<!-- HTML недоступен: {exc} -->"
        (folder / "frame.html").write_text(html, encoding="utf-8")
        try:
            await page.screenshot(path=str(folder / "screen.jpg"), type="jpeg", quality=70, timeout=5_000)
        except PlaywrightError as exc:
            logger.debug("Скриншот не сохранён: %s", exc)
        (folder / "llm_view.txt").write_text(
            LLMConnector.build_user_message(state, DecisionContext()), encoding="utf-8",
        )
        (folder / "state.json").write_text(state.model_dump_json(indent=1), encoding="utf-8")

        frames = []
        for f in page.frames:
            if f.is_detached():
                continue
            frames.append(f.url)
            self._frames_seen[f.url] = self._frames_seen.get(f.url, 0) + 1
        meta = {
            "n": self.snapshots,
            "time": datetime.now().isoformat(timespec="seconds"),
            "where": where,                       # task | iframe | main
            "page_url": page.url,
            "frame_url": frame.url,
            "frames": frames,
            "task_id": state.task_identifier,
            "task_preview": state.task_preview,
            "elements": len(state.elements),
            "alerts": state.alerts,
        }
        self._write_json(folder / "meta.json", meta)
        self._current = _Snapshot(folder=folder, state=state, signature=signature,
                                  frame_url=frame.url, meta=meta)
        logger.info(
            "📸 #%d [%s] «%s» — элементов: %d%s", self.snapshots, where,
            state.task_preview[:60] or frame.url[:60], len(state.elements),
            f", сообщения: {' | '.join(state.alerts)[:80]}" if state.alerts else "",
        )

    def _close_current(self) -> None:
        current, self._current = self._current, None
        if current is None:
            return
        self._write_json(current.folder / "actions.json", current.events)
        self._summary.append({
            "n": current.meta["n"],
            "folder": current.folder.name,
            "where": current.meta["where"],
            "task_preview": current.meta["task_preview"],
            "frame_url": current.meta["frame_url"],
            "actions": [self._short(ev) for ev in current.events if ev["type"] != "input"],
        })

    # ------------------------------------------------------------------
    # Действия человека
    # ------------------------------------------------------------------

    async def _collect_events(self) -> None:
        pages = list(self._browser.context.pages)
        for page_index, page in enumerate(pages):
            frames = page.frames if page is self._browser.page else [page.main_frame]
            for frame in frames:
                if frame.is_detached():
                    continue
                try:
                    events = await frame.evaluate(_JS_PULL_EVENTS)
                except PlaywrightError:
                    continue
                for event in events:
                    event["page"] = page_index
                    self._attach(event)

    def _attach(self, event: dict[str, Any]) -> None:
        current = self._current
        if current is not None and event.get("uid") and event.get("url") == current.frame_url:
            try:
                index = int(str(event["uid"]).rsplit("-", 1)[1])
            except (IndexError, ValueError):
                index = -1
            if 0 <= index < len(current.state.elements):
                el = current.state.elements[index]
                event["element"] = {"index": el.index, "kind": el.kind.value, "text": el.label(), "path": el.path}
        if event["type"] in ("click", "change", "enter", "submit"):
            logger.info("   👆 %s", self._short(event))
        (current.events if current is not None else self._orphan_events).append(event)

    @staticmethod
    def _short(event: dict[str, Any]) -> str:
        el = event.get("element")
        target = f"[{el['index']}] {el['kind']} «{el['text']}»" if el else f"{event.get('tag')} «{event.get('text', '')[:60]}»"
        value = event.get("value")
        extra = f" = «{str(value)[:60]}»" if value not in (None, "") and event["type"] != "click" else ""
        where = f" (вкладка {event['page']})" if event.get("page") else ""
        return f"{event['type']} {target}{extra}{where}"

    # ------------------------------------------------------------------
    # Вкладки и переходы (поиск в интернете и т.п.)
    # ------------------------------------------------------------------

    def _watch_page(self, page: Page) -> None:
        if id(page) in self._watched_pages:
            return
        self._watched_pages.add(id(page))
        page.on("framenavigated", lambda frame: self._on_navigated(page, frame))

    def _on_new_page(self, page: Page) -> None:
        self._watch_page(page)
        self._log_navigation(page, page.url, "new_tab")

    def _on_navigated(self, page: Page, frame: Frame) -> None:
        if frame is page.main_frame:
            self._log_navigation(page, frame.url, "navigated")

    def _log_navigation(self, page: Page, url: str, kind: str) -> None:
        if not url or url == "about:blank":
            return
        self._nav_count += 1
        pages = self._browser.context.pages
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "page": pages.index(page) if page in pages else -1,
            "kind": kind,
            "url": url,
        }
        external = page is not self._browser.page
        if external and self._external_shots < _MAX_EXTERNAL_SHOTS:
            self._external_shots += 1
            record["shot"] = f"external/{self._external_shots:03d}.jpg"
            task = asyncio.ensure_future(self._shoot_external(page, self._session_dir / record["shot"]))
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        with (self._session_dir / "navigation.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("   🌐 %s: %s", "новая вкладка" if kind == "new_tab" else "переход", url[:120])

    @staticmethod
    async def _shoot_external(page: Page, path: Path) -> None:
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=10_000)
            await asyncio.sleep(1.0)
            path.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(path), type="jpeg", quality=60, timeout=5_000)
        except PlaywrightError:
            pass

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


async def record() -> list[Path]:
    """Точка входа режима записи (python main.py --record)."""
    started = time.monotonic()
    parts = await Recorder().run()
    logger.info("Запись длилась %.0f с", time.monotonic() - started)
    return parts
