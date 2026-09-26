"""media.py — фото и аудио задания для модели.

Фото. v3 отправлял модели скриншот ОДНОЙ «главной» картинки. В реальных заданиях их
десятки: две карусели отелей по 10–20 фото, блоки фото банкомата под каждым вариантом.
Теперь каждое фото [ФОТО n] скачивается в исходном качестве (со страницы задания —
с её авторизацией; чужие CDN без CORS — через запрос браузерного контекста с теми же
куками), уменьшается в браузере (OffscreenCanvas, без Pillow) и уходит модели:
до VISION_SINGLE_MAX фото — по одному, больше — коллажами с крупными номерами,
совпадающими с [ФОТО n] в тексте страницы. Результат кэшируется на задание.

Аудио. Запись [АУДИО n] скачивается и расшифровывается через audio.transcriptions
того же API (ProxyAPI/OpenAI) — параллельно с воспроизведением. Требование
«Прослушайте звонок до конца» выполняется буквально: перед отправкой ответа агент
доигрывает запись до конца (AUDIO_PLAY_TO_END).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from playwright.async_api import Error as PlaywrightError, Frame

from config import (
    AUDIO_MAX_WAIT,
    AUDIO_TRANSCRIBE,
    LLM_VISION_DETAIL,
    VISION_CELL,
    VISION_IMAGE_SIDE,
    VISION_MAX_IMAGES,
    VISION_SINGLE_MAX,
)
from dom_parser import photo_groups
from models import MediaAudio, MediaImage, PageState, VisionImage

logger = logging.getLogger("twork.media")

# (bytes, имя файла, mime) → текст расшифровки или None
Transcriber = Callable[[bytes, str, str], Awaitable[Optional[str]]]

_MAX_AUDIO_BYTES = 24 * 1024 * 1024        # лимит audio.transcriptions — 25 МБ

# Получить картинки (по метке <img> или адресу) и отрисовать их одиночными JPEG или
# коллажами с номерами. Всё в памяти страницы: DOM задания не меняется.
_JS_RENDER = r"""
async (args) => {
    const b64ToBytes = (b64) => {
        const bin = atob(b64); const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out;
    };
    const blobToB64 = async (blob) => {
        const buf = new Uint8Array(await blob.arrayBuffer());
        let s = '';
        for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
        return btoa(s);
    };
    const clean = (bmp) => {            // чужая картинка без CORS «пачкает» холст — такую не отдать
        try {
            const c = new OffscreenCanvas(1, 1); const x = c.getContext('2d');
            x.drawImage(bmp, 0, 0, 1, 1); x.getImageData(0, 0, 1, 1); return true;
        } catch (e) { return false; }
    };
    const bitmapOf = async (item) => {
        if (item.b64) {
            try { return await createImageBitmap(new Blob([b64ToBytes(item.b64)], { type: item.mime || 'image/jpeg' })); }
            catch (e) { return null; }
        }
        const el = item.uid ? document.querySelector(`[data-agent-media="${item.uid}"]`) : null;
        if (el && el.complete && el.naturalWidth > 0) {
            try { const b = await createImageBitmap(el); if (clean(b)) return b; } catch (e) {}
        }
        if (item.src) {
            // свой домен — с куками (вложения задания требуют авторизации); чужой CDN — без
            // них: CORS с credentials не принимает «Access-Control-Allow-Origin: *»
            const same = new URL(item.src, location.href).origin === location.origin;
            for (const credentials of same ? ['include'] : ['omit', 'include']) {
                try {
                    const r = await fetch(item.src, { credentials });
                    if (r.ok) return await createImageBitmap(await r.blob());
                } catch (e) {}
            }
        }
        return null;
    };
    const results = [];
    for (const job of args.jobs) {
        const bitmaps = [];
        const failed = [];
        for (const item of job.items) {
            const b = await bitmapOf(item);
            if (b && b.width > 0) bitmaps.push([item.n, b]); else failed.push(item.n);
        }
        if (!bitmaps.length) { results.push({ ok: false, failed }); continue; }
        let canvas;
        if (job.kind === 'single') {
            const [, b] = bitmaps[0];
            const k = Math.min(1, job.maxSide / Math.max(b.width, b.height));
            canvas = new OffscreenCanvas(Math.max(1, Math.round(b.width * k)), Math.max(1, Math.round(b.height * k)));
            canvas.getContext('2d').drawImage(b, 0, 0, canvas.width, canvas.height);
        } else {
            const cols = Math.min(job.cols, bitmaps.length);
            const rows = Math.ceil(bitmaps.length / cols);
            const cell = job.cell, gap = 6;
            canvas = new OffscreenCanvas(cols * cell + (cols - 1) * gap, rows * cell + (rows - 1) * gap);
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height);
            bitmaps.forEach(([n, b], i) => {
                const x0 = (i % cols) * (cell + gap), y0 = Math.floor(i / cols) * (cell + gap);
                const k = Math.min(cell / b.width, cell / b.height);
                const w = b.width * k, h = b.height * k;
                ctx.drawImage(b, x0 + (cell - w) / 2, y0 + (cell - h) / 2, w, h);
                const label = String(n);
                const fs = Math.round(cell * 0.09);
                ctx.font = `bold ${fs}px sans-serif`;
                const tw = ctx.measureText(label).width;
                ctx.fillStyle = 'rgba(0,0,0,0.78)';
                ctx.fillRect(x0, y0, tw + fs * 0.9, fs * 1.45);
                ctx.fillStyle = '#ffeb3b';
                ctx.fillText(label, x0 + fs * 0.45, y0 + fs * 1.08);
            });
        }
        const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality: job.quality });
        results.push({ ok: true, b64: await blobToB64(blob), placed: bitmaps.map(([n]) => n), failed,
                       w: canvas.width, h: canvas.height });
    }
    return results;
}
"""

_JS_FETCH = r"""
async ({ src, limit }) => {
    try {
        const same = new URL(src, location.href).origin === location.origin;
        const r = await fetch(src, { credentials: same ? 'include' : 'omit' });
        if (!r.ok) return { ok: false, error: 'HTTP ' + r.status };
        const blob = await r.blob();
        if (blob.size > limit) return { ok: false, error: 'файл больше лимита: ' + blob.size };
        const buf = new Uint8Array(await blob.arrayBuffer());
        let s = '';
        for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
        return { ok: true, b64: btoa(s), type: blob.type || r.headers.get('content-type') || '' };
    } catch (e) {
        return { ok: false, error: String(e) };
    }
}
"""

_JS_AUDIO_STATE = r"""
(uid) => {
    const m = document.querySelector(`[data-agent-media="${uid}"]`);
    if (!m) return null;
    const d = Number.isFinite(m.duration) ? m.duration : 0;
    return { paused: m.paused, ended: m.ended || (d > 0 && m.currentTime >= d - 0.25),
             current: m.currentTime || 0, duration: d, ready: m.readyState,
             error: m.error ? String(m.error.code) : '' };
}
"""

_JS_AUDIO_PLAY = r"""
async (uid) => {
    const m = document.querySelector(`[data-agent-media="${uid}"]`);
    if (!m) return 'нет элемента';
    try { await m.play(); return 'ok'; } catch (e) { return String(e); }
}
"""

_EXT_BY_MIME = {"mpeg": "mp3", "mp3": "mp3", "webm": "webm", "ogg": "ogg", "wav": "wav",
                "x-wav": "wav", "mp4": "m4a", "aac": "aac", "x-m4a": "m4a", "flac": "flac"}


@dataclass
class _Transcript:
    text: Optional[str]
    note: str = ""


class MediaManager:
    """Фото (vision) и аудио (расшифровка, воспроизведение) текущего задания."""

    def __init__(self, browser, transcriber: Optional[Transcriber] = None) -> None:
        self._browser = browser
        self._transcriber = transcriber
        self._vision_cache: dict[tuple, tuple[list[VisionImage], list[str]]] = {}
        self._transcripts: dict[str, _Transcript] = {}
        self._pending: dict[str, asyncio.Task] = {}
        self._play_attempts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Фото
    # ------------------------------------------------------------------

    async def vision_images(self, frame: Frame, state: PageState) -> tuple[list[VisionImage], list[str]]:
        """Все фото задания для vision-модели и заметки («[ФОТО 3] не загрузилось»)."""
        images = state.images[:VISION_MAX_IMAGES]
        if not images:
            return [], []
        key = (state.content_hash, tuple(i.src for i in images))
        cached = self._vision_cache.get(key)
        if cached is not None:
            return cached
        started = time.monotonic()
        jobs, captions = self._plan_jobs(state, images)
        results = await self._render(frame, jobs)

        # чужие CDN без CORS: скачиваем запросом браузерного контекста (те же куки) и повторяем
        retry_ns = sorted({n for r in results for n in (r.get("failed") or [])})
        if retry_ns:
            by_n = {i.n: i for i in images}
            fetched = await self._fetch_many(frame, [by_n[n] for n in retry_ns if n in by_n])
            if fetched:
                for job in jobs:
                    for item in job["items"]:
                        if item["n"] in fetched:
                            item["b64"], item["mime"] = fetched[item["n"]]
                redo = [i for i, r in enumerate(results) if r.get("failed")]
                again = await self._render(frame, [jobs[i] for i in redo])
                for i, r in zip(redo, again):
                    results[i] = r

        out: list[VisionImage] = []
        failed: set[int] = set()
        for caption, result in zip(captions, results):
            failed.update(result.get("failed") or [])
            if result.get("ok") and result.get("b64"):
                placed = result.get("placed") or []
                if len(placed) > 1:
                    caption = f"ФОТО {_runs(placed)}"
                out.append(VisionImage(caption=caption, b64=result["b64"], detail=LLM_VISION_DETAIL))
        notes: list[str] = []
        if failed:
            notes.append(f"Не удалось получить фото: {_runs(sorted(failed))} (на странице они, вероятно, "
                         "не загрузились).")
        if len(state.images) > len(images):
            notes.append(f"Приложены первые {len(images)} фото из {len(state.images)}.")
        logger.info("Фото для модели: %d шт. → %d изображений за %.1f с%s",
                    len(images), len(out), time.monotonic() - started,
                    f", не получено: {_runs(sorted(failed))}" if failed else "")
        result = (out, notes)
        if out or failed:
            self._vision_cache[key] = result
        return result

    @staticmethod
    def _plan_jobs(state: PageState, images: list[MediaImage]) -> tuple[list[dict], list[str]]:
        """До VISION_SINGLE_MAX фото — по одному (детали: грязь, надписи на экране);
        больше — коллажами 2×2 (до 16 фото) или 3×3, не смешивая блоки фото."""
        def item(img: MediaImage) -> dict:
            return {"n": img.n, "uid": img.uid, "src": img.src}

        wanted = {i.n for i in images}
        if len(images) <= VISION_SINGLE_MAX:
            jobs = [{"kind": "single", "items": [item(i)], "maxSide": VISION_IMAGE_SIDE, "quality": 0.85}
                    for i in images]
            return jobs, [f"ФОТО {i.n}" for i in images]
        per = 4 if len(images) <= 16 else 9
        cols = 2 if per == 4 else 3
        cell = VISION_CELL if per == 4 else int(VISION_CELL * 0.75)
        by_n = {i.n: i for i in images}
        jobs, captions = [], []
        for group in photo_groups(state):
            group = [n for n in group if n in wanted]
            for start in range(0, len(group), per):
                chunk = group[start:start + per]
                if len(chunk) == 1:
                    jobs.append({"kind": "single", "items": [item(by_n[chunk[0]])],
                                 "maxSide": VISION_IMAGE_SIDE, "quality": 0.85})
                else:
                    jobs.append({"kind": "grid", "items": [item(by_n[n]) for n in chunk],
                                 "cols": cols, "cell": cell, "quality": 0.82})
                captions.append(f"ФОТО {_runs(chunk)}")
        return jobs, captions

    async def _render(self, frame: Frame, jobs: list[dict]) -> list[dict]:
        if not jobs:
            return []
        try:
            return await asyncio.wait_for(frame.evaluate(_JS_RENDER, {"jobs": jobs}), timeout=90)
        except (PlaywrightError, asyncio.TimeoutError) as exc:
            logger.warning("Фото не обработаны: %s", str(exc).splitlines()[0][:160] if str(exc) else exc)
            return [{"ok": False, "failed": [i["n"] for i in job["items"]]} for job in jobs]

    async def _fetch_many(self, frame: Frame, images: list[MediaImage]) -> dict[int, tuple[str, str]]:
        """Скачать картинки запросом контекста браузера (куки те же, CORS не действует)."""
        sem = asyncio.Semaphore(6)

        async def one(img: MediaImage) -> Optional[tuple[int, str, str]]:
            async with sem:
                data, mime = await self.fetch_bytes(frame, img.src, in_page=False)
                if not data:
                    return None
                return img.n, base64.b64encode(data).decode("ascii"), mime or "image/jpeg"

        done = await asyncio.gather(*(one(i) for i in images))
        return {n: (b64, mime) for n, b64, mime in (d for d in done if d)}

    async def fetch_bytes(self, frame: Frame, src: str, *, in_page: bool = True,
                          limit: int = _MAX_AUDIO_BYTES) -> tuple[Optional[bytes], str]:
        """Файл задания: сначала fetch внутри фрейма (его авторизация), затем запрос контекста."""
        if in_page:
            try:
                res = await asyncio.wait_for(frame.evaluate(_JS_FETCH, {"src": src, "limit": limit}), timeout=60)
                if res and res.get("ok"):
                    return base64.b64decode(res["b64"]), str(res.get("type") or "")
                logger.debug("fetch в странице не удался (%s): %s", src[-60:], res and res.get("error"))
            except (PlaywrightError, asyncio.TimeoutError) as exc:
                logger.debug("fetch в странице не удался: %s", exc)
        try:
            response = await self._browser.context.request.get(
                src, headers={"Referer": frame.url}, timeout=30_000, fail_on_status_code=False,
            )
            if response.ok:
                body = await response.body()
                if len(body) <= limit:
                    return body, response.headers.get("content-type", "")
            logger.debug("Запрос %s: HTTP %s", src[-60:], response.status)
        except PlaywrightError as exc:
            logger.debug("Запрос %s не удался: %s", src[-60:], exc)
        return None, ""

    # ------------------------------------------------------------------
    # Аудио: расшифровка
    # ------------------------------------------------------------------

    def start_transcription(self, frame: Frame, state: PageState) -> None:
        """Запустить скачивание и расшифровку всех записей задания (в фоне)."""
        if not AUDIO_TRANSCRIBE or self._transcriber is None:
            return
        for audio in state.audios:
            if not audio.src or audio.src in self._transcripts or audio.src in self._pending:
                continue
            logger.info("АУДИО %d: скачиваю и расшифровываю (%s)", audio.n, audio.src[-40:])
            self._pending[audio.src] = asyncio.create_task(self._transcribe(frame, audio))

    async def _transcribe(self, frame: Frame, audio: MediaAudio) -> None:
        started = time.monotonic()
        data, mime = await self.fetch_bytes(frame, audio.src)
        if not data:
            self._transcripts[audio.src] = _Transcript(None, "запись не удалось скачать")
            return
        subtype = (mime.split("/")[-1].split(";")[0] or "mpeg").strip().lower()
        filename = f"audio.{_EXT_BY_MIME.get(subtype, 'mp3')}"
        try:
            text = await self._transcriber(data, filename, mime or "audio/mpeg")  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 — сбой расшифровки не должен ронять агента
            logger.warning("Расшифровка АУДИО %d не удалась: %s", audio.n, exc)
            text = None
        if text:
            logger.info("АУДИО %d расшифровано за %.1f с (%d симв.)", audio.n, time.monotonic() - started, len(text))
            self._transcripts[audio.src] = _Transcript(text)
        else:
            self._transcripts[audio.src] = _Transcript(None, "расшифровка недоступна")

    async def transcripts(self, state: PageState, *, wait: float = 45.0) -> list[str]:
        """Расшифровки [АУДИО n] для промпта. Ждёт незавершённые не дольше wait секунд."""
        pending = [self._pending[a.src] for a in state.audios if a.src in self._pending]
        if pending:
            done, _ = await asyncio.wait(pending, timeout=wait)
            for task in done:
                if task.exception() is not None:
                    logger.warning("Расшифровка завершилась с ошибкой: %s", task.exception())
        for src in [s for s, t in self._pending.items() if t.done()]:
            self._pending.pop(src, None)
        out: list[str] = []
        for audio in state.audios:
            t = self._transcripts.get(audio.src)
            if t is None:
                out.append(f"[АУДИО {audio.n}] расшифровка ещё готовится")
            elif t.text:
                out.append(f"[АУДИО {audio.n}] расшифровка:\n{t.text}")
            else:
                out.append(f"[АУДИО {audio.n}] {t.note}")
        return out

    # ------------------------------------------------------------------
    # Аудио: воспроизведение
    # ------------------------------------------------------------------

    async def ensure_playing(self, frame: Frame, state: PageState) -> None:
        """Запустить недослушанные записи (автовоспроизведение бывает заблокировано)."""
        for audio in state.audios:
            if audio.ended or not audio.paused:
                continue
            tries = self._play_attempts.get(audio.src, 0)
            if tries >= 3:                  # плеер не запускается (ошибка записи) — не долбим кнопку
                continue
            self._play_attempts[audio.src] = tries + 1
            await self._play(frame, audio)

    async def _play(self, frame: Frame, audio: MediaAudio) -> None:
        if audio.play_uid:
            locator = frame.locator(f'[data-agent-media="{audio.play_uid}"]')
            try:
                if await locator.count():
                    outcome = await self._browser.click_locator(locator.first, f"play АУДИО {audio.n}")
                    if outcome.ok:
                        logger.info("АУДИО %d: воспроизведение (кнопка плеера)", audio.n)
                        await asyncio.sleep(0.5)
                        st = await frame.evaluate(_JS_AUDIO_STATE, audio.uid)
                        if st and not st.get("paused"):
                            return
            except PlaywrightError as exc:
                logger.debug("Кнопка play: %s", exc)
        try:
            result = await frame.evaluate(_JS_AUDIO_PLAY, audio.uid)
            logger.info("АУДИО %d: воспроизведение (%s)", audio.n, result)
        except PlaywrightError as exc:
            logger.debug("audio.play(): %s", exc)

    async def wait_finished(self, frame: Frame, state: PageState) -> bool:
        """Доиграть записи до конца («Прослушайте звонок до конца»). True — дослушано."""
        for audio in state.audios:
            st = await self._state(frame, audio)
            if st is None or st["ended"]:
                continue
            left = max(0.0, st["duration"] - st["current"]) if st["duration"] else 60.0
            deadline = time.monotonic() + min(AUDIO_MAX_WAIT, left + 20)
            logger.info("АУДИО %d: дослушиваю до конца (осталось ~%.0f с)", audio.n, left)
            restarts = 0
            last_log = time.monotonic()
            while time.monotonic() < deadline:
                st = await self._state(frame, audio)
                if st is None or st["ended"]:
                    break
                if st["error"]:
                    logger.warning("АУДИО %d: ошибка плеера (код %s)", audio.n, st["error"])
                    break
                if st["paused"]:
                    if restarts >= 3:
                        logger.warning("АУДИО %d: воспроизведение останавливается — не жду дальше", audio.n)
                        break
                    restarts += 1
                    await self._play(frame, audio)
                if time.monotonic() - last_log > 15:
                    last_log = time.monotonic()
                    logger.info("АУДИО %d: %.0f / %.0f с", audio.n, st["current"], st["duration"])
                await asyncio.sleep(1.0)
            else:
                logger.warning("АУДИО %d: не доиграла за отведённое время", audio.n)
                return False
        return True

    async def _state(self, frame: Frame, audio: MediaAudio) -> Optional[dict]:
        try:
            return await frame.evaluate(_JS_AUDIO_STATE, audio.uid)
        except PlaywrightError:
            return None

    def close(self) -> None:
        """Отменить незавершённые расшифровки (агент остановлен)."""
        for task in self._pending.values():
            task.cancel()
        self._pending.clear()

    def forget_task(self) -> None:
        """Кэш фото — только на текущее задание; расшифровки живут дольше (повтор записи)."""
        self._vision_cache.clear()
        self._play_attempts.clear()
        if len(self._transcripts) > 50:
            for src in list(self._transcripts)[:-20]:
                self._transcripts.pop(src, None)


def _runs(numbers: list[int]) -> str:
    """[1, 2, 3, 5] → «1–3, 5»."""
    nums = sorted(set(numbers))
    if not nums:
        return ""
    parts: list[str] = []
    start = prev = nums[0]
    for n in nums[1:] + [None]:  # type: ignore[list-item]
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}–{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(parts)
