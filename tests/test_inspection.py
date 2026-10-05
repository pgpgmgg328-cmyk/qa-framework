"""Исследование новой площадки: ожидание «страница успокоилась» (XHR/fetch, DOM, лоадеры) и
опись компонентов интерфейса в записи (components.json). Страницы — синтетические, в духе
React-SPA; к интерфейсу конкретной площадки они отношения не имеют."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import pytest
from playwright.async_api import BrowserContext, Route

from browser_controller import BrowserController
from dom_parser import DomParser
from models import ElementKind
from recorder import Recorder
from tests.helpers import run

SPA = """<!doctype html><html lang="ru"><head><meta charset="utf-8"></head><body>
<div id="app"><div class="card-skeleton" style="width:300px;height:80px;background:#eee"></div></div>
<div role="progressbar" aria-valuenow="3" aria-valuemax="14" style="width:200px;height:8px;background:#ccc"></div>
<script>
  fetch('/api/poll').catch(() => {});                       // long-poll: ответа не будет
  fetch('/api/task').then((r) => r.json()).then((task) => {
    setTimeout(() => fetch('/api/options').then((r) => r.json()).then((o) => {
      document.getElementById('app').innerHTML = '<h1>' + task.title + '</h1>' + o.items.map(
        (x) => '<label><input type="radio" name="a"> ' + x + '</label>').join('');
      window.__ready = true;
    }), 300);
  });
</script></body></html>"""

SPINNER = """<!doctype html><html><body><p>Ждём</p>
<div class="page-spinner" style="width:24px;height:24px;background:#999"></div></body></html>"""

PROGRESS_ONLY = """<!doctype html><html><body><p>Задание 3 из 14</p>
<div role="progressbar" aria-valuenow="3" aria-valuemax="14" style="width:200px;height:8px;background:#ccc"></div>
</body></html>"""

WIDGETS = """<!doctype html><html lang="ru"><head><meta charset="utf-8"></head><body>
<h2>Выберите ответ</h2>
<div role="radiogroup" class="ui-radio-group">
  <div role="radio" aria-checked="true" tabindex="0" class="ui-radio ui-radio_checked" style="cursor:pointer">
    <span class="ui-radio__label">Да</span></div>
  <div role="radio" aria-checked="false" tabindex="-1" class="ui-radio" style="cursor:pointer">
    <span class="ui-radio__label">Нет</span></div>
</div>
<span role="checkbox" aria-checked="false" aria-disabled="true" class="ui-checkbox">Недоступно</span>
<button data-testid="submit-button" class="ui-button ui-button_primary">Отправить</button>
<div role="dialog" aria-modal="true" class="ui-modal"><p>Точно отправить?</p>
  <button class="ui-button">Отмена</button></div>
<div class="list-skeleton" style="width:100px;height:20px;background:#eee"></div>
<iframe src="https://widgets.test/inner.html" style="width:300px;height:150px"></iframe>
</body></html>"""

INNER = """<!doctype html><html><body><input type="checkbox" checked> Внутри фрейма</body></html>"""


def _routes(release: asyncio.Event):
    async def handler(route: Route) -> None:
        path = route.request.url.split("widgets.test", 1)[-1]
        if path == "/api/poll":
            await release.wait()
            try:
                await route.abort()
            except Exception:  # noqa: BLE001 — браузер уже закрыт
                pass
            return
        if path == "/api/task":
            await asyncio.sleep(1.0)
            await route.fulfill(status=200, content_type="application/json", body=json.dumps({"title": "Вопрос"}))
            return
        if path == "/api/options":
            await asyncio.sleep(0.5)
            await route.fulfill(status=200, content_type="application/json",
                                body=json.dumps({"items": ["Да", "Нет"]}))
            return
        pages = {"/spa.html": SPA, "/spinner.html": SPINNER, "/progress.html": PROGRESS_ONLY,
                 "/widgets.html": WIDGETS, "/inner.html": INNER}
        await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=pages.get(path, "<p>?</p>"))

    async def install(context: BrowserContext) -> None:
        await context.route("https://widgets.test/**", handler)

    return install


def test_waits_for_xhr_fetch_dom_and_skeletons():
    """React-форма догружается после загрузки страницы: вопрос, через паузу — варианты; пока их нет,
    виден скелетон. Ожидание не заканчивается в паузе между запросами, не ждёт вечный long-poll и
    не принимает прогресс «3 из 14» за загрузку."""
    async def scenario():
        release = asyncio.Event()
        controller = BrowserController(on_context=_routes(release), headless=True,
                                       start_url="https://widgets.test/spa.html")
        async with controller:
            started = time.monotonic()
            ok = await controller.wait_for_network_idle_and_dom(timeout_ms=10_000, long_request_s=1.0)
            elapsed = time.monotonic() - started
            ready = await controller.page.evaluate("() => !!window.__ready")
            state = await DomParser(controller.page.main_frame).parse(quiet=True)
            release.set()
        return ok, elapsed, ready, state

    ok, elapsed, ready, state = run(scenario())
    assert ok and ready and elapsed < 8
    options = [e.text for e in state.elements if e.kind == ElementKind.OPTION]
    assert options == ["Да", "Нет"] and "Вопрос" in state.task_text


def test_visible_spinner_keeps_waiting_and_progress_bar_does_not():
    async def scenario():
        release = asyncio.Event()
        controller = BrowserController(on_context=_routes(release), headless=True,
                                       start_url="https://widgets.test/spinner.html")
        async with controller:
            spinner = await controller.wait_for_network_idle_and_dom(timeout_ms=1500)
            await controller.page.goto("https://widgets.test/progress.html")
            started = time.monotonic()
            progress = await controller.wait_for_network_idle_and_dom(timeout_ms=5000)
            elapsed = time.monotonic() - started
            await controller.page.goto("https://widgets.test/spinner.html")
            extra = await controller.wait_for_network_idle_and_dom(timeout_ms=1500, extra_loaders="p")
            release.set()
        return spinner, progress, elapsed, extra

    spinner, progress, elapsed, extra = run(scenario())
    assert spinner is False                                   # спиннер так и не пропал
    assert progress is True and elapsed < 3                  # «3 из 14» — не загрузка
    assert extra is False                                    # лоадер площадки задан селектором


def test_components_inventory_describes_the_interface(tmp_path):
    """components.json — по нему подбирается поддержка новой площадки: роли и aria-состояния
    переключателей, классы и data-атрибуты кнопок, диалоги, скелетоны, фреймы (их HTML — отдельно)."""
    async def scenario():
        release = asyncio.Event()
        controller = BrowserController(on_context=_routes(release), headless=True,
                                       start_url="https://widgets.test/widgets.html")
        async with controller:
            page = controller.page
            await page.wait_for_load_state("load")
            recorder = Recorder.__new__(Recorder)
            await recorder._save_components(page, page.main_frame, tmp_path)
            release.set()

    run(scenario())
    data = json.loads((tmp_path / "components.json").read_text(encoding="utf-8"))
    main = next(f for f in data["frames"] if f["main"])
    inner = next(f for f in data["frames"] if not f["main"])
    inv = main["inventory"]
    assert main["snapshot"] and inv["roles"]["radio"] == 2 and inv["roles"]["dialog"] == 1
    assert inv["states"]["aria-checked=true"] == 1 and inv["states"]["aria-disabled=true"] == 1
    assert inv["data"]["data-testid"] == 1 and inv["classes"]["ui-radio"] == 2
    radios = [c for c in inv["choices"] if c["role"] == "radio"]
    assert [(r["text"], r["state"]["aria-checked"], r["cursor"]) for r in radios] == [
        ("Да", "true", "pointer"), ("Нет", "false", "pointer")]
    submit = next(b for b in inv["buttons"] if b["text"] == "Отправить")
    assert submit["testid"] == "submit-button" and "ui-button_primary" in submit["cls"]
    assert inv["dialogs"][0]["state"]["aria-modal"] == "true"
    assert any("list-skeleton" in x["cls"] for x in inv["loaders"])
    assert inv["iframes"][0]["src"] == "https://widgets.test/inner.html"
    assert inner["html"] == "frames/01.html" and "Внутри фрейма" in (tmp_path / "frames" / "01.html").read_text(
        encoding="utf-8")
    assert inner["inventory"]["choices"][0]["checked"] is True


@pytest.mark.parametrize("argv, ok", [
    (["--record", "--url", "https://profit.ozon.ru"], True),
    (["--url", "https://profit.ozon.ru"], False),             # агент там решать не умеет
    (["--record", "--url", "profit.ozon.ru"], False),          # без https://
])
def test_url_option_is_for_recording_only(monkeypatch, argv, ok):
    import main

    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    if ok:
        assert main.parse_args().url == "https://profit.ozon.ru"
    else:
        with pytest.raises(SystemExit):
            main.parse_args()
