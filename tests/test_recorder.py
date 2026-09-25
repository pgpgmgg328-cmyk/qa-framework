"""Режим записи: снимки экранов, действия человека, вкладки и упаковка в zip."""

from __future__ import annotations

import asyncio
import json
import zipfile

from browser_controller import BrowserController
from recorder import Recorder
from tests.helpers import install_routes, main_url, run


async def _wait(predicate, timeout: float = 15.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("не дождались условия")
        await asyncio.sleep(0.1)


def test_recorder_captures_screens_actions_and_tabs(tmp_path):
    async def scenario():
        controller = BrowserController(on_context=install_routes, headless=True)
        recorder = Recorder(browser=controller, out_dir=tmp_path, interval=0.3)
        task = asyncio.create_task(recorder.run())
        await _wait(lambda: recorder.snapshots >= 1)

        # «человек» раскрывает ветку и выбирает вариант — настоящими кликами мыши
        frame = await controller.find_target_frame()
        await frame.locator(".child__expand").nth(1).click()
        await _wait(lambda: recorder.snapshots >= 2)
        await frame.get_by_text("Смартфоны").click()
        await _wait(lambda: recorder.snapshots >= 3)
        page_log = await frame.evaluate("() => window.__log")

        # поиск «в интернете» во второй вкладке
        tab = await controller.context.new_page()
        await tab.goto(main_url("intro"))
        await asyncio.sleep(2.0)

        recorder.stop()
        parts = await task
        return parts, page_log

    parts, page_log = run(scenario())
    assert page_log == ["toggle:Электроника"]            # рекордер сам ничего не нажимал

    session = next(p for p in tmp_path.iterdir() if p.is_dir())
    shots = sorted(p for p in session.iterdir() if p.is_dir() and p.name[:3].isdigit())
    assert len(shots) >= 3
    for name in ("frame.html", "screen.jpg", "llm_view.txt", "state.json", "meta.json", "actions.json"):
        assert (shots[0] / name).exists(), name
    assert "data-agent-id" in (shots[0] / "frame.html").read_text(encoding="utf-8")
    assert "[FOLDER закрыта] «Электроника»" in (shots[0] / "llm_view.txt").read_text(encoding="utf-8")

    # действия привязаны к элементам снимка, который человек видел в момент клика
    first = json.loads((shots[0] / "actions.json").read_text(encoding="utf-8"))
    assert any(a["type"] == "click" and a.get("element", {}).get("text") == "Электроника" for a in first)
    second = json.loads((shots[1] / "actions.json").read_text(encoding="utf-8"))
    assert any(a.get("element", {}).get("text") == "Смартфоны" and a.get("element", {}).get("kind") == "OPTION"
               for a in second)
    assert any(a["type"] == "change" and a.get("checked") is True for a in second)

    navigation = [json.loads(line) for line in (session / "navigation.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any("task=intro" in n["url"] for n in navigation)
    assert (session / "external" / "001.jpg").exists()

    summary = json.loads((session / "session.json").read_text(encoding="utf-8"))
    assert len(summary["snapshots"]) == len(shots)
    assert parts and all(p.exists() for p in parts)
    with zipfile.ZipFile(parts[0]) as archive:
        names = archive.namelist()
    assert any(n.endswith("frame.html") for n in names) and any(n.endswith("session.json") for n in names)
