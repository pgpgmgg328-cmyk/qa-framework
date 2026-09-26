"""Видимое окно (как у пользователя): без эмуляции viewport, клики мышью точные.

Открывает настоящее окно браузера, поэтому по умолчанию пропускается.
Запуск: RUN_HEADED_TESTS=1 python -m pytest tests/test_headed.py
(на Linux без монитора — под Xvfb: xvfb-run -a ...).
"""

from __future__ import annotations

import os

import pytest

from browser_controller import BrowserController
from dom_parser import DomParser
from tests.helpers import install_routes, run

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HEADED_TESTS") != "1", reason="открывает окно браузера; RUN_HEADED_TESTS=1",
)


def _one(state, text):
    return next(e for e in state.elements if e.text == text)


def test_headed_window_uses_real_size_and_clicks_land():
    async def scenario():
        # conftest задаёт PAGE_ZOOM=75%: в видимом окне он игнорируется (и не должен ронять запуск)
        controller = BrowserController(on_context=install_routes, headless=False)
        async with controller:
            assert controller.page.viewport_size is None          # страница подстраивается под окно
            size = await controller.viewport()
            assert size["width"] > 300 and size["height"] > 300

            frame = None
            for _ in range(50):
                frame = await controller.find_target_frame()
                if frame is not None:
                    break
                await controller.page.wait_for_timeout(100)
            assert frame is not None
            await frame.wait_for_load_state("load")

            state = await DomParser(frame).parse(quiet=True)
            first = await controller.click_element(frame, _one(state, "Электроника"), prefer_toggle=True)
            await controller.wait_settle(frame)
            state = await DomParser(frame).parse(quiet=True)
            second = await controller.click_element(frame, _one(state, "Смартфоны"))
            await controller.wait_settle(frame)
            state = await DomParser(frame).parse(quiet=True)
            third = await controller.click_element(frame, _one(state, "Завершить"))
            await controller.page.wait_for_timeout(200)

            assert [first.method, second.method, third.method] == ["mouse", "mouse", "mouse"]
            assert await frame.evaluate("() => window.__log") == ["toggle:Электроника", "submit:phones"]
    run(scenario())
