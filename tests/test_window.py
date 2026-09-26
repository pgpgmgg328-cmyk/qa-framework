"""Свёрнутое окно: сторож окна (логика без WinAPI) и клики, когда страница не отрисовывается."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Optional

from browser_controller import BrowserController
from dom_parser import DomParser
from tests.helpers import install_routes, run, workspace_url
from window_guard import (
    OFFSCREEN_LEFT,
    SW_SHOWMAXIMIZED,
    SW_SHOWNOACTIVATE,
    SW_SHOWNORMAL,
    WPF_RESTORETOMAXIMIZED,
    Placement,
    WindowGuard,
)

HWND = 4242
MAXIMIZED_NORMAL_RECT = (200, 100, 1000, 700)       # «обычный» размер развёрнутого окна


class FakeWindows:
    """Подмена user32: одно окно браузера, человек сворачивает и выбирает его."""

    def __init__(self, *, maximized: bool) -> None:
        self.alive = True
        self.minimized = False
        self.front = 0
        self.placement = Placement(
            flags=WPF_RESTORETOMAXIMIZED if maximized else 0, show=SW_SHOWMAXIMIZED if maximized else SW_SHOWNORMAL,
            min_pos=(-1, -1), max_pos=(-1, -1), normal=MAXIMIZED_NORMAL_RECT)
        self.calls: list[Placement] = []

    # --- человек
    def user_minimizes(self) -> None:
        self.minimized = True
        self.front = 777                                  # фокус ушёл в другую программу
        self.placement = replace(self.placement, show=2)  # SW_SHOWMINIMIZED

    def user_selects_window(self) -> None:
        self.front = HWND

    # --- API
    def find_window(self, title_part: str) -> Optional[int]:
        return HWND

    def is_window(self, hwnd: int) -> bool:
        return self.alive

    def is_minimized(self, hwnd: int) -> bool:
        return self.minimized

    def foreground(self) -> int:
        return self.front

    def get_placement(self, hwnd: int) -> Placement:
        return self.placement

    def set_placement(self, hwnd: int, placement: Placement) -> None:
        self.calls.append(placement)
        self.placement = placement
        self.minimized = False
        if placement.show != SW_SHOWNOACTIVATE:
            self.front = hwnd

    def work_area(self, hwnd: int) -> tuple[int, int, int, int]:
        return (0, 0, 1920, 1040)


def test_minimized_window_goes_offscreen_and_comes_back_maximized():
    win = FakeWindows(maximized=True)
    guard = WindowGuard(win, HWND)
    assert guard.tick() is None and win.calls == []

    win.user_minimizes()
    assert guard.tick() == "hidden"
    hidden = win.calls[-1]
    assert hidden.show == SW_SHOWNOACTIVATE and win.front == 777       # фокус не перехвачен
    left, top, right, bottom = hidden.normal
    assert left == OFFSCREEN_LEFT and (right - left, bottom - top) == (1920, 1040)   # размер как на экране
    assert not win.minimized                                             # окно отрисовывается
    assert guard.tick() is None and len(win.calls) == 1                  # человек в другой программе

    win.user_selects_window()                                            # кнопка на панели задач
    assert guard.tick() == "restored"
    back = win.calls[-1]
    assert back.show == SW_SHOWMAXIMIZED and back.normal == MAXIMIZED_NORMAL_RECT
    assert guard.tick() is None and not guard.hidden

    win.alive = False
    assert guard.tick() == "gone"


def test_normal_window_keeps_its_size_and_second_minimize_stays_hidden():
    win = FakeWindows(maximized=False)
    guard = WindowGuard(win, HWND)
    win.user_minimizes()
    assert guard.tick() == "hidden"
    left, top, right, bottom = win.calls[-1].normal
    assert (right - left, bottom - top) == (800, 600)
    # окно, убранное за край, стало активным и его свернули кнопкой на панели — снова за край,
    # а сохранённое «как было» не затирается
    win.front = HWND
    win.minimized = True
    assert guard.tick() is None and win.calls[-1].show == SW_SHOWNOACTIVATE and guard.hidden
    win.front = HWND
    assert guard.tick() == "restored"
    assert win.calls[-1].show == SW_SHOWNORMAL and win.calls[-1].normal == MAXIMIZED_NORMAL_RECT
    # агент завершает работу, пока окно за краем — возвращает его
    win.user_minimizes()
    guard.tick()
    guard.restore()
    assert not guard.hidden and win.calls[-1].show == SW_SHOWNORMAL


def test_click_without_rendering_goes_straight_to_js(monkeypatch):
    """Свёрнутое окно: проверки Playwright ждали бы кадр отрисовки ~5 с — клик сразу из JS."""

    async def not_rendering(locator, timeout_ms=250):
        return False

    monkeypatch.setattr(BrowserController, "_locator_rendering", staticmethod(not_rendering))
    monkeypatch.setattr("browser_controller.TARGET_URL", workspace_url("quiz"))

    async def scenario():
        async with BrowserController(on_context=install_routes) as browser:
            frame = None
            for _ in range(50):
                frame = await browser.find_target_frame()
                if frame is not None:
                    break
                await browser.page.wait_for_timeout(100)
            state = await DomParser(frame).parse(quiet=True)
            yes = next(e for e in state.visible_elements if e.text == "Да")
            started = time.monotonic()
            outcome = await browser.click_element(frame, yes)
            elapsed = time.monotonic() - started
            after = await DomParser(frame).parse(quiet=True)
            return outcome, elapsed, after.by_key(yes.key)

    outcome, elapsed, yes = run(scenario())
    assert outcome.ok and outcome.method == "js" and "не отрисовывается" in outcome.detail
    assert elapsed < 1.0 and yes.is_selected


def test_controller_finds_window_by_title_and_guards_it(monkeypatch):
    """Контроллер находит окно по временной метке в заголовке (заголовок страницы потом
    возвращается) и следит за ним: свернули — за край экрана, выбрали — на место."""
    import asyncio
    import os

    win = FakeWindows(maximized=True)
    markers: list[str] = []
    real_find = win.find_window

    def find_window(title_part: str):
        markers.append(title_part)
        return real_find(title_part)

    win.find_window = find_window
    monkeypatch.setattr("browser_controller.win_api", lambda: win)
    monkeypatch.setattr("browser_controller.TARGET_URL", workspace_url("quiz"))

    async def scenario():
        async with BrowserController(on_context=install_routes) as browser:
            title = await browser.page.title()
            watcher = asyncio.create_task(browser._watch_window())
            for _ in range(40):
                if browser._guard is not None:
                    break
                await asyncio.sleep(0.1)
            restored_title = await browser.page.title()
            win.user_minimizes()
            await asyncio.sleep(0.8)
            hidden = win.calls[-1] if win.calls else None
            win.user_selects_window()
            await asyncio.sleep(0.8)
            back = win.calls[-1]
            watcher.cancel()
            return title, restored_title, hidden, back

    title, restored_title, hidden, back = run(scenario())
    assert markers and markers[0] == f"twork-agent-{os.getpid()}" and restored_title == title
    assert hidden is not None and hidden.show == SW_SHOWNOACTIVATE
    assert back.show == SW_SHOWMAXIMIZED and not win.minimized
