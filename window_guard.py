"""window_guard.py — свёрнутое окно браузера не тормозит агента (Windows).

Свёрнутое окно Chrome не отрисовывает страницу: анимации сайта (появление и закрытие окон,
списков) не завершаются, проверки перед кликом ждут кадра отрисовки и падают по таймауту.
Агент в таком окне работает в разы медленнее и может застрять на окне «Тренировка».

Окно, которое закрыто другими окнами или стоит за краем экрана, работает на полной скорости
(Playwright запускает Chrome с --disable-backgrounding-occluded-windows). Поэтому сторож:
- заметил, что окно браузера свернули, — разворачивает его за край экрана без перехвата
  фокуса (SW_SHOWNOACTIVATE): для человека окно просто исчезло, как при сворачивании;
- заметил, что человек снова выбрал окно (кнопка на панели задач, Alt+Tab), — возвращает
  его на прежнее место и в прежнем виде (развёрнутым, если оно было развёрнуто).

Только Windows (вызовы user32 через ctypes); на других системах агент лишь предупреждает
в логе, что окно свёрнуто. Логика отделена от WinAPI — её проверяют тесты с подменой API.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, replace
from typing import Optional, Protocol

logger = logging.getLogger("twork.window")

SW_SHOWNORMAL = 1
SW_SHOWMAXIMIZED = 3
SW_SHOWNOACTIVATE = 4
WPF_RESTORETOMAXIMIZED = 0x0002
# запрос уходит в очередь окна браузера — сторож не ждёт его UI-поток
WPF_ASYNCWINDOWPLACEMENT = 0x0004
# За краем любого монитора (координаты рабочей области; Windows сама кладёт свёрнутые окна в -32000)
OFFSCREEN_LEFT, OFFSCREEN_TOP = -30000, 0


@dataclass(frozen=True)
class Placement:
    """WINDOWPLACEMENT: состояние окна и его «обычный» прямоугольник (left, top, right, bottom)."""
    flags: int
    show: int
    min_pos: tuple[int, int]
    max_pos: tuple[int, int]
    normal: tuple[int, int, int, int]


class WindowApi(Protocol):
    def find_window(self, title_part: str) -> Optional[int]: ...
    def is_window(self, hwnd: int) -> bool: ...
    def is_minimized(self, hwnd: int) -> bool: ...
    def foreground(self) -> int: ...
    def get_placement(self, hwnd: int) -> Placement: ...
    def set_placement(self, hwnd: int, placement: Placement) -> None: ...
    def work_area(self, hwnd: int) -> tuple[int, int, int, int]: ...


class WindowGuard:
    """Один вызов tick() — одна проверка окна (агент вызывает её раз в полсекунды)."""

    def __init__(self, api: WindowApi, hwnd: int) -> None:
        self._api = api
        self.hwnd = hwnd
        self._saved: Optional[Placement] = None     # как окно стояло до «сворачивания»

    @property
    def hidden(self) -> bool:
        return self._saved is not None

    def tick(self) -> Optional[str]:
        """«hidden» — окно убрано за край экрана, «restored» — возвращено, «gone» — окна
        больше нет (браузер закрыт), None — ничего не изменилось."""
        api, hwnd = self._api, self.hwnd
        if not api.is_window(hwnd):
            return "gone"
        if self._saved is not None:
            if api.is_minimized(hwnd):
                self._move_offscreen(self._saved)        # свернули снова (кнопкой на панели задач)
                return None
            if api.foreground() == hwnd:                 # человек выбрал окно — вернуть на место
                saved, self._saved = self._saved, None
                api.set_placement(hwnd, _shown(saved))
                return "restored"
            return None
        if api.is_minimized(hwnd):
            self._saved = api.get_placement(hwnd)
            self._move_offscreen(self._saved)
            return "hidden"
        return None

    def restore(self) -> None:
        """Вернуть окно на экран (агент завершает работу — окно не должно остаться за краем)."""
        if self._saved is not None and self._api.is_window(self.hwnd):
            saved, self._saved = self._saved, None
            self._api.set_placement(self.hwnd, _shown(saved))

    def _move_offscreen(self, saved: Placement) -> None:
        # размер — как был на экране: у развёрнутого окна это рабочая область монитора
        if saved.flags & WPF_RESTORETOMAXIMIZED:
            left, top, right, bottom = self._api.work_area(self.hwnd)
        else:
            left, top, right, bottom = saved.normal
        width, height = max(right - left, 400), max(bottom - top, 300)
        self._api.set_placement(self.hwnd, Placement(
            flags=WPF_ASYNCWINDOWPLACEMENT, show=SW_SHOWNOACTIVATE, min_pos=saved.min_pos, max_pos=saved.max_pos,
            normal=(OFFSCREEN_LEFT, OFFSCREEN_TOP, OFFSCREEN_LEFT + width, OFFSCREEN_TOP + height),
        ))


def _shown(saved: Placement) -> Placement:
    """Как окно стояло до сворачивания: развёрнутым или в обычном прямоугольнике."""
    show = SW_SHOWMAXIMIZED if saved.flags & WPF_RESTORETOMAXIMIZED else SW_SHOWNORMAL
    return replace(saved, flags=WPF_ASYNCWINDOWPLACEMENT, show=show)


def win_api() -> Optional[WindowApi]:
    """WinAPI-реализация (только Windows)."""
    if sys.platform != "win32":
        return None
    try:
        return _User32()
    except (OSError, AttributeError) as exc:
        logger.debug("user32 недоступен: %s", exc)
        return None


class _User32:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ct, self._wt = ctypes, wintypes
        self._u = ctypes.WinDLL("user32", use_last_error=True)

        class WINDOWPLACEMENT(ctypes.Structure):
            _fields_ = [("length", wintypes.UINT), ("flags", wintypes.UINT), ("showCmd", wintypes.UINT),
                        ("ptMinPosition", wintypes.POINT), ("ptMaxPosition", wintypes.POINT),
                        ("rcNormalPosition", wintypes.RECT)]

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        self._WP, self._MI = WINDOWPLACEMENT, MONITORINFO
        u = self._u
        u.GetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)]
        u.SetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(WINDOWPLACEMENT)]
        u.IsWindow.argtypes = [wintypes.HWND]
        u.IsIconic.argtypes = [wintypes.HWND]
        u.GetForegroundWindow.restype = wintypes.HWND
        u.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        u.MonitorFromWindow.restype = wintypes.HANDLE
        u.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFO)]
        u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.IsWindowVisible.argtypes = [wintypes.HWND]

    def find_window(self, title_part: str) -> Optional[int]:
        ct, wt, u = self._ct, self._wt, self._u
        found: list[int] = []

        @ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
        def callback(hwnd, _lparam):
            length = u.GetWindowTextLengthW(hwnd)
            if length and u.IsWindowVisible(hwnd):
                buf = ct.create_unicode_buffer(length + 1)
                u.GetWindowTextW(hwnd, buf, length + 1)
                if title_part in buf.value:
                    found.append(int(hwnd))
                    return False
            return True

        u.EnumWindows(callback, 0)
        return found[0] if found else None

    def is_window(self, hwnd: int) -> bool:
        return bool(self._u.IsWindow(hwnd))

    def is_minimized(self, hwnd: int) -> bool:
        return bool(self._u.IsIconic(hwnd))

    def foreground(self) -> int:
        return int(self._u.GetForegroundWindow() or 0)

    def get_placement(self, hwnd: int) -> Placement:
        wp = self._WP()
        wp.length = self._ct.sizeof(self._WP)
        if not self._u.GetWindowPlacement(hwnd, self._ct.byref(wp)):
            raise OSError(self._ct.get_last_error(), "GetWindowPlacement")
        r = wp.rcNormalPosition
        return Placement(flags=wp.flags, show=wp.showCmd,
                         min_pos=(wp.ptMinPosition.x, wp.ptMinPosition.y),
                         max_pos=(wp.ptMaxPosition.x, wp.ptMaxPosition.y),
                         normal=(r.left, r.top, r.right, r.bottom))

    def set_placement(self, hwnd: int, placement: Placement) -> None:
        wt = self._wt
        wp = self._WP()
        wp.length = self._ct.sizeof(self._WP)
        wp.flags, wp.showCmd = placement.flags, placement.show
        wp.ptMinPosition = wt.POINT(*placement.min_pos)
        wp.ptMaxPosition = wt.POINT(*placement.max_pos)
        wp.rcNormalPosition = wt.RECT(*placement.normal)
        if not self._u.SetWindowPlacement(hwnd, self._ct.byref(wp)):
            raise OSError(self._ct.get_last_error(), "SetWindowPlacement")

    def work_area(self, hwnd: int) -> tuple[int, int, int, int]:
        monitor = self._u.MonitorFromWindow(hwnd, 2)          # MONITOR_DEFAULTTONEAREST
        info = self._MI()
        info.cbSize = self._ct.sizeof(self._MI)
        if not monitor or not self._u.GetMonitorInfoW(monitor, self._ct.byref(info)):
            return (0, 0, 1280, 800)
        r = info.rcWork
        return (r.left, r.top, r.right, r.bottom)
