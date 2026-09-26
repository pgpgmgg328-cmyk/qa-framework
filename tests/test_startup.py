"""Запуск браузера: недоступный TARGET_URL, сбой при старте, подсказки об ошибках."""

from __future__ import annotations

import pytest

from browser_controller import BrowserController
from main import startup_hint
from tests.helpers import install_routes, main_url, run


def test_unreachable_target_url_does_not_abort_start(monkeypatch):
    # как ERR_NAME_NOT_RESOLVED у пользователя: адрес не открывается
    monkeypatch.setattr("browser_controller.TARGET_URL", "https://no-such-host.test/")

    async def scenario():
        controller = BrowserController(on_context=install_routes)
        async with controller:
            assert not controller.is_closed()
            # окно живо — нужный адрес можно открыть вручную, и агент найдёт задание.
            # Пауза как у человека: Chromium сначала сам открывает свою страницу ошибки
            await controller.page.wait_for_timeout(1000)
            await controller.page.goto(main_url("task_tree"))
            frame = None
            for _ in range(50):
                frame = await controller.find_target_frame()
                if frame is not None:
                    break
                await controller.page.wait_for_timeout(100)
            assert frame is not None
    run(scenario())


def test_failed_start_closes_browser():
    async def broken_hook(_context):
        raise RuntimeError("сбой при запуске")

    async def scenario():
        controller = BrowserController(on_context=broken_hook)
        with pytest.raises(RuntimeError):
            async with controller:
                pass
        assert controller.is_closed()      # браузер и драйвер не остались висеть
    run(scenario())


def test_startup_hints_for_known_errors():
    missing = Exception("BrowserType.launch: Executable doesn't exist at C:\\ms-playwright\\chromium")
    assert "playwright install chromium" in startup_hint(missing)
    busy = Exception("BrowserType.launch_persistent_context: Failed to create a ProcessSingleton")
    assert "уже открыт" in startup_hint(busy)
    assert startup_hint(Exception("что-то совсем другое")) is None
