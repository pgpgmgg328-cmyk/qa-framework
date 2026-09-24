"""Помощники тестов: раздача фикстур через page.route без HTTP-сервера.

Главная страница — https://t-work.test/index.html?task=<имя>, внутри неё
кросс-доменный iframe https://klecks-operator.test/task/<имя>.html — как на T-Work.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Coroutine
from urllib.parse import parse_qs, urlsplit

from playwright.async_api import BrowserContext, Route

FIXTURES = Path(__file__).resolve().parent / "fixtures"

WRAPPER = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>T-Work</title></head>
<body style="margin:0"><header style="height:60px">T-Work · шапка сайта</header>
<iframe src="{src}" style="width:1000px;height:700px;border:0"></iframe></body></html>"""


def main_url(task: str) -> str:
    return f"https://t-work.test/index.html?task={task}"


def fixture_url(task: str) -> str:
    return f"https://klecks-operator.test/task/{task}.html"


async def install_routes(context: BrowserContext) -> None:
    async def handle(route: Route) -> None:
        url = urlsplit(route.request.url)
        if url.hostname == "t-work.test":
            task = parse_qs(url.query).get("task", ["task_tree"])[0]
            await route.fulfill(status=200, content_type="text/html; charset=utf-8",
                                body=WRAPPER.format(src=fixture_url(task)))
        elif url.hostname == "klecks-operator.test":
            body = (FIXTURES / Path(url.path).name).read_text(encoding="utf-8")
            await route.fulfill(status=200, content_type="text/html; charset=utf-8", body=body)
        else:
            await route.abort()

    await context.route("**/*", handle)


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


RouteInstaller = Callable[[BrowserContext], Awaitable[None]]
