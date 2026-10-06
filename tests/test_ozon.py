"""Ozon Profit (PLATFORM=ozon): задание в главном фрейме task.ozon.ru, окно инструкции без
role="dialog", несколько блоков с radio и одна «Отправить», экран «закончились задачи».
Страницы — копия структуры живого сайта с вымышленными товарами (tests/fixtures/ozon_*.html)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pytest
from playwright.async_api import async_playwright

import agent as agent_module
from adapters import get_platform
from adapters.ozon_profit import OzonProfitAdapter
from adapters.twork import TWorkAdapter
from agent import Agent, StepResult, _url_tail
from browser_controller import BrowserController
from dom_parser import DomParser, render_page
from knowledge import KnowledgeBase
from models import ActionType, DecisionContext, ElementKind, LLMDecision, PageState, ParsedElement, PlannedAction
from tests.helpers import (OZON_LOGIN_URL, OZON_PROJECTS_URL, OZON_TASK_URL, launch_options, ozon_routes, run,
                           use_platform)

# ---------------------------------------------------------------------------
# Помощники
# ---------------------------------------------------------------------------


@asynccontextmanager
async def ozon_page(url: str, answers: Optional[list] = None):
    async with async_playwright() as p:
        browser = await p.chromium.launch(**launch_options())
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        await ozon_routes(answers if answers is not None else [])(context)
        page = await context.new_page()
        await page.goto(url)
        try:
            yield page
        finally:
            await browser.close()


def options(state: PageState, text: str) -> list[ParsedElement]:
    return [e for e in state.visible_elements if e.kind == ElementKind.OPTION and e.text == text]


def click(el: ParsedElement) -> PlannedAction:
    return PlannedAction(action=ActionType.CLICK, target_index=el.index, target_text=el.text)


def answer(*clicks: PlannedAction, state: PageState) -> LLMDecision:
    """Пакет: выбрать варианты и отправить — как ответ модели с несколькими actions."""
    send = next(e for e in state.visible_elements if e.text == "Отправить")
    first, rest = clicks[0], list(clicks[1:])
    return LLMDecision(reasoning="сценарий", action=first.action, target_index=first.target_index,
                       target_text=first.target_text,
                       next_actions=rest + [PlannedAction(action=ActionType.SUBMIT, target_index=send.index)])


class ScriptedLLM:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.calls: list[tuple[PageState, DecisionContext]] = []

    async def decide(self, state: PageState, context: DecisionContext) -> LLMDecision:
        self.calls.append((state, context))
        return self.policy(state, context)


def make_agent(tmp_path, policy, answers: list) -> tuple[Agent, ScriptedLLM]:
    llm = ScriptedLLM(policy)
    browser = BrowserController(on_context=ozon_routes(answers))
    return Agent(browser=browser, llm=llm, knowledge=KnowledgeBase(str(tmp_path / "knowledge"))), llm


# ---------------------------------------------------------------------------
# Настройки площадки
# ---------------------------------------------------------------------------


def test_platform_is_chosen_by_key_and_twork_stays_default():
    assert get_platform("") is TWorkAdapter and get_platform("OZON") is OzonProfitAdapter
    with pytest.raises(ValueError, match="twork | ozon"):
        get_platform("avito")
    s = OzonProfitAdapter.settings
    assert s["ALLOW_MAIN_FRAME"] is True and s["FRAME_KEYWORDS"] == ()
    assert s["TARGET_URL"] == "https://task.ozon.ru"


def _config_in_subprocess(**env: str) -> dict:
    """config.py читает .env один раз при импорте — проверяем его в отдельном процессе."""
    code = ("import json, config; print(json.dumps({k: getattr(config, k) for k in ('PLATFORM_NAME', "
            "'TARGET_URL', 'FRAME_KEYWORDS', 'ALLOW_MAIN_FRAME', 'TASK_URL_KEYWORDS', 'FINISH_BUTTON_TEXTS', "
            "'DIALOG_SELECTORS', 'PAGE_SKIP_SELECTORS', 'POOL_URL_RE')}, ensure_ascii=False))")
    out = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                         env={**os.environ, **env}, capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_platform_settings_override_values_copied_from_twork_env_example():
    """В .env, скопированном из .env.example, есть TARGET_URL и FRAME_KEYWORDS для T-Work: с
    PLATFORM=ozon берутся значения Ozon; без PLATFORM — всё как раньше."""
    copied = {"TARGET_URL": "https://twork.tbank.ru", "FRAME_KEYWORDS": "klecks-operator,task"}
    ozon = _config_in_subprocess(PLATFORM="ozon", **copied)
    assert ozon["TARGET_URL"] == "https://task.ozon.ru" and ozon["FRAME_KEYWORDS"] == []
    assert ozon["ALLOW_MAIN_FRAME"] is True and ozon["TASK_URL_KEYWORDS"] == ["/task/"]
    assert ozon["FINISH_BUTTON_TEXTS"] == ["отправить"] and ozon["POOL_URL_RE"]
    assert '[data-testid="ChatUrlButton"]' in ozon["PAGE_SKIP_SELECTORS"]      # регистр сохранён
    twork = _config_in_subprocess(PLATFORM="", **copied)
    assert twork["PLATFORM_NAME"] == "twork" and twork["TARGET_URL"] == "https://twork.tbank.ru"
    assert twork["FRAME_KEYWORDS"] == ["klecks-operator", "task"] and twork["ALLOW_MAIN_FRAME"] is False
    assert twork["DIALOG_SELECTORS"] == [] and twork["PAGE_SKIP_SELECTORS"] == [] and twork["POOL_URL_RE"] == ""
    # своё значение в .env сильнее площадки
    own = _config_in_subprocess(PLATFORM="ozon", TARGET_URL="https://task.ozon.ru/?activeOnly=true")
    assert own["TARGET_URL"] == "https://task.ozon.ru/?activeOnly=true"


def test_task_url_is_checked_without_domain(monkeypatch):
    """«/task» в домене task.ozon.ru — не страница задания: иначе список проектов не узнаётся."""
    assert _url_tail("https://task.ozon.ru/?sortBy=X") == "/?sortby=x"
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL)
    assert not agent_module._is_task_url("https://task.ozon.ru/")
    assert not agent_module._is_task_url("https://profit.ozon.ru/cabinet/tasks")
    assert agent_module._is_task_url("https://task.ozon.ru/task/bd75")
    # T-Work: как раньше
    monkeypatch.setattr(agent_module, "TASK_URL_KEYWORDS", ("/task",))
    assert agent_module._is_task_url("https://klecks-operator.tbank.ru/klecks/task?scenario=1")
    assert not agent_module._is_task_url("https://klecks-operator.tbank.ru/klecks/orders.html")


# ---------------------------------------------------------------------------
# Снимок страницы
# ---------------------------------------------------------------------------


def test_task_page_snapshot_hides_site_header_and_reads_instruction_window(monkeypatch):
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL)

    async def scenario():
        async with ozon_page(OZON_TASK_URL) as page:
            await page.wait_for_function("() => !!document.querySelector('[data-testid=InstructionPopup]')")
            with_popup = await DomParser(page.main_frame).parse()
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(200)
            plain = await DomParser(page.main_frame).parse()
            return with_popup, plain

    with_popup, plain = run(scenario())
    # окно инструкции без role="dialog" — диалог; крестик без текста — «Закрыть»
    assert with_popup.dialog_open
    in_dialog = [e for e in with_popup.visible_elements if e.container == "dialog"]
    assert any(e.text == "Закрыть" and e.kind == ElementKind.BUTTON for e in in_dialog)
    assert "Нет нужного варианта" in "\n".join(with_popup.dialog_lines)

    assert not plain.dialog_open
    text, shown = render_page(plain)
    labels = {e.text for e in shown}
    # шапка сайта, «Все задания», «Чат с заказчиком» модели не видны
    for hidden in ("Статистика", "Обучение", "Ozon ID", "Чат с заказчиком", "Вcе задания"):
        assert hidden not in labels and hidden not in text
    assert "Инструкция" in labels and "Отправить" in labels
    assert plain.pool_title == "Выбор одинаковых названий товаров"
    # варианты — строками под своим товаром, у каждого блока своя группа
    lines = text.splitlines()
    jam = lines.index("#### Джем Ягодная Поляна Клубничный дой-пак 400 г")
    juice = lines.index("#### Сок Солнечный Сад томат с солью 1 л")
    assert jam < juice
    assert any("«Джем Ягодная Поляна клубничный 400г»" in line for line in lines[jam:juice])
    assert len(options(plain, "Нет нужного варианта")) == 2
    assert all(e.choice_type == "radio" for e in options(plain, "Нет нужного варианта"))


def test_projects_list_is_orders_list_and_done_screen_ends_order(monkeypatch):
    use_platform(monkeypatch, OzonProfitAdapter, OZON_PROJECTS_URL)

    async def scenario():
        async with ozon_page(OZON_PROJECTS_URL) as page:
            projects = await DomParser(page.main_frame).parse()
            await page.goto(OZON_TASK_URL + "?instruction=0")
            await page.wait_for_selector("[data-testid=TasksSendButton]")
            task = await DomParser(page.main_frame).parse()
            for _ in range(2):        # две страницы заданий без ответов → «закончились задачи»
                await page.click("[data-testid=TasksSendButton]")
                await page.wait_for_timeout(600)
            done = await DomParser(page.main_frame).parse()
            return projects, task, done

    projects, task, done = run(scenario())
    assert Agent._is_orders_list(projects) and not Agent._is_order_done(projects)
    assert not Agent._is_orders_list(task) and not Agent._is_order_done(task)
    assert "Вы молодец!" in done.task_text and Agent._is_order_done(done)


# ---------------------------------------------------------------------------
# Агент целиком
# ---------------------------------------------------------------------------


def test_agent_reads_instruction_solves_pages_and_stops_when_tasks_end(tmp_path, monkeypatch):
    """Инструкция открылась сама — агент прочитал её в базу знаний и закрыл; на первой странице
    два блока (выбор названия), на второй — сравнение пары; ответы ушли одной «Отправить» на
    страницу; «закончились задачи» — заказ выполнен."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL)
    answers: list = []

    def policy(state: PageState, ctx: DecisionContext) -> LLMDecision:
        if "Джем Ягодная Поляна" in state.task_text:
            jam = options(state, "Джем Ягодная Поляна клубничный 400г")[0]
            no_juice = options(state, "Нет нужного варианта")[-1]
            return answer(click(jam), click(no_juice), state=state)
        return answer(click(options(state, "Да")[0]), state=state)

    async def scenario():
        agent, llm = make_agent(tmp_path, policy, answers)
        await asyncio.wait_for(agent.run(), timeout=90)
        return agent, llm

    agent, llm = run(scenario())
    assert answers == [["plu:1001", "__no_match__"], ["1"]]
    assert agent._tasks_done == 2 and len(llm.calls) == 2
    # модель решала без окна поверх страницы, с инструкцией в знаниях
    assert all(not state.dialog_open for state, _ in llm.calls)
    assert "Нет нужного варианта" in llm.calls[0][1].knowledge
    text, _ = render_page(llm.calls[0][0])
    assert "Чат с заказчиком" not in text and "Статистика" not in text
    # вид задания — проект /task/<id>: товары на страницах разные, а инструкция прочитана один раз
    # и пришла модели и на второй странице
    assert llm.calls[0][0].pool_key == llm.calls[1][0].pool_key
    assert "Нет нужного варианта" in llm.calls[1][1].knowledge
    files = list((tmp_path / "knowledge").glob("*.md"))
    assert len(files) == 1 and "Сравнение названий товаров" in files[0].read_text(encoding="utf-8")


def test_option_under_fixed_send_bar_is_clicked(tmp_path, monkeypatch):
    """Панель «Отправить» закреплена внизу экрана и перекрывает нижние варианты (на сайте — до
    10 товаров на странице): клик всё равно выбирает нужный вариант."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL + "?instruction=0")

    async def scenario():
        agent, _ = make_agent(tmp_path, lambda *a: LLMDecision.skip("-"), [])
        async with agent._browser:
            frame = agent._browser.page.main_frame
            await frame.wait_for_selector("[data-testid=TasksSendButton]")
            state = await DomParser(frame).parse()
            covered = next(e for e in state.visible_elements if e.kind == ElementKind.OPTION and e.occluded)
            outcome = await agent._browser.click_element(frame, covered)
            after = await DomParser(frame).parse()
            return covered, outcome, after

    covered, outcome, after = run(scenario())
    assert outcome.ok
    chosen = [e.text for e in after.visible_elements if e.is_selected]
    assert chosen == [covered.text]


def test_agent_does_not_touch_login_or_projects_list(tmp_path, monkeypatch):
    """Вход в Ozon ID и список проектов: агент ждёт человека, модель не зовёт, поле телефона пустое."""
    monkeypatch.setattr(agent_module, "FRAME_LOAD_WAIT", 0.1)

    async def steps(url: str) -> tuple[list, ScriptedLLM, str]:
        use_platform(monkeypatch, OzonProfitAdapter, url)
        agent, llm = make_agent(tmp_path, lambda *a: LLMDecision.skip("-"), [])
        async with agent._browser:
            page = agent._browser.page
            await page.wait_for_load_state("load")
            results = [await agent._step() for _ in range(3)]
            phone = await page.evaluate("() => (document.getElementById('phone') || {}).value || ''")
        return results, llm, phone

    for url in (OZON_LOGIN_URL, OZON_PROJECTS_URL):
        results, llm, phone = run(steps(url))
        assert results == [StepResult.WAITING] * 3 and llm.calls == [] and phone == ""
