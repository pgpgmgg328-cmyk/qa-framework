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
from agent import Agent, StepResult, TaskIdentity, _url_tail
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
    # адрес T-Work в любом виде: с «/», из адресной строки, прежний t-work.ru
    for old in ("https://twork.tbank.ru/", "https://twork.tbank.ru/orders", "https://t-work.ru"):
        assert _config_in_subprocess(PLATFORM="ozon", TARGET_URL=old)["TARGET_URL"] == "https://task.ozon.ru"
        assert _config_in_subprocess(PLATFORM="", TARGET_URL=old)["TARGET_URL"] == old


def test_main_frame_switched_off_on_ozon_is_a_startup_error():
    """ALLOW_MAIN_FRAME=false (строка из .env.example) на Ozon: задание агент не нашёл бы никогда —
    не тихое ожидание, а понятная ошибка при запуске."""
    code = "import config; config.validate_config(require_llm=False)"
    root = Path(__file__).resolve().parents[1]
    bad = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, encoding="utf-8",
                         env={**os.environ, "PLATFORM": "ozon", "ALLOW_MAIN_FRAME": "false"})
    assert bad.returncode != 0 and "ALLOW_MAIN_FRAME" in bad.stderr
    ok = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, encoding="utf-8",
                        env={**os.environ, "PLATFORM": "", "ALLOW_MAIN_FRAME": "false"})
    assert ok.returncode == 0, ok.stderr


def test_task_url_is_checked_without_domain(monkeypatch):
    """«/task» в домене task.ozon.ru — не страница задания: иначе список проектов не узнаётся."""
    assert _url_tail("https://task.ozon.ru/?sortBy=X") == "/"
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL)
    assert not agent_module._is_task_url("https://task.ozon.ru/")
    # адрес возврата на задание в параметрах страницы входа — не страница задания
    assert not agent_module._is_task_url("https://sso.ozon.ru/auth/ozonid?redirect=https://task.ozon.ru/task/bd75")
    assert not agent_module._is_task_url("https://sso.ozon.ru/auth#https://task.ozon.ru/task/bd75")
    # инструкция проекта, открытая во вкладке
    assert not agent_module._is_task_url("https://task.ozon.ru/task/bd75/instruction")
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


def test_bad_selector_from_env_is_dropped_alone(monkeypatch):
    """Селектор, который браузер не понимает (синтаксис Playwright :has-text), отбрасывается один —
    шапка по-прежнему скрыта, окно инструкции по-прежнему диалог."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL)
    import dom_parser
    monkeypatch.setattr(dom_parser, "PAGE_SKIP_SELECTORS",
                        OzonProfitAdapter.settings["PAGE_SKIP_SELECTORS"] + ('button:has-text("Обучение")',))
    monkeypatch.setattr(dom_parser, "DIALOG_SELECTORS",
                        OzonProfitAdapter.settings["DIALOG_SELECTORS"] + ('div:has-text("Инструкция")',))

    async def scenario():
        async with ozon_page(OZON_TASK_URL) as page:
            await page.wait_for_function("() => !!document.querySelector('[data-testid=InstructionPopup]')")
            return await DomParser(page.main_frame).parse()

    state = run(scenario())
    assert state.dialog_open
    labels = {e.text for e in state.visible_elements}
    assert not labels & {"Обучение", "Статистика", "Чат с заказчиком"}


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

    login_back = OZON_LOGIN_URL + "?redirect=https://task.ozon.test/task/demo"
    for url in (OZON_LOGIN_URL, login_back, OZON_PROJECTS_URL):
        results, llm, phone = run(steps(url))
        assert results == [StepResult.WAITING] * 3 and llm.calls == [] and phone == "", url


def test_login_window_buttons_are_not_pressed(tmp_path, monkeypatch):
    """Вход в окне поверх страницы: «Продолжить» в нём агент не нажимает (как кнопку окна новостей)."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_LOGIN_URL + "/dialog")
    monkeypatch.setattr(agent_module, "FRAME_LOAD_WAIT", 0.1)

    async def scenario():
        agent, llm = make_agent(tmp_path, lambda *a: LLMDecision.skip("-"), [])
        async with agent._browser:
            page = agent._browser.page
            await page.wait_for_load_state("load")
            results = [await agent._step() for _ in range(3)]
            sent = await page.evaluate("() => window.__sent || 0")
        return results, llm, sent

    results, llm, sent = run(scenario())
    assert results == [StepResult.WAITING] * 3 and llm.calls == [] and sent == 0


@pytest.mark.parametrize("variant", ["expired", "expired-code"])
def test_login_window_over_task_page_waits_for_human(tmp_path, monkeypatch, variant):
    """Сессия истекла — вход окном поверх страницы задания (телефон type=tel; код из SMS в полях
    type=text): кнопки окна не нажимаются, модель не зовётся, в поля ничего не вводится."""
    use_platform(monkeypatch, OzonProfitAdapter, f"https://task.ozon.test/task/demo/{variant}")
    monkeypatch.setattr(agent_module, "FRAME_LOAD_WAIT", 0.1)

    async def scenario():
        agent, llm = make_agent(tmp_path, lambda *a: LLMDecision.skip("-"), [])
        async with agent._browser:
            page = agent._browser.page
            await page.wait_for_load_state("load")
            results = [await agent._step() for _ in range(3)]
            sent = await page.evaluate("() => window.__sent || 0")
            typed = await page.evaluate("() => [...document.querySelectorAll('input')].map((i) => i.value).join('')")
        return results, llm, sent, typed

    results, llm, sent, typed = run(scenario())
    assert results == [StepResult.WAITING] * 3 and llm.calls == [] and sent == 0 and typed == ""


def test_task_window_about_phone_number_is_not_a_login(tmp_path, monkeypatch):
    """Окно задания «найдите номер телефона и впишите в поле» — работа для модели, а не вход."""
    use_platform(monkeypatch, OzonProfitAdapter, "https://task.ozon.test/task/demo/org-dialog")

    async def scenario():
        agent, llm = make_agent(tmp_path, lambda *a: LLMDecision.skip("-"), [])
        async with agent._browser:
            await agent._browser.page.wait_for_load_state("load")
            for _ in range(3):
                await agent._step()
                if llm.calls:
                    break
        return llm

    llm = run(scenario())
    assert llm.calls and llm.calls[0][0].dialog_open


def test_budget_exhausted_does_not_send_partial_page(tmp_path, monkeypatch):
    """Лимит шагов на задание кончился, а ответ выбран не во всех вопросах страницы: агент не
    отправляет неполный ответ (пустые блоки — неверные ответы), а просит человека."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL + "?instruction=0")
    monkeypatch.setattr(agent_module, "MAX_STEPS_PER_TASK", 2)
    monkeypatch.setattr(agent_module, "MAX_IDLE_SECONDS", 1)
    answers: list = []

    def policy(state: PageState, ctx: DecisionContext) -> LLMDecision:
        jam = options(state, "Джем Ягодная Поляна клубничный 400г")[0]
        if jam.is_selected:
            return LLMDecision.skip("ищу ответ для второго товара")
        return LLMDecision(reasoning="сценарий", action=ActionType.CLICK, target_index=jam.index, target_text=jam.text)

    async def scenario():
        agent, llm = make_agent(tmp_path, policy, answers)
        async with agent._browser:
            page = agent._browser.page
            await page.wait_for_selector("[data-testid=TasksSendButton]")
            for _ in range(6):
                await agent._step()
            state = await DomParser(page.main_frame).parse()
        return state

    state = run(scenario())
    assert answers == []
    assert Agent._unanswered_groups(state) == 1
    assert {e.group for e in state.visible_elements if e.kind == ElementKind.OPTION} == {"result_0_0", "result_0_1"}


def test_model_submit_with_unanswered_block_is_held_back(tmp_path, monkeypatch):
    """Модель отметила ответ только в первом блоке и нажала «Отправить»: агент не отправляет, а
    говорит модели, какой вопрос без ответа; ответ на все блоки — уходит."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL + "?instruction=0")
    answers: list = []

    def policy(state: PageState, ctx: DecisionContext) -> LLMDecision:
        jam = options(state, "Джем Ягодная Поляна клубничный 400г")[0]
        if not any("ответ нужен в КАЖДОМ" in n for n in ctx.notes):
            return answer(click(jam), state=state)                 # забыла второй товар
        no_juice = options(state, "Нет нужного варианта")[-1]
        return answer(click(no_juice), state=state)

    async def scenario():
        agent, llm = make_agent(tmp_path, policy, answers)
        async with agent._browser:
            await agent._browser.page.wait_for_selector("[data-testid=TasksSendButton]")
            for _ in range(6):
                await agent._step()
                if answers:
                    break
        return llm

    llm = run(scenario())
    assert answers == [["plu:1001", "__no_match__"]]
    notes = [n for _, ctx in llm.calls for n in ctx.notes if "ответ нужен в КАЖДОМ" in n]
    assert notes and "Без ответа: 1" in notes[0] and "Сок Солнечный Сад" in notes[0]


def test_rejected_answer_toast_is_a_wrong_answer(tmp_path, monkeypatch):
    """Тренировка Ozon не принимает неверный ответ: задание то же, всплывает «Ошибка — Для отправки
    ответа необходимо решить все задания». Агент понимает это как «неверно», говорит модели и не
    повторяет ответ; верный — уходит."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL + "?instruction=0&check=1")
    answers: list = []

    def policy(state: PageState, ctx: DecisionContext) -> LLMDecision:
        if "Джем Ягодная Поляна" not in state.task_text:
            return answer(click(options(state, "Да")[0]), state=state)
        no_juice = options(state, "Нет нужного варианта")[-1]
        jam = options(state, "Джем Ягодная Поляна клубничный 300г" if not ctx.feedback
                      else "Джем Ягодная Поляна клубничный 400г")[0]
        return answer(click(jam), click(no_juice), state=state)

    async def scenario():
        agent, llm = make_agent(tmp_path, policy, answers)
        await asyncio.wait_for(agent.run(), timeout=90)
        return agent, llm

    agent, llm = run(scenario())
    assert answers == [["plu:1001", "__no_match__"], ["1"]] and agent._wrong_total == 1
    fixed = next(ctx for _, ctx in llm.calls if ctx.feedback)
    assert any("НЕВЕРНЫЙ" in f and "необходимо решить все задания" in f for f in fixed.feedback)
    assert fixed.wrong_answers and "300г" in fixed.wrong_answers[0]


def test_revealed_questions_gallery_thumbnails_and_skip_button(monkeypatch):
    """«Да» открывает новые вопросы — это то же задание; миниатюры галереи (ещё не загружены) — фото
    для модели, одно фото дважды не считается; «Пропустить» модель не видит."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL)

    async def scenario():
        async with ozon_page(OZON_TASK_URL + "?instruction=0&mode=reveal") as page:
            await page.wait_for_selector("[data-testid=TasksSendButton]")
            before = await DomParser(page.main_frame).parse()
            await page.get_by_text("Да", exact=True).click()
            await page.wait_for_selector("text=Нет ошибок")
            after = await DomParser(page.main_frame).parse()
            return before, after

    before, after = run(scenario())
    assert TaskIdentity.of(before).same_task(TaskIdentity.of(after), submitted=False)
    srcs = [i.src for i in before.images]
    assert len(srcs) == 4 and sum("/lazy/thumb-" in s for s in srcs) == 3
    text, shown = render_page(before)
    assert not any("Пропустить" in e.text for e in shown) and "Отправить" in {e.text for e in shown}


def test_agent_keeps_answer_when_question_is_revealed(tmp_path, monkeypatch):
    """Ответ «Да» открыл вопрос — агент не считает это новым заданием: память и план сохраняются,
    модель отвечает на открывшийся вопрос, ответ уходит целиком."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL + "?instruction=0&mode=reveal&check=1")
    answers: list = []

    def policy(state: PageState, ctx: DecisionContext) -> LLMDecision:
        ok = options(state, "Нет ошибок")
        if not ok:
            return LLMDecision(reasoning="сценарий", action=ActionType.CLICK, plan="фото читаемое, сверю цену",
                               target_index=options(state, "Да")[0].index, target_text="Да")
        return answer(click(ok[0]), state=state)

    async def scenario():
        agent, llm = make_agent(tmp_path, policy, answers)
        await asyncio.wait_for(agent.run(), timeout=90)
        return llm

    llm = run(scenario())
    assert answers == [["yes", "ok"]]
    second = llm.calls[1][1]
    assert second.plan == "фото читаемое, сверю цену" and second.history


def test_agent_that_has_worked_takes_no_new_tabs(tmp_path, monkeypatch):
    """Агент уже решал (вкладку проекта закрыли — он вернулся в кабинет), а человек открыл проект
    (экзамен) в новой вкладке для себя: агент его не забирает."""
    use_platform(monkeypatch, OzonProfitAdapter, "https://profit.ozon.test/cabinet/")
    monkeypatch.setattr(agent_module, "FRAME_LOAD_WAIT", 0.1)
    answers: list = []

    async def scenario():
        agent, llm = make_agent(tmp_path, lambda *a: LLMDecision.skip("-"), answers)
        agent._tab_follow = False                         # как после первого задания
        async with agent._browser:
            browser = agent._browser
            cabinet = browser.page
            async with browser.context.expect_page() as info:
                await cabinet.click("#go")
            users_tab = await info.value
            await users_tab.wait_for_selector("[data-testid=TasksSendButton]")
            results = [await agent._step() for _ in range(3)]
            return results, browser.page is cabinet, llm

    results, stayed, llm = run(scenario())
    assert stayed and results == [StepResult.WAITING] * 3 and llm.calls == [] and answers == []


def test_project_in_users_own_tab_is_not_taken(tmp_path, monkeypatch):
    """Агент ждёт на списке проектов, а человек в другой вкладке сам решает проект (экзамен): агент
    его вкладку не забирает и ответов не отправляет."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_PROJECTS_URL)
    monkeypatch.setattr(agent_module, "FRAME_LOAD_WAIT", 0.1)
    answers: list = []

    async def scenario():
        agent, llm = make_agent(tmp_path, lambda *a: LLMDecision.skip("-"), answers)
        async with agent._browser:
            browser = agent._browser
            agents_tab = browser.page
            users_tab = await browser.context.new_page()
            await users_tab.goto(OZON_TASK_URL + "?instruction=0")
            await users_tab.wait_for_selector("[data-testid=TasksSendButton]")
            results = [await agent._step() for _ in range(3)]
            return results, browser.page is agents_tab, llm

    results, stayed, llm = run(scenario())
    assert stayed and results == [StepResult.WAITING] * 3 and llm.calls == [] and answers == []


def test_closing_project_tab_returns_to_cabinet_only(tmp_path, monkeypatch):
    """Вкладку проекта закрыли: агент продолжает в кабинете (другой сайт), но не во вкладке того же
    сайта заданий — там может работать человек; если остались только такие, окно для агента закрыто."""
    use_platform(monkeypatch, OzonProfitAdapter, "https://profit.ozon.test/cabinet/")

    async def scenario():
        agent, _ = make_agent(tmp_path, lambda *a: LLMDecision.skip("-"), [])
        async with agent._browser:
            browser = agent._browser
            cabinet = browser.page
            project = await browser.context.new_page()
            await project.goto(OZON_TASK_URL + "?instruction=0")
            browser.adopt_page(project)
            await project.close()
            back_to_cabinet = not browser.is_closed() and browser.page is cabinet

            users_tab = await browser.context.new_page()
            await users_tab.goto(OZON_PROJECTS_URL)
            project = await browser.context.new_page()
            await project.goto(OZON_TASK_URL + "?instruction=0")
            browser.adopt_page(project)
            await cabinet.close()
            await project.close()
            stopped = browser.is_closed()
            return back_to_cabinet, stopped

    back_to_cabinet, stopped = run(scenario())
    assert back_to_cabinet and stopped


def test_project_opened_in_new_tab_is_followed(tmp_path, monkeypatch):
    """Кабинет открывает проект в новой вкладке — агент переходит в неё и решает задание."""
    use_platform(monkeypatch, OzonProfitAdapter, "https://profit.ozon.test/cabinet/")

    async def scenario():
        agent, llm = make_agent(tmp_path, lambda *a: LLMDecision.skip("-"), [])
        async with agent._browser:
            browser = agent._browser
            async with browser.context.expect_page() as info:
                await browser.page.click("#go")                  # человек открыл проект
            tab = await info.value
            await tab.wait_for_selector("[data-testid=TasksSendButton]")
            for _ in range(4):
                await agent._step()
                if llm.calls:
                    break
            return browser.page is tab, llm

    followed, llm = run(scenario())
    assert followed and llm.calls and "Джем Ягодная Поляна" in llm.calls[0][0].task_text


NOTIFICATIONS = ("<div class='ozi__backdrop__backdrop__H2ks_'></div><div class='ozi__window__root__lcYqb'>"
                 "<div class='ozi__window__window__lcYqb' data-testid='NotificationsPopup'>"
                 "<div class='ozi-heading-400'>Уведомления</div>"
                 + "".join(f"<p>Начислено вознаграждение за проект «Проверка ценников {i}». Выплата поступит на "
                           "карту в течение трёх рабочих дней. Подробности — в разделе «Статистика».</p>"
                           for i in range(8))
                 + "</div><button type='button' class='ozi__window__closeIcon__lcYqb' onclick=\"document."
                   "getElementById('ozi-window-teleport-target').innerHTML=''\">×</button></div>")


def test_notifications_window_does_not_replace_instruction(tmp_path, monkeypatch):
    """Инструкция проекта прочитана; на следующей странице человек открыл «Уведомления» — окно Ozon
    того же вида, длинное: в базе знаний остаётся инструкция, а не уведомления."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL)

    def policy(state: PageState, ctx: DecisionContext) -> LLMDecision:
        firsts: dict[str, ParsedElement] = {}           # первый вариант в каждом блоке
        for e in state.visible_elements:
            if e.kind == ElementKind.OPTION:
                firsts.setdefault(e.group, e)
        return answer(*(click(e) for e in firsts.values()), state=state)

    async def scenario():
        agent, llm = make_agent(tmp_path, policy, [])
        async with agent._browser:
            page = agent._browser.page
            while not llm.calls:
                await agent._step()                # инструкция прочитана, первая страница отправлена
            await page.wait_for_function("() => document.body.innerText.includes('Морс Северный Бор')")
            await page.evaluate("(h) => { document.getElementById('ozi-window-teleport-target').innerHTML = h; }",
                                NOTIFICATIONS)
            for _ in range(3):
                await agent._step()
        return "\n".join(f.read_text(encoding="utf-8") for f in (tmp_path / "knowledge").glob("*.md"))

    saved = run(scenario())
    assert "Сравнение названий товаров" in saved and "Начислено вознаграждение" not in saved


def test_photo_question_waits_for_a_loaded_photo():
    """Вариант «Фото не загружается»: пока на странице нет ни одного загруженного фото (Ozon дорисовывает
    фото позже вопроса), агент не отвечает; незагруженные миниатюры при загруженном фото — не повод ждать."""
    from models import MediaImage

    option = ParsedElement(index=0, kind=ElementKind.OPTION, text="Фото не загружается")
    state = PageState(elements=[option])
    loaded = MediaImage(n=1, src="https://cdn.test/a.jpg", width=640, height=480)
    lazy_thumb = MediaImage(n=2, src="https://cdn.test/b.jpg", width=0, height=0)
    assert Agent._photo_pending(state)
    assert Agent._photo_pending(state.model_copy(update={"images": [lazy_thumb]}))
    assert not Agent._photo_pending(state.model_copy(update={"images": [loaded, lazy_thumb]}))
    other = state.model_copy(update={"elements": [ParsedElement(index=0, kind=ElementKind.OPTION, text="Да")]})
    assert not Agent._photo_pending(other)


def test_knowledge_of_ozon_project_survives_restart(tmp_path):
    """Вид задания Ozon — ключ по адресу проекта («u…»): инструкция и разборы ошибок, записанные в
    knowledge/, читаются при следующем запуске (раньше файл отбрасывался: «нет метки pool»)."""
    from knowledge import PoolKnowledge

    kb = KnowledgeBase(str(tmp_path))
    pool = PoolKnowledge(key="u1e264f4abcd", title="Выбор одинаковых названий товаров", signature=["f:да"])
    kb._pools[pool.key] = pool
    kb.save_instruction(pool, "Сравните бренд, вкус и вес. " * 20, "диалог «Инструкция»")
    kb.add_lesson(pool, "Задание «джем»: 300 г вместо 400 г — другой товар.")
    again = KnowledgeBase(str(tmp_path))
    again._load()
    restored = again._pools.get("u1e264f4abcd")
    assert restored is not None and "Сравните бренд" in restored.instruction and restored.lessons


def test_ozon_product_photos_are_downloaded_resized(monkeypatch):
    """Фото товаров Ozon (ir.ozone.ru) скачиваются копией …/wc1000/… (24 КБ вместо 0,5–2 МБ): при
    медленной связи с Ozon оригиналы шли по 10–30 с. Остальные адреса — как есть."""
    import media

    monkeypatch.setattr(media, "IMAGE_URL_REWRITE", OzonProfitAdapter.settings["IMAGE_URL_REWRITE"])
    assert (media.download_url("https://ir.ozone.ru/s3/multimedia-1-v/7080087955.jpg")
            == "https://ir.ozone.ru/s3/multimedia-1-v/wc1000/7080087955.jpg")
    already = "https://ir.ozone.ru/s3/multimedia-1-v/wc500/7080087955.jpg"
    assert media.download_url(already) == already
    other = "https://cdn1.ozonusercontent.com/s3/ozon-crowd-public-storage/135_d05.jpg"
    assert media.download_url(other) == other
    monkeypatch.setattr(media, "IMAGE_URL_REWRITE", "")
    assert media.download_url("https://ir.ozone.ru/s3/multimedia-1-v/7080087955.jpg").endswith("/7080087955.jpg")
