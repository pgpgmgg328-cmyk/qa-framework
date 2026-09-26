"""Задания со структурой T-Work (flex_task.html): reader-view, медиа, поиск, инструкции,
тренировка с «Неверный ответ», остановка на списке заказов. Реальный Chromium,
«LLM» — сценарная (решения детерминированы, чтобы проверять именно агента)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable, Optional

from playwright.async_api import async_playwright

import agent as agent_module
from agent import Agent, StepResult
from browser_controller import BrowserController
from dom_parser import DomParser, geo_facts, render_page
from knowledge import KnowledgeBase
from media import MediaManager
from models import ActionType, DecisionContext, ElementKind, LLMDecision, PageState, ParsedElement
from tests.helpers import install_routes, run, workspace_url

# ---------------------------------------------------------------------------
# Помощники
# ---------------------------------------------------------------------------


@asynccontextmanager
async def flex_frame(scenario: str, **kw):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        await install_routes(context)
        page = await context.new_page()
        await page.goto(workspace_url(scenario, **kw))
        frame = None
        for _ in range(50):
            frame = next((f for f in page.frames if "/klecks/task" in f.url), None)
            if frame is not None:
                await frame.wait_for_load_state("load")
                break
            await page.wait_for_timeout(100)
        assert frame is not None
        await page.wait_for_timeout(300)
        try:
            yield context, page, frame
        finally:
            await browser.close()


def find(state: PageState, text: str, kind: Optional[ElementKind] = None) -> Optional[ParsedElement]:
    for e in state.visible_elements:
        label = e.text or e.placeholder
        if label == text and (kind is None or e.kind == kind):
            return e
    return None


def click(el: ParsedElement, **kw) -> LLMDecision:
    return LLMDecision(reasoning="сценарий", action=ActionType.CLICK, target_index=el.index,
                       target_text=el.text or el.placeholder, **kw)


def submit(state: PageState, **kw) -> LLMDecision:
    button = find(state, "Завершить задание", ElementKind.BUTTON)
    return LLMDecision(reasoning="сценарий", action=ActionType.SUBMIT,
                       target_index=button.index if button else None, **kw)


class ScriptedLLM:
    """Сценарная «модель»: policy(state, context, llm) → решение."""

    def __init__(self, policy: Callable[[PageState, DecisionContext, "ScriptedLLM"], LLMDecision]) -> None:
        self.policy = policy
        self.calls: list[tuple[PageState, DecisionContext]] = []
        self.transcribed: list[tuple[int, str, str]] = []

    async def decide(self, state: PageState, context: DecisionContext) -> LLMDecision:
        self.calls.append((state, context))
        return self.policy(state, context, self)

    async def transcribe(self, data: bytes, filename: str, mime: str) -> Optional[str]:
        self.transcribed.append((len(data), filename, mime))
        return "[0:00–0:01] Здравствуйте, абонент не может ответить. Оставьте сообщение после сигнала."


def make_agent(tmp_path, policy, url: str) -> tuple[Agent, ScriptedLLM]:
    llm = ScriptedLLM(policy)
    browser = BrowserController(on_context=install_routes)
    agent = Agent(browser=browser, llm=llm, knowledge=KnowledgeBase(str(tmp_path / "knowledge")))
    return agent, llm


# ---------------------------------------------------------------------------
# Reader-view
# ---------------------------------------------------------------------------


def test_hotel_page_reader_view():
    async def scenario():
        async with flex_frame("hotels") as (_, _page, frame):
            state = await DomParser(frame).parse()
            text, shown = render_page(state)
            lines = text.splitlines()
            # вопрос — строкой выше своих вариантов; кнопки карусели «1»,«2»,«3» скрыты
            q = lines.index("#### Отели совпадают?")
            assert "[OPTION radio] «Да»" in lines[q + 1] and "[OPTION radio] «Нет»" in lines[q + 2]
            assert not any(e.text in ("1", "2", "3") for e in shown)
            assert any(e.text in ("1", "2", "3") and e.aux for e in state.elements)
            # фото карточек: две группы, у каждой строка [ФОТО …]
            assert "[ФОТО 1–3]" in lines and "[ФОТО 4–5]" in lines
            assert len(state.images) == 5
            # ссылки на карту с координатами и вычисленное расстояние
            maps = [e for e in shown if e.text == "Открыть карту"]
            assert len(maps) == 2 and all(e.href.startswith("https://www.google.com/maps?ll=") for e in maps)
            facts = geo_facts(state)
            assert len(facts) == 1 and "расстояние ≈ 1 м" in facts[0]
            # текст с переносами <br> и полями карточек
            assert "Название отеля: Отель Магнолия" in lines and "Адрес: Mira street 5" in lines
            assert "Выйти из задания" not in text          # стоп-список
            assert state.pool_title == "Отели совпадают?"
    run(scenario())


def test_product_dropdown_caption_popup_and_table():
    async def scenario():
        async with flex_frame("product") as (_, _page, frame):
            state = await DomParser(frame).parse()
            color = find(state, "Цвет", ElementKind.DROPDOWN)
            assert color is not None and color.value == "" and color.caption == "Цвет"
            text, _ = render_page(state)
            assert "| Модель | A05 |" in text and "# Выбор характеристики из списка" in text
            assert "[DROPDOWN закрыт] «Цвет» (не выбрано)" in text
            await frame.locator(f'[data-agent-id="{color.uid}"]').click()
            opened = await DomParser(frame).parse()
            options = [e for e in opened.visible_elements if e.container == "popup"]
            # все 21 значение — включая прокручиваемый хвост под overflow:hidden-обёрткой
            assert len(options) == 21 and options[-1].text == "Черный"
            text, _ = render_page(opened)
            assert "═══ ОТКРЫТЫЙ ВЫПАДАЮЩИЙ СПИСОК ═══" in text
            await frame.locator(f'[data-agent-id="{options[-1].uid}"]').click()
            chosen = await DomParser(frame).parse()
            color2 = find(chosen, "Цвет", ElementKind.DROPDOWN)
            assert color2.value == "Черный" and color2.key == color.key     # ключ памяти не меняется
            assert chosen.task_identifier == state.task_identifier
    run(scenario())


# ---------------------------------------------------------------------------
# Агент целиком
# ---------------------------------------------------------------------------


def test_training_feedback_lesson_and_stop_on_orders_list(tmp_path, monkeypatch):
    """Новости сайта закрыты; неверный ответ → подсказка → исправление; урок сохранён;
    после последнего задания платформа вернула на список заказов → агент остановился."""
    monkeypatch.setattr("browser_controller.TARGET_URL", workspace_url("hotels", popup=True))

    def policy(state: PageState, ctx: DecisionContext, llm: ScriptedLLM) -> LLMDecision:
        yes, no = find(state, "Да"), find(state, "Нет")
        first_task = "Отель Магнолия" in state.task_text
        wanted = yes if first_task else no
        if not first_task or ctx.feedback:
            if wanted.is_selected:
                return submit(state)
            return click(wanted, plan="по подсказке платформы")
        # первое задание: намеренно ошибаемся
        if no.is_selected:
            return submit(state)
        return click(no, plan="кажется, разные отели")

    async def scenario():
        agent, llm = make_agent(tmp_path, policy, workspace_url("hotels", popup=True))
        await agent.run()
        return agent, llm

    agent, llm = run(scenario())
    assert agent._tasks_done == 2 and agent._wrong_total == 1
    # после «Неверный ответ» то же задание: память сохранена, подсказка передана модели
    fixes = [ctx for state, ctx in llm.calls if ctx.feedback]
    assert fixes, "модель не получила ответ платформы"
    feedback = fixes[0]
    assert any("Неверный ответ" in f for f in feedback.feedback)
    assert any("Правильный ответ: Да" in f for f in feedback.feedback)
    assert feedback.wrong_answers == ["«Нет»"]
    assert feedback.plan == "кажется, разные отели"
    assert any("НЕВЕРНЫЙ ОТВЕТ" in h for h in feedback.history)
    # урок в базе знаний и в промпте следующего задания этого вида
    files = list((tmp_path / "knowledge").glob("*.md"))
    assert len(files) == 1 and "Правильный ответ: Да" in files[0].read_text(encoding="utf-8")
    last_ctx = llm.calls[-1][1]
    assert "Уроки из прошлых ошибок" in last_ctx.knowledge
    # второе задание начато с чистой памятью
    second = [ctx for state, ctx in llm.calls if "Хостел Кедр" in state.task_text]
    assert second and second[0].history == [] and second[0].wrong_answers == []


def test_audio_is_transcribed_and_played_to_the_end(tmp_path, monkeypatch):
    monkeypatch.setattr("browser_controller.TARGET_URL", workspace_url("robot"))

    def policy(state: PageState, ctx: DecisionContext, llm: ScriptedLLM) -> LLMDecision:
        assert ctx.transcripts and "Оставьте сообщение" in ctx.transcripts[0]
        machine = find(state, "Результат неправильный. Был автоответчик")
        if machine.is_selected:
            return submit(state)
        return click(machine)

    async def scenario():
        agent, llm = make_agent(tmp_path, policy, workspace_url("robot"))
        await agent.run()
        return agent, llm

    agent, llm = run(scenario())
    # запись скачана со страницы задания (с её авторизацией) и расшифрована один раз
    assert len(llm.transcribed) == 1 and llm.transcribed[0][0] > 10_000
    assert llm.transcribed[0][1] == "audio.wav"
    # плеер виден модели одной строкой, его кнопки — нет
    state = llm.calls[0][0]
    text, shown = render_page(state)
    assert "[АУДИО 1:" in text and not any(e.text == "1x" for e in shown)
    # ответ принят с первой отправки: агент дослушал запись до «Завершить задание»
    assert agent._tasks_done == 1 and agent._wrong_total == 0
    assert agent._memory.submit_failures == 0


def test_web_research_fills_org_form(tmp_path, monkeypatch):
    monkeypatch.setattr("browser_controller.TARGET_URL", workspace_url("org"))

    def policy(state: PageState, ctx: DecisionContext, llm: ScriptedLLM) -> LLMDecision:
        url_field = find(state, "Введите полный URL на Яндекс Картах")
        id_field = find(state, "Введите ID из URL Яндекс Карт")
        if not ctx.research:
            return LLMDecision(reasoning="ищу", action=ActionType.WEB, query="lesnoydom.example")
        if len(ctx.research) == 1:
            return LLMDecision(reasoning="открываю карточку", action=ActionType.WEB,
                               query="https://yandex.ru/maps/org/lesnoy-dom/1234567890/")
        if not url_field.value:
            return LLMDecision(reasoning="копирую", action=ActionType.TYPE, target_index=url_field.index,
                               target_text=url_field.placeholder,
                               type_text="https://yandex.ru/maps/org/lesnoy-dom/1234567890/")
        if not id_field.value:
            return LLMDecision(reasoning="ID", action=ActionType.TYPE, target_index=id_field.index,
                               target_text=id_field.placeholder, type_text="1234567890")
        for name in ("Название организации", "URL", "Адрес организации"):
            box = find(state, name)
            if not box.is_selected:
                return click(box)
        return submit(state)

    async def scenario():
        agent, llm = make_agent(tmp_path, policy, workspace_url("org"))
        await agent.run()
        return agent, llm

    agent, llm = run(scenario())
    assert agent._tasks_done == 1 and agent._wrong_total == 0
    research = [ctx.research for _, ctx in llm.calls if len(ctx.research) >= 2][0]
    assert "Адрес страницы: https://yandex.ru/search/?text=lesnoydom.example" in research[0]
    assert "Лесной дом — Яндекс Карты → https://yandex.ru/maps/org/lesnoy-dom/1234567890/" in research[0]
    assert "Адрес страницы: https://yandex.ru/maps/org/lesnoy-dom/1234567890/" in research[1]
    assert "Цветочная улица, 7к1" in research[1]
    final = llm.calls[-1][0]                                   # снимок, на котором модель нажала submit
    assert find(final, "Введите ID из URL Яндекс Карт").value == "1234567890"
    assert find(final, "Введите полный URL на Яндекс Картах").value == "https://yandex.ru/maps/org/lesnoy-dom/1234567890/"


def test_instruction_and_tooltips_are_read_once_and_used(tmp_path, monkeypatch):
    monkeypatch.setattr("browser_controller.TARGET_URL", workspace_url("atm"))

    def policy(state: PageState, ctx: DecisionContext, llm: ScriptedLLM) -> LLMDecision:
        assert "обе боковые стороны" in ctx.knowledge                 # инструкция
        assert "Нет хотя бы одной поверхности." in ctx.knowledge       # подсказка «?»
        partial = find(state, "Фото присутствуют частично")
        side = find(state, "Боковые поверхности - недостаточно фото")
        if not partial.is_selected:
            return click(partial)
        if not side.is_selected:
            return click(side)
        return submit(state)

    async def scenario():
        agent, llm = make_agent(tmp_path, policy, workspace_url("atm"))
        await agent.run()
        return agent, llm

    agent, llm = run(scenario())
    assert agent._tasks_done == 1
    content = next((tmp_path / "knowledge").glob("*.md")).read_text(encoding="utf-8")
    assert "## Инструкция" in content and "обе боковые стороны" in content
    assert "«Все фото в наличии»: Есть фото лицевой части" in content
    assert "## Заметки" in content


def test_exit_dialog_is_answered_with_stay(tmp_path, monkeypatch):
    monkeypatch.setattr("browser_controller.TARGET_URL", workspace_url("hotels"))

    def policy(state, ctx, llm):
        raise AssertionError("диалог выхода закрывается без LLM")

    async def scenario():
        agent, _ = make_agent(tmp_path, policy, workspace_url("hotels"))
        agent._pool = None
        async with agent._browser:
            frame = None
            for _ in range(50):
                frame = await agent._browser.find_target_frame()
                if frame is not None:
                    break
                await agent._browser.page.wait_for_timeout(100)
            await frame.wait_for_load_state("load")
            await frame.evaluate("() => window.openExitDialog()")
            await agent._browser.page.wait_for_timeout(200)
            result = await agent._step()
            await agent._browser.page.wait_for_timeout(200)
            return result, await frame.evaluate("() => window.__log")

    result, log = run(scenario())
    assert result == StepResult.ACTED and log == ["exit:stay"]


def test_agent_waits_on_orders_list_before_any_task(tmp_path, monkeypatch):
    """Запуск со списка заказов: агент ждёт, пока человек выберет заказ, и сам «Приступить» не жмёт."""
    monkeypatch.setattr("browser_controller.TARGET_URL", "https://t-work.test/workspace.html?orders=1")
    monkeypatch.setattr(agent_module, "FRAME_LOAD_WAIT", 0.1)

    async def scenario():
        agent, llm = make_agent(tmp_path, lambda *a: LLMDecision.skip("-"), "")
        async with agent._browser:
            results = []
            for _ in range(20):
                frame = await agent._browser.find_target_frame()
                if frame is not None:
                    break
                await agent._browser.page.wait_for_timeout(100)
            for _ in range(3):
                results.append(await agent._step())
            log = await frame.evaluate("() => window.__log")
        return results, llm, log

    results, llm, log = run(scenario())
    assert results == [StepResult.IDLE] * 3 and llm.calls == [] and log == []


def test_media_manager_fetches_authorized_attachments(tmp_path):
    async def scenario():
        async with flex_frame("atm") as (context, page, frame):
            state = await DomParser(frame).parse()

            class Browser:
                def __init__(self, ctx):
                    self.context = ctx

            media = MediaManager(Browser(context))
            images, notes = await media.vision_images(frame, state)
            return state, images, notes

    state, images, notes = run(scenario())
    assert len(state.images) == 3 and all(i.src.startswith("https://klecks-operator.test/") for i in state.images)
    assert [i.caption for i in images] == ["ФОТО 1", "ФОТО 2", "ФОТО 3"] and notes == []
    assert all(len(i.b64) > 500 for i in images)
