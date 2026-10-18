"""Режим помощника (--assist): человек решает сам; в текстовых заданиях модель подсвечивает ответ и
объясняет его в панели; задания с фото, записью и полем для ввода — без подсказки и без модели."""

from __future__ import annotations

from adapters.ozon_profit import OzonProfitAdapter
from assistant import Assistant
from browser_controller import BrowserController
from dom_parser import ASSIST_PANEL_ID, DomParser
from knowledge import KnowledgeBase
from models import ActionType, DecisionContext, ElementKind, LLMDecision, PageState, PlannedAction
from tests.helpers import OZON_TASK_URL, ozon_routes, run, use_platform


class ScriptedLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[PageState, DecisionContext]] = []

    async def decide(self, state: PageState, context: DecisionContext) -> LLMDecision:
        self.calls.append((state, context))
        jam = next(e for e in state.elements if e.text == "Джем Ягодная Поляна клубничный 400г")
        no_juice = [e for e in state.elements if e.text == "Нет нужного варианта"][-1]
        send = next(e for e in state.elements if e.text == "Отправить")
        return LLMDecision(
            reasoning="Тот же бренд, вкус и вес; томатного сока 1 л в списке нет", confidence=0.9,
            action=ActionType.CLICK, target_index=jam.index, target_text=jam.text,
            next_actions=[PlannedAction(action=ActionType.CLICK, target_index=no_juice.index, target_text=no_juice.text),
                          PlannedAction(action=ActionType.SUBMIT, target_index=send.index)])


def make(tmp_path, answers: list) -> tuple[Assistant, ScriptedLLM]:
    llm = ScriptedLLM()
    browser = BrowserController(on_context=ozon_routes(answers))
    return Assistant(browser=browser, llm=llm, knowledge=KnowledgeBase(str(tmp_path / "knowledge"))), llm


async def ticks(assistant: Assistant, until: str, limit: int = 40) -> list[str]:
    seen = []
    for _ in range(limit):
        seen.append(await assistant.tick())
        if seen[-1] == until:
            break
        await assistant._browser.page.wait_for_timeout(150)
    return seen


def test_text_task_is_highlighted_and_nothing_is_clicked(tmp_path, monkeypatch):
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL + "?instruction=0")
    answers: list = []

    async def scenario():
        assistant, llm = make(tmp_path, answers)
        async with assistant._browser:
            page = assistant._browser.page
            await page.wait_for_selector("[data-testid=TasksSendButton]")
            seen = await ticks(assistant, "подсказка")
            hinted = await page.eval_on_selector_all("[data-ai-hint]", "els => els.map((e) => e.innerText.trim())")
            panel = await page.inner_text(f"#{ASSIST_PANEL_ID}")
            checked = await page.eval_on_selector_all("input:checked", "els => els.length")
            # человек выбирает ответ — модель не спрашивается заново, подсказка остаётся
            await page.get_by_text("Джем Ягодная Поляна клубничный 400г", exact=True).click()
            for _ in range(4):
                await assistant.tick()
            state = await DomParser(page.main_frame).parse(quiet=True)
            return seen, hinted, panel, checked, llm, state

    seen, hinted, panel, checked, llm, state = run(scenario())
    assert seen[-1] == "подсказка" and len(llm.calls) == 1
    assert hinted == ["Джем Ягодная Поляна клубничный 400г", "Нет нужного варианта"]
    assert "Ответ: «Джем Ягодная Поляна клубничный 400г»; «Нет нужного варианта»" in panel
    assert "Почему: Тот же бренд" in panel and "Уверенность: 90%" in panel
    assert checked == 0 and answers == []                      # сам ничего не выбрал и не отправил
    assert "Уверенность" not in state.task_text                # панель — не часть страницы задания
    assert not any(e.text.startswith("ИИ") for e in state.elements)


def test_task_with_photo_gets_no_hint_and_no_model_call(tmp_path, monkeypatch):
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL + "?instruction=0")

    async def scenario():
        assistant, llm = make(tmp_path, [])
        async with assistant._browser:
            page = assistant._browser.page
            await page.wait_for_selector("[data-testid=TasksSendButton]")
            await ticks(assistant, "подсказка")
            await page.click("[data-testid=TasksSendButton]")        # человек отправил страницу
            await page.wait_for_selector("text=Морс Северный Бор")
            seen = await ticks(assistant, "не текстовое")
            panel = await page.inner_text(f"#{ASSIST_PANEL_ID}")
            hinted = await page.eval_on_selector_all("[data-ai-hint]", "els => els.length")
            return seen, panel, hinted, llm

    seen, panel, hinted, llm = run(scenario())
    assert seen[-1] == "не текстовое" and len(llm.calls) == 1        # только первая, текстовая страница
    assert "на странице фото" in panel and hinted == 0


def test_instruction_opened_by_human_is_saved(tmp_path, monkeypatch):
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL)        # окно инструкции откроется само

    async def scenario():
        assistant, _ = make(tmp_path, [])
        async with assistant._browser:
            page = assistant._browser.page
            await page.wait_for_selector("[data-testid=InstructionPopup]")
            return await ticks(assistant, "открыто окно")

    seen = run(scenario())
    assert seen[-1] == "открыто окно"
    saved = "\n".join(f.read_text(encoding="utf-8") for f in (tmp_path / "knowledge").glob("*.md"))
    assert "Сравнение названий товаров" in saved


def test_twork_task_in_frame_gets_hints(tmp_path, monkeypatch):
    """T-Work: задание во фрейме klecks-operator — подсветка и панель внутри фрейма, без кликов."""
    from tests.helpers import install_routes, workspace_url

    monkeypatch.setattr("browser_controller.TARGET_URL", workspace_url("quiz"))

    class YesYes:
        calls = 0

        async def decide(self, state: PageState, context: DecisionContext) -> LLMDecision:
            YesYes.calls += 1
            yes = [e for e in state.elements if e.text == "Да" and e.kind == ElementKind.OPTION]
            return LLMDecision(reasoning="холодильник новый, гарантия 2 года", confidence=0.95,
                               action=ActionType.CLICK, target_index=yes[0].index, target_text="Да",
                               next_actions=[PlannedAction(action=ActionType.CLICK, target_index=yes[1].index,
                                                           target_text="Да")])

    async def scenario():
        assistant = Assistant(browser=BrowserController(on_context=install_routes), llm=YesYes(),
                              knowledge=KnowledgeBase(str(tmp_path / "knowledge")))
        async with assistant._browser:
            seen = await ticks(assistant, "подсказка")
            frame = await assistant._browser.find_target_frame()
            hinted = await frame.eval_on_selector_all("[data-ai-hint]", "els => els.length")
            panel = await frame.inner_text(f"#{ASSIST_PANEL_ID}")
            checked = await frame.eval_on_selector_all("input:checked", "els => els.length")
            return seen, hinted, panel, checked

    seen, hinted, panel, checked = run(scenario())
    assert seen[-1] == "подсказка" and YesYes.calls == 1
    assert hinted == 2 and "Ответ: «Да»; «Да»" in panel and checked == 0


def test_platform_flag_is_read_before_settings():
    """--platform на один запуск: читается до config.py (настройки площадки берутся при импорте)."""
    import subprocess
    import sys
    from pathlib import Path

    from main import _platform_from_argv

    assert _platform_from_argv(["--assist", "--platform", "twork"]) == "twork"
    assert _platform_from_argv(["--platform=ozon", "--assist"]) == "ozon"
    assert _platform_from_argv(["--assist"]) == ""
    out = subprocess.run([sys.executable, "main.py", "--platform", "avito", "--assist"],
                         cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 2 and "PLATFORM='avito'" in out.stderr + out.stdout


def test_yandex_task_in_foreign_frame_gets_hints(tmp_path, monkeypatch):
    """Яндекс Задания: задание во фрейме другого домена (iframe-tasks.yandex), «Отправить» и «Выйти» —
    на основной странице. Подсказки — внутри фрейма; подсказки клавиш («1», «q»…) модель не видит;
    «Есть замечания» открывает флажки — модель спрашивается заново и подсказывает нарушение."""
    from adapters.yandex_tasks import YandexTasksAdapter
    from dom_parser import render_page
    from tests.helpers import YANDEX_TASK_URL, install_yandex_routes

    use_platform(monkeypatch, YandexTasksAdapter, YANDEX_TASK_URL)

    class Moderator:
        def __init__(self) -> None:
            self.calls: list[PageState] = []

        async def decide(self, state: PageState, context: DecisionContext) -> LLMDecision:
            self.calls.append(state)
            ok = [e for e in state.elements if e.text == "Всё хорошо"]
            bad = [e for e in state.elements if e.text == "Есть замечания"]
            steps = [(ok[0], "Всё хорошо"), (bad[1], "Есть замечания")]
            banned = [e for e in state.visible_elements if e.text == "Запрещенная услуга"]
            if banned:
                steps.append((banned[0], "Запрещенная услуга"))
            first, *rest = steps
            return LLMDecision(reasoning="курсовые на заказ — запрещённая услуга", confidence=0.85,
                               action=ActionType.CLICK, target_index=first[0].index, target_text=first[1],
                               next_actions=[PlannedAction(action=ActionType.CLICK, target_index=e.index, target_text=t)
                                             for e, t in rest])

    async def scenario():
        llm = Moderator()
        assistant = Assistant(browser=BrowserController(on_context=install_yandex_routes), llm=llm,
                              knowledge=KnowledgeBase(str(tmp_path / "knowledge")))
        async with assistant._browser:
            await ticks(assistant, "подсказка")
            frame = await assistant._browser.find_target_frame()
            first = await frame.eval_on_selector_all("[data-ai-hint]", "els => els.map((e) => e.innerText.trim())")
            # человек отметил «Есть замечания» во второй задаче — открылись флажки нарушений
            await frame.locator(".task").nth(1).get_by_text("Есть замечания").click()
            for _ in range(20):
                await assistant.tick()
                if len(llm.calls) > 1 and await frame.eval_on_selector_all("[data-ai-hint]", "els => els.length") == 3:
                    break
                await assistant._browser.page.wait_for_timeout(150)
            second = await frame.eval_on_selector_all("[data-ai-hint]", "els => els.map((e) => e.innerText.trim())")
            sent = await assistant._browser.page.evaluate("() => window.__sent || 0")
            return frame.url, first, second, llm, sent

    url, first, second, llm, sent = run(scenario())
    assert "iframe-tasks.yandex" in url
    assert first == ["Всё хорошо", "Есть замечания"]
    assert len(llm.calls) == 2 and "Запрещенная услуга" in second and sent == 0
    text, shown = render_page(llm.calls[0])
    lines = text.splitlines()
    assert "q" not in lines and "1" not in lines                       # подсказки клавиш — не текст задания
    assert not any(e.text in ("Отправить", "Выйти") for e in shown)    # кнопки основной страницы — не во фрейме
    assert "Курсовая на заказ недорого" in text and "Название услуги" in text


def test_photo_task_shows_how_to_solve_from_instruction(tmp_path, monkeypatch):
    """Задание с фото — ответ за человеком, но панель показывает выжимку инструкции «как решать»:
    модель делает её один раз на вид заданий, дальше она берётся из базы знаний."""
    use_platform(monkeypatch, OzonProfitAdapter, OZON_TASK_URL)       # окно инструкции откроется само

    class Briefer(ScriptedLLM):
        def __init__(self) -> None:
            super().__init__()
            self.briefs: list[str] = []

        async def brief_instruction(self, instruction: str, model: str = "") -> str:
            self.briefs.append(instruction)
            return "• Сравнивайте бренд, вкус и вес\n• Другой вес — другой товар"

    async def scenario():
        llm = Briefer()
        assistant = Assistant(browser=BrowserController(on_context=ozon_routes([])), llm=llm,
                              knowledge=KnowledgeBase(str(tmp_path / "knowledge")))
        async with assistant._browser:
            page = assistant._browser.page
            await page.wait_for_selector("[data-testid=InstructionPopup]")
            await ticks(assistant, "открыто окно")                     # инструкцию прочитал
            await page.keyboard.press("Escape")
            await ticks(assistant, "подсказка")
            await page.click("[data-testid=TasksSendButton]")
            await page.wait_for_selector("text=Морс Северный Бор")
            for _ in range(30):
                await assistant.tick()
                panel = await page.inner_text(f"#{ASSIST_PANEL_ID}")
                if "Другой вес" in panel:
                    break
                await page.wait_for_timeout(150)
            return panel, llm

    panel, llm = run(scenario())
    assert "Задание ваше — как решать" in panel and "• Сравнивайте бренд, вкус и вес" in panel
    assert "на странице фото" in panel
    assert len(llm.briefs) == 1 and "Сравнение названий товаров" in llm.briefs[0]
    saved = "\n".join(f.read_text(encoding="utf-8") for f in (tmp_path / "knowledge").glob("*.md"))
    assert "## Как решать" in saved and "Другой вес — другой товар" in saved
    restored = KnowledgeBase(str(tmp_path / "knowledge"))
    restored._load()
    assert any("Другой вес" in p.brief for p in restored._pools.values())        # переживает перезапуск
