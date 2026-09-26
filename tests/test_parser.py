"""DomParser и BrowserController на фикстурах: реальный Chromium, кросс-доменный
iframe, масштаб 75% через device_scale_factor (как в боевом конфиге)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from playwright.async_api import async_playwright

import browser_controller
from browser_controller import BrowserController
from dom_parser import DomParser, build_elements_prompt
from models import ElementKind, FolderState
from tests.helpers import install_routes, main_url, run


@asynccontextmanager
async def task_frame(task: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1707, "height": 960}, device_scale_factor=0.75,
        )
        await install_routes(context)
        page = await context.new_page()
        await page.goto(main_url(task))
        frame = None
        for _ in range(50):
            frame = next((f for f in page.frames if "klecks-operator" in f.url), None)
            if frame is not None:
                await frame.wait_for_load_state("load")
                break
            await page.wait_for_timeout(100)
        assert frame is not None, "фрейм задания не загрузился"
        try:
            yield page, frame
        finally:
            await browser.close()


def one(state, text):
    found = [e for e in state.elements if e.text == text]
    assert len(found) == 1, f"ожидался один элемент «{text}», найдено {len(found)}"
    return found[0]


def test_tree_initial_snapshot():
    async def scenario():
        async with task_frame("task_tree") as (_, frame):
            state = await DomParser(frame).parse()
            # ни скрытых radio, ни display:none-кнопки, ни детей свёрнутых веток, ни дублей
            assert [e.text for e in state.elements] == ["Одежда", "Электроника", "Завершить", "Завершить смену"]
            for name in ("Одежда", "Электроника"):
                folder = one(state, name)
                assert folder.kind == ElementKind.FOLDER
                assert folder.folder_state == FolderState.CLOSED
                assert folder.has_toggle
            finish = one(state, "Завершить")
            assert finish.kind == ElementKind.BUTTON and finish.is_disabled
            assert state.task_text.startswith("Выберите категорию для товара: смартфон")
            assert state.image_src and state.hidden_elements > 0
            # таймер виден модели в тексте страницы, но отпечаток задания от него не зависит
            await frame.evaluate("() => { document.querySelector('.timer').textContent = '00:41'; }")
            ticked = await DomParser(frame).parse()
            assert "00:41" in ticked.task_text
            assert ticked.task_identifier == state.task_identifier
    run(scenario())


def test_expand_paths_duplicates_and_stable_task_id():
    async def scenario():
        async with task_frame("task_tree") as (_, frame):
            parser = DomParser(frame)
            before = await parser.parse()
            for name in ("Электроника", "Одежда"):
                folder = one(before, name)
                await frame.locator(f'[data-agent-toggle="{folder.uid}"]').click()
            after = await parser.parse()

            assert one(after, "Электроника").folder_state == FolderState.OPEN
            phones = one(after, "Смартфоны")
            assert phones.kind == ElementKind.OPTION and phones.choice_type == "radio"
            assert phones.path == ["Электроника"] and phones.depth == 1

            others = [e for e in after.elements if e.text == "Другое"]
            assert [e.path for e in others] == [["Одежда"], ["Электроника"]]
            assert len({e.key for e in others}) == 2          # ключи памяти различаются
            prompt = build_elements_prompt(after.elements)
            assert "«Другое» ⟨путь: Одежда⟩" in prompt and "«Другое» ⟨путь: Электроника⟩" in prompt

            # раскрытие папок не меняет отпечаток задания, но меняет состояние
            assert after.task_identifier == before.task_identifier
            assert after.state_hash != before.state_hash
    run(scenario())


def test_stale_stamp_is_detected():
    async def scenario():
        async with task_frame("task_tree") as (_, frame):
            state = await DomParser(frame).parse()
            uid = one(state, "Одежда").uid
            assert await frame.locator(f'[data-agent-id="{uid}"]').count() == 1
            # Angular пересоздал узел → метка исчезла → клик не уйдёт в чужой элемент
            await frame.evaluate("""uid => {
                const el = document.querySelector(`[data-agent-id="${uid}"]`);
                const copy = el.cloneNode(true); copy.removeAttribute('data-agent-id'); el.replaceWith(copy);
            }""", uid)
            assert await frame.locator(f'[data-agent-id="{uid}"]').count() == 0
            fresh = await DomParser(frame).parse()
            assert fresh.generation != state.generation
            assert await frame.locator(f'[data-agent-id="{uid}"]').count() == 0
    run(scenario())


def test_taiga_like_semantics():
    async def scenario():
        async with task_frame("taiga_like") as (_, frame):
            state = await DomParser(frame).parse()
            texts = [e.text or e.placeholder for e in state.elements]

            chars = one(state, "Характеристики")
            assert chars.kind == ElementKind.FOLDER and chars.folder_state == FolderState.OPEN and chars.has_toggle
            material = one(state, "Материал")
            assert material.kind == ElementKind.FOLDER and material.folder_state == FolderState.CLOSED
            assert "Пластик" not in texts                    # display:none-ветка
            assert "Свернуть" not in texts and "Развернуть" not in texts   # стрелки слиты со строками

            # v2 делал чекбоксы с tui-svg «папками»
            waterproof = one(state, "Водонепроницаемый")
            assert waterproof.kind == ElementKind.OPTION and waterproof.choice_type == "checkbox"
            assert waterproof.is_selected and waterproof.path == ["Характеристики"]
            assert not one(state, "Беспроводной").is_selected
            # class="inactive" не «выбран»; SVG с class*=list-item не роняет парсер
            assert not one(state, "Устаревший вариант").is_selected

            assert "Невидимая кнопка" not in texts           # opacity:0 у предка
            shadow_button = one(state, "Кнопка в тени")     # открытый shadow DOM
            assert shadow_button.kind == ElementKind.BUTTON
            await frame.locator(f'[data-agent-id="{shadow_button.uid}"]').click()

            comment = next(e for e in state.elements if e.placeholder == "Комментарий")
            assert comment.kind == ElementKind.INPUT and comment.input_type == "text"
            size = next(e for e in state.elements if e.placeholder == "Размер")
            assert size.input_type == "select" and size.options == ["S", "M", "L"] and size.value == "M"
            color = [e for e in state.elements if (e.text or e.placeholder) == "Цвет"]
            assert len(color) == 1 and color[0].kind == ElementKind.DROPDOWN   # readonly input слит

            assert "Отметьте все подходящие" in state.task_text
            assert state.hint_text.startswith("Подсказка")
            assert state.alerts == ["Выберите хотя бы один вариант"]
            assert not state.loading and state.local_loading == 0
            # маленький спиннер рядом с полем — «местная» загрузка, работу не блокирует
            await frame.evaluate("() => { document.querySelector('.spinner').hidden = false; }")
            small = await DomParser(frame).parse()
            assert not small.loading and small.local_loading == 1
            # оверлей на весь экран — блокирующая загрузка
            await frame.evaluate("() => { document.querySelector('.spinner').style.cssText = "
                                 "'position:fixed;inset:0;width:auto;height:auto'; }")
            assert (await DomParser(frame).parse()).loading
    run(scenario())


def test_flat_tree_with_aria_level():
    async def scenario():
        async with task_frame("flat_aria") as (_, frame):
            state = await DomParser(frame).parse()
            electronics = one(state, "Электроника")
            assert electronics.kind == ElementKind.FOLDER and electronics.folder_state == FolderState.OPEN
            phones = one(state, "Смартфоны")
            assert phones.kind == ElementKind.OPTION and phones.depth == 1 and phones.path == ["Электроника"]
            accessories = one(state, "Аксессуары")
            assert accessories.folder_state == FolderState.CLOSED and accessories.path == ["Электроника"]
            clothes = one(state, "Одежда")
            assert clothes.depth == 0 and clothes.path == []
    run(scenario())


def test_controller_clicks_land_on_target_with_zoom():
    """Полная цепочка BrowserController (trial-клик + мышь) при PAGE_ZOOM=75%
    внутри кросс-доменного iframe. В v2 CSS-zoom уводил такие клики мимо цели."""
    async def scenario():
        controller = BrowserController(on_context=install_routes)
        async with controller:
            frame = None
            for _ in range(50):
                frame = await controller.find_target_frame()
                if frame is not None:
                    break
                await controller.page.wait_for_timeout(100)
            assert frame is not None
            await frame.wait_for_load_state("load")

            state = await DomParser(frame).parse()
            outcome = await controller.click_element(frame, one(state, "Электроника"), prefer_toggle=True)
            assert outcome.ok and outcome.method == "mouse"
            await controller.wait_settle(frame)

            state = await DomParser(frame).parse()
            outcome = await controller.click_element(frame, one(state, "Смартфоны"))
            assert outcome.ok and outcome.method == "mouse"
            await controller.wait_settle(frame)

            state = await DomParser(frame).parse()
            assert one(state, "Смартфоны").is_selected
            finish = one(state, "Завершить")
            assert not finish.is_disabled
            assert (await controller.click_element(frame, finish)).ok
            await controller.page.wait_for_timeout(100)
            log = await frame.evaluate("() => window.__log")
            assert log == ["toggle:Электроника", "submit:phones"]
    assert browser_controller.PAGE_ZOOM_FACTOR == 0.75
    run(scenario())


def test_js_cap_keeps_submit_button(monkeypatch):
    """При обрезке большого дерева кнопка «Завершить» (конец документа) не пропадает."""
    import dom_parser

    monkeypatch.setattr(dom_parser, "_JS_MAX_ELEMENTS", 4)

    async def scenario():
        async with task_frame("task_tree") as (_, frame):
            state = await DomParser(frame).parse()
            for name in ("Электроника", "Одежда"):
                await frame.locator(f'[data-agent-toggle="{one(state, name).uid}"]').click()
            capped = await DomParser(frame).parse()
            texts = [e.text for e in capped.elements]
            assert "Завершить" in texts and "Завершить смену" in texts
            assert len(capped.elements) == 4 and capped.total_elements == 9
    run(scenario())
