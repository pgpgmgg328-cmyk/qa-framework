"""Модульные тесты без браузера: разбор ответа LLM, автокоррекция цели,
память задания, согласованность легенды промпта с форматом строк."""

from __future__ import annotations

import pytest

import openrouter_connector as oc
from agent import Agent, _clean_target_text
from dom_parser import DomParser, build_elements_prompt, select_for_prompt
from models import (
    ActionType,
    DecisionContext,
    ElementKind,
    FolderState,
    LLMDecision,
    MediaImage,
    Notice,
    PageState,
    ParsedElement,
)
from task_memory import TaskMemory


def el(index, text, kind=ElementKind.OPTION, *, path=(), state=FolderState.NA, **kw) -> ParsedElement:
    return ParsedElement(index=index, uid=f"t-{index}", text=text, kind=kind, folder_state=state,
                         path=list(path), depth=len(path), **kw)


def page(*elements, task_id="task-1", state_hash="h1") -> PageState:
    items = list(elements)
    DomParser._assign_keys(items)
    return PageState(elements=items, task_identifier=task_id, state_hash=state_hash)


def tree_state(**kw) -> PageState:
    return page(
        el(0, "Одежда", ElementKind.FOLDER, state=FolderState.OPEN),
        el(1, "Другое", path=("Одежда",)),
        el(2, "Электроника", ElementKind.FOLDER, state=FolderState.OPEN),
        el(3, "Смартфоны", path=("Электроника",)),
        el(4, "Другое", path=("Электроника",)),
        el(5, "Завершить", ElementKind.BUTTON),
        **kw,
    )


# --------------------------------------------------------------------- LLMDecision

def test_decision_coerces_dirty_json():
    d = LLMDecision.model_validate({
        "action": "SELECT", "target_index": "[5]", "target_text": 12, "confidence": 95,
        "reasoning": None,
    })
    assert d.action == ActionType.CLICK and d.target_index == 5
    assert d.target_text == "12" and d.confidence == pytest.approx(0.95) and d.reasoning == ""


def test_parse_response_handles_fences_and_rejects_unknown_action():
    raw = '```json\n{"action": "open", "target_index": 2, "target_text": "Электроника", "confidence": 0.8}\n```'
    assert oc.LLMConnector.parse_response(raw).action == ActionType.OPEN
    with pytest.raises(oc.DecisionParseError):
        oc.LLMConnector.parse_response('{"action": "dance"}')
    with pytest.raises(oc.DecisionParseError):
        oc.LLMConnector.parse_response("никакого JSON")


def test_json_schema_is_strict_compatible():
    schema = oc.DECISION_JSON_SCHEMA["schema"]
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["action"]["enum"] == [a.value for a in ActionType]
    # порядок полей: рассуждения генерируются ДО действия
    keys = list(schema["properties"])
    assert keys.index("reasoning") < keys.index("action")


# --------------------------------------------------------------------- промпт

def test_every_rendered_tag_is_explained_in_system_prompt():
    samples = [
        el(0, "a", ElementKind.FOLDER, state=FolderState.OPEN),
        el(1, "b", ElementKind.FOLDER, state=FolderState.CLOSED, selectable=True),
        el(2, "c", ElementKind.DROPDOWN, state=FolderState.CLOSED),
        el(3, "d", ElementKind.DROPDOWN, state=FolderState.OPEN, container="popup"),
        el(4, "e", ElementKind.OPTION, choice_type="checkbox", is_selected=True),
        el(5, "f", ElementKind.BUTTON, is_disabled=True, occluded=True, container="dialog"),
        el(6, "g", ElementKind.OTHER),
    ]
    for sample in samples:
        line = sample.prompt_line(show_path=True)
        assert sample.type_tag().split()[0].strip("[") in oc.SYSTEM_PROMPT
        for token in ("✓ВЫБРАН", "⛔НЕАКТИВЕН", "⚠ПЕРЕКРЫТ", "(можно выбрать)", "(в диалоге)", "(во всплывающем списке)"):
            if token in line:
                assert token in oc.SYSTEM_PROMPT
    assert "[FOLDER закрыта]" in oc.SYSTEM_PROMPT and "[FOLDER раскрыта]" in oc.SYSTEM_PROMPT
    assert "ПАПКА" not in oc.SYSTEM_PROMPT        # v2: легенда и строки расходились


def test_user_message_contains_history_forbidden_and_notes():
    ctx = DecisionContext(
        history=["1. open «Электроника» → ✓ раскрыто, появилось 3: «Смартфоны»"],
        forbidden=["click «Другое» ⟨Одежда⟩ — 2 раз(а) без эффекта"],
        notes=["Осталось шагов на это задание: 3."], step_in_task=4,
    )
    ctx.plan = "Смартфон → Электроника › Смартфоны, затем submit"
    ctx.wrong_answers = ["«Ноутбуки»"]
    ctx.knowledge = "Инструкция к заданиям этого вида:\nВыбирайте самую узкую категорию."
    state = tree_state()
    state.notices = [Notice(kind="error", text="Выберите вариант")]
    state.reader = ["Выберите категорию для товара", "\x01N0\x01"] + [f"\x00E{e.index}\x00" for e in state.elements]
    text = oc.LLMConnector.build_user_message(state, ctx)
    for part in ("═══ ЗНАНИЯ О ВИДЕ ЗАДАНИЙ", "Выбирайте самую узкую", "═══ СТРАНИЦА ЗАДАНИЯ",
                 "‼ Выберите вариант", "  [3] [OPTION] «Смартфоны»", "═══ ТВОЙ ПЛАН", "Электроника › Смартфоны",
                 "═══ НЕВЕРНЫЕ ОТВЕТЫ", "✗ «Ноутбуки»", "═══ ИСТОРИЯ (шаг 4", "✓ раскрыто",
                 "═══ НЕ ПОВТОРЯТЬ", "═══ ВНИМАНИЕ"):
        assert part in text, part
    # знания — первыми, страница — до истории
    assert text.index("ЗНАНИЯ") < text.index("СТРАНИЦА") < text.index("ИСТОРИЯ")


def test_truncation_keeps_buttons_and_folders():
    many = [el(i, f"Вариант {i}", path=("Каталог",)) for i in range(300)]
    many.append(el(300, "Завершить", ElementKind.BUTTON))
    shown, omitted = select_for_prompt(many, limit=50)
    assert len(shown) == 50 and omitted == 251
    assert any(e.text == "Завершить" for e in shown)   # v2 обрезал её вместе с хвостом списка
    assert "ещё 251 элементов" in build_elements_prompt(many, limit=50)


# --------------------------------------------------------------------- автокоррекция

def resolve(decision_kw, state) -> ParsedElement | None:
    agent = Agent.__new__(Agent)          # без браузера и LLM
    agent._memory = TaskMemory()
    return agent._resolve_target(LLMDecision(reasoning="", **decision_kw), state)


def test_resolve_prefers_index_confirmed_by_text_for_duplicates():
    state = tree_state()
    # v2 взял бы первое «Другое» (ветка «Одежда»), хотя модель указала [4]
    target = resolve({"action": "click", "target_index": 4, "target_text": "Другое"}, state)
    assert target.index == 4 and target.path == ["Электроника"]


def test_resolve_fixes_shifted_index_by_text():
    state = tree_state()
    target = resolve({"action": "click", "target_index": 1, "target_text": "Смартфоны"}, state)
    assert target.index == 3


def test_resolve_respects_action_kind_and_cleans_copied_line():
    state = tree_state()
    target = resolve({"action": "open", "target_index": 3,
                      "target_text": "[2] [FOLDER раскрыта] «Электроника» ✓ВЫБРАН"}, state)
    assert target.index == 2 and target.kind == ElementKind.FOLDER
    assert _clean_target_text("[OPTION radio] «Смартфоны» ⟨путь: Электроника⟩") == "Смартфоны"


def test_resolve_falls_back_to_index_when_text_is_paraphrased():
    state = tree_state()
    target = resolve({"action": "click", "target_index": 3, "target_text": "мобильные телефоны"}, state)
    assert target.index == 3


def test_resolve_returns_none_for_hallucinated_target():
    state = tree_state()
    assert resolve({"action": "click", "target_index": 99, "target_text": "Холодильники"}, state) is None


# --------------------------------------------------------------------- память

def test_memory_verifies_effects_and_forbids_repeats():
    mem = TaskMemory()
    mem.reset("task-1")
    before = page(el(0, "Электроника", ElementKind.FOLDER, state=FolderState.CLOSED), state_hash="a")
    mem.expect(ActionType.OPEN, before.elements[0], before)
    after = page(
        el(0, "Электроника", ElementKind.FOLDER, state=FolderState.OPEN),
        el(1, "Смартфоны", path=("Электроника",)),
        state_hash="b",
    )
    mem.verify(after)
    assert mem.history[-1].result.startswith("✓ раскрыто, появилось 1: «Смартфоны»")

    option = after.elements[1]
    for _ in range(2):                                   # два клика без эффекта
        mem.expect(ActionType.CLICK, option, after)
        mem.verify(after)
    assert mem.history[-1].result == "✗ без видимого эффекта"
    assert mem.is_forbidden(ActionType.CLICK, option.key)
    assert any("Смартфоны" in line for line in mem.forbidden_lines())

    # ключ, а не индекс: тот же элемент с другим индексом остаётся под запретом
    shifted = page(el(0, "Одежда", ElementKind.FOLDER), el(1, "Электроника", ElementKind.FOLDER),
                   el(2, "Смартфоны", path=("Электроника",)))
    assert mem.is_forbidden(ActionType.CLICK, shifted.elements[2].key)
    assert not mem.is_forbidden(ActionType.CLICK, shifted.elements[0].key)


def test_memory_reset_on_new_task():
    mem = TaskMemory()
    mem.reset("task-1")
    mem.add(ActionType.CLICK, el(0, "x"), result="✓")
    mem.no_effect[(ActionType.CLICK, "row||x")] = 5
    mem.reset("task-2")
    assert mem.task_id == "task-2" and not mem.history and not mem.no_effect


# --------------------------------------------------------------------- отпечаток задания

def _fp(text: str, image: str = "img.png", notices: tuple[str, ...] = ()) -> tuple[str, str, str, str]:
    reader = [text] + [f"\x01N{i}\x01" for i in range(len(notices))]
    images = [MediaImage(n=1, src=image)] if image else []
    return DomParser._fingerprint("https://x/task", reader, [], images, [], [])


@pytest.mark.parametrize("tick_a, tick_b", [
    ("Осталось 04:59", "Осталось 04:58"),
    ("осталось 59 сек.", "осталось 58 сек."),
    ("Таймер: 3 мин", "Таймер: 2 мин"),
])
def test_fingerprint_ignores_timers(tick_a, tick_b):
    a = _fp(f"Выберите категорию. {tick_a}")
    b = _fp(f"Выберите категорию. {tick_b}")
    assert a[0] == b[0] and a[2] == b[2]


def test_fingerprint_changes_with_task_content():
    a = _fp("Товар №12345")
    b = _fp("Товар №12346")
    c = _fp("Товар №12345", image="other.png")
    assert len({a[0], b[0], c[0]}) == 3
    assert a[2] != b[2] and a[3] == b[3]         # отличаются только цифры → «мягкий» хэш совпадает
    assert a[2] == c[2]                           # тот же текст, другое фото → тот же хэш текста


def test_fingerprint_ignores_platform_messages():
    """«Неверный ответ» и подсказка после отправки — то же задание (память не сбрасывается)."""
    plain = _fp("Оцените чистоту банкомата")
    feedback = _fp("Оцените чистоту банкомата", notices=("Неверный ответ", "Правильный ответ: Да"))
    assert plain[0] == feedback[0]


# --------------------------------------------------------------------- предохранители

class _FakeBrowser:
    def __init__(self):
        self.clicks = []

    async def click_element(self, frame, el, *, prefer_toggle=False):
        from browser_controller import ActionOutcome
        self.clicks.append((el.text, prefer_toggle))
        return ActionOutcome(ok=True, method="mouse")


def test_open_on_open_folder_is_refused_once_then_executed():
    from tests.helpers import run

    agent = Agent.__new__(Agent)
    agent._memory = TaskMemory()
    agent._browser = _FakeBrowser()
    state = tree_state()
    decision = LLMDecision(reasoning="", action="open", target_index=2, target_text="Электроника")
    target = agent._resolve_target(decision, state)
    run(agent._execute(None, decision, target, state))
    assert agent._browser.clicks == [] and "уже раскрыта" in agent._memory.history[-1].result
    run(agent._execute(None, decision, target, state))     # модель настаивает — эвристика могла ошибиться
    assert agent._browser.clicks == [("Электроника", True)]


def test_selected_checkbox_can_be_unchecked_but_radio_is_protected():
    from tests.helpers import run

    agent = Agent.__new__(Agent)
    agent._memory = TaskMemory()
    agent._browser = _FakeBrowser()
    state = page(
        el(0, "Радио", is_selected=True, choice_type="radio"),
        el(1, "Чекбокс", is_selected=True, choice_type="checkbox"),
    )
    for idx, text in ((0, "Радио"), (1, "Чекбокс")):
        decision = LLMDecision(reasoning="", action="click", target_index=idx, target_text=text)
        run(agent._execute(None, decision, agent._resolve_target(decision, state), state))
    assert agent._browser.clicks == [("Чекбокс", False)]


# --------------------------------------------------------------------- стоп-список кнопок

def test_deny_list_buttons_are_hidden_and_never_resolved():
    state = page(
        el(0, "Смартфоны"),
        el(1, "Завершить смену", ElementKind.BUTTON),
        el(2, "Выйти", ElementKind.BUTTON),
        el(3, "Выходная обувь"),                        # категория, а не кнопка — остаётся
        el(4, "Сменить категорию", ElementKind.BUTTON),  # «смен» раньше задевало и её
        el(5, "Начать работу", ElementKind.BUTTON),      # «работу» раньше задевало и её
    )
    prompt = build_elements_prompt(state.elements)
    assert "Завершить смену" not in prompt and "«Выйти»" not in prompt
    message = oc.LLMConnector.build_user_message(state, DecisionContext())
    assert "Завершить смену" not in message and "«Выйти»" not in message
    assert "Выходная обувь" in prompt and "Сменить категорию" in prompt and "Начать работу" in prompt
    # ни по номеру, ни по тексту, ни по нечёткому совпадению «Завершить» ≈ «Завершить смену»
    assert resolve({"action": "click", "target_index": 1, "target_text": "Завершить смену"}, state) is None
    assert resolve({"action": "click", "target_index": None, "target_text": "Завершить"}, state) is None


def test_execute_refuses_denied_button_even_if_resolved():
    from tests.helpers import run

    agent = Agent.__new__(Agent)
    agent._memory = TaskMemory()
    agent._browser = _FakeBrowser()
    state = page(el(0, "Завершить смену", ElementKind.BUTTON))
    decision = LLMDecision(reasoning="", action="click", target_index=0)
    run(agent._execute(None, decision, state.elements[0], state))
    assert agent._browser.clicks == []
    assert "стоп-списка" in agent._memory.history[-1].result
