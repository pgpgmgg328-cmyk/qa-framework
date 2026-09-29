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
    item = schema["properties"]["actions"]["items"]
    assert set(item["required"]) == set(item["properties"]) and item["additionalProperties"] is False
    assert item["properties"]["action"]["enum"] == [a.value for a in ActionType]
    # порядок полей: рассуждения генерируются ДО действий
    keys = list(schema["properties"])
    assert keys.index("observation") < keys.index("plan") < keys.index("reasoning") < keys.index("actions")


def test_parse_batch_and_value_mapping():
    raw = ('{"observation": "о", "plan": "п", "reasoning": "р", "confidence": 0.8, "actions": ['
           '{"action": "type", "target_index": 2, "target_text": "URL", "value": "https://x.example/1"},'
           '{"action": "web", "target_index": null, "target_text": null, "value": "кафе Сочи"},'
           '{"action": "scroll", "target_index": null, "target_text": null, "value": "down"}]}')
    d = oc.LLMConnector.parse_response(raw)
    steps = d.steps()
    assert [s.action for s in steps] == [ActionType.TYPE, ActionType.WEB, ActionType.SCROLL]
    assert steps[0].type_text == "https://x.example/1" and steps[1].query == "кафе Сочи"
    assert steps[2].scroll_direction == "down" and d.plan == "п"
    assert oc.LLMConnector.parse_response('{"actions": []}').action == ActionType.SKIP
    with pytest.raises(oc.DecisionParseError):
        oc.LLMConnector.parse_response('{"actions": [{"action": "click"}, {"action": "dance"}]}')


def test_batch_order_rules():
    def step(action, text=""):
        return LLMDecision(action=action, target_text=text or None)

    submit, a, b = step("submit", "Завершить"), step("click", "Да"), step("click", "Нет")
    assert Agent._order_batch([submit, a, b]) == [a, b, submit]            # submit — последним
    opened = step("open", "Цвет")
    assert Agent._order_batch([a, opened, b, submit]) == [a, opened]        # после open — новый взгляд
    assert Agent._order_batch([step("skip")]) == [step("skip")]
    assert Agent._order_batch([step("skip"), a]) == [a]


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

def _fp(text: str, image: str = "img.png", notices: tuple[str, ...] = ()) -> tuple[str, ...]:
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


def test_page_hints_follow_the_page():
    """Советы, вынесенные из системного промпта, приходят на страницах, где они нужны."""
    from models import MediaAudio

    def hints(reader, *elements, context=None, **kw):
        state = page(*elements)
        state.reader = list(reader)
        for name, value in kw.items():
            setattr(state, name, value)
        return {k for k, text in oc.HINTS.items() if text in oc.page_hints(state, context or DecisionContext())}

    radio = dict(choice_type="radio")
    hotels = hints(["Название отеля: Магнолия", "#### Отели совпадают?"],
                   el(0, "Да", **radio), el(1, "Нет", **radio), el(2, "Завершить задание", ElementKind.BUTTON))
    assert hotels == {"compare"}
    assert hints(["Оцените фото"], el(0, "Хорошо", **radio), images=[MediaImage(n=1, src="a.jpg")]) == {"photos"}
    assert hints(["Прослушайте звонок"], el(0, "Робот", **radio), audios=[MediaAudio(n=1, src="a.mp3")]) == {"audio"}
    color = el(0, "Выберите значение", ElementKind.DROPDOWN, caption="Цвет")
    assert hints(["Товар: футболка"], color) == {"dropdown"}
    assert hints(["Товар: футболка"], color, popup_lines=["Белый", "Другое"]) == {"dropdown", "special"}
    # меню кнопки — не список значений ответа
    assert hints(["Товар"], el(0, "Опции завершения задания", ElementKind.DROPDOWN)) == set()
    assert hints(["Выберите категорию"], el(0, "Одежда", ElementKind.FOLDER, state=FolderState.CLOSED)) == {"tree"}
    assert "web" in hints(["Найдите организацию на Яндекс Картах"], el(0, "Ссылка", ElementKind.INPUT))
    assert "web" not in hints(["Нельзя использовать поиск в интернете"], el(0, "Да", **radio))
    assert "web" in hints(["Оцените фото"], el(0, "Да", **radio), context=DecisionContext(research=["…"]))
    # текст страницы и подписи элементов не склеиваются в одну строку
    assert "web" in hints(["Ответ изменить нельзя"], el(0, "Поиск по каталогу", ElementKind.BUTTON))


def test_missing_key_message_names_the_file_problem(tmp_path, monkeypatch):
    """Ключ не прочитан — сообщение говорит, что именно не так с файлом .env."""
    import sys

    import config

    monkeypatch.setattr(config, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(sys, "platform", "win32")
    example = tmp_path / ".env.example"
    example.write_text("# образец\nOPENROUTER_API_KEY=\nLLM_MODEL=gpt-4o\n", encoding="utf-8")
    assert "copy .env.example .env" in config._missing_key_hint()
    example.write_text("# образец\nOPENROUTER_API_KEY=sk-test\n", encoding="utf-8")   # ключ вписан в образец
    with pytest.raises(ValueError, match=r"ren \.env\.example \.env"):
        config.validate_config()
    (tmp_path / ".env.txt").write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")  # Блокнот добавил .txt
    assert "ren .env.txt .env" in config._missing_key_hint()
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=\n", encoding="utf-8")
    hint = config._missing_key_hint()
    assert "впишите ключ OpenRouter в файл" in hint and ".env.example" in hint
    example.write_text("# старый образец\nOPENAI_API_KEY=sk-test\n", encoding="utf-8")   # прежнее имя ключа
    assert "ключ вписан в .env.example" in config._missing_key_hint()
    config.validate_config(require_llm=False)                                    # режим записи — без ключа


def test_exam_mode_and_outcome_are_recognized():
    from agent import _exam_outcome

    exam = PageState(reader=["#### Экзамен", "1 из 2 заданий", "### Оценка качества выполненного клининга"])
    training = PageState(reader=["#### Тренировка", "3 из 14 заданий"])
    assert Agent._page_mode(exam) == "exam" and Agent._page_mode(training) == "training"
    assert Agent._page_mode(PageState(reader=["### Выполните задание"])) == ""
    assert _exam_outcome("Экзамен не пройден. Ошибок: 3") is False
    assert _exam_outcome("Поздравляем! Экзамен сдан") is True
    assert _exam_outcome("Тренировка завершена. Нажмите «Начать»") is None


def test_inspection_tasks_get_single_photos_and_checklist_hint():
    from media import MediaManager

    photos = [MediaImage(n=i, src=f"https://x/{i}.jpg") for i in range(1, 13)]
    cleaning = PageState(pool_title="Оценка качества выполненного клининга", images=photos,
                         reader=["Оцени чистоту банкомата по фото."])
    hotels = PageState(pool_title="Отели совпадают?", images=photos, reader=["Сравни фото отелей."])
    jobs, captions = MediaManager._plan_jobs(cleaning, photos)
    assert len(jobs) == 12 and all(j["kind"] == "single" for j in jobs)       # каждое фото отдельно
    jobs, _ = MediaManager._plan_jobs(hotels, photos)
    assert len(jobs) == 3 and all(j["kind"] == "grid" for j in jobs)          # сравнение — коллажами
    hint = oc.HINTS["inspection"]
    assert hint in oc.page_hints(cleaning, DecisionContext()) and oc.HINTS["photos"] not in oc.page_hints(
        cleaning, DecisionContext())
    assert "ФОТО n" in hint and "у основания" in hint


def test_only_last_web_page_is_shown_in_full(tmp_path):
    """В запрос к модели полностью идёт только последняя открытая страница поиска."""
    from knowledge import KnowledgeBase
    from media import MediaManager
    from research import WebResult
    from tests.helpers import run

    agent = Agent.__new__(Agent)
    agent._memory = TaskMemory()
    agent._knowledge = KnowledgeBase(str(tmp_path))
    agent._pool = None
    agent._media = MediaManager(None, None)
    agent._memory.web_results = [
        WebResult(query="первый", url="https://yandex.ru/search/?text=a", title="Поиск", text="ТЕКСТ-1"),
        WebResult(query="второй", url="https://yandex.ru/maps/org/x/1/", title="Карточка", text="ТЕКСТ-2"),
    ]
    context = run(agent._build_context(None, PageState()))
    assert "ТЕКСТ-1" not in context.research[0] and "https://yandex.ru/search/?text=a" in context.research[0]
    assert "ТЕКСТ-2" in context.research[1]


class _NoLLM:
    """Заглушка модели: для правил лестницы и экзамена вызовы модели не нужны."""


def test_exam_gate_and_ladder_decide_early_on_few_training_tasks(tmp_path, monkeypatch):
    """Порог 80% из 3 заданий: одна ошибка в первых заданиях — порог уже недостижим (экзамен
    не начинать / перейти к следующей модели); без ошибок — ждать данных."""
    import agent as agent_module
    from knowledge import KnowledgeBase

    kb = KnowledgeBase(str(tmp_path))
    agent = Agent(llm=_NoLLM(), knowledge=kb)
    agent._pool = kb.for_state(PageState(pool_key="k1", pool_title="Клининг", pool_signature=["h:клининг"]))

    agent._order_train = [1, 1]
    assert agent._training_accuracy() is None                       # 1 из 1: 3 из 3 ещё возможно
    agent._order_train = [1, 2]
    assert agent._training_accuracy() == (1, 2)                     # максимум 2 из 3 < 80%
    agent._order_train = [4, 5]
    assert agent._training_accuracy() == (4, 5)

    monkeypatch.setattr(agent_module, "LLM_MODELS", ("cheap", "mid", "strong"))
    pool = agent._pool
    kb.model_attempt(pool, "cheap", True)
    agent._climb_ladder(pool, "cheap")
    assert agent._model_for(pool) == "cheap"
    kb.model_attempt(pool, "cheap", False)
    agent._climb_ladder(pool, "cheap")
    assert agent._model_for(pool) == "mid" and agent._training_accuracy() is None   # у mid данных нет
    agent._climb_ladder(pool, "mid", reason="экзамен не пройден")
    agent._climb_ladder(pool, "strong", reason="экзамен не пройден")          # выше некуда
    assert agent._model_for(pool) == "strong"
    monkeypatch.setattr(agent_module, "LLM_MODELS", ("other-cheap", "other-strong"))
    assert agent._model_for(pool) == "other-cheap"                  # лестницу в .env поменяли


def test_ladder_returns_to_the_more_accurate_cheaper_model(tmp_path, monkeypatch):
    """Лог пользователя: Gemini ошиблась 1 раз из 4, вид перешёл к gpt-5.4-mini, а та ошибалась
    чаще. Выше идти некуда — агент возвращает модель, которая была точнее (и дешевле), и дальше
    между ними не прыгает."""
    import agent as agent_module
    from knowledge import KnowledgeBase

    kb = KnowledgeBase(str(tmp_path))
    agent = Agent(llm=_NoLLM(), knowledge=kb)
    pool = kb.for_state(PageState(pool_key="k2", pool_title="Звонок", pool_signature=["h:звонок"]))
    monkeypatch.setattr(agent_module, "LLM_MODELS", ("gemini", "gpt"))
    for ok in (True, True, True, False):
        kb.model_attempt(pool, "gemini", ok)
        agent._climb_ladder(pool, "gemini")
    assert agent._model_for(pool) == "gpt"
    kb.model_attempt(pool, "gpt", False)
    agent._climb_ladder(pool, "gpt")
    assert agent._model_for(pool) == "gemini"                         # 3 из 4 точнее, чем 0 из 1
    kb.model_attempt(pool, "gemini", False)
    agent._climb_ladder(pool, "gemini")
    assert agent._model_for(pool) == "gemini"                         # не прыгает обратно к gpt
    assert "- текущая модель: gemini" in pool.path.read_text(encoding="utf-8")


def test_catcher_offers_the_largest_pdf_first():
    """Крошечный PDF (1 КБ — заглушка) модель на T-Work не прочитала; настоящая инструкция — большая."""
    from documents import CaughtDocument, DocumentCatcher

    class Context:
        def on(self, event, handler):
            pass

    catcher = DocumentCatcher(Context())
    catcher.documents = [CaughtDocument("https://x/stub.pdf", "application/pdf", b"%PDF-1.4 stub"),
                         CaughtDocument("https://x/big.pdf", "application/pdf", b"%PDF-1.7 " + b"x" * 5000)]
    assert [d.url for d in catcher.pdfs()] == ["https://x/big.pdf", "https://x/stub.pdf"]


def test_order_summary_tells_how_long_the_key_limit_lasts():
    class KeyLLM:
        def budget_left(self):
            return 10.0

    agent = Agent(llm=KeyLLM())
    assert agent._budget_note(0.2) == "; на ключе OpenRouter осталось ≈ $10.00 — примерно на 50 таких заказов"
    assert agent._budget_note() == "; на ключе OpenRouter осталось ≈ $10.00"
    assert Agent(llm=_NoLLM())._budget_note(0.2) == ""                  # лимит ключа не задан
