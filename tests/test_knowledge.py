"""База знаний, идентичность задания, адреса поиска — без браузера."""

from __future__ import annotations

from agent import TaskIdentity
from knowledge import KnowledgeBase
from models import PageState
from research import WebResult, to_url


def pool_state(key: str = "abc123def456", title: str = "Отели совпадают?", sig=("h:отели совпадают?", "f:да", "f:нет")):
    return PageState(pool_key=key, pool_title=title, pool_signature=list(sig))


def test_knowledge_roundtrip_keeps_user_notes(tmp_path):
    kb = KnowledgeBase(str(tmp_path))
    pool = kb.for_state(pool_state())
    kb.save_instruction(pool, "Основные поля — название и адрес. " * 5, "диалог «Инструкция»")
    kb.save_instruction(pool, "Страница 2: телефоны не решают совпадение. " * 3, "диалог", append=True)
    kb.save_tooltips(pool, {"Да": "Один и тот же объект"})
    kb.add_lesson(pool, "Ответ «Нет» — неверно. Подсказка: совпадают адреса.")
    kb.add_lesson(pool, "ответ «нет» — неверно.  подсказка: совпадают адреса")      # дубль
    path = next(tmp_path.glob("*.md"))
    # пользователь дописал свои правила
    path.write_text(path.read_text(encoding="utf-8").replace("## Заметки\n", "## Заметки\nТелефоны не сравнивать.\n"),
                    encoding="utf-8")

    fresh = KnowledgeBase(str(tmp_path))
    again = fresh.for_state(pool_state())
    assert "название и адрес" in again.instruction and "Страница 2" in again.instruction
    assert again.tooltips == {"Да": "Один и тот же объект"}
    assert len(again.lessons) == 1 and again.notes == "Телефоны не сравнивать."
    text = fresh.prompt_text(again)
    assert text.index("Заметки пользователя") < text.index("Уроки") < text.index("Инструкция к заданиям")
    # запись агента не затирает заметки пользователя
    fresh.add_lesson(again, "Новый урок")
    assert "Телефоны не сравнивать." in path.read_text(encoding="utf-8")


def test_similar_form_reuses_knowledge(tmp_path):
    kb = KnowledgeBase(str(tmp_path))
    pool = kb.for_state(pool_state())
    kb.save_instruction(pool, "Правило " * 30, "диалог")
    other = KnowledgeBase(str(tmp_path)).for_state(
        pool_state(key="ffff00001111", sig=("h:отели совпадают?", "f:да", "f:нет", "f:не знаю")))
    assert other.instruction.startswith("Правило")                      # 3 из 4 признаков совпали
    unrelated = KnowledgeBase(str(tmp_path)).for_state(pool_state(key="999", sig=("h:звонок",)))
    assert unrelated.instruction == ""


def test_general_notes_are_always_included(tmp_path):
    (tmp_path / "_general.md").write_text("Если фото не загрузилось — фото нет.", encoding="utf-8")
    kb = KnowledgeBase(str(tmp_path))
    assert "фото нет" in kb.prompt_text(None)


def test_task_identity_rules():
    base = TaskIdentity("text-1", "loose-1", frozenset())
    loading = TaskIdentity("text-1", "loose-1", frozenset({"a.jpg"}))
    loaded = TaskIdentity("text-1", "loose-1", frozenset({"a.jpg", "b.jpg"}))
    next_photo = TaskIdentity("text-1", "loose-1", frozenset({"c.jpg"}))
    counter = TaskIdentity("text-2", "loose-1", frozenset({"a.jpg", "b.jpg"}))
    other = TaskIdentity("text-3", "loose-3", frozenset())
    assert base.same_task(loading, submitted=False) and loading.same_task(loaded, submitted=False)
    assert not loaded.same_task(next_photo, submitted=True)       # тот же текст, другие фото — новое задание
    assert loaded.same_task(counter, submitted=False)             # изменились только цифры (счётчик)
    assert not loaded.same_task(counter, submitted=True)          # «2 из 14» после отправки — новое
    assert not base.same_task(other, submitted=False)


def test_search_urls():
    assert to_url("https://yandex.ru/maps/org/x/1/") == "https://yandex.ru/maps/org/x/1/"
    assert to_url("www.example.ru") == "https://www.example.ru"
    assert to_url("example.ru Санкт-Петербург").startswith("https://yandex.ru/search/?text=example.ru+")
    rendered = WebResult(query="q", url="https://yandex.ru/maps/org/x/1/?ll=1%2C2&z=9", title="X",
                         text="строка 1\nстрока 2", links=[("X", "https://x.example/")]).render(1, full=True)
    assert "Адрес без параметров: https://yandex.ru/maps/org/x/1/" in rendered
    assert "   | строка 2" in rendered and "   - X → https://x.example/" in rendered
