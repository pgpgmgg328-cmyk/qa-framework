"""models.py v3 — Pydantic v2 схемы данных агента.

Изменения относительно v2:
- ParsedElement получил стабильные идентификаторы: `uid` (метка data-agent-id в DOM,
  по ней кликает BrowserController) и `key` (ключ «путь + текст» для памяти агента,
  переживает смену индексов при раскрытии папок).
- Добавлены path/value/input_type/container/occluded — контекст для LLM.
- Легенда типов для системного промпта генерируется из тех же констант, что и
  prompt_line(), поэтому формат строк и описание в промпте больше не расходятся
  (в v2 промпт описывал «[FOLDER ЗАКРЫТА]», а в списке было «[ПАПКА ЗАКРЫТА]»).
- LLMDecision терпим к «грязному» JSON: строки вместо чисел, confidence=95,
  синонимы действий — вместо падения шага с ValidationError.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Тип действия
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Все доступные действия агента."""
    CLICK  = "click"   # выбрать вариант (OPTION) / нажать кнопку (BUTTON)
    OPEN   = "open"    # раскрыть папку (FOLDER) или выпадающий список (DROPDOWN)
    SUBMIT = "submit"  # отправить ответ (нажать «Завершить/Отправить»)
    TYPE   = "type"    # ввести текст в поле / выбрать пункт нативного select
    SCROLL = "scroll"  # прокрутить список (виртуальный скролл / ленивая подгрузка)
    SKIP   = "skip"    # ничего не делать на этом шаге (идёт загрузка)


# Синонимы, которые модели иногда возвращают вместо канонических значений
_ACTION_SYNONYMS: dict[str, ActionType] = {
    "select": ActionType.CLICK, "choose": ActionType.CLICK, "press": ActionType.CLICK,
    "expand": ActionType.OPEN, "unfold": ActionType.OPEN,
    "input": ActionType.TYPE, "fill": ActionType.TYPE, "write": ActionType.TYPE,
    "finish": ActionType.SUBMIT, "send": ActionType.SUBMIT, "complete": ActionType.SUBMIT,
    "wait": ActionType.SKIP, "none": ActionType.SKIP, "noop": ActionType.SKIP,
}


# ---------------------------------------------------------------------------
# Тип элемента
# ---------------------------------------------------------------------------

class ElementKind(str, Enum):
    """Семантический тип интерактивного элемента."""
    FOLDER   = "FOLDER"    # папка / ветка дерева — её можно раскрыть
    OPTION   = "OPTION"    # конечный вариант — кликнуть для выбора
    DROPDOWN = "DROPDOWN"  # выпадающий список (tui-select, combobox)
    INPUT    = "INPUT"     # поле ввода (включая нативный select)
    BUTTON   = "BUTTON"    # кнопка действия (submit, ok и т.д.)
    OTHER    = "OTHER"     # прочее кликабельное


class FolderState(str, Enum):
    """Состояние папки / выпадающего списка."""
    OPEN   = "OPEN"    # раскрыта
    CLOSED = "CLOSED"  # закрыта
    NA     = "NA"      # не применимо


# ---------------------------------------------------------------------------
# Единый словарь обозначений для prompt_line() и системного промпта
# ---------------------------------------------------------------------------

TAG_FOLDER_CLOSED   = "[FOLDER закрыта]"
TAG_FOLDER_OPEN     = "[FOLDER раскрыта]"
TAG_DROPDOWN_CLOSED = "[DROPDOWN закрыт]"
TAG_DROPDOWN_OPEN   = "[DROPDOWN открыт]"
TAG_OPTION          = "[OPTION]"
TAG_BUTTON          = "[BUTTON]"
TAG_OTHER           = "[OTHER]"

FLAG_SELECTED   = "✓ВЫБРАН"
FLAG_DISABLED   = "⛔НЕАКТИВЕН"
FLAG_OCCLUDED   = "⚠ПЕРЕКРЫТ"
FLAG_SELECTABLE = "(можно выбрать)"
FLAG_IN_DIALOG  = "(в диалоге)"
FLAG_IN_POPUP   = "(во всплывающем списке)"

PROMPT_LEGEND = f"""\
  {TAG_FOLDER_CLOSED}  — ветка дерева свёрнута; action "open" раскроет её содержимое.
  {TAG_FOLDER_OPEN} — ветка раскрыта; её содержимое — строки НИЖЕ с бо́льшим отступом.
  {TAG_OPTION} / [OPTION radio] / [OPTION checkbox] — конечный вариант; action "click" выбирает его
                     (radio — один вариант из группы, checkbox — можно отметить несколько).
  {TAG_DROPDOWN_CLOSED} / {TAG_DROPDOWN_OPEN} — выпадающий список; "open" открывает его,
                     затем "click" по варианту с пометкой {FLAG_IN_POPUP}.
  [INPUT text] / [INPUT textarea] / [INPUT number] / [INPUT select] … — поле ввода; action "type"
                     (для [INPUT select] type_text = точный текст одного из вариантов).
  {TAG_BUTTON}          — кнопка; "click". Кнопку отправки ответа нажимай через action "submit".
  {TAG_OTHER}           — прочий кликабельный элемент."""

PROMPT_FLAGS = f"""\
  {FLAG_SELECTED}      — элемент уже выбран/отмечен.
  {FLAG_DISABLED}   — нажать нельзя (для кнопки отправки: ответ ещё не заполнен).
  {FLAG_OCCLUDED}     — сейчас закрыт другим элементом (оверлей, спиннер, диалог).
  {FLAG_SELECTABLE}  — у папки есть собственный переключатель: её саму можно выбрать как ответ.
  {FLAG_IN_DIALOG} / {FLAG_IN_POPUP} — элемент модального окна / открытого выпадающего списка.
  ⟨путь: A › B⟩     — родительские папки; показывается у элементов с одинаковым текстом."""


def normalize_text(value: Any) -> str:
    """Нормализация текста для сравнения: регистр, ё, кавычки, пробелы."""
    text = str(value or "")
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"[«»\"“”„']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip(".:;,").strip().lower()


# ---------------------------------------------------------------------------
# Парсер — единица элемента
# ---------------------------------------------------------------------------

class ParsedElement(BaseModel):
    """Один интерактивный элемент страницы (одна «строка» для LLM)."""

    index:        int          = Field(...,  description="Номер в текущем снимке (для LLM)")
    uid:          str          = Field("",   description="Метка data-agent-id в DOM (для клика)")
    key:          str          = Field("",   description="Стабильный ключ «группа|путь|текст» для памяти")
    kind:         ElementKind  = Field(...,  description="Семантический тип")
    folder_state: FolderState  = Field(FolderState.NA, description="Состояние папки/списка")
    tag:          str          = Field("",   description="HTML-тег")
    text:         str          = Field("",   description="Видимый текст (без текста вложенных строк)")
    is_selected:  bool         = Field(False, description="Выбран?")
    is_disabled:  bool         = Field(False, description="Заблокирован?")
    depth:        int          = Field(0,    description="Глубина вложенности в дереве")
    path:         list[str]    = Field(default_factory=list, description="Родительские папки")
    placeholder:  str          = Field("",   description="placeholder / подпись поля")
    value:        str          = Field("",   description="Текущее значение поля ввода")
    input_type:   str          = Field("",   description="text/textarea/number/select/…")
    choice_type:  str          = Field("",   description="radio/checkbox для OPTION")
    options:      list[str]    = Field(default_factory=list, description="Варианты нативного select")
    container:    str          = Field("",   description="dialog/popup/''")
    selectable:   bool         = Field(False, description="Папку можно выбрать как ответ")
    has_toggle:   bool         = Field(False, description="Есть отдельная кнопка-стрелка раскрытия")
    has_select:   bool         = Field(False, description="Есть видимый radio/checkbox для выбора")
    in_viewport:  bool         = Field(True, description="Сейчас в видимой области фрейма")
    occluded:     bool         = Field(False, description="Центр элемента перекрыт другим элементом")

    # --- представление для LLM -------------------------------------------

    def type_tag(self) -> str:
        if self.kind == ElementKind.FOLDER:
            return TAG_FOLDER_OPEN if self.folder_state == FolderState.OPEN else TAG_FOLDER_CLOSED
        if self.kind == ElementKind.DROPDOWN:
            return TAG_DROPDOWN_OPEN if self.folder_state == FolderState.OPEN else TAG_DROPDOWN_CLOSED
        if self.kind == ElementKind.INPUT:
            return f"[INPUT {self.input_type or 'text'}]"
        if self.kind == ElementKind.OPTION:
            return f"[OPTION {self.choice_type}]" if self.choice_type else TAG_OPTION
        if self.kind == ElementKind.BUTTON:
            return TAG_BUTTON
        return TAG_OTHER

    def label(self) -> str:
        """Человекочитаемая подпись для логов и истории."""
        return self.text or self.placeholder or f"#{self.index}"

    def prompt_line(self, *, show_path: bool = False) -> str:
        """Одна строка для LLM-промпта. Формат описан в PROMPT_LEGEND/PROMPT_FLAGS."""
        indent = "  " * min(self.depth, 8)
        parts = [f"{indent}[{self.index}] {self.type_tag()}"]

        text = self.text or self.placeholder
        if text:
            parts.append(f"«{text}»")

        if self.kind == ElementKind.INPUT:
            parts.append(f"значение=«{self.value[:80]}»" if self.value else "(пусто)")
            if self.options:
                parts.append("варианты: " + " | ".join(self.options[:20]))
        elif self.kind == ElementKind.DROPDOWN and self.value:
            parts.append(f"значение=«{self.value[:80]}»")

        if self.is_selected:
            parts.append(FLAG_SELECTED)
        if self.kind == ElementKind.FOLDER and self.selectable and not self.is_selected:
            parts.append(FLAG_SELECTABLE)
        if self.is_disabled:
            parts.append(FLAG_DISABLED)
        if self.occluded:
            parts.append(FLAG_OCCLUDED)
        if self.container == "dialog":
            parts.append(FLAG_IN_DIALOG)
        elif self.container == "popup":
            parts.append(FLAG_IN_POPUP)
        if show_path and self.path:
            parts.append(f"⟨путь: {' › '.join(self.path)}⟩")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Состояние страницы
# ---------------------------------------------------------------------------

class PageState(BaseModel):
    """Атомарный снимок фрейма (собран одним evaluate)."""

    task_text:       str                 = Field("", description="Текст задания")
    hint_text:       str                 = Field("", description="Подсказка / инструкция")
    elements:        list[ParsedElement] = Field(default_factory=list)
    task_identifier: str                 = Field("", description="Отпечаток задания (не меняется при раскрытии папок)")
    task_preview:    str                 = Field("", description="Короткое описание задания для логов")
    state_hash:      str                 = Field("", description="Хэш интерактивного состояния (для проверки эффекта)")
    has_captcha:     bool                = Field(False)
    loading:         bool                = Field(False, description="Виден спиннер/лоадер")
    alerts:          list[str]           = Field(default_factory=list, description="Ошибки/уведомления страницы")
    overlay_text:    str                 = Field("", description="Текст оверлея, перекрывающего элементы")
    scroll_hints:    list[str]           = Field(default_factory=list)
    image_src:       str                 = Field("", description="src главной картинки задания")
    generation:      str                 = Field("", description="Поколение меток data-agent-id")
    frame_url:       str                 = Field("")
    total_elements:  int                 = Field(0)
    hidden_elements: int                 = Field(0, description="Совпавших по селектору, но невидимых")

    def by_index(self, index: Optional[int]) -> Optional[ParsedElement]:
        if index is None:
            return None
        for el in self.elements:
            if el.index == index:
                return el
        return None

    def by_key(self, key: Optional[str]) -> Optional[ParsedElement]:
        if not key:
            return None
        for el in self.elements:
            if el.key == key:
                return el
        return None

    @property
    def keys(self) -> set[str]:
        return {el.key for el in self.elements}


# ---------------------------------------------------------------------------
# Решение LLM
# ---------------------------------------------------------------------------

class LLMDecision(BaseModel):
    """Решение, принятое LLM (устойчиво к «грязным» типам в JSON)."""

    goal:             str           = Field("", description="Что требуется в задании")
    observation:      str           = Field("", description="Что уже сделано / релевантные элементы")
    reasoning:        str           = Field("", description="Почему выбрано это действие")
    action:           ActionType    = Field(..., description="Действие")
    target_index:     Optional[int] = Field(None, description="Индекс целевого элемента")
    target_text:      Optional[str] = Field(
        None,
        description=(
            "Точный видимый текст элемента. "
            "Python-скрипт использует его для проверки и автокоррекции target_index."
        ),
    )
    type_text:        Optional[str] = Field(None, description="Текст для ввода")
    scroll_direction: Optional[str] = Field(None, description="down/up для action=scroll")
    confidence:       float         = Field(1.0, ge=0.0, le=1.0)

    @field_validator("action", mode="before")
    @classmethod
    def _coerce_action(cls, value: Any) -> Any:
        if isinstance(value, ActionType):
            return value
        # str(Enum) в Python 3.11 даёт 'ActionType.OPEN', поэтому берём .value
        raw = str(value.value if isinstance(value, Enum) else (value or "")).strip().lower()
        if raw in ActionType._value2member_map_:
            return raw
        if raw in _ACTION_SYNONYMS:
            return _ACTION_SYNONYMS[raw]
        raise ValueError(f"неизвестное действие {value!r}")

    @field_validator("target_index", mode="before")
    @classmethod
    def _coerce_index(cls, value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, str):
            match = re.search(r"-?\d+", value)
            if not match:
                return None
            value = match.group()
        try:
            index = int(float(value))
        except (TypeError, ValueError):
            return None
        return index if index >= 0 else None

    @field_validator("target_text", "type_text", "scroll_direction", mode="before")
    @classmethod
    def _coerce_optional_str(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("goal", "observation", "reasoning", mode="before")
    @classmethod
    def _coerce_str(cls, value: Any) -> Any:
        return "" if value is None else str(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: Any) -> Any:
        try:
            conf = float(str(value).strip().rstrip("%"))
        except (TypeError, ValueError):
            return 0.5
        if conf > 1.0:
            conf = conf / 100.0  # модель ответила в процентах
        return min(max(conf, 0.0), 1.0)

    @classmethod
    def skip(cls, reason: str) -> "LLMDecision":
        return cls(reasoning=reason, action=ActionType.SKIP, confidence=0.0)


# ---------------------------------------------------------------------------
# Контекст решения: история, запреты, бюджет — то, чего не было в v2
# ---------------------------------------------------------------------------

@dataclass
class DecisionContext:
    """Всё, что LLM должна знать помимо снимка страницы."""

    history:     list[str] = field(default_factory=list)   # «open «Электроника» → ✓ раскрыта …»
    forbidden:   list[str] = field(default_factory=list)   # действия, не давшие эффекта
    notes:       list[str] = field(default_factory=list)   # предупреждения агента (бюджет, петли)
    step_in_task: int = 1
    steps_left:  int = 0
    image_b64:   Optional[str] = None                      # скриншот картинки задания (vision)
    temperature: Optional[float] = None                    # повышается при зацикливании
