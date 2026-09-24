"""models.py — Pydantic v2 схемы данных агента."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Тип действия
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Все доступные действия агента."""
    CLICK  = "click"   # выбрать конечный элемент (OPTION)
    OPEN   = "open"    # раскрыть папку (FOLDER/DROPDOWN в состоянии CLOSED)
    SUBMIT = "submit"  # отправить форму (нажать «завершить»)
    TYPE   = "type"    # ввести текст в поле
    SKIP   = "skip"    # пропустить задание


# ---------------------------------------------------------------------------
# Тип элемента
# ---------------------------------------------------------------------------

class ElementKind(str, Enum):
    """Семантический тип интерактивного элемента."""
    FOLDER  = "FOLDER"   # папка / ветка дерева — её нужно развернуть
    OPTION  = "OPTION"   # конечный вариант — кликнуть для выбора
    INPUT   = "INPUT"    # поле ввода — вводить текст
    BUTTON  = "BUTTON"   # кнопка действия (submit, ok и т.д.)
    OTHER   = "OTHER"    # прочее


class FolderState(str, Enum):
    """Состояние папки/ветки."""
    OPEN   = "OPEN"    # раскрыта
    CLOSED = "CLOSED"  # закрыта
    NA     = "NA"      # не применимо


# ---------------------------------------------------------------------------
# Парсер — единица элемента
# ---------------------------------------------------------------------------

class ParsedElement(BaseModel):
    """Oдин интерактивный элемент страницы."""

    index:       int         = Field(...,  description="Порядковый номер")
    kind:        ElementKind = Field(...,  description="Семантический тип")
    folder_state: FolderState = Field(FolderState.NA, description="Состояние папки")
    tag:         str         = Field("",   description="HTML-тег")
    text:        str         = Field("",   description="Видимый текст")
    is_selected: bool        = Field(False, description="Выбран?")
    is_disabled: bool        = Field(False, description="Заблокирован?")
    depth:       int         = Field(0,    description="Глубина вложенности в дереве")
    placeholder: str         = Field("",   description="placeholder")

    def prompt_line(self) -> str:
        """Одна строка для LLM-промпта, максимально информативна."""
        indent = "  " * self.depth

        # Статус папки
        if self.kind == ElementKind.FOLDER:
            if self.folder_state == FolderState.OPEN:
                state_tag = "[ПАПКА РАСКРЫТА]"
            else:
                state_tag = "[ПАПКА ЗАКРЫТА]"
        else:
            state_tag = f"[{self.kind.value}]"

        parts = [f"{indent}[{self.index}] {state_tag}"]

        text = self.text or self.placeholder
        if text:
            parts.append(f'"{text}"')

        if self.is_selected:
            parts.append("✓Выбран")
        if self.is_disabled:
            parts.append("❌Заблокирован")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Состояние страницы
# ---------------------------------------------------------------------------

class PageState(BaseModel):
    """Cнимок фрейма в данный момент."""

    task_text:       str                  = Field("", description="Текст задания")
    hint_text:       str                  = Field("", description="Подсказка")
    elements:        list[ParsedElement]  = Field(default_factory=list)
    task_identifier: str                  = Field("", description="Уникальный ID задания")
    has_captcha:     bool                 = Field(False)


# ---------------------------------------------------------------------------
# Решение LLM
# ---------------------------------------------------------------------------

class LLMDecision(BaseModel):
    """Rешение, принятое LLM."""

    reasoning:    str        = Field(..., description="Цепочка рассуждений")
    action:       ActionType = Field(..., description="Действие")
    target_index: Optional[int] = Field(None, description="Индекс целевого элемента")
    target_text:  Optional[str] = Field(
        None,
        description=(
            "Точный видимый текст элемента. "
            "Python-скрипт использует его для автокоррекции target_index."
        ),
    )
    type_text:    Optional[str] = Field(None, description="Текст для ввода")
    confidence:   float         = Field(1.0, ge=0.0, le=1.0)
