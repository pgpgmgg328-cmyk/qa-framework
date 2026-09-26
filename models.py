"""models.py v4 — Pydantic v2 схемы данных агента.

v4: страница задания целиком (reader-view: текст, таблицы, ссылки, [ФОТО n], [АУДИО n]
и элементы на своих местах), уведомления платформы, медиа, действие web (поиск в
интернете), план решения в ответе модели.

Изменения v3 относительно v2:
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
    WEB    = "web"     # поиск в интернете / открыть адрес во вкладке поиска (query)
    SKIP   = "skip"    # ничего не делать на этом шаге (идёт загрузка)


# Синонимы, которые модели иногда возвращают вместо канонических значений
_ACTION_SYNONYMS: dict[str, ActionType] = {
    "select": ActionType.CLICK, "choose": ActionType.CLICK, "press": ActionType.CLICK,
    "expand": ActionType.OPEN, "unfold": ActionType.OPEN,
    "input": ActionType.TYPE, "fill": ActionType.TYPE, "write": ActionType.TYPE,
    "finish": ActionType.SUBMIT, "send": ActionType.SUBMIT, "complete": ActionType.SUBMIT,
    "wait": ActionType.SKIP, "none": ActionType.SKIP, "noop": ActionType.SKIP,
    "search": ActionType.WEB, "browse": ActionType.WEB, "visit": ActionType.WEB,
    "open_url": ActionType.WEB, "research": ActionType.WEB, "google": ActionType.WEB,
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
  {TAG_FOLDER_CLOSED} / {TAG_FOLDER_OPEN} — ветка дерева: open раскрывает, содержимое — строки ниже с бо́льшим отступом.
  {TAG_OPTION} / [OPTION radio] / [OPTION checkbox] — вариант ответа: click выбирает (radio — один в группе, checkbox — несколько).
  {TAG_DROPDOWN_CLOSED} / {TAG_DROPDOWN_OPEN} — выпадающий список: open, затем click по пункту {FLAG_IN_POPUP}.
  [INPUT text] / [INPUT textarea] / [INPUT select] … — поле: type (для select — точный текст варианта).
  {TAG_BUTTON} — кнопка или ссылка; {TAG_OTHER} — прочий кликабельный элемент."""

PROMPT_FLAGS = f"""\
  {FLAG_SELECTED} — уже выбран; {FLAG_DISABLED} — нажать нельзя (у кнопки отправки: ответ не заполнен);
  {FLAG_OCCLUDED} — закрыт оверлеем или диалогом; {FLAG_SELECTABLE} — папку можно выбрать как ответ;
  {FLAG_IN_DIALOG} / {FLAG_IN_POPUP} — в диалоге / в открытом списке; ⟨путь: A › B⟩ — родительские папки
  (у одинаковых названий); ⟨поле: X⟩ — подпись поля; → https://… — адрес ссылки."""


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
    caption:      str          = Field("",   description="Подпись поля на странице («Цвет» у списка значений)")
    href:         str          = Field("",   description="Адрес ссылки (a[href])")
    aux:          bool         = Field(False, description="Служебный (плеер, переключатели карусели) — LLM не показывается")

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
        return self.text or self.placeholder or self.caption or f"#{self.index}"

    def prompt_line(self, *, show_path: bool = False, show_caption: bool = True) -> str:
        """Одна строка для LLM-промпта. Формат описан в PROMPT_LEGEND/PROMPT_FLAGS.

        show_caption=False — подпись поля уже стоит строкой выше в тексте страницы."""
        indent = "  " * min(self.depth, 8)
        parts = [f"{indent}[{self.index}] {self.type_tag()}"]

        text = self.text or self.placeholder
        if text:
            parts.append(f"«{text}»")
        if show_caption and self.caption and normalize_text(self.caption) != normalize_text(text):
            parts.append(f"⟨поле: {self.caption}⟩")

        if self.kind == ElementKind.INPUT:
            parts.append(f"значение=«{self.value[:300]}»" if self.value else "(пусто)")
            if self.options:
                parts.append("варианты: " + " | ".join(self.options[:40]))
        elif self.kind == ElementKind.DROPDOWN:
            if self.value:
                parts.append(f"значение=«{self.value[:120]}»")
            elif self.caption or self.placeholder:
                parts.append("(не выбрано)")
        if self.href:
            parts.append(f"→ {self.href[:300]}")

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

class Notice(BaseModel):
    """Сообщение платформы: ошибка («Неверный ответ»), подсказка («Правильный ответ: …»), инфо."""

    kind:  str = Field("info", description="error | warning | hint | success | info")
    text:  str = Field("")
    where: str = Field("inline", description="inline | toast | dialog")

    def render(self) -> str:
        mark = {"error": "‼", "warning": "⚠", "hint": "💡", "success": "✓"}.get(self.kind, "ℹ")
        return f"{mark} {self.text}"


class MediaImage(BaseModel):
    """Фото/картинка задания — [ФОТО n] в тексте страницы."""

    n:       int  = Field(..., description="Номер для LLM (с 1)")
    uid:     str  = Field("", description="Метка data-agent-media")
    src:     str  = Field("", description="Абсолютный адрес")
    alt:     str  = Field("")
    loaded:  bool = Field(True, description="Загрузилось в странице (naturalWidth > 0)")
    width:   int  = Field(0, description="Исходная ширина")
    height:  int  = Field(0, description="Исходная высота")


class MediaAudio(BaseModel):
    """Аудиозапись задания — [АУДИО n] в тексте страницы."""

    n:        int   = Field(...)
    uid:      str   = Field("", description="Метка data-agent-media на <audio>/<video>")
    play_uid: str   = Field("", description="Метка кнопки play/pause плеера, если есть")
    src:      str   = Field("")
    duration: float = Field(0.0)
    current:  float = Field(0.0)
    paused:   bool  = Field(True)
    ended:    bool  = Field(False)


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
    # --- v4: страница целиком -------------------------------------------------
    # Строки reader-view: текст/заголовки/таблицы как есть; элементы, фото и аудио —
    # плейсхолдеры \x00E<index>\x00, \x00I<n>\x00, \x00A<n>\x00 (подставляются при рендере)
    reader:          list[str]           = Field(default_factory=list, description="Основная страница")
    dialog_lines:    list[str]           = Field(default_factory=list, description="Открытый диалог")
    popup_lines:     list[str]           = Field(default_factory=list, description="Открытый выпадающий список")
    toast_lines:     list[str]           = Field(default_factory=list, description="Всплывающие уведомления")
    notices:         list[Notice]        = Field(default_factory=list)
    images:          list[MediaImage]    = Field(default_factory=list)
    audios:          list[MediaAudio]    = Field(default_factory=list)
    headings:        list[str]           = Field(default_factory=list)
    pool_key:        str                 = Field("", description="Вид задания (структура формы без данных)")
    pool_title:      str                 = Field("", description="Название вида задания для файла знаний")
    pool_signature:  list[str]           = Field(default_factory=list, description="Заголовки и поля формы")
    dialog_open:     bool                = Field(False)
    dialog_loading:  bool                = Field(False, description="В диалоге крутится загрузка")
    dialog_frames:   int                 = Field(0, description="iframe внутри диалога (документ инструкции)")
    local_loading:   int                 = Field(0, description="Мелкие лоадеры (галерея), не мешают работе")
    page_url:        str                 = Field("", description="URL главной страницы (не фрейма)")
    content_hash:    str                 = Field("", description="Хэш текста задания (без медиа)")
    loose_hash:      str                 = Field("", description="Хэш текста задания без цифр")
    form_hash:       str                 = Field("", description="Хэш состава формы (варианты, поля)")
    media_srcs:      list[str]           = Field(default_factory=list, description="Адреса фото и аудио")

    @property
    def visible_elements(self) -> list["ParsedElement"]:
        return [e for e in self.elements if not e.aux]

    def notice_texts(self, *kinds: str) -> list[str]:
        return [n.text for n in self.notices if not kinds or n.kind in kinds]

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

# Действия, после которых страница меняется настолько, что нужен новый взгляд модели:
# в пакете действий они могут стоять только последними
PAGE_CHANGING_ACTIONS = frozenset({ActionType.OPEN, ActionType.WEB, ActionType.SCROLL})


class _ActionFields(BaseModel):
    """Поля одного действия (общие для решения LLM и шагов пакета)."""

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
    query:            Optional[str] = Field(None, description="Поисковый запрос или URL для action=web")
    scroll_direction: Optional[str] = Field(None, description="down/up для action=scroll")

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

    @field_validator("target_text", "type_text", "query", "scroll_direction", mode="before")
    @classmethod
    def _coerce_optional_str(cls, value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class PlannedAction(_ActionFields):
    """Следующее действие пакета (после первого)."""

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "PlannedAction":
        return cls.model_validate(split_value(raw))


def split_value(raw: dict[str, Any]) -> dict[str, Any]:
    """Поле value из ответа модели → type_text / query / scroll_direction по типу действия.
    Старые поля (type_text, query, scroll_direction) тоже принимаются."""
    data = dict(raw)
    value = data.pop("value", None)
    if value is not None:
        action = str(data.get("action") or "").strip().lower()
        action = _ACTION_SYNONYMS.get(action, action)
        action = action.value if isinstance(action, ActionType) else action
        key = {"web": "query", "scroll": "scroll_direction"}.get(action, "type_text")
        data.setdefault(key, value)
        if data.get(key) is None:
            data[key] = value
    return data


class LLMDecision(_ActionFields):
    """Решение, принятое LLM (устойчиво к «грязным» типам в JSON).

    Поля действия — ПЕРВОЕ действие; next_actions — следующие действия пакета
    (модель может за один ответ выбрать ответы на все вопросы и отправить)."""

    goal:             str           = Field("", description="Что требуется в задании (v3; теперь в observation)")
    observation:      str           = Field("", description="Что требуется, ключевые факты, что уже сделано")
    reasoning:        str           = Field("", description="Почему выбраны эти действия")
    plan:             str           = Field("", description="План решения: выводы, найденные факты, что осталось")
    confidence:       float         = Field(1.0, ge=0.0, le=1.0)
    next_actions:     list[PlannedAction] = Field(default_factory=list, description="Следующие действия пакета")

    @field_validator("goal", "observation", "reasoning", "plan", mode="before")
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

    def steps(self) -> list["LLMDecision"]:
        """Все действия пакета по порядку — каждое как отдельное решение (для исполнителя)."""
        first = self.model_copy(update={"next_actions": []})
        rest = [
            LLMDecision(
                action=a.action, target_index=a.target_index, target_text=a.target_text,
                type_text=a.type_text, query=a.query, scroll_direction=a.scroll_direction,
                reasoning=self.reasoning, confidence=self.confidence,
            )
            for a in self.next_actions
        ]
        return [first] + rest


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
    image_b64:   Optional[str] = None                      # скриншот фрейма (LLM_VISION=frame / фоллбэк)
    temperature: Optional[float] = None                    # повышается при зацикливании
    # --- v4 ---
    images:      list["VisionImage"] = field(default_factory=list)   # фото задания (по одному / коллажи)
    image_notes: list[str] = field(default_factory=list)   # «[ФОТО 3] не загрузилось» и т.п.
    transcripts: list[str] = field(default_factory=list)   # расшифровки [АУДИО n]
    knowledge:   str = ""                                  # инструкция вида заданий + уроки
    research:    list[str] = field(default_factory=list)   # результаты action=web
    plan:        str = ""                                  # план модели с прошлого шага
    feedback:    list[str] = field(default_factory=list)   # «Неверный ответ» после отправки, подсказки
    wrong_answers: list[str] = field(default_factory=list)  # ответы, признанные неверными


@dataclass
class VisionImage:
    """Изображение для vision-модели: одиночное фото или коллаж с номерами."""

    caption: str            # «ФОТО 1–4» — подпись перед картинкой в сообщении
    b64: str                # JPEG, base64
    detail: str = "high"
