"""task_memory.py — память агента в рамках одного задания.

В v2 память состояла из множества ИНДЕКСОВ (_banned, _open_counters). После
раскрытия любой папки индексы сдвигались, и запрет «переезжал» на другие
элементы — в том числе на правильный ответ, а LLM получала «Не выбирай
индексы [3, 7]», указывающие уже на чужие строки. Кроме того, у LLM не было
истории: на каждом шаге она видела только текущий снимок.

Здесь всё привязано к стабильному ключу элемента (группа|путь|текст), а каждое
действие получает ФАКТИЧЕСКИЙ результат, проверенный по следующему снимку:
«✓ раскрыта, появилось 3: …», «✗ без видимого эффекта», «⚠ выбор снят».
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from config import MAX_NO_EFFECT, MAX_OPEN_RETRIES
from models import ActionType, ElementKind, FolderState, PageState, ParsedElement, normalize_text


@dataclass
class HistoryEntry:
    """Одна строка истории для промпта."""
    step: int
    action: ActionType
    label: str = ""
    path: tuple[str, ...] = ()
    note: str = ""
    result: str = "…"

    def render(self) -> str:
        target = f" «{self.label}»" if self.label else ""
        where = f" ⟨{' › '.join(self.path)}⟩" if self.path else ""
        note = f" {self.note}" if self.note else ""
        return f"{self.step}. {self.action.value}{target}{where}{note} → {self.result}"


@dataclass
class PendingCheck:
    """Действие, эффект которого проверяется по следующему снимку."""
    action: ActionType
    key: str
    entry: HistoryEntry
    task_id: str
    state_hash: str
    keys_before: frozenset[str]
    was_selected: bool
    was_open: bool
    typed: Optional[str] = None


@dataclass
class TaskMemory:
    """Состояние агента по текущему заданию. Сбрасывается при смене отпечатка задания."""

    task_id: str = ""
    steps: int = 0
    history: list[HistoryEntry] = field(default_factory=list)
    no_effect: Counter = field(default_factory=Counter)        # (action, key) → сколько раз без эффекта
    open_attempts: Counter = field(default_factory=Counter)    # key → попыток раскрыть
    insist: Counter = field(default_factory=Counter)           # (action, key) → отклонённых «лишних» команд
    labels: dict[str, tuple[str, tuple[str, ...]]] = field(default_factory=dict)
    start_clicks: Counter = field(default_factory=Counter)     # state_hash → авто-кликов «Приступить»
    pending: Optional[PendingCheck] = None
    consecutive_skips: int = 0
    invalid_targets: int = 0
    repeated_forbidden: int = 0
    submit_failures: int = 0
    loading_waits: int = 0
    image_b64: Optional[str] = None
    image_src: str = ""
    image_checked: bool = False

    def reset(self, task_id: str) -> None:
        """Полный сброс при смене задания (раскрытие папок задание не меняет)."""
        self.__init__()  # type: ignore[misc]
        self.task_id = task_id

    # ------------------------------------------------------------------
    # История
    # ------------------------------------------------------------------

    def add(
        self,
        action: ActionType,
        el: Optional[ParsedElement],
        *,
        result: str = "…",
        note: str = "",
    ) -> HistoryEntry:
        entry = HistoryEntry(
            step=len(self.history) + 1,
            action=action,
            label=el.label() if el else "",
            path=tuple(el.path) if el else (),
            note=note,
            result=result,
        )
        if el is not None:
            self.labels[el.key] = (entry.label, entry.path)
        self.history.append(entry)
        return entry

    def expect(
        self,
        action: ActionType,
        el: Optional[ParsedElement],
        state: PageState,
        *,
        typed: Optional[str] = None,
        note: str = "",
        key: Optional[str] = None,
    ) -> None:
        """Записать действие и отложить проверку его эффекта до следующего снимка."""
        entry = self.add(action, el, note=note)
        self.pending = PendingCheck(
            action=action,
            key=key if key is not None else (el.key if el else ""),
            entry=entry,
            task_id=state.task_identifier,
            state_hash=state.state_hash,
            keys_before=frozenset(state.keys),
            was_selected=bool(el and el.is_selected),
            was_open=bool(el and el.folder_state == FolderState.OPEN),
            typed=typed,
        )

    def verify(self, state: PageState) -> None:
        """Сравнить новый снимок с ожиданием и записать фактический результат."""
        pending, self.pending = self.pending, None
        if pending is None:
            return
        entry = pending.entry
        if state.task_identifier != pending.task_id:
            entry.result = "задание сменилось"
            return

        after = state.by_key(pending.key)
        new_keys = state.keys - pending.keys_before
        gone_keys = pending.keys_before - state.keys
        changed = state.state_hash != pending.state_hash
        new_labels = self._labels_of(state, new_keys)

        if pending.action == ActionType.OPEN:
            if new_keys:
                entry.result = f"✓ раскрыто, появилось {len(new_keys)}: {new_labels}"
            elif after is not None and after.folder_state == FolderState.OPEN and not pending.was_open:
                entry.result = "✓ раскрыто (новых элементов не видно — ветка пуста или ещё грузится)"
            elif gone_keys:
                entry.result = (f"⚠ исчезло {len(gone_keys)} элементов — похоже, папка была раскрыта "
                                "и свернулась; раскрой её снова, если там был нужный вариант")
            else:
                entry.result = "✗ без видимого эффекта"
                self.no_effect[(pending.action, pending.key)] += 1

        elif pending.action == ActionType.CLICK:
            if after is not None and after.is_selected and not pending.was_selected:
                entry.result = "✓ выбран"
            elif after is not None and pending.was_selected and not after.is_selected:
                entry.result = "⚠ выбор снят"
            elif new_keys or gone_keys or changed:
                extra = f", появилось: {new_labels}" if new_keys else ""
                entry.result = f"✓ страница изменилась{extra}"
            else:
                entry.result = "✗ без видимого эффекта"
                self.no_effect[(pending.action, pending.key)] += 1

        elif pending.action == ActionType.TYPE:
            if after is None:
                entry.result = "✓ введено (поле исчезло — форма обновилась)"
            elif normalize_text(after.value) == normalize_text(pending.typed):
                entry.result = "✓ введено"
            else:
                entry.result = f"⚠ в поле сейчас: «{after.value[:60]}»"

        elif pending.action == ActionType.SCROLL:
            if new_keys:
                entry.result = f"✓ появились новые элементы ({len(new_keys)}): {new_labels}"
            elif changed:
                entry.result = "✓ прокручено"
            else:
                entry.result = "✗ ничего не изменилось (вероятно, конец списка)"
                self.no_effect[(pending.action, pending.key)] += 1

    @staticmethod
    def _labels_of(state: PageState, keys: set[str], limit: int = 6) -> str:
        labels = [f"«{el.label()}»" for el in state.elements if el.key in keys]
        head = ", ".join(labels[:limit])
        return head + (f" и ещё {len(labels) - limit}" if len(labels) > limit else "")

    def history_lines(self, limit: int) -> list[str]:
        return [entry.render() for entry in self.history[-limit:]]

    # ------------------------------------------------------------------
    # Запреты (вместо _banned по индексам)
    # ------------------------------------------------------------------

    def mark_no_effect(self, action: ActionType, key: str) -> None:
        if key:
            self.no_effect[(action, key)] += 1

    def is_forbidden(self, action: ActionType, key: str) -> bool:
        if not key:
            return False
        if self.no_effect[(action, key)] >= MAX_NO_EFFECT:
            return True
        return action == ActionType.OPEN and self.open_attempts[key] >= MAX_OPEN_RETRIES

    def forbidden_lines(self) -> list[str]:
        lines: list[str] = []
        for (action, key), count in self.no_effect.items():
            if count < MAX_NO_EFFECT:
                continue
            label, path = self.labels.get(key, (key.split("|")[-1], ()))
            where = f" ⟨{' › '.join(path)}⟩" if path else ""
            lines.append(f"{action.value} «{label}»{where} — {count} раз(а) без эффекта")
        for key, count in self.open_attempts.items():
            if count >= MAX_OPEN_RETRIES and self.no_effect[(ActionType.OPEN, key)] < MAX_NO_EFFECT:
                label, path = self.labels.get(key, (key.split("|")[-1], ()))
                where = f" ⟨{' › '.join(path)}⟩" if path else ""
                lines.append(f"open «{label}»{where} — уже {count} попытки раскрыть")
        return lines

    # ------------------------------------------------------------------
    # Утилиты
    # ------------------------------------------------------------------

    @staticmethod
    def selected_options(state: PageState) -> list[ParsedElement]:
        return [
            el for el in state.elements
            if el.is_selected and el.kind in (ElementKind.OPTION, ElementKind.FOLDER)
        ]
