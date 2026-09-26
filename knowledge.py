"""knowledge.py — база знаний по видам заданий (папка knowledge/).

Для каждого вида заданий (одинаковая структура формы: заголовки, варианты, поля) агент
ведёт markdown-файл:
  ## Инструкция              — текст «Подробной инструкции» / диалога инструкции;
  ## Пояснения к вариантам   — подсказки «?» у вариантов ответа;
  ## Уроки из тренировки     — «Неверный ответ» + подсказка/правильный ответ платформы;
  ## Заметки                 — ВАШИ правила: агент их не меняет и отдаёт модели.
Всё это модель получает в каждом задании этого вида. Файл _general.md (если создать)
модель получает во всех заданиях.

Файлы можно читать и править вручную между запусками.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import KNOWLEDGE_DIR, KNOWLEDGE_PROMPT_CHARS
from models import PageState, normalize_text

logger = logging.getLogger("twork.knowledge")

_SECTIONS = {
    "инструкция": "instruction",
    "пояснения к вариантам": "tooltips",
    "уроки из тренировки": "lessons",
    "заметки": "notes",
}
_MAX_LESSONS = 40
_HEADER_HELP = (
    "Файл ведёт агент: разделы «Инструкция», «Пояснения к вариантам» и «Уроки из тренировки» "
    "он обновляет сам. Свои правила для этого вида заданий пишите в раздел «Заметки» — агент "
    "его не меняет и передаёт модели в каждом таком задании."
)


@dataclass
class PoolKnowledge:
    key: str
    title: str
    signature: list[str] = field(default_factory=list)
    path: Optional[Path] = None
    instruction: str = ""
    instruction_meta: str = ""
    tooltips: dict[str, str] = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)
    notes: str = ""
    instruction_attempted: bool = False     # в этом запуске уже пытались открыть инструкцию
    tooltips_attempted: bool = False

    @property
    def has_instruction(self) -> bool:
        return len(self.instruction.strip()) >= 80


class KnowledgeBase:
    def __init__(self, directory: str = KNOWLEDGE_DIR) -> None:
        self._dir = Path(directory)
        self._pools: dict[str, PoolKnowledge] = {}
        self._loaded = False
        self._general = ""

    # ------------------------------------------------------------------
    # Загрузка / поиск
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self._dir.is_dir():
            return
        general = self._dir / "_general.md"
        if general.is_file():
            self._general = general.read_text(encoding="utf-8").strip()
        for path in sorted(self._dir.glob("*.md")):
            if path.name.startswith("_"):
                continue
            try:
                pool = _parse(path)
            except (OSError, ValueError) as exc:
                logger.warning("Файл знаний %s не прочитан: %s", path.name, exc)
                continue
            if pool.key:
                self._pools[pool.key] = pool
        if self._pools:
            logger.info("База знаний: %d вид(ов) заданий в %s", len(self._pools), self._dir)

    def for_state(self, state: PageState) -> Optional[PoolKnowledge]:
        """Знания для вида задания: точное совпадение ключа или похожая структура формы."""
        if not state.pool_key:
            return None
        self._load()
        pool = self._pools.get(state.pool_key)
        if pool is not None:
            return pool
        sig = set(state.pool_signature)
        best, best_score = None, 0.0
        for candidate in self._pools.values():
            other = set(candidate.signature)
            if not sig or not other:
                continue
            score = len(sig & other) / len(sig | other)
            if score > best_score:
                best, best_score = candidate, score
        if best is not None and best_score >= 0.6:
            logger.info("Вид задания «%s» похож на «%s» (%.0f%%) — использую его знания",
                        state.pool_title, best.title, best_score * 100)
            self._pools[state.pool_key] = best
            return best
        pool = PoolKnowledge(key=state.pool_key, title=state.pool_title or "Задание",
                             signature=list(state.pool_signature))
        self._pools[state.pool_key] = pool
        return pool

    # ------------------------------------------------------------------
    # Обновление
    # ------------------------------------------------------------------

    def save_instruction(self, pool: PoolKnowledge, text: str, source: str, *, append: bool = False) -> None:
        """append=True — следующая страница той же инструкции (кнопка «Далее» в диалоге)."""
        text = _clean_block(text)
        if len(text) < 80:
            return
        if normalize_text(text) in normalize_text(pool.instruction):
            return
        if append and pool.instruction:
            text = f"{pool.instruction}\n\n{text}"
        pool.instruction = text
        pool.instruction_meta = f"Прочитано: {datetime.now():%Y-%m-%d %H:%M}, источник: {source}"
        logger.info("📘 Инструкция «%s» сохранена (%d симв.)", pool.title, len(text))
        self._write(pool)

    def save_tooltips(self, pool: PoolKnowledge, tips: dict[str, str]) -> None:
        fresh = {k: v for k, v in tips.items() if v and pool.tooltips.get(k) != v}
        if not fresh:
            return
        pool.tooltips.update(fresh)
        logger.info("📘 Пояснения к вариантам «%s»: %d", pool.title, len(fresh))
        self._write(pool)

    def add_lesson(self, pool: PoolKnowledge, lesson: str) -> None:
        lesson = re.sub(r"\s+", " ", lesson).strip()
        if not lesson or any(normalize_text(lesson) == normalize_text(x) for x in pool.lessons):
            return
        pool.lessons.append(lesson)
        pool.lessons = pool.lessons[-_MAX_LESSONS:]
        logger.info("📘 Урок для «%s»: %s", pool.title, lesson[:160])
        self._write(pool)

    # ------------------------------------------------------------------
    # Для промпта
    # ------------------------------------------------------------------

    def prompt_text(self, pool: Optional[PoolKnowledge], limit: int = KNOWLEDGE_PROMPT_CHARS) -> str:
        self._load()
        parts: list[str] = []
        if self._general:
            parts.append("Общие правила (knowledge/_general.md):\n" + self._general)
        if pool is not None:
            if pool.notes.strip():
                parts.append("Заметки пользователя для этого вида заданий:\n" + pool.notes.strip())
            if pool.lessons:
                parts.append("Уроки из прошлых ошибок (платформа сообщила правильный ответ/подсказку):\n"
                             + "\n".join(f"- {x}" for x in pool.lessons[-20:]))
            if pool.tooltips:
                parts.append("Пояснения к вариантам ответа (подсказки «?» на странице):\n"
                             + "\n".join(f"- «{k}»: {v}" for k, v in pool.tooltips.items()))
            if pool.instruction:
                parts.append("Инструкция к заданиям этого вида:\n" + pool.instruction)
        text = "\n\n".join(parts)
        if len(text) > limit:
            text = text[:limit] + "\n[… инструкция сокращена: полный текст в папке knowledge …]"
        return text

    # ------------------------------------------------------------------
    # Файлы
    # ------------------------------------------------------------------

    def _write(self, pool: PoolKnowledge) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            if pool.path is None:
                pool.path = self._dir / f"{_slug(pool.title)}-{pool.key[:8]}.md"
            content = _render(pool)
            tmp = pool.path.with_suffix(".md.tmp")
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, pool.path)
        except OSError as exc:
            logger.warning("Не удалось записать файл знаний: %s", exc)


def _slug(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip()
    s = re.sub(r"\s+", "_", s)
    return (s or "task")[:60]


def _clean_block(text: str) -> str:
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            if out and out[-1] != "":
                out.append("")
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _render(pool: PoolKnowledge) -> str:
    lines = [
        f"# {pool.title}",
        f"<!-- pool: {pool.key} -->",
        f"<!-- signature: {json.dumps(pool.signature, ensure_ascii=False)} -->",
        "",
        f"_{_HEADER_HELP}_",
        "",
        "## Инструкция",
    ]
    if pool.instruction:
        if pool.instruction_meta:
            lines.append(f"_{pool.instruction_meta}_")
            lines.append("")
        lines.append(pool.instruction)
    lines += ["", "## Пояснения к вариантам"]
    lines += [f"- «{k}»: {v}" for k, v in pool.tooltips.items()]
    lines += ["", "## Уроки из тренировки"]
    lines += [f"- {x}" for x in pool.lessons]
    lines += ["", "## Заметки", pool.notes.strip(), ""]
    return "\n".join(lines)


def _parse(path: Path) -> PoolKnowledge:
    text = path.read_text(encoding="utf-8")
    key = re.search(r"<!--\s*pool:\s*([0-9a-f]+)\s*-->", text)
    if not key:
        raise ValueError("нет метки <!-- pool: … -->")
    sig = re.search(r"<!--\s*signature:\s*(\[.*?\])\s*-->", text)
    title = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    pool = PoolKnowledge(
        key=key.group(1), title=title.group(1).strip() if title else path.stem,
        signature=json.loads(sig.group(1)) if sig else [], path=path,
    )
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            current = _SECTIONS.get(m.group(1).strip().lower())
            if current:
                sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)
    body = sections.get("instruction", [])
    if body and re.match(r"^_Прочитано:.*_$", body[0].strip()):
        pool.instruction_meta = body[0].strip().strip("_")
        body = body[1:]
    pool.instruction = _clean_block("\n".join(body))
    for line in sections.get("tooltips", []):
        m = re.match(r"^-\s*«(.+?)»:\s*(.+)$", line.strip())
        if m:
            pool.tooltips[m.group(1)] = m.group(2)
    pool.lessons = [ln.strip()[1:].strip() for ln in sections.get("lessons", []) if ln.strip().startswith("-")]
    pool.notes = _clean_block("\n".join(sections.get("notes", [])))
    return pool
