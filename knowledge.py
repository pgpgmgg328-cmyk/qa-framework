"""knowledge.py — база знаний по видам заданий (папка knowledge/).

Для каждого вида заданий (одинаковая структура формы: заголовки, варианты, поля) агент
ведёт markdown-файл:
  ## Инструкция              — текст «Подробной инструкции» / диалога инструкции;
  ## Пояснения к вариантам   — подсказки «?» у вариантов ответа;
  ## Уроки из тренировки     — «Неверный ответ» + подсказка/правильный ответ платформы;
  ## Статистика тренировки   — сколько ответов верны с первого раза, какие ответы оказывались
                               правильными, результаты экзаменов;
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
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import KNOWLEDGE_DIR, KNOWLEDGE_LESSON_CHARS, KNOWLEDGE_PROMPT_CHARS
from models import PageState, normalize_text

logger = logging.getLogger("twork.knowledge")

_SECTIONS = {
    "инструкция": "instruction",
    "пояснения к вариантам": "tooltips",
    "уроки из тренировки": "lessons",
    "статистика тренировки": "stats",
    "заметки": "notes",
}
_MAX_LESSONS = 40
_HEADER_HELP = (
    "Файл ведёт агент: разделы «Инструкция», «Пояснения к вариантам», «Уроки из тренировки» и "
    "«Статистика тренировки» он обновляет сам. Свои правила для этого вида заданий пишите в раздел «Заметки» — агент "
    "его не меняет и передаёт модели в каждом таком задании. Строка «Модель агента: <название>» в «Заметках» "
    "закрепляет модель для этого вида заданий."
)
_PIN_RE = re.compile(r"(?im)^\s*модель агента\s*:\s*([a-z0-9][a-z0-9_.:/@+-]*)[^\n]*\n?")
# урок прежних версий без самого задания и без подсказки: «ответ X — неверно» модель применяла ко
# всем заданиям подряд — такие в запрос не идут
_BARE_LESSON_RE = re.compile(r"^Задание «[^»]*»: ответ .+ — неверно\.$")
# как агент читал инструкцию: вкладки с инструкцией, прочитанные до v4.8, — это PDF, у которого
# схемы и таблицы остались «одной строкой»; такие агент перечитает один раз
READ_MARK = "чтение v2"


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
    # тренировка: заданий с первым ответом / из них верно с первого раза; какие ответы платформа
    # приняла (правильные); результаты экзаменов
    train_total: int = 0
    train_first_ok: int = 0
    train_answers: dict[str, int] = field(default_factory=dict)
    exams: list[str] = field(default_factory=list)
    # лестница моделей: какой моделью агент решает этот вид и её точность в тренировке
    model: str = ""
    model_stats: dict[str, list[int]] = field(default_factory=dict)   # модель → [верно, всего]
    instruction_attempted: bool = False     # в этом запуске уже пытались открыть инструкцию
    instruction_unread: bool = False        # инструкцию открыли, но прочитать не удалось (PDF, ошибка)
    tooltips_attempted: bool = False

    @property
    def has_instruction(self) -> bool:
        return len(self.instruction.strip()) >= 80 and not self.instruction_outdated

    @property
    def instruction_outdated(self) -> bool:
        """Инструкция из вкладки, прочитанная прежним способом (PDF: схемы — одной строкой)."""
        meta = self.instruction_meta
        return "источник: вкладка" in meta and READ_MARK not in meta

    @property
    def pinned_model(self) -> str:
        """Модель, закреплённая человеком в «Заметках»: строка «Модель агента: <название>»."""
        m = _PIN_RE.search(self.notes)
        return m.group(1) if m else ""

    def training_line(self) -> str:
        if not self.train_total:
            return ""
        return (f"с первого раза верно {self.train_first_ok} из {self.train_total} "
                f"({self.train_first_ok * 100 // self.train_total}%)")


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
        meta = f"Прочитано: {datetime.now():%Y-%m-%d %H:%M}, источник: {source}, {READ_MARK}"
        if pool.instruction_outdated:
            append = False                      # перечитанная инструкция заменяет прежнюю
        elif normalize_text(text) in normalize_text(pool.instruction):
            return
        if append and pool.instruction:
            text = f"{pool.instruction}\n\n{text}"
        before = len(pool.instruction) if pool.instruction_outdated else 0
        pool.instruction = text
        pool.instruction_meta = meta
        pool.instruction_unread = False
        if before:
            logger.info("📘 Инструкция «%s» перечитана: было %d симв., стало %d", pool.title, before, len(text))
        else:
            logger.info("📘 Инструкция «%s» сохранена (%d симв.)", pool.title, len(text))
        self._write(pool)

    def save_tooltips(self, pool: PoolKnowledge, tips: dict[str, str]) -> None:
        fresh = {k: v for k, v in tips.items() if v and pool.tooltips.get(k) != v}
        if not fresh:
            return
        pool.tooltips.update(fresh)
        logger.info("📘 Пояснения к вариантам «%s»: %d", pool.title, len(fresh))
        self._write(pool)

    def training_attempt(self, pool: PoolKnowledge, first_ok: bool) -> None:
        """Первый ответ на задание тренировки: верен ли он."""
        pool.train_total += 1
        pool.train_first_ok += int(first_ok)
        self._write(pool)

    def training_answer(self, pool: PoolKnowledge, answer: str) -> None:
        """Ответ, который платформа в тренировке приняла (правильный)."""
        answer = re.sub(r"\s+", " ", answer).strip()
        if answer:
            pool.train_answers[answer] = pool.train_answers.get(answer, 0) + 1
            self._write(pool)

    def model_attempt(self, pool: PoolKnowledge, model: str, first_ok: bool) -> None:
        """Первый ответ тренировки, данный моделью model."""
        stats = pool.model_stats.setdefault(model, [0, 0])
        stats[0] += int(first_ok)
        stats[1] += 1
        self._write(pool)

    def set_model(self, pool: PoolKnowledge, model: str) -> None:
        pool.model = model
        self._write(pool)

    def ensure_file(self, pool: PoolKnowledge) -> Optional[Path]:
        """Файл вида заданий (создать, если его ещё нет) — например, чтобы человек вписал инструкцию."""
        if pool.path is None or not pool.path.exists():
            self._write(pool)
        return pool.path

    def exam_result(self, pool: PoolKnowledge, passed: bool) -> None:
        pool.exams.append(f"{'пройден' if passed else 'не пройден'} ({datetime.now():%Y-%m-%d %H:%M})")
        pool.exams = pool.exams[-10:]
        self._write(pool)

    def add_lesson(self, pool: PoolKnowledge, lesson: str) -> None:
        lesson = re.sub(r"\s+", " ", lesson).strip()
        if not lesson or any(normalize_text(lesson) == normalize_text(x) for x in pool.lessons):
            return
        pool.lessons.append(lesson)
        pool.lessons = pool.lessons[-_MAX_LESSONS:]
        logger.info("📘 Разбор ошибки для «%s»: %s", pool.title, lesson[:300])
        self._write(pool)

    # ------------------------------------------------------------------
    # Для промпта
    # ------------------------------------------------------------------

    def prompt_text(self, pool: Optional[PoolKnowledge], limit: int = KNOWLEDGE_PROMPT_CHARS) -> str:
        """Знания для запроса. Инструкция — сразу после правил пользователя и целиком (длинные
        разборы ошибок её не вытесняют, а начало запроса меньше меняется и берётся из кэша);
        разборы ошибок — самые свежие, сколько войдёт."""
        self._load()
        head: list[str] = []
        if self._general:
            head.append("Общие правила (knowledge/_general.md):\n" + self._general)
        if pool is None:
            return _cut("\n\n".join(head), limit)
        notes = _PIN_RE.sub("", pool.notes).strip()          # выбор модели — не для модели
        if notes:
            head.append("Заметки пользователя для этого вида заданий:\n" + notes)
        tail: list[str] = []
        if pool.tooltips:
            tail.append("Пояснения к вариантам ответа (подсказки «?» на странице):\n"
                        + "\n".join(f"- «{k}»: {v}" for k, v in pool.tooltips.items()))
        if pool.train_answers:
            top = sorted(pool.train_answers.items(), key=lambda kv: -kv[1])[:6]
            tail.append(
                "Какие ответы в тренировке этого вида оказались правильными (для калибровки, а не вместо "
                "проверки): " + "; ".join(f"{a} — {n}" for a, n in top)
                + (f". Твоя точность в тренировке: {pool.training_line()} — проверяй внимательнее."
                   if pool.train_total and pool.train_first_ok < pool.train_total else "."))
        lessons = [x for x in pool.lessons if not _BARE_LESSON_RE.match(x)]
        fixed = sum(len(x) + 2 for x in head + tail)
        room = max(limit - fixed, 0)
        lesson_room = min(sum(len(x) + 3 for x in lessons), KNOWLEDGE_LESSON_CHARS) if lessons else 0
        parts = list(head)
        if pool.instruction:
            header = "Инструкция к заданиям этого вида:\n"
            parts.append(header + _cut(pool.instruction, max(room - lesson_room - len(header) - 2, 400)))
        parts += tail
        left = limit - sum(len(x) + 2 for x in parts)
        if lessons and left > 200:
            chosen: list[str] = []
            for lesson in reversed(lessons):                 # свежие важнее
                if len(lesson) + 3 > left - 160:
                    break
                chosen.insert(0, lesson)
                left -= len(lesson) + 3
            if chosen:
                parts.append("Разборы ошибок из тренировки — конкретные задания этого вида. Применяй разбор, "
                             "только если в новом задании те же признаки; на другие задания его не переноси:\n"
                             + "\n".join(f"- {x}" for x in chosen))
        return _cut("\n\n".join(parts), limit)

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


def import_previous(directory: str, project_dir: Path) -> int:
    """Новую версию распаковали в новую папку — база знаний пуста: уроки, статистика тренировок (по
    ней агент решает, начинать ли экзамен) и инструкции остались в папке прежней версии. Копирует
    файлы из самой свежей базы знаний в соседних папках «twork…» (там ничего не меняется)."""
    target = Path(directory)
    marker = target / "_imported.txt"           # перенос был: базу потом очистили сами — не повторять
    if marker.exists() or (target.is_dir() and any(not f.name.startswith("_") for f in target.glob("*.md"))):
        return 0
    here = project_dir.resolve()
    found: list[tuple[float, Path]] = []
    try:
        for folder in project_dir.parent.iterdir():
            if not folder.is_dir() or "twork" not in folder.name.lower() or folder.resolve() == here:
                continue
            nested = [c / "knowledge" for c in folder.iterdir() if c.is_dir() and c.resolve() != here]
            for kdir in [folder / "knowledge", *nested]:
                files = [f for f in kdir.glob("*.md") if not f.name.startswith("_") and _is_pool_file(f)]
                if files and kdir.resolve() != target.resolve():
                    found.append((max(f.stat().st_mtime for f in files), kdir))
    except OSError as exc:
        logger.debug("Папки прежних версий не просмотрены: %s", exc)
        return 0
    if not found:
        return 0
    source = max(found)[1]
    copied = 0
    try:
        target.mkdir(parents=True, exist_ok=True)
        for f in source.glob("*.md"):
            if not (target / f.name).exists():
                shutil.copy2(f, target / f.name)
                copied += 1
        marker.write_text(f"База знаний скопирована из {source}\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Базу знаний прежней версии скопировать не удалось: %s", exc)
    if copied:
        logger.info("📚 База знаний пуста — скопировал %d файл(ов) из папки прежней версии %s (уроки, статистика "
                    "тренировок, инструкции)", copied, source)
    return copied


def _is_pool_file(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as fh:
            return "<!-- pool:" in fh.read(4096)
    except OSError:
        return False


def _slug(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE).strip()
    s = re.sub(r"\s+", "_", s)
    return (s or "task")[:60]


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[… сокращено: полный текст — в папке knowledge …]"


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
    lines += ["", "## Статистика тренировки"]
    if pool.train_total:
        lines.append(f"- с первого раза верно: {pool.train_first_ok} из {pool.train_total}")
    lines += [f"- правильный ответ {a}: {n}" for a, n in sorted(pool.train_answers.items(), key=lambda kv: -kv[1])]
    lines += [f"- экзамен: {x}" for x in pool.exams]
    if pool.model:
        lines.append(f"- текущая модель: {pool.model}")
    lines += [f"- модель {m}: с первого раза верно {ok} из {total}" for m, (ok, total) in pool.model_stats.items()]
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
    for line in sections.get("stats", []):
        line = line.strip()
        if m := re.match(r"^-\s*с первого раза верно:\s*(\d+)\s+из\s+(\d+)", line):
            pool.train_first_ok, pool.train_total = int(m.group(1)), int(m.group(2))
        elif m := re.match(r"^-\s*правильный ответ\s+(.+):\s*(\d+)$", line):
            pool.train_answers[m.group(1).strip()] = int(m.group(2))
        elif m := re.match(r"^-\s*экзамен:\s*(.+)$", line):
            pool.exams.append(m.group(1).strip())
        elif m := re.match(r"^-\s*текущая модель:\s*(\S+)", line):
            pool.model = m.group(1).strip()
        elif m := re.match(r"^-\s*модель\s+(\S+):\s*с первого раза верно\s*(\d+)\s+из\s+(\d+)", line):
            pool.model_stats[m.group(1)] = [int(m.group(2)), int(m.group(3))]
    pool.notes = _clean_block("\n".join(sections.get("notes", [])))
    return pool
