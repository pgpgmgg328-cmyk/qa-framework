"""agent.py v4 — главный цикл агента.

Агент решает задания любого вида, пока платформа не вернёт его на список заказов.
Шаблонов под виды заданий нет: страница задания целиком (текст, таблицы, ссылки, фото,
аудио, поля) уходит модели, а модель сама решает, как выполнить задание.

Шаг агента:
  1. Всплывающие окна сайта (новости T-Work) поверх фрейма — закрыть.
  2. Найти видимый фрейм задания; капча — ждать ручного решения.
  3. Снимок DomParser (reader-view, элементы размечены метками data-agent-id).
  4. Список заказов («Приступить») после выполненных заданий — остановка.
  5. Новое задание (TaskIdentity) → сброс памяти, знания о виде задания.
  6. Диалоги: «Выйти из задания?» → «остаться»; инструкция → прочитать, сохранить,
     закрыть; «Тренировка … Начать» → начать.
  7. Инструкция и подсказки «?» вида задания — прочитать один раз за запуск.
  8. Аудио — расшифровка в фоне и воспроизведение; фото — скачивание для модели.
  9. Решение LLM (знания, страница, фото, аудио, поиск, план, история, неверные ответы).
 10. Выполнение по метке; web — поиск в отдельной вкладке; submit — дослушать аудио,
     нажать и дождаться: следующее задание или «Неверный ответ» (тогда урок + исправление).
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from urllib.parse import urlsplit

from playwright.async_api import Error as PlaywrightError, Frame

from browser_controller import BrowserController, is_connection_lost
from config import (
    ACTION_WAIT,
    ALLOW_MAIN_FRAME,
    AUDIO_PLAY_TO_END,
    AUDIO_TO_MODEL,
    BATCH_ACTIONS,
    CAPTCHA_TIMEOUT,
    CLOSE_BROWSER_WHEN_DONE,
    EXAM_MIN_ACCURACY,
    EXAM_MIN_TASKS,
    LADDER_MIN_ACCURACY,
    LADDER_MIN_TASKS,
    DIALOG_CLOSE_TEXTS,
    DIALOG_SELECTORS,
    EXIT_CANCEL_TEXTS,
    FINISH_BUTTON_TEXTS,
    FINISH_DENY_SUBSTRINGS,
    FRAME_LOAD_WAIT,
    INSTRUCTION_WAIT,
    LLM_HISTORY_SIZE,
    LLM_MODELS,
    LLM_TEMPERATURE,
    LLM_VISION,
    MAX_BATCH_ACTIONS,
    MAX_IDLE_SECONDS,
    MAX_STEPS,
    MAX_STEPS_PER_TASK,
    MAX_WEB_PER_TASK,
    NON_TASK_URL_KEYWORDS,
    ORDERS_BUTTON_TEXTS,
    ORDERS_DONE_TEXTS,
    PLATFORM,
    READ_INSTRUCTIONS,
    READ_TOOLTIPS,
    START_BUTTON_TEXTS,
    STOP_ON_ORDERS_LIST,
    SUBMIT_WAIT,
    TARGET_URL,
    TASK_URL_KEYWORDS,
    WEB_RESEARCH,
    WRONG_ANSWER_TEXTS,
)
from documents import DocumentCatcher, describe_documents, describe_page, document_urls, meaningful, page_text
from dom_parser import DomParser, _plain_text, is_denied_button
from knowledge import KnowledgeBase, PoolKnowledge
from media import MediaManager
from models import (
    FLAG_DISABLED,
    FLAG_IN_DIALOG,
    FLAG_IN_POPUP,
    FLAG_OCCLUDED,
    FLAG_SELECTABLE,
    FLAG_SELECTED,
    PAGE_CHANGING_ACTIONS,
    ActionType,
    DecisionContext,
    ElementKind,
    FolderState,
    LLMDecision,
    PageState,
    ParsedElement,
    normalize_text,
)
from openrouter_connector import LLMConnector
from research import WebResearch
from task_memory import TaskMemory

logger = logging.getLogger("twork.agent")


# Модель ответила «skip» (ждать): следующий вызов — после изменения страницы, но не позже чем через
# столько секунд (первое ожидание подряд / следующие)
SKIP_WAIT_FIRST, SKIP_WAIT_NEXT = 8.0, 20.0
OUTAGE_PAUSE = 5.0          # OpenRouter недоступен: пауза перед следующей попыткой шага, с
# оценка точности модели для экзамена: столько «заданий» точности всей тренировки добавляется к её
# собственным — пока у модели 1–3 задания, одна ошибка не превращает оценку в 66% или 0%
_EXAM_PRIOR = 5
AUDIO_SETTLE = 1.5          # запись доиграла — пауза, пока страница отметит прослушивание, с


class StepResult(str, Enum):
    ACTED = "acted"     # шаг с действием — расходует MAX_STEPS
    IDLE = "idle"       # ожидание (нет фрейма, капча, загрузка) — не расходует
    WAITING = "waiting" # список заказов: ждём, пока человек откроет заказ (без лимита простоя)
    STOP = "stop"


# Какие типы элементов допустимы для действия (в v2 автокоррекция могла
# «исправить» open на OPTION, а type — на BUTTON)
_ACTION_KINDS: dict[ActionType, set[ElementKind]] = {
    ActionType.OPEN:   {ElementKind.FOLDER, ElementKind.DROPDOWN},
    ActionType.CLICK:  {ElementKind.OPTION, ElementKind.BUTTON, ElementKind.FOLDER,
                        ElementKind.DROPDOWN, ElementKind.OTHER},
    ActionType.TYPE:   {ElementKind.INPUT},
    ActionType.SUBMIT: {ElementKind.BUTTON},
    ActionType.SCROLL: set(ElementKind),
}
# Мягкий фоллбэк: open по строке, которую парсер не распознал как папку
_FALLBACK_KINDS: dict[ActionType, set[ElementKind]] = {
    ActionType.OPEN: {ElementKind.OPTION, ElementKind.OTHER},
}
_MATCH_THRESHOLD = 0.85
_WRONG_RE = re.compile(r"(неверн|неправильн|ошибк[аи] в ответе|incorrect|wrong)", re.IGNORECASE)
# Режим задания на панели страницы: «Тренировка» (есть подсказки) или «Экзамен» (без подсказок)
_MODE_RE = re.compile(r"^#*\s*(тренировка|экзамен)\s*$", re.IGNORECASE)
_EXAM_FAILED_RE = re.compile(r"(не\s+(пройден|сдан|прош[её]л|прошли|удалось)|провал)", re.IGNORECASE)
_EXAM_PASSED_RE = re.compile(r"(пройден|сдан|прош[её]л|прошли|поздравля)", re.IGNORECASE)


def _exam_outcome(text: str) -> Optional[bool]:
    """Окно с итогом экзамена: False — «Экзамен не пройден», True — пройден, None — не итог."""
    if "экзамен" not in text.lower():
        return None
    if _EXAM_FAILED_RE.search(text):
        return False
    if _EXAM_PASSED_RE.search(text):
        return True
    return None
_INSTRUCTION_RE = re.compile(r"инструкц", re.IGNORECASE)
_PHOTO_FAIL_RE = re.compile(r"фото.{0,20}не\s*(загруж|открыва|отображ)", re.IGNORECASE)
# окно входа в аккаунт (сессия истекла): в аккаунт входит только человек
# («номер телефона» сюда не входит: «найдите номер телефона организации» — обычное задание)
_LOGIN_RE = re.compile(r"(войти|войдите|вход в (аккаунт|профиль|ozon)|авториз|ozon id|код из (sms|смс)|"
                       r"(sms|смс)[- ]код|код подтверждения|введите пароль)", re.IGNORECASE)
_NEW_TAB_RE = re.compile(r"нов(ой|ую|ом) (вкладк|окн)", re.IGNORECASE)     # «Открыть в новой вкладке»

# Всплывающее окно на ГЛАВНОЙ странице сайта (новости, объявления) поверх фрейма задания
# own — окна самой площадки (DIALOG_SELECTORS): когда задание в главном фрейме (Ozon), инструкцию
# в таком окне агент читает как диалог задания, а не закрывает как новость
_JS_PAGE_POPUP = r"""
(own) => {
    const SEL = '[role="dialog"], [aria-modal="true"], dialog[open], tui-dialog, [class*="modal" i]';
    const ownSel = (own || []).filter((s) => { try { document.createElement('div').matches(s); return true; } catch (e) { return false; } }).join(', ');
    const ownOf = (d) => !!ownSel && (!!d.closest(ownSel) || !!d.querySelector(ownSel));
    const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim();
    const visible = (el) => {
        const r = el.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) return false;
        return el.checkVisibility ? el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true }) : true;
    };
    document.querySelectorAll('[data-agent-popup]').forEach((b) => b.removeAttribute('data-agent-popup'));
    const frames = Array.from(document.querySelectorAll('iframe'));
    for (const d of document.querySelectorAll(SEL)) {
        let box = d.getBoundingClientRect();
        for (const ch of d.children) {
            const r = ch.getBoundingClientRect();
            if (r.width * r.height > box.width * box.height) box = r;
        }
        if (box.width < 150 || box.height < 100 || !visible(d)) continue;
        if (frames.some((f) => d.contains(f))) continue;         // это оболочка самого задания
        if (ownOf(d)) continue;
        // окно с полями ввода — форма (вход в аккаунт: «Продолжить» отправило бы SMS), а не новость
        if (d.querySelector('input:not([type="hidden"]):not([type="checkbox"]):not([type="radio"]), textarea')) continue;
        const buttons = [];
        d.querySelectorAll('button, [role="button"], a[href]').forEach((b, i) => {
            if (!visible(b)) return;
            b.setAttribute('data-agent-popup', String(i));
            buttons.push({ i, text: norm(b.innerText || b.getAttribute('aria-label') || b.title || ''),
                           aria: norm((b.getAttribute('aria-label') || '') + ' ' + (b.getAttribute('class') || '')) });
        });
        if (buttons.length) return { text: norm(d.innerText).slice(0, 300), buttons };   // подложка без кнопок — дальше
    }
    return null;
}
"""

# Подсказки «?» у вариантов ответа: иконки-триггеры внутри строк формы
_JS_TOOLTIP_TRIGGERS = r"""
() => {
    const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim();
    const SEL = 'tui-tooltip, [tuitooltip], [tuihint], [data-tooltip], [class*="tooltip" i]';
    document.querySelectorAll('[data-agent-tip]').forEach((x) => x.removeAttribute('data-agent-tip'));
    const out = [], rows = new Set();
    let n = 0;
    for (const t of document.querySelectorAll(SEL)) {
        const r = t.getBoundingClientRect();
        if (r.width < 4 || r.height < 4 || r.width > 48 || r.height > 48) continue;
        const row = t.closest('[data-agent-id]');
        if (!row || rows.has(row)) continue;
        if (t.checkVisibility && !t.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) continue;
        rows.add(row);
        t.setAttribute('data-agent-tip', String(n));
        out.push({ n, uid: row.getAttribute('data-agent-id') });
        n++;
        if (n >= 16) break;
    }
    return out;
}
"""

_JS_HINT_TEXT = r"""
() => {
    const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim();
    const out = [];
    for (const h of document.querySelectorAll('tui-hint, [role="tooltip"], [class*="tooltip" i][class*="content" i]')) {
        const r = h.getBoundingClientRect();
        if (r.width < 4 || r.height < 4) continue;
        const t = norm(h.innerText);
        if (t) out.push(t);
    }
    return out.join(' ').slice(0, 800);
}
"""


def _similarity(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.8 + 0.15 * min(len(a), len(b)) / max(len(a), len(b))
    return difflib.SequenceMatcher(None, a, b).ratio()


def _clean_target_text(text: str) -> str:
    """LLM иногда копирует строку целиком: «[3] [OPTION] «Смартфоны» ✓ВЫБРАН»."""
    t = re.sub(r"^\s*\[\d+\]\s*", "", text or "")
    t = re.sub(r"^\s*\[[^\]]*\]\s*", "", t)
    quoted = re.search(r"«([^»]+)»", t)
    if quoted:
        t = quoted.group(1)
    for flag in (FLAG_SELECTED, FLAG_DISABLED, FLAG_OCCLUDED, FLAG_SELECTABLE, FLAG_IN_DIALOG, FLAG_IN_POPUP):
        t = t.replace(flag, "")
    t = re.sub(r"⟨[^⟩]*⟩", "", t)
    t = re.sub(r"\s→\s\S+$", "", t)
    return t.strip()


def _short(exc: BaseException) -> str:
    return str(exc).strip().splitlines()[0][:160] if str(exc).strip() else exc.__class__.__name__


def _label_matches(label: str, texts: tuple[str, ...]) -> bool:
    label = normalize_text(label)
    return any(label == t or label.startswith(t + " ") for t in texts)


def _is_wrong_answer(text: str) -> bool:
    """Сообщение платформы о неверном ответе: «Неверный ответ» или текст площадки (WRONG_ANSWER_TEXTS)."""
    return bool(_WRONG_RE.search(text)) or any(t in normalize_text(text) for t in WRONG_ANSWER_TEXTS)


def _url_tail(url: str) -> str:
    """Путь адреса — без домена («/task» в task.ozon.ru — это домен) и без параметров: у страницы
    входа в параметрах бывает адрес возврата на задание (…/auth?redirect=https://task…/task/1)."""
    return urlsplit(url or "").path.lower()


def _site_root(url: str) -> str:
    parts = urlsplit(url or "")
    return f"{parts.scheme}://{parts.netloc}" if parts.netloc else url


def _is_task_url(url: str) -> bool:
    tail = _url_tail(url)
    return (any(keyword in tail for keyword in TASK_URL_KEYWORDS)
            and not any(keyword in tail for keyword in NON_TASK_URL_KEYWORDS))


@dataclass(frozen=True)
class TaskIdentity:
    """Что считать «тем же заданием».

    Один хэш не годится: фото карусели догружаются (набор адресов растёт), счётчик
    символов в поле меняет цифры, а в заданиях «оцените фото» текст одинаков у всех
    заданий — различаются только фото."""
    content: str
    loose: str
    media: frozenset[str]
    form: str = ""
    lines: frozenset[str] = frozenset()    # строки текста страницы (без цифр и меток элементов)

    @classmethod
    def of(cls, state: PageState) -> "TaskIdentity":
        lines = frozenset(
            line for line in (re.sub(r"\d+", "", normalize_text(_plain_text([raw]))) for raw in state.reader)
            if len(line) >= 3
        )
        return cls(state.content_hash, state.loose_hash, frozenset(state.media_srcs), state.form_hash, lines)

    def same_task(self, other: "TaskIdentity", *, submitted: bool) -> bool:
        media_related = (not self.media or not other.media or bool(self.media & other.media))
        if not media_related:
            return False
        if self.content == other.content:
            # форма изменилась без отправки — это наш же выбор открыл/скрыл поле
            return self.form == other.form or not submitted
        if not submitted and self._revealed(other):
            return True
        return self.loose == other.loose and not submitted   # только цифры: таймер, счётчик

    def _revealed(self, other: "TaskIdentity") -> bool:
        """Ответ открыл (или скрыл) поля, а прежний текст страницы весь на месте: Ozon «Фото ценника
        читаемое? — Да» добавляет вопросы о ценах. Новое задание прежний текст заменяет."""
        small, big = sorted((self.lines, other.lines), key=len)
        return len(small) >= 3 and small <= big


class Agent:
    """Автономный агент для платформы T-Work v4."""

    def __init__(
        self,
        *,
        browser: Optional[BrowserController] = None,
        llm: Optional[LLMConnector] = None,
        knowledge: Optional[KnowledgeBase] = None,
    ) -> None:
        # зависимости можно подменить (тесты, другой провайдер LLM)
        self._browser = browser or BrowserController()
        self._llm = llm or LLMConnector()
        self._memory = TaskMemory()
        self._identity: Optional[TaskIdentity] = None
        self._task_identifier: str = ""
        self._submitted = False             # после нажатия «Завершить» новое задание ожидаемо
        self._tasks_done = 0
        self._wrong_total = 0
        self._seen_task = False             # агент уже был в задании (для итога заказа на списке заказов)
        # переход во вкладку с заданием — один раз, до первого задания (см. _follow_task_tab)
        self._tab_follow = True
        self._task_site = ""                # сайт заданий, где агент работал (для подсказки в кабинете)
        self._login_logged = False          # сообщение об окне входа уже выведено
        self._order_mark: Optional[tuple] = None   # начало заказа: (заданий, ошибок, токены, время)
        self._order_train = [0, 0]          # тренировка в этом заказе: верно с первого раза, всего
        self._order_exam: Optional[bool] = None    # итог экзамена в этом заказе
        self._exam_block_logged: set[str] = set()
        self._captcha_suppressed_until = 0.0
        self._last_idle_log = 0.0
        self._popup_attempts: dict[str, int] = {}
        self._media = MediaManager(self._browser, getattr(self._llm, "transcribe", None))
        self._web = WebResearch(self._browser)
        self._knowledge = knowledge or KnowledgeBase()
        self._pool: Optional[PoolKnowledge] = None
        self._frame_shot: Optional[str] = None
        self._frame_shot_task = ""
        self._doc_catcher: Optional[DocumentCatcher] = None   # документы, пока открыта инструкция
        self._instruction_seen = ""                           # что было на вкладке инструкции (для лога)
        self._instruction_urls: list[str] = []                # адреса вкладки и фреймов инструкции (PDF)
        # учёт токенов: у LLMConnector есть usage (сценарные «LLM» тестов — без него)
        usage = getattr(self._llm, "usage", None)
        self._task_usage_start = usage.snapshot() if usage is not None else None

    # ------------------------------------------------------------------
    # Главная точка входа
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Запустить агента: работает до возврата на список заказов (или лимитов)."""
        started = time.monotonic()
        check = getattr(self._llm, "check_model", None)
        if check is not None:
            await check()          # модели нет у OpenRouter — ModelUnavailable, браузер не открываем
            self._task_usage_start = self._llm.usage.snapshot()      # проверку не считаем заданием
        async with self._browser:
            logger.info("=" * 60)
            logger.info("АГЕНТ v4.9 ЗАПУЩЕН. Решаю задания открытого заказа; после заказа %s.",
                        "завершаю работу" if CLOSE_BROWSER_WHEN_DONE else
                        "жду следующий (закончить — закройте окно браузера или Ctrl+C)")
            logger.info("Площадка: %s. Шагов с действием максимум: %d, на одно задание: %d",
                        PLATFORM.title, MAX_STEPS, MAX_STEPS_PER_TASK)
            logger.info("=" * 60)
            acted = 0
            idle_since: Optional[float] = None
            stuck = ""          # агент остановился сам (простой, лимит шагов) — окно не закрываем
            while acted < MAX_STEPS:
                if self._browser.is_closed():
                    logger.warning("Окно браузера закрыто — остановка")
                    break
                try:
                    result = await self._step()
                except PlaywrightError as exc:
                    if self._browser.is_closed() or is_connection_lost(exc):
                        logger.warning("Браузер закрыт — остановка")
                        break
                    # «Execution context was destroyed» и т.п.: фрейм перезагрузился посреди шага
                    logger.warning("Ошибка Playwright в шаге (%s) — повтор на свежем снимке", _short(exc))
                    await asyncio.sleep(1.0)
                    continue
                except Exception as exc:  # noqa: BLE001 — один сбойный шаг не должен ронять агента
                    if is_connection_lost(exc):
                        logger.warning("Связь с браузером потеряна — остановка")
                        break
                    logger.exception("Непредвиденная ошибка в шаге: %s", exc)
                    await asyncio.sleep(2.0)
                    continue

                if result == StepResult.STOP:
                    break
                if result == StepResult.WAITING:
                    idle_since = None       # ждать следующий заказ можно сколько угодно
                elif result == StepResult.ACTED:
                    acted += 1
                    idle_since = None
                else:
                    idle_since = idle_since or time.monotonic()
                    if time.monotonic() - idle_since > MAX_IDLE_SECONDS:
                        logger.error("Нет активности %.0f с — остановка", MAX_IDLE_SECONDS)
                        stuck = "нет активности"
                        break
                await asyncio.sleep(0.2)
            else:
                logger.warning("ДОСТИГНУТ ЛИМИТ ШАГОВ (%d)", MAX_STEPS)
                stuck = "лимит шагов"
            await self._web.close()
            self._media.close()
            self._log_task_usage()
            minutes = (time.monotonic() - started) / 60
            logger.info(
                "ИТОГ: отправлено заданий=%d, из них платформа признала неверными=%d, "
                "шагов с действием=%d, время %.1f мин",
                self._tasks_done, self._wrong_total, acted, minutes,
            )
            usage = getattr(self._llm, "usage", None)
            if usage is not None and usage.calls:
                per_task = ""
                if self._tasks_done:
                    per_task = (f"; в среднем на задание: вход {usage.prompt // self._tasks_done}, "
                                f"выход {usage.completion // self._tasks_done}")
                logger.info("ТОКЕНЫ за запуск: %s%s%s", usage.render(), per_task, self._budget_note())
            if stuck and not CLOSE_BROWSER_WHEN_DONE and not self._browser.is_closed():
                logger.warning("Агент остановлен (%s). Окно браузера оставлено открытым, чтобы было видно, "
                               "на чём он остановился. Закройте окно браузера (или нажмите Ctrl+C), когда "
                               "закончите.", stuck)
                await self._browser.wait_closed()

    # ------------------------------------------------------------------
    # Один шаг агента
    # ------------------------------------------------------------------

    async def _step(self) -> StepResult:
        # 0. Задание в главном фрейме (Ozon): проект, открытый из кабинета в новой вкладке, — туда
        if ALLOW_MAIN_FRAME:
            self._follow_task_tab()

        # 1. Окно новостей сайта поверх задания. Когда задание в главном фрейме, а в окне не
        # страница задания (вход в Ozon ID, кабинет), окна не трогаем: «Продолжить» в окне входа
        # отправило бы SMS
        if (not ALLOW_MAIN_FRAME or _is_task_url(self._browser.page.url)) and await self._dismiss_page_popup():
            return StepResult.IDLE

        # 2. Фрейм задания
        frame = await self._browser.find_target_frame()
        if frame is None:
            # вход в аккаунт, другая страница сайта — ждём без лимита простоя (модель не вызывается)
            self._log_idle("Фрейм задания не найден — жду (если нужен вход в аккаунт, войдите в окне "
                           "браузера и откройте заказ кнопкой «Приступить»)")
            await asyncio.sleep(FRAME_LOAD_WAIT)
            return StepResult.WAITING

        if time.monotonic() > self._captcha_suppressed_until and await self._browser.page_has_captcha():
            await self._wait_captcha_solved()
            return StepResult.IDLE

        # 3. Снимок
        state = await DomParser(frame).parse()

        # 4. Список заказов (или «задачи в проекте закончились»)
        if self._is_orders_list(state) or self._is_order_done(state):
            return await self._on_orders_list()
        # Задание в главном фрейме (Ozon): на других страницах сайта — вход в аккаунт, статистика —
        # агент ничего не делает
        if frame.parent_frame is None and not _is_task_url(state.frame_url):
            if self._tab_follow:
                self._log_idle("Открыта страница сайта, а не задание — жду (если нужен вход в аккаунт, войдите "
                               "в окне браузера сами и откройте задание кнопкой «Приступить»)")
            else:
                self._log_idle("Открыта страница сайта, а не задание. Проекты в других вкладках агент после начала "
                               "работы не берёт (их можно решать самому): чтобы он продолжил, откройте список "
                               f"проектов в ЭТОЙ вкладке ({self._task_site or _site_root(TARGET_URL)}) и нажмите "
                               "«Приступить»")
            await asyncio.sleep(FRAME_LOAD_WAIT)
            return StepResult.WAITING
        if frame.parent_frame is None:
            self._task_site = _site_root(state.frame_url)       # сюда агента возвращать после кабинета

        self._refresh_task_context(frame, state)
        self._memory.verify(state)          # фактический результат прошлого действия → в историю

        if state.has_captcha and time.monotonic() > self._captcha_suppressed_until:
            await self._wait_captcha_solved()
            return StepResult.IDLE

        # 4а. Окно входа в аккаунт поверх страницы задания в главном фрейме (Ozon: сессия истекла):
        # в аккаунт входит только человек
        login = frame.parent_frame is None and self._login_window(state)
        if login:
            # ждём сколько нужно (как на странице входа): человек войдёт — агент продолжит сам
            if not self._login_logged:
                self._login_logged = True
                logger.warning("Сайт просит войти в аккаунт — войдите в окне браузера сами, агент подождёт "
                               "и продолжит после входа")
            await asyncio.sleep(FRAME_LOAD_WAIT)
            return StepResult.WAITING
        self._login_logged = False

        # 5. Диалоги поверх задания
        handled = await self._handle_dialog(frame, state)
        if handled is not None:
            return handled

        # 5а. Экзамен после плохой тренировки не начинаем и не решаем (см. EXAM_MIN_ACCURACY)
        if await self._exam_gate(state):
            return StepResult.WAITING

        # 5б. Заставка «Тренировка … [Начать]» — сразу. Под ней часто ещё крутится загрузка
        # задания (и не уходит, пока не нажата «Начать»): раньше агент сначала ждал её до ~25 с,
        # и человек успевал нажать «Начать» сам
        if state.dialog_open and await self._maybe_click_start(frame, state):
            return StepResult.ACTED

        # 6. Загрузка на всю страницу: ждём, но не бесконечно
        if state.loading and self._memory.loading_waits < 5:
            self._memory.loading_waits += 1
            logger.info("Идёт загрузка — жду стабилизации DOM")
            await self._browser.wait_settle(frame)
            await asyncio.sleep(0.5)
            return StepResult.IDLE
        if not state.loading:
            self._memory.loading_waits = 0

        # 7. Локальный флоу «Начать / ОК» (заставки и диалоги тренировки)
        if await self._maybe_click_start(frame, state):
            return StepResult.ACTED

        # 8. Нет активных элементов — ждём
        if not any(not e.is_disabled for e in state.visible_elements):
            self._log_idle("Активных элементов нет — жду загрузки/следующего задания")
            await asyncio.sleep(max(ACTION_WAIT, 1.0))
            return StepResult.IDLE
        if not self._seen_task:
            self._seen_task = True
            self._tab_follow = False        # агент начал работать — дальше только своя вкладка
            self._mark_order_start()

        # 8а. Среди ответов есть «Фото не загружается», а фото на странице ещё нет или оно грузится:
        # Ozon дорисовывает фото позже вопроса — модель выбрала бы «не загружается». Ждём до ~10 с
        if self._photo_pending(state) and self._memory.photo_waits < 5:
            self._memory.photo_waits += 1
            logger.info("Фото задания ещё не загрузилось — жду, прежде чем отвечать")
            await asyncio.sleep(2.0)
            return StepResult.IDLE

        # 9. Знания о виде задания: инструкция и подсказки «?» (один раз за запуск)
        if await self._maybe_open_instruction(frame, state):
            return StepResult.IDLE
        await self._maybe_read_tooltips(frame, state)

        # 10. Аудио: расшифровка в фоне, воспроизведение
        if state.audios:
            self._media.start_transcription(frame, state)
            await self._media.ensure_playing(frame, state)

        # 11. Бюджет шагов на задание
        self._memory.steps += 1
        if self._memory.steps > MAX_STEPS_PER_TASK:
            return await self._handle_budget_exhausted(frame, state)

        # 12. Решение LLM
        logger.info(
            "%s задание %s · шаг %d/%d · «%s»%s %s",
            "-" * 10, state.task_identifier[:8], self._memory.steps, MAX_STEPS_PER_TASK,
            state.task_preview[:50], f" · {self._model_for(self._pool)}" if len(self._ladder) > 1 else "",
            "-" * 10,
        )
        context = await self._build_context(frame, state)       # ждёт и расшифровку записей
        if state.audios and self._media.transcripts_failed(state):
            # без расшифровки ответ на задание со звонком — угадывание
            return await self._wait_for_human(
                "запись не удалось расшифровать ни одной моделью — ответ без неё был бы угадыванием")
        decision = await self._llm.decide(state, context)
        if getattr(self._llm, "unavailable", False):
            # OpenRouter недоступен (блокировка, нет связи, нет денег) — агент ждал и не решал: шаг
            # не расходует бюджет задания (иначе по его исчерпании ушёл бы случайный ответ) и не
            # считается пропуском модели
            self._memory.steps -= 1
            await asyncio.sleep(OUTAGE_PAUSE)
            return StepResult.ACTED
        if decision.plan:
            self._memory.plan = decision.plan
        if decision.observation and not all(step.action == ActionType.SKIP for step in decision.steps()):
            self._memory.observation = decision.observation

        # 13. Сверка целей и выполнение пакета действий
        await self._run_batch(frame, decision, state)
        await self._browser.wait_settle(frame)
        return StepResult.ACTED

    # ------------------------------------------------------------------
    # Список заказов и окна сайта
    # ------------------------------------------------------------------

    @staticmethod
    def _is_orders_list(state: PageState) -> bool:
        """Список заказов: фрейм не на странице задания и есть кнопки «Приступить»."""
        if _is_task_url(state.frame_url):
            return False
        return any(
            e.kind == ElementKind.BUTTON and _label_matches(e.text, ORDERS_BUTTON_TEXTS)
            for e in state.elements
        )

    @staticmethod
    def _photo_pending(state: PageState) -> bool:
        """В ответах есть «Фото не загружается», а загруженного фото на странице нет."""
        about_photo = any(e.kind == ElementKind.OPTION and _PHOTO_FAIL_RE.search(e.text or "")
                          for e in state.visible_elements)
        # ни одного загруженного фото (миниатюры за краем экрана не в счёт — они не грузятся до прокрутки)
        return about_photo and not any(i.width for i in state.images)

    @staticmethod
    def _login_window(state: PageState) -> bool:
        """Открыто окно входа в аккаунт: поле пароля или телефона, или поле ввода в окне, где речь о
        входе, коде из SMS, пароле (поля кода часто type=text)."""
        if not state.dialog_open:
            return False
        fields = [e for e in state.visible_elements if e.kind == ElementKind.INPUT and e.container == "dialog"]
        if any(e.input_type in ("password", "tel") for e in fields):
            return True
        return bool(fields) and bool(_LOGIN_RE.search(_plain_text(state.dialog_lines)))

    @staticmethod
    def _is_order_done(state: PageState) -> bool:
        """«В текущем проекте закончились задачи» (ORDERS_DONE_TEXTS) на месте задания, полей ответа
        нет — заказ выполнен, как при возврате на список заказов."""
        if not ORDERS_DONE_TEXTS:
            return False
        text = normalize_text(state.task_text)
        if not any(normalize_text(t) in text for t in ORDERS_DONE_TEXTS):
            return False
        answer_kinds = (ElementKind.OPTION, ElementKind.FOLDER, ElementKind.INPUT, ElementKind.DROPDOWN)
        return not any(e.kind in answer_kinds and e.container == "" for e in state.visible_elements)

    async def _on_orders_list(self) -> StepResult:
        """Список заказов: сам заказ агент не выбирает. После выполненного заказа — итог и
        ожидание следующего (браузер не закрывается), либо выход (CLOSE_BROWSER_WHEN_DONE)."""
        if self._seen_task:
            self._seen_task = False
            self._log_task_usage()
            self._log_order_summary()
            done = ("❌ Экзамен не пройден — платформа вернула на список заказов" if self._order_exam is False
                    else "✅ Платформа вернула на список заказов — заказ выполнен")
            if STOP_ON_ORDERS_LIST and CLOSE_BROWSER_WHEN_DONE:
                logger.info("%s, агент завершает работу", done)
                return StepResult.STOP
            where = (" в этой же вкладке («К списку проектов» → «Приступить»; проекты в других вкладках агент "
                     "не трогает — их можно решать самому)" if ALLOW_MAIN_FRAME else " («Приступить»)")
            logger.info("%s. Браузер остаётся открытым: откройте следующий заказ%s — агент продолжит сам. "
                        "Закончить работу — закройте окно браузера или нажмите Ctrl+C в этом окне.", done, where)
            self._last_idle_log = time.monotonic()
        else:
            self._log_idle("Открыт список заказов. Выберите заказ и нажмите «Приступить» — агент начнёт решать "
                           "задания (заказ выбираете вы)")
        await asyncio.sleep(FRAME_LOAD_WAIT)
        return StepResult.WAITING

    def _mark_order_start(self) -> None:
        usage = getattr(self._llm, "usage", None)
        self._order_mark = (self._tasks_done, self._wrong_total,
                            usage.snapshot() if usage is not None else None, time.monotonic())
        self._order_train = [0, 0]
        self._order_exam = None

    # ------------------------------------------------------------------
    # Тренировка и экзамен
    # ------------------------------------------------------------------

    @staticmethod
    def _page_mode(state: PageState) -> str:
        """«training» / «exam» по панели режима страницы, «» — не видно."""
        for line in _plain_text(state.reader).splitlines()[:15]:
            m = _MODE_RE.match(line.strip())
            if m:
                return "exam" if m.group(1).lower() == "экзамен" else "training"
        return ""

    def _record_training(self, mode: str, *, accepted: bool, answer: str) -> None:
        """Первый ответ на задание тренировки: верен ли он; принятый ответ — правильный.
        Вне тренировки правильность неизвестна (экзамен подсказок не даёт)."""
        pool, mem = self._pool, self._memory
        if accepted:
            self._save_case_lesson(accepted=answer)
        if mode == "exam" or (accepted and mode != "training"):
            return
        model = self._model_for(pool)
        if accepted:
            if not mem.wrong_answers:
                first_ok = not mem.human_answer         # ответ модели поправил человек — ошибка модели
                self._order_train[0] += int(first_ok)
                self._order_train[1] += 1
                if pool is not None:
                    self._knowledge.training_attempt(pool, first_ok)
                    self._knowledge.model_attempt(pool, model, first_ok)
                    self._climb_ladder(pool, model)
            # после двух неверных ответов платформа принимает любой третий — он не «правильный»
            if (pool is not None and len(mem.wrong_answers) < 2
                    and " = «" not in answer and answer.startswith("«")):
                self._knowledge.training_answer(pool, answer)
        elif len(mem.wrong_answers) == 1:
            self._order_train[1] += 1
            if pool is not None:
                self._knowledge.training_attempt(pool, False)
                self._knowledge.model_attempt(pool, model, False)
                self._climb_ladder(pool, model)

    def _save_case_lesson(self, *, accepted: str = "") -> None:
        """Разбор ошибки в базу знаний — один на задание: что было в задании (как его увидела
        модель), какие ответы платформа отклонила, её подсказка и верный ответ. Такой разбор
        модель применяет к похожим заданиям, а не ко всем подряд (как голое «ответ X — неверно»).
        Ответ, выбранный человеком в окне браузера, — тоже урок (в экзамене платформа молчит)."""
        pool, mem = self._pool, self._memory
        if pool is None or mem.lesson_saved or not (mem.wrong_answers or mem.human_answer):
            return
        mem.lesson_saved = True
        case = re.sub(r"\s+", " ", mem.case or mem.observation).strip().rstrip(".")
        parts = [f"Задание: {case[:320]}." if case else f"Задание «{pool.title[:70]}»."]
        if mem.wrong_answers:
            parts.append("Неверно: " + "; ".join(mem.wrong_answers) + ".")
        if mem.hints:
            parts.append("Подсказка платформы: " + " ".join(mem.hints)[:700])
        if mem.human_answer and accepted == mem.human_answer:
            parts.append(f"Человек (он слушал/смотрел сам) поправил ответ на {mem.human_answer} — это верный ответ.")
        elif mem.human_answer:
            parts.append(f"Человек выбирал ответ {mem.human_answer}, отправлен {accepted or 'другой'} — "
                         "верен ли он, не известно.")
        elif accepted and len(mem.wrong_answers) < 2:
            parts.append(f"Верный ответ: {accepted}.")
        elif accepted:
            parts.append(f"Третий ответ {accepted} принят, но после двух ошибок платформа пропускает "
                         "дальше с любым ответом — верен ли он, не известно.")
        else:
            parts.append("Верный ответ не известен.")
        self._knowledge.add_lesson(pool, " ".join(parts))

    # ------------------------------------------------------------------
    # Лестница моделей: самая дешёвая модель, которая справляется с этим видом заданий
    # ------------------------------------------------------------------

    @property
    def _ladder(self) -> tuple[str, ...]:
        """Лестница моделей: LLM_MODEL из .env без моделей, недоступных при проверке на старте."""
        models = getattr(getattr(self, "_llm", None), "models", None)
        return tuple(models) if models else LLM_MODELS

    def _model_for(self, pool: Optional[PoolKnowledge]) -> str:
        """Модель для вида заданий: закреплённая в «Заметках» → выбранная в тренировке (если она
        есть в нынешней лестнице) → самая дешёвая, которая не провалила тренировку этого вида."""
        ladder = self._ladder
        if pool is None:
            return ladder[0]
        if pool.pinned_model:
            return pool.pinned_model
        if pool.model in ladder:
            return pool.model
        return next((m for m in ladder if not self._ruled_out(pool, m)), ladder[-1])

    @staticmethod
    def _ruled_out(pool: PoolKnowledge, model: str) -> bool:
        """Модель не справляется с видом: точность тренировки ниже порога LADDER_MIN_ACCURACY —
        после LADDER_MIN_TASKS заданий или раньше, если порог уже недостижим (даже если
        оставшиеся до минимума ответы будут верными)."""
        ok, total = pool.model_stats.get(model, [0, 0])
        need = max(LADDER_MIN_TASKS, total)
        return bool(total) and ok + need - total < LADDER_MIN_ACCURACY * need

    def _climb_ladder(self, pool: PoolKnowledge, model: str, *, reason: str = "") -> None:
        """Перейти к следующей модели лестницы, если текущая часто ошибается в тренировке
        (или не сдала экзамен). Выше справляющихся нет — остаться на самой точной из опробованных
        (сильная модель не всегда точнее дешёвой), но сравнивать не раньше, чем текущая решит
        LADDER_MIN_TASKS заданий: по одному-трём заданиям точность не видна. Закреплённую
        человеком модель агент не меняет."""
        ladder = self._ladder
        if pool.pinned_model or model not in ladder:
            return
        if not reason and not self._ruled_out(pool, model):
            return
        above = [m for m in ladder[ladder.index(model) + 1:] if not self._ruled_out(pool, m)]
        if above:
            if not reason:
                ok, total = pool.model_stats[model]
                reason = f"в тренировке с первого раза верно {ok} из {total}"
            self._knowledge.set_model(pool, above[0])
            logger.warning("📈 Вид «%s»: модель %s не справляется (%s) — дальше этот вид решает %s",
                           pool.title, model, reason, above[0])
            return
        if reason:                      # экзамен не сдан, а сильнее модели нет — выбирать не из чего
            return
        if pool.model_stats.get(model, [0, 0])[1] < LADDER_MIN_TASKS:
            return                      # у этой модели ещё мало заданий, чтобы сравнивать

        def score(name: str) -> float:  # доля верных с первого раза, сглаженная на малом числе заданий
            ok, total = pool.model_stats.get(name, [0, 0])
            return (ok + 1) / (total + 2)

        tried = [m for m in ladder if pool.model_stats.get(m, [0, 0])[1] > 0]
        best = max(tried, key=lambda m: (score(m), -ladder.index(m)))
        if best == model or score(best) <= score(model):
            return
        (ok, total), (bok, btotal) = pool.model_stats[model], pool.model_stats[best]
        self._knowledge.set_model(pool, best)
        logger.warning("📉 Вид «%s»: модель %s точнее не стала (с первого раза верно %d из %d, у %s — %d из %d) — "
                       "дальше этот вид решает %s", pool.title, model, ok, total, best, bok, btotal, best)

    def _exam_estimate(self) -> Optional[tuple[float, str]]:
        """Ожидаемая точность на экзамене и откуда она (для лога). По тренировке этого заказа
        (если в нём меньше EXAM_MIN_TASKS заданий — по всем тренировкам этого вида). С лестницей
        моделей — точность той модели, которая будет решать экзамен, но пока у неё мало заданий,
        оценка тянется к точности всей тренировки (одна ошибка из трёх ещё не значит 66%).
        None — заданий мало, и ошибок среди них не больше, чем допускает порог: можно начинать."""
        pool = self._pool
        ok, total = self._order_train
        if total < EXAM_MIN_TASKS and pool is not None and pool.train_total >= EXAM_MIN_TASKS:
            ok, total = pool.train_first_ok, pool.train_total
        if not total or (total < EXAM_MIN_TASKS
                         and ok + EXAM_MIN_TASKS - total >= EXAM_MIN_ACCURACY * EXAM_MIN_TASKS):
            return None
        # сглаживание: по 4–5 заданиям точность видна плохо, а проваленный экзамен закрывает
        # задания этого вида — «4 из 5» считается как ≈ 71%, «3 из 4» — как ≈ 67%
        overall = (ok + 1) / (total + 2)
        source = f"в тренировке с первого раза верно {ok} из {total}"
        if pool is not None and len(self._ladder) > 1:
            model = self._model_for(pool)
            mok, mtotal = pool.model_stats.get(model, [0, 0])
            if mtotal and (mok, mtotal) != (ok, total):
                estimate = (mok + _EXAM_PRIOR * overall) / (mtotal + _EXAM_PRIOR)
                return estimate, f"{source}; у модели {model}, которая будет решать экзамен, — {mok} из {mtotal}"
        return overall, source

    async def _exam_gate(self, state: PageState) -> bool:
        """Экзамен после тренировки с низкой точностью не начинать (окно «Экзамен … Начать»)
        и не решать (страница в режиме «Экзамен») — человек решает сам или меняет настройку."""
        if EXAM_MIN_ACCURACY <= 0:
            return False
        dialog_text = normalize_text(_plain_text(state.dialog_lines))
        dialog_buttons = [e for e in state.visible_elements
                          if e.container == "dialog" and e.kind == ElementKind.BUTTON]
        exam_start = (state.dialog_open and "экзамен" in dialog_text
                      and any(_label_matches(b.text, START_BUTTON_TEXTS) for b in dialog_buttons))
        if not exam_start and self._page_mode(state) != "exam":
            return False
        estimate = self._exam_estimate()
        pool = self._pool
        if estimate is None and pool is not None and pool.instruction_unread and not pool.instruction.strip():
            # ни инструкции, ни тренировки — экзамен вслепую; не сдан — задания этого вида закроются
            if pool.key not in self._exam_block_logged:
                self._exam_block_logged.add(pool.key)
                logger.warning(
                    "⛔ ЭКЗАМЕН НЕ НАЧИНАЮ: инструкцию к этому виду заданий прочитать не удалось, а тренировки "
                    "этого вида у агента нет — это решение вслепую, а после проваленного экзамена задания этого "
                    "вида закрываются. Пройдите экзамен сами (агент не мешает и ждёт), сначала пройдите с агентом "
                    "тренировку или впишите текст инструкции в раздел «Инструкция» файла %s",
                    self._knowledge.ensure_file(pool) or "этого вида в папке knowledge")
            await asyncio.sleep(FRAME_LOAD_WAIT)
            return True
        if estimate is None or estimate[0] + 1e-9 >= EXAM_MIN_ACCURACY:
            return False
        key = self._pool.key if self._pool is not None else ""
        if key not in self._exam_block_logged:
            self._exam_block_logged.add(key)
            accuracy, source = estimate
            logger.warning(
                "⛔ ЭКЗАМЕН НЕ НАЧИНАЮ: %s — ожидаемая точность ≈ %d%%, а нужно не меньше %d%%. С такой "
                "точностью экзамен, скорее всего, не будет сдан — денег за него не будет, а токены уйдут. "
                "Пройдите экзамен сами (агент не мешает и ждёт) или, чтобы агент решал его, поставьте в .env "
                "EXAM_MIN_ACCURACY=0", source, int(accuracy * 100), round(EXAM_MIN_ACCURACY * 100))
        await asyncio.sleep(FRAME_LOAD_WAIT)
        return True

    def _log_order_summary(self) -> None:
        if self._order_mark is None:
            return
        tasks0, wrong0, usage0, started = self._order_mark
        self._order_mark = None
        line = (f"ИТОГ ЗАКАЗА: отправлено заданий={self._tasks_done - tasks0}, из них неверных="
                f"{self._wrong_total - wrong0}, время {(time.monotonic() - started) / 60:.1f} мин")
        ok, total = self._order_train
        if total:
            line += f"; тренировка: с первого раза верно {ok} из {total}"
        if self._order_exam is not None:
            line += f"; экзамен: {'пройден' if self._order_exam else 'НЕ ПРОЙДЕН'}"
        if len(self._ladder) > 1:
            line += f"; модель: {self._model_for(self._pool)}"
        usage = getattr(self._llm, "usage", None)
        if usage is not None and usage0 is not None:
            spent = usage.since(usage0)
            if spent.calls:
                line += f"; {spent.render()}"
            line += self._budget_note(spent.cost if spent.calls else 0.0)
        logger.info(line)

    def _budget_note(self, order_cost: float = 0.0) -> str:
        """Остаток лимита ключа OpenRouter и на сколько таких заказов его хватит."""
        budget = getattr(self._llm, "budget_left", None)
        left = budget() if callable(budget) else None
        if left is None:
            return ""
        note = f"; на ключе OpenRouter осталось ≈ ${left:.2f}"
        if order_cost > 0:
            note += f" — примерно на {int(left / order_cost + 1e-9)} таких заказов"
        return note

    def _follow_task_tab(self) -> None:
        """Вкладка агента — вне сайта заданий (кабинет profit.ozon.ru, вход), а в другой вкладке того
        же окна открыто задание (кабинет открывает проект в новой вкладке): работать там.

        Если агент уже на сайте заданий (список проектов, задание, «закончились задачи»), чужие
        вкладки этого сайта он не забирает: в них человек может решать проект (экзамен) сам.
        Следующий проект — в вкладке агента: «К списку проектов» → «Приступить».
        Переход — только один раз и только пока агент ещё не работал (запустили агента, он ждёт в
        кабинете, человек открыл проект): иначе из кабинета, куда агент вернулся после закрытой
        вкладки, он забрал бы проект, который человек открыл для себя."""
        browser = self._browser
        current = browser.page.url
        if not self._tab_follow or _is_task_url(current):
            return
        here = (urlsplit(current).hostname or "").lower()
        search_tab = getattr(self._web, "_page", None)
        for page in reversed(browser.context.pages):
            if page is browser.page or page is search_tab or page.is_closed() or not _is_task_url(page.url):
                continue
            if (urlsplit(page.url).hostname or "").lower() == here:
                continue
            logger.info("Задание открыто в другой вкладке (%s) — перехожу в неё", page.url[:80])
            browser.adopt_page(page)
            self._tab_follow = False
            return

    async def _dismiss_page_popup(self) -> bool:
        """Новости/объявления сайта (например, «Одноразовые пароли для TWork») открываются
        поверх фрейма задания и перехватывают клики. Закрываем кнопкой «Закрыть/Далее/OK»."""
        page = self._browser.page
        try:
            popup = await page.main_frame.evaluate(_JS_PAGE_POPUP, list(DIALOG_SELECTORS))
        except PlaywrightError:
            return False
        if not popup:
            return False
        text = str(popup.get("text") or "")
        key = normalize_text(text)[:120]
        if self._popup_attempts.get(key, 0) >= 3:
            return False
        buttons = popup.get("buttons") or []

        def pick(texts: tuple[str, ...]) -> Optional[dict]:
            for b in buttons:
                label = str(b.get("text") or "")
                if label and _label_matches(label, texts) and not any(d in normalize_text(label)
                                                                      for d in FINISH_DENY_SUBSTRINGS):
                    return b
            return None

        button = pick(DIALOG_CLOSE_TEXTS) or pick(EXIT_CANCEL_TEXTS)
        if button is None:
            button = next((b for b in buttons if not b.get("text")
                           and re.search(r"(закрыть|close|cross)", str(b.get("aria") or ""), re.IGNORECASE)), None)
        if button is None:
            return False
        self._popup_attempts[key] = self._popup_attempts.get(key, 0) + 1
        logger.info("Закрываю всплывающее окно сайта «%s» кнопкой «%s»", text[:60], button.get("text") or "×")
        locator = page.main_frame.locator(f'[data-agent-popup="{button["i"]}"]')
        try:
            if await locator.count():
                await self._browser.click_locator(locator.first, str(button.get("text") or "закрыть"))
                await asyncio.sleep(0.6)
                return True
        except PlaywrightError as exc:
            logger.debug("Окно сайта не закрылось: %s", _short(exc))
        return False

    # ------------------------------------------------------------------
    # Память и знания задания
    # ------------------------------------------------------------------

    def _refresh_task_context(self, frame: Frame, state: PageState) -> None:
        """Сбрасывать память ТОЛЬКО при смене задания (см. TaskIdentity)."""
        identity = TaskIdentity.of(state)
        if self._identity is None or not self._identity.same_task(identity, submitted=self._submitted):
            if self._identity is not None:
                self._log_task_usage()
            logger.info(
                "СМЕНА ЗАДАНИЯ: %s → %s «%s»",
                self._task_identifier[:8] or "—", state.task_identifier[:8], state.task_preview[:60],
            )
            self._save_case_lesson()          # задание сменилось без принятого ответа
            self._memory.reset(state.task_identifier)
            self._media.forget_task()
            self._stop_catching_documents()
            self._frame_shot = None
            pool = self._knowledge.for_state(state)
            if pool is not None and pool is not self._pool:
                known = ("инструкцию перечитаю: прежде схемы и таблицы из PDF записывались одной строкой"
                         if pool.instruction_outdated else
                         "есть инструкция" if pool.has_instruction else "инструкции пока нет")
                logger.info("Вид задания: «%s» (%s, уроков: %d)", pool.title, known, len(pool.lessons))
            self._pool = pool
        self._identity = identity
        self._task_identifier = state.task_identifier
        self._submitted = False

    def _log_task_usage(self) -> None:
        """Расход токенов на прошлое задание (по данным API: вход, из кэша, выход)."""
        usage = getattr(self._llm, "usage", None)
        if usage is None or self._task_usage_start is None:
            return
        spent = usage.since(self._task_usage_start)
        self._task_usage_start = usage.snapshot()
        if spent.calls:
            logger.info("Токены на задание %s: %s", self._task_identifier[:8] or "—", spent.render())

    async def _build_context(self, frame: Frame, state: PageState) -> DecisionContext:
        mem = self._memory
        steps_left = MAX_STEPS_PER_TASK - mem.steps
        notes: list[str] = []
        if steps_left <= 5:
            notes.append(
                f"Осталось шагов на это задание: {steps_left}. Если ответ уже заполнен — submit; "
                "иначе заверши заполнение лучшими доступными вариантами."
            )
        if mem.consecutive_skips >= 2:
            notes.append(f"Ты пропустил {mem.consecutive_skips} шага подряд — выбери конкретное действие.")
        if mem.invalid_targets:
            notes.append("Прошлое действие ссылалось на несуществующий элемент: бери номер и текст "
                         "ТОЛЬКО из текущей страницы.")
        if mem.repeated_forbidden:
            notes.append("Ты повторил действие из НЕ ПОВТОРЯТЬ — выбери другой элемент или другое действие.")
        notes += mem.batch_notes
        mem.batch_notes = []
        if mem.web_queries and sum(mem.web_queries.values()) >= MAX_WEB_PER_TASK:
            notes.append("Лимит поисковых запросов на задание исчерпан — отвечай по уже найденным данным.")

        # При зацикливании слегка «встряхиваем» детерминированную модель
        temperature = None
        if mem.consecutive_skips >= 3 or mem.repeated_forbidden >= 1:
            temperature = max(LLM_TEMPERATURE, 0.4)

        images, image_notes = [], []
        image_b64 = None
        if LLM_VISION in ("auto", "image") and state.images:
            images, image_notes = await self._media.vision_images(frame, state)
        if LLM_VISION == "frame" or (LLM_VISION in ("auto", "image") and not state.images and state.image_src):
            image_b64 = await self._frame_screenshot(frame, state)

        transcripts = await self._media.transcripts(state) if state.audios else []
        # сама запись — модели, которая умеет слушать (коннектор отдаст её только такой модели)
        audio = self._media.audio_clips(state) if AUDIO_TO_MODEL and state.audios else []
        # полностью — только последняя открытая страница; факты с прежних модель переносит в plan
        research = [
            r.render(i + 1, full=i == len(mem.web_results) - 1) for i, r in enumerate(mem.web_results)
        ]
        return DecisionContext(
            history=mem.history_lines(LLM_HISTORY_SIZE),
            forbidden=mem.forbidden_lines(),
            notes=notes,
            step_in_task=mem.steps,
            steps_left=steps_left,
            image_b64=image_b64,
            temperature=temperature,
            images=images,
            image_notes=image_notes,
            transcripts=transcripts,
            knowledge=self._knowledge.prompt_text(self._pool),
            model=self._model_for(self._pool),
            research=research,
            plan=mem.plan,
            feedback=list(mem.feedback),
            wrong_answers=list(mem.wrong_answers),
            audio=audio,
        )

    async def _frame_screenshot(self, frame: Frame, state: PageState) -> Optional[str]:
        """Скриншот фрейма (LLM_VISION=frame или картинка без <img>, например canvas)."""
        if LLM_VISION == "frame" or self._frame_shot_task != state.task_identifier:
            self._frame_shot = await self._browser.screenshot_b64(frame, "frame")
            self._frame_shot_task = state.task_identifier
        return self._frame_shot

    # ------------------------------------------------------------------
    # Диалоги поверх задания
    # ------------------------------------------------------------------

    async def _handle_dialog(self, frame: Frame, state: PageState) -> Optional[StepResult]:
        """Диалоги, которые агент закрывает сам. None — диалога нет или решает LLM."""
        if not state.dialog_open:
            return None
        dialog = [e for e in state.visible_elements if e.container == "dialog"]
        buttons = [e for e in dialog if e.kind == ElementKind.BUTTON and not e.is_disabled]
        text = _plain_text(state.dialog_lines)

        # «Выйти из задания?» — агент задания не бросает: «Нет, остаться»
        denied = [e for e in state.elements if e.container == "dialog" and is_denied_button(e)]
        if denied:
            cancel = next((b for b in buttons if _label_matches(b.text, EXIT_CANCEL_TEXTS)), None)
            if cancel is not None:
                logger.warning("Открыт диалог «%s» — отвечаю «%s»", text[:60], cancel.label())
                await self._browser.click_element(frame, cancel)
                await self._browser.wait_settle(frame)
                return StepResult.ACTED

        # Инструкция к заданию: дождаться загрузки, прочитать, сохранить, закрыть.
        # Заставка «Тренировка … изучите инструкцию … [Начать]» — не инструкция: её
        # закрывает локальный флоу кнопкой «Начать».
        start_button = any(_label_matches(b.text, START_BUTTON_TEXTS) for b in buttons)

        # «Экзамен не пройден / пройден» — записать итог и закрыть окно без модели
        outcome = None if start_button else _exam_outcome(text)
        if outcome is not None and buttons:
            key = "exam:" + normalize_text(text)[:80]
            if self._memory.dialog_attempts[key] < 3:
                if self._memory.dialog_attempts[key] == 0:
                    self._order_exam = outcome
                    if self._pool is not None:
                        self._knowledge.exam_result(self._pool, outcome)
                        if not outcome:
                            self._climb_ladder(self._pool, self._model_for(self._pool), reason="экзамен не пройден")
                    if outcome:
                        logger.info("✅ ЭКЗАМЕН ПРОЙДЕН: «%s»", text[:120])
                    else:
                        logger.warning("❌ ЭКЗАМЕН НЕ ПРОЙДЕН: «%s»", text[:120])
                self._memory.dialog_attempts[key] += 1
                button = next((b for b in buttons if _label_matches(b.text, DIALOG_CLOSE_TEXTS)), buttons[0])
                logger.info("Закрываю окно итога экзамена кнопкой «%s»", button.label())
                await self._browser.click_element(frame, button)
                await self._browser.wait_settle(frame)
                return StepResult.ACTED
        size = len(re.sub(r"\s+", "", text))
        titled = bool(_INSTRUCTION_RE.search(text[:120]))
        instruction = (state.dialog_loading or state.dialog_frames > 0 or size > 700
                       or (titled and not start_button))
        if (instruction and not titled and self._memory.instruction_pages == 0
                and self._pool is not None and self._pool.has_instruction):
            # инструкция вида уже прочитана, а это большое окно без слова «инструкция» (Ozon:
            # «Уведомления» — окно того же вида) — не следующая её страница: инструкцию оно не заменяет
            instruction = False
        if instruction:
            key = normalize_text(text)[:80]
            if self._memory.dialog_attempts[key] < 3:
                self._memory.dialog_attempts[key] += 1
                return await self._read_instruction_dialog(frame, state)
        return None

    async def _read_instruction_dialog(self, frame: Frame, state: PageState) -> StepResult:
        deadline = time.monotonic() + INSTRUCTION_WAIT
        if state.dialog_loading:
            logger.info("Открыта инструкция — жду загрузки (до %.0f с)", INSTRUCTION_WAIT)
        while state.dialog_loading and time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            state = await DomParser(frame).parse(quiet=True)
            if not state.dialog_open:
                return StepResult.IDLE
        text = _plain_text(state.dialog_lines)
        if state.dialog_frames:
            text = "\n".join(filter(None, [text, await self._read_child_frames(frame)]))
        self._instruction_urls += [c.url for c in frame.child_frames if c.url.startswith(("http://", "https://"))
                                   and c.url not in self._instruction_urls]
        failed = any("ошибка загрузки" in n.lower() for n in state.notice_texts())
        title = (text.strip().splitlines() or ["инструкция"])[0][:80]
        body = text
        if len(re.sub(r"\s+", "", body)) < 200 and not failed:
            extra = await self._read_instruction_tab(frame, state)
            if not meaningful(extra):
                extra = await self._instruction_from_documents() or extra
            if extra:
                body = f"{text}\n{extra}"
        if self._pool is not None:
            self._pool.instruction_attempted = True
            if len(re.sub(r"\s+", "", body)) >= 200:
                # вторая и следующие страницы (кнопка «Далее») дописываются, а не заменяют первую
                self._knowledge.save_instruction(self._pool, body, f"диалог «{title}»",
                                                 append=self._memory.instruction_pages > 0)
                self._memory.instruction_pages += 1
                self._memory.add(ActionType.CLICK, None, note="(авто)",
                                 result=f"📘 прочитана инструкция «{title[:60]}» — она в разделе ЗНАНИЯ")
        if failed:
            if self._pool is not None and not self._pool.instruction.strip():
                self._pool.instruction_unread = True
            logger.warning("Инструкция не загрузилась (сообщение платформы) — продолжаю без неё")
        elif len(re.sub(r"\s+", "", body)) < 200:
            if self._pool is not None:
                self._pool.instruction_unread = True
            logger.warning("Инструкция открыта, но текста в ней не найдено (%d симв.). Вкладка: %s. Скачано: %s",
                           len(body), self._instruction_seen or "не открывалась",
                           describe_documents(self._doc_catcher))
        self._stop_catching_documents()
        await self._close_dialog(frame, state)
        return StepResult.ACTED

    async def _read_child_frames(self, frame: Frame) -> str:
        """Текст документов во вложенных iframe (инструкция бывает отдельной страницей)."""
        texts: list[str] = []
        for child in frame.child_frames:
            try:
                element = await child.frame_element()
                inside = await element.evaluate(
                    "(f) => !!f.closest('[role=\"dialog\"], [aria-modal=\"true\"], dialog[open], tui-dialog')"
                )
                if not inside:
                    continue
                await child.wait_for_load_state("load", timeout=10_000)
                texts.append(await child.evaluate("() => document.body ? document.body.innerText : ''"))
            except PlaywrightError as exc:
                logger.debug("Вложенный документ не прочитан: %s", _short(exc))
        return "\n".join(t for t in texts if t and t.strip())

    async def _read_instruction_tab(self, frame: Frame, state: PageState) -> str:
        """Инструкция без текста в диалоге: кнопка «открыть в новой вкладке» → читаем вкладку."""
        close_like = DIALOG_CLOSE_TEXTS + START_BUTTON_TEXTS
        candidates = [
            e for e in state.visible_elements
            if e.container == "dialog" and e.kind in (ElementKind.BUTTON, ElementKind.OTHER)
            and not _label_matches(e.text, close_like)
            and (_INSTRUCTION_RE.search(e.text or "") or _NEW_TAB_RE.search(e.text or ""))
        ]
        if not candidates:
            return ""
        context = self._browser.context
        if self._doc_catcher is None:
            self._start_catching_documents()
        before = set(context.pages)
        await self._browser.click_element(frame, candidates[0])
        await asyncio.sleep(2.0)
        new_pages = [p for p in context.pages if p not in before]
        text = ""
        for page in new_pages:
            text = await self._read_tab(page) or text
        await self._browser.ensure_front()
        return text.strip()

    async def _read_tab(self, page) -> str:
        """Текст вкладки с инструкцией: страница догружает его скриптом — ждём до INSTRUCTION_WAIT
        секунд, читаем все фреймы. Вкладку закрываем."""
        text = ""
        try:
            await page.wait_for_load_state("load", timeout=15_000)
            catcher = self._doc_catcher
            text = await page_text(page, INSTRUCTION_WAIT,
                                   give_up=(lambda: bool(catcher.documents)) if catcher is not None else None)
            self._instruction_urls += [u for u in document_urls([page]) if u not in self._instruction_urls]
            if meaningful(text):
                logger.info("Инструкция прочитана из вкладки %s (%d симв.)", page.url[:80], len(text))
            else:
                logger.info("Вкладка с инструкцией %s: текста на странице нет (%d симв.) — ищу документ",
                            page.url[:80], len(text))
                self._instruction_seen = f"{page.url[:80]}: {await describe_page(page)}"
        except PlaywrightError as exc:
            logger.info("Вкладку с инструкцией прочитать не удалось (%s)", _short(exc))
        finally:
            try:
                await page.close()
            except PlaywrightError:
                pass
        return text

    def _start_catching_documents(self) -> None:
        self._stop_catching_documents()
        self._instruction_seen = ""
        self._instruction_urls = []
        try:
            self._doc_catcher = DocumentCatcher(self._browser.context)
        except PlaywrightError as exc:
            logger.debug("Документы инструкции не отслеживаются: %s", _short(exc))

    def _stop_catching_documents(self) -> None:
        if self._doc_catcher is not None:
            self._doc_catcher.close()
            self._doc_catcher = None

    async def _instruction_from_documents(self) -> str:
        """Инструкция не текстом страницы: JSON с текстом от сервера или PDF (его переписывает
        модель — один раз для вида заданий, дальше текст в knowledge/)."""
        catcher = self._doc_catcher
        if catcher is None:
            return ""
        await catcher.settle()
        text = catcher.json_text()
        if meaningful(text):
            logger.info("📘 Инструкция взята из ответа сервера (%d симв.)", len(text))
            return text
        reader = getattr(self._llm, "read_document", None)
        if reader is None or not (catcher.pdfs() or self._instruction_urls):
            return ""
        frame = await self._browser.find_target_frame()
        docs = await catcher.full_pdfs(self._instruction_urls, referer=frame.url if frame else self._browser.page.url,
                                       frame=frame)
        if not docs and catcher.pdfs():
            logger.warning("📘 PDF инструкции пришёл не целиком (%s) и целиком не скачался — модели его не "
                           "отправляю (она вернула бы ошибку)", describe_documents(catcher))
        for doc in docs[:2]:
            logger.info("📘 Инструкция — PDF (%d КБ): переписываю её текст моделью (один раз для вида заданий)",
                        max(len(doc.data) // 1024, 1))
            text = await reader(doc.data, "instruction.pdf", "application/pdf") or ""
            if meaningful(text):
                return text
        return ""

    async def _close_dialog(self, frame: Frame, state: PageState) -> None:
        fresh = await DomParser(frame).parse(quiet=True)
        buttons = [e for e in fresh.visible_elements
                   if e.container == "dialog" and e.kind == ElementKind.BUTTON and not e.is_disabled]
        button = next((b for b in buttons if _label_matches(b.text, DIALOG_CLOSE_TEXTS)), None) \
            or next((b for b in buttons if _label_matches(b.text, START_BUTTON_TEXTS)), None)
        if button is not None:
            logger.info("Закрываю диалог кнопкой «%s»", button.label())
            await self._browser.click_element(frame, button)
        else:
            logger.info("Закрываю диалог клавишей Escape")
            try:
                await self._browser.page.keyboard.press("Escape")
            except PlaywrightError:
                pass
        await self._browser.wait_settle(frame)

    # ------------------------------------------------------------------
    # Инструкция и подсказки вида задания
    # ------------------------------------------------------------------

    async def _maybe_open_instruction(self, frame: Frame, state: PageState) -> bool:
        """Открыть «Подробную инструкцию» один раз для вида задания без сохранённой инструкции."""
        pool = self._pool
        if not READ_INSTRUCTIONS or pool is None or pool.has_instruction or pool.instruction_attempted:
            return False
        if state.dialog_open or state.loading:
            return False
        link = next((
            e for e in state.visible_elements
            if e.kind in (ElementKind.BUTTON, ElementKind.OTHER) and not e.occluded and not e.is_disabled
            and e.container == "" and _INSTRUCTION_RE.search(e.text or "") and len(e.text) <= 60
        ), None)
        pool.instruction_attempted = True
        if link is None:
            return False
        logger.info("📘 Открываю «%s», чтобы прочитать правила задания", link.label())
        context = self._browser.context
        self._start_catching_documents()
        before = set(context.pages)
        outcome = await self._browser.click_element(frame, link)
        if not outcome.ok:
            self._stop_catching_documents()
            return False
        await self._browser.wait_settle(frame)
        await asyncio.sleep(0.5)
        # инструкция открылась в новой вкладке, а не диалогом
        new_pages = [p for p in context.pages if p not in before]
        for page in new_pages:
            text = await self._read_tab(page)
            if not meaningful(text):
                text = await self._instruction_from_documents()
            if meaningful(text):
                self._knowledge.save_instruction(pool, text, f"вкладка {page.url[:80]}")
            else:
                pool.instruction_unread = True
                logger.warning("Инструкция во вкладке без текста — %s; %s", self._instruction_seen,
                               describe_documents(self._doc_catcher))
            await self._browser.ensure_front()
        if new_pages:
            self._stop_catching_documents()
        return True

    async def _maybe_read_tooltips(self, frame: Frame, state: PageState) -> None:
        """Прочитать подсказки «?» у вариантов ответа (наведение мыши) — один раз для вида."""
        pool = self._pool
        if not READ_TOOLTIPS or pool is None or pool.tooltips_attempted or state.dialog_open or state.loading:
            return
        pool.tooltips_attempted = True
        try:
            triggers = await frame.evaluate(_JS_TOOLTIP_TRIGGERS)
        except PlaywrightError:
            return
        if not triggers:
            return
        by_uid = {e.uid: e for e in state.elements}
        tips: dict[str, str] = {}
        page = self._browser.page
        started = time.monotonic()
        for trig in triggers:
            if time.monotonic() - started > 25:
                break
            row = by_uid.get(str(trig.get("uid")))
            if row is None or row.aux:
                continue
            locator = frame.locator(f'[data-agent-tip="{trig["n"]}"]')
            try:
                await locator.scroll_into_view_if_needed(timeout=2_000)
                box = await locator.bounding_box(timeout=1_000)
                if not box:
                    continue
                await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=4)
                await asyncio.sleep(0.9)
                text = str(await frame.evaluate(_JS_HINT_TEXT) or "").strip()
                await page.mouse.move(5, 5, steps=2)
                await asyncio.sleep(0.2)
            except PlaywrightError:
                continue
            if text and normalize_text(text) != normalize_text(row.label()):
                tips[row.label()] = text
        if tips:
            self._knowledge.save_tooltips(pool, tips)

    # ------------------------------------------------------------------
    # Сверка цели (автокоррекция v3)
    # ------------------------------------------------------------------

    def _resolve_target(self, decision: LLMDecision, state: PageState) -> Optional[ParsedElement]:
        """Найти элемент, который имела в виду LLM.

        Индекс, подтверждённый текстом, приоритетнее; поиск по тексту учитывает тип
        элемента, а при дублях берёт ближайший к названному индексу и не запрещённый.
        Служебные элементы (плеер, карусель) и кнопки стоп-списка не выбираются никогда.
        """
        action = decision.action
        if action in (ActionType.SKIP, ActionType.WEB):
            return None
        kinds = _ACTION_KINDS[action]
        fallback = _FALLBACK_KINDS.get(action, set())
        by_index = state.by_index(decision.target_index)
        if by_index is not None and (is_denied_button(by_index) or by_index.aux):
            logger.warning("LLM указала недоступный элемент «%s» — игнорирую", by_index.label())
            by_index = None
        text = _clean_target_text(decision.target_text or "")

        def label(e: ParsedElement) -> str:
            return e.text or e.placeholder or e.caption

        # 1. Индекс и текст согласованы
        if by_index is not None and by_index.kind in kinds and (
            not text or _similarity(label(by_index), text) >= 0.9
        ):
            return by_index

        # 2. Поиск по тексту среди элементов подходящего типа
        if text:
            for allowed in (kinds, fallback):
                scored = [
                    (_similarity(label(e), text), e)
                    for e in state.elements if e.kind in allowed and not is_denied_button(e) and not e.aux
                ]
                good = [(s, e) for s, e in scored if s >= _MATCH_THRESHOLD]
                if not good:
                    continue
                best = max(s for s, _ in good)
                top = [e for s, e in good if s >= best - 1e-6]
                ref = decision.target_index if decision.target_index is not None else 0
                top.sort(key=lambda e: (self._memory.is_forbidden(action, e.key), abs(e.index - ref)))
                chosen = top[0]
                if len(top) > 1:
                    logger.info(
                        "Автокоррекция: %d элементов с текстом «%s», выбран [%d] ⟨%s⟩",
                        len(top), text, chosen.index, " › ".join(chosen.path) or "корень",
                    )
                if by_index is None or chosen.index != by_index.index:
                    logger.info(
                        "Автокоррекция: idx %s → %d (по тексту «%s», сходство %.2f)",
                        decision.target_index, chosen.index, text, best,
                    )
                return chosen

        # 3. Текст не найден (перефразирован) — доверяем индексу, если тип подходит
        if by_index is not None and (by_index.kind in kinds or by_index.kind in fallback):
            if text:
                logger.warning(
                    "Текст «%s» не найден — использую индекс %d («%s»)", text, by_index.index, by_index.label(),
                )
            return by_index

        if action not in (ActionType.SUBMIT, ActionType.SCROLL):
            logger.warning("Цель не найдена: idx=%s text=%r", decision.target_index, decision.target_text)
        return None

    # ------------------------------------------------------------------
    # Пакет действий
    # ------------------------------------------------------------------

    @staticmethod
    def _order_batch(steps: list[LLMDecision]) -> list[LLMDecision]:
        """Порядок пакета: submit — только последним; после open/web/scroll действия
        отбрасываются (страница меняется, нужен новый взгляд модели); лимит длины."""
        if not BATCH_ACTIONS:
            return steps[:1]
        body = [s for s in steps if s.action not in (ActionType.SKIP, ActionType.SUBMIT)]
        submit = next((s for s in steps if s.action == ActionType.SUBMIT), None)
        if not body and submit is None:
            return steps[:1]                              # только skip
        ordered: list[LLMDecision] = []
        for step in body:
            ordered.append(step)
            if step.action in PAGE_CHANGING_ACTIONS:
                return ordered[:MAX_BATCH_ACTIONS]
        if submit is not None and len(ordered) < MAX_BATCH_ACTIONS:
            ordered.append(submit)                        # не влез в лимит — отправит следующим шагом
        return ordered[:MAX_BATCH_ACTIONS]

    @staticmethod
    def _step_label(step: LLMDecision) -> str:
        what = step.target_text or step.type_text or step.query or ""
        return f"{step.action.value}" + (f" «{what[:40]}»" if what else "")

    async def _run_batch(self, frame: Frame, decision: LLMDecision, state: PageState) -> None:
        """Выполнить действия по очереди. Перед каждым следующим: новый снимок, проверка
        эффекта предыдущего и «ничего неожиданного» (задание то же, не открылся диалог, нет
        новых ошибок, не появились и не исчезли поля). Иначе пакет останавливается, и модель
        на следующем шаге видит новое состояние и заметку, что осталось не выполненным."""
        mem = self._memory
        all_steps = decision.steps()
        steps = self._order_batch(all_steps)
        dropped = [s for s in all_steps if s.action != ActionType.SKIP and not any(s is t for t in steps)]
        if dropped:
            mem.batch_notes.append(
                "Не выполнено (после open/web/scroll нужен новый взгляд на страницу, submit — только "
                "последним): " + ", ".join(self._step_label(s) for s in dropped)
            )

        # цели — по тому снимку, который видела модель (номера из него)
        planned: list[tuple[LLMDecision, Optional[ParsedElement]]] = []
        for i, step in enumerate(steps):
            target = self._resolve_target(step, state)
            if i > 0 and target is None and step.action in (ActionType.CLICK, ActionType.OPEN, ActionType.TYPE):
                note = f"Действие {self._step_label(step)} не выполнено: элемента [{step.target_index}] нет на странице."
                rest = steps[i + 1:]
                if rest:
                    note += " Следующие действия пакета тоже не выполнены: " + ", ".join(self._step_label(s) for s in rest)
                mem.batch_notes.append(note)
                break
            planned.append((step, target))
        if len(planned) > 1:
            logger.info("Пакет из %d действий: %s", len(planned), " → ".join(self._step_label(s) for s, _ in planned))

        current = state
        last_status, last_target = "", None
        for i, (step, target) in enumerate(planned):
            if i > 0:
                await self._browser.wait_settle(frame)
                fresh = await DomParser(frame).parse(quiet=True)
                mem.verify(fresh)               # итог прошлого действия — до проверки пакета (она его читает)
                reason = self._batch_break_reason(current, fresh, last_status, last_target)
                if reason == "задание сменилось":
                    # Ozon после клика перерисовывает карточку: за 350 мс тишины страница бывает
                    # недорисована и выглядит другой — проверяем ещё раз, когда она догрузится
                    await self._browser.wait_for_network_idle_and_dom(frame, timeout_ms=3000)
                    fresh = await DomParser(frame).parse(quiet=True)
                    reason = self._batch_break_reason(current, fresh, last_status, last_target)
                    if reason == "задание сменилось":
                        a, b = TaskIdentity.of(current), TaskIdentity.of(fresh)
                        logger.info("Пакет: задание выглядит другим — исчезли строки %s, появились %s, фото общих: %s",
                                    sorted(a.lines - b.lines)[:3], sorted(b.lines - a.lines)[:3],
                                    len(a.media & b.media))
                if reason is None and target is not None:
                    remapped = fresh.by_key(target.key)
                    if remapped is None:
                        reason = f"элемент «{target.label()}» исчез со страницы"
                    target = remapped
                if reason is not None:
                    rest = ", ".join(self._step_label(s) for s, _ in planned[i:])
                    logger.info("Пакет остановлен: %s. Не выполнено: %s", reason, rest)
                    mem.batch_notes.append(f"Пакет действий остановлен: {reason}. Не выполнено: {rest}. "
                                           "Посмотри на страницу заново.")
                    return
                current = fresh
            last_status = await self._execute(frame, step, target, current)
            last_target = target
            if last_status in ("done", "fail"):
                return

    def _batch_break_reason(
        self, before: PageState, after: PageState, status: str, target: Optional[ParsedElement],
    ) -> Optional[str]:
        """Почему нельзя продолжать пакет после очередного действия (None — можно)."""
        if status == "continue":
            result = self._memory.history[-1].result if self._memory.history else ""
            toggled_off = (target is not None and target.is_selected and target.choice_type == "checkbox"
                           and result.startswith("⚠ выбор снят"))
            if not (result.startswith("✓") or toggled_off):
                return f"«{target.label() if target else '?'}» → {result}"
        if not TaskIdentity.of(before).same_task(TaskIdentity.of(after), submitted=False):
            return "задание сменилось"
        if after.dialog_open and not before.dialog_open:
            return "открылся диалог"
        fresh_errors = set(after.notice_texts("error", "warning")) - set(before.notice_texts("error", "warning"))
        if fresh_errors:
            return "сообщение платформы: " + " | ".join(sorted(fresh_errors))[:200]

        def form_keys(state: PageState) -> dict[str, str]:
            return {e.key: e.label() for e in state.elements
                    if not e.aux and e.container not in ("popup", "dialog", "toast")}

        # Новое поле/вариант может требовать ответа — модель должна его увидеть до отправки.
        # Исчезнувшие элементы новых обязанностей не создают (если исчезла цель следующего
        # действия — пакет остановится при поиске цели).
        was, now = form_keys(before), form_keys(after)
        added = [now[k] for k in now if k not in was]
        if added:
            return "на странице появились " + ", ".join(f"«{x}»" for x in added[:3])
        return None

    # ------------------------------------------------------------------
    # Выполнение действия
    # ------------------------------------------------------------------

    async def _execute(
        self,
        frame: Frame,
        decision: LLMDecision,
        target: Optional[ParsedElement],
        state: PageState,
    ) -> str:
        """Выполнить одно действие. Возвращает исход для пакета действий:
        continue — выполнено, эффект проверяется следующим снимком; noop — делать нечего
        (вариант уже выбран); done — действие завершает пакет (submit, web, open, scroll,
        skip); fail — не выполнено."""
        action = decision.action
        mem = self._memory
        logger.info(
            "ВЫПОЛНЯЕМ: %s → %s conf=%.2f",
            action.value,
            f"[{target.index}] «{target.label()}»" if target else (decision.query or "—"),
            decision.confidence,
        )

        if action == ActionType.SKIP:
            mem.consecutive_skips += 1
            mem.add(action, None, result="ожидание")
            # модель ждёт (загрузка, ответ платформы): следующий вызов — когда страница изменится,
            # а не каждые пару секунд — ожидание не должно стоить токенов
            await self._wait_page_change(frame, state,
                                         timeout=SKIP_WAIT_FIRST if mem.consecutive_skips == 1 else SKIP_WAIT_NEXT)
            return "done"
        mem.consecutive_skips = 0

        if action == ActionType.WEB:
            await self._do_web(decision.query or decision.type_text or decision.target_text or "")
            return "done"

        if action == ActionType.SUBMIT:
            await self._do_submit(frame, state, target)
            return "done"

        if action == ActionType.SCROLL:
            direction = decision.scroll_direction or "down"
            key = target.key if target else f"__page__|{direction}"
            if mem.is_forbidden(action, key):
                mem.repeated_forbidden += 1
                mem.add(action, target, result="⛔ прокрутка уже ничего не меняла")
                return "fail"
            outcome = await self._browser.scroll(frame, target, direction)
            if outcome.ok:
                mem.expect(action, target, state, note=direction, key=key)
            else:
                mem.add(action, target, result=f"✗ {outcome.detail}")
            return "done"

        if target is None:
            mem.invalid_targets += 1
            mem.add(action, None, result=(
                f"✗ элемент не найден (номер={decision.target_index}, текст=«{decision.target_text or ''}»)"
            ))
            return "fail"
        mem.invalid_targets = 0

        if is_denied_button(target):
            mem.add(action, target, result="⛔ кнопка из стоп-списка — агент её не нажимает")
            return "fail"
        if target.is_disabled:
            mem.add(action, target, result="✗ элемент неактивен — нажать нельзя")
            mem.mark_no_effect(action, target.key)
            return "fail"
        if mem.is_forbidden(action, target.key):
            mem.repeated_forbidden += 1
            mem.add(action, target, result="⛔ уже пробовали без эффекта — выбери другое действие")
            return "fail"
        mem.repeated_forbidden = 0

        # Ссылка на внешний сайт: переход увёл бы фрейм задания со страницы (задание
        # потерялось бы) — открываем её во вкладке поиска и показываем модели как web
        if action == ActionType.CLICK and target.href and self._is_external(target.href, frame):
            await self._do_web(target.href, label=target.label())
            return "done"

        if action == ActionType.OPEN:
            if (target.kind in (ElementKind.FOLDER, ElementKind.DROPDOWN)
                    and target.folder_state == FolderState.OPEN
                    and mem.insist[(action, target.key)] == 0):
                # Первый раз не кликаем (клик свернул бы папку), а подсказываем. Если модель
                # настаивает — эвристика состояния могла ошибиться, выполняем.
                mem.insist[(action, target.key)] += 1
                mem.add(action, target, result="уже раскрыта — её содержимое ниже, с бо́льшим отступом")
                return "done"
            mem.open_attempts[target.key] += 1
            outcome = await self._browser.click_element(frame, target, prefer_toggle=True)
        elif action == ActionType.CLICK:
            if (target.is_selected and target.kind in (ElementKind.OPTION, ElementKind.FOLDER)
                    and target.choice_type != "checkbox" and mem.insist[(action, target.key)] == 0):
                # Checkbox можно снять сознательно, поэтому для него клик выполняется.
                mem.insist[(action, target.key)] += 1
                mem.add(action, target, result="уже ✓ВЫБРАН — повторный клик снял бы выбор; если всё готово — submit")
                return "noop"
            if target.kind == ElementKind.BUTTON and self._is_finish_button(target):
                await self._do_submit(frame, state, target)
                return "done"
            outcome = await self._browser.click_element(frame, target)
        elif action == ActionType.TYPE:
            if decision.type_text is None:
                mem.add(action, target, result="✗ пустой type_text")
                return "fail"
            outcome = await self._browser.type_into(frame, target, decision.type_text)
        else:
            logger.error("Неизвестное действие: %s", action.value)
            return "fail"

        if outcome.stale:
            mem.add(action, target, result="⚠ элемент перерисовался до клика — повтор на свежем снимке")
            return "fail"
        if not outcome.ok:
            mem.add(action, target, result=f"✗ {outcome.detail}")
            mem.mark_no_effect(action, target.key)
            return "fail"
        note = ""
        if action == ActionType.TYPE:
            note = f"«{decision.type_text[:80]}»" if decision.type_text else "(очистить поле)"
        if outcome.method == "js":
            note = (note + " (js-клик)").strip()
        mem.expect(action, target, state, typed=decision.type_text, note=note)
        # open меняет страницу (раскрытый список/ветка) — после него нужен новый взгляд модели
        return "done" if action == ActionType.OPEN else "continue"

    async def _wait_page_change(self, frame: Frame, state: PageState, timeout: float) -> None:
        """Ждать изменения страницы (текст без цифр, поля, выбор, загрузка, диалог, сообщения)
        до timeout секунд — без вызовов модели. Таймеры и счётчики изменением не считаются."""
        def signature(s: PageState) -> tuple:
            return (s.loose_hash, s.form_hash, s.loading, s.dialog_open, tuple(sorted(s.notice_texts())),
                    tuple((e.key, e.is_selected, e.is_disabled) for e in s.visible_elements))

        before = signature(state)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            try:
                fresh = await DomParser(frame).parse(quiet=True)
            except PlaywrightError:
                return                    # фрейм перезагружается — дальше решает основной цикл
            if signature(fresh) != before:
                return

    @staticmethod
    def _is_external(href: str, frame: Frame) -> bool:
        try:
            return urlsplit(href).netloc.lower() != urlsplit(frame.url).netloc.lower()
        except ValueError:
            return True

    async def _do_web(self, query: str, *, label: str = "") -> None:
        mem = self._memory
        query = (query or "").strip()
        if not WEB_RESEARCH:
            mem.add(ActionType.WEB, None, note=f"«{query[:80]}»",
                    result="✗ поиск в интернете отключён (WEB_RESEARCH=false)")
            return
        if not query:
            mem.add(ActionType.WEB, None, result="✗ пустой query — укажи запрос или адрес")
            return
        key = normalize_text(query)
        if sum(mem.web_queries.values()) >= MAX_WEB_PER_TASK:
            mem.add(ActionType.WEB, None, note=f"«{query[:80]}»", result="⛔ лимит запросов на задание исчерпан")
            return
        if mem.web_queries[key] >= 2:
            mem.repeated_forbidden += 1
            mem.add(ActionType.WEB, None, note=f"«{query[:80]}»",
                    result="⛔ этот запрос уже выполнялся — его результаты в разделе ВЕБ-ПОИСК")
            return
        mem.web_queries[key] += 1
        result = await self._web.run(query)
        mem.web_results.append(result)
        note = f"«{label or query[:100]}»"
        if result.error:
            mem.add(ActionType.WEB, None, note=note, result=f"✗ {result.error}")
        else:
            mem.add(ActionType.WEB, None, note=note,
                    result=f"✓ прочитано: {result.url[:160]} (результат #{len(mem.web_results)} в ВЕБ-ПОИСК)")

    # ------------------------------------------------------------------
    # Отправка ответа
    # ------------------------------------------------------------------

    @staticmethod
    def _is_finish_button(el: ParsedElement, *, strict: bool = True) -> bool:
        label = normalize_text(el.text)
        if not label or any(deny in label for deny in FINISH_DENY_SUBSTRINGS):
            return False
        if not strict:
            return True
        return any(label == t or label.startswith(t + " ") for t in FINISH_BUTTON_TEXTS)

    def _find_finish_button(self, state: PageState) -> Optional[ParsedElement]:
        buttons = [e for e in state.elements
                   if e.kind == ElementKind.BUTTON and not e.aux and self._is_finish_button(e)]
        if not buttons:
            return None

        def rank(e: ParsedElement) -> tuple:
            label = normalize_text(e.text)
            priority = next(
                (i for i, t in enumerate(FINISH_BUTTON_TEXTS) if label == t or label.startswith(t + " ")), 99,
            )
            return (e.is_disabled, e.container != "dialog", priority, e.index)

        return min(buttons, key=rank)

    @staticmethod
    def _describe_answer(state: PageState) -> str:
        """Текущий ответ в форме: отмеченные варианты, значения полей и списков."""
        parts: list[str] = []
        for e in state.visible_elements:
            if e.container in ("dialog", "popup", "toast"):
                continue
            if e.is_selected and e.kind in (ElementKind.OPTION, ElementKind.FOLDER):
                parts.append(f"«{e.label()}»")
            elif e.kind in (ElementKind.INPUT, ElementKind.DROPDOWN) and e.value:
                parts.append(f"{e.text or e.placeholder or e.caption or 'поле'} = «{e.value[:120]}»")
        return ", ".join(parts) or "(ничего не выбрано)"

    async def _do_submit(self, frame: Frame, state: PageState, target: Optional[ParsedElement]) -> None:
        """Нажать кнопку отправки и дождаться исхода: следующее задание или ответ платформы.

        v2 сбрасывал память сразу после клика, даже если форма не прошла валидацию.
        v4: «Неверный ответ» (тренировка) — отмечается в памяти, превращается в урок
        для этого вида заданий, модель исправляет ответ на следующем шаге.
        """
        mem = self._memory
        rejected = self._describe_answer(state)
        if rejected in mem.wrong_answers:
            # платформа этот ответ уже признала неверным — отправлять его снова бессмысленно
            logger.warning("Не отправляю: ответ %s платформа уже признала неверным — модель выберет другой", rejected)
            mem.add(ActionType.SUBMIT, None,
                    result=f"⛔ не отправлено: ответ {rejected} уже признан неверным — выбери другой вариант")
            mem.batch_notes.append(f"Ответ {rejected} платформа уже признала неверным. Не отправляй его снова — "
                                   "выбери другой вариант.")
            mem.submit_failures += 1
            return
        # «Прослушайте звонок до конца»: запись доигрывается ДО нажатия
        if AUDIO_PLAY_TO_END and any(not a.ended for a in state.audios):
            fresh = await DomParser(frame).parse(quiet=True)
            await self._media.wait_finished(frame, fresh)
            await asyncio.sleep(AUDIO_SETTLE)      # страница отмечает прослушивание не сразу
            state = await DomParser(frame).parse(quiet=True)
            target = state.by_key(target.key) if target is not None else None
            chosen = self._describe_answer(state)
            if chosen != rejected and chosen != "(ничего не выбрано)":
                # пока запись доигрывала, ответ поменял человек (он тоже слушал) — его ответ главный
                logger.warning("👤 Пока запись доигрывала, ответ в окне поменяли: %s → %s. Это сделал человек — "
                               "отправляю его ответ", rejected, chosen)
                mem.human_answer = chosen
                mem.case = mem.case or mem.observation
                mem.batch_notes.append(
                    f"Человек в окне браузера сам выбрал ответ {chosen} вместо {rejected} — он слушал запись. "
                    "Ответ человека верный: не меняй его, только отправь задание (submit).")

        # Ozon: на странице до 10 вопросов и одна «Отправить» — пустой блок ушёл бы неверным ответом
        try:
            # новый снимок заново размечает элементы: дальше — только по нему (кнопка по ключу)
            fresh = await DomParser(frame).parse(quiet=True)
            state, target = fresh, (fresh.by_key(target.key) if target is not None else None)
        except PlaywrightError:
            fresh = state
        missing = self._unanswered_groups(fresh)
        if missing:
            hint = "; ".join(f"варианты «{e.label()}»…" for e in self._first_unanswered_options(fresh)[:5])
            logger.warning("Не отправляю: ответ выбран не во всех вопросах страницы (без ответа: %d)", missing)
            mem.add(ActionType.SUBMIT, None, result=f"⛔ не отправлено: без ответа вопросов — {missing}")
            mem.batch_notes.append(f"На странице несколько вопросов, ответ нужен в КАЖДОМ. Без ответа: {missing} "
                                   f"({hint}). Выбери вариант в каждом из них, потом отправь.")
            mem.submit_failures += 1
            return

        answer = self._describe_answer(state)
        mode = self._page_mode(state)
        if target is not None and target.kind == ElementKind.BUTTON and self._is_finish_button(target, strict=False):
            button: Optional[ParsedElement] = target
        else:
            button = self._find_finish_button(state)

        errors_before = set(state.notice_texts("error", "warning"))
        if button is None:
            logger.info("SUBMIT: кнопка в снимке не найдена — поиск по тексту")
            if not await self._browser.click_by_text(frame, FINISH_BUTTON_TEXTS, deny=FINISH_DENY_SUBSTRINGS):
                mem.add(ActionType.SUBMIT, None, result="✗ кнопка отправки не найдена")
                mem.submit_failures += 1
                return
        elif button.is_disabled:
            mem.add(ActionType.SUBMIT, button,
                    result="✗ кнопка неактивна — ответ ещё не заполнен (проверь ✓ВЫБРАН и поля)")
            mem.submit_failures += 1
            return
        else:
            outcome = await self._browser.click_element(frame, button)
            if not outcome.ok:
                mem.add(ActionType.SUBMIT, button, result=f"✗ {outcome.detail}")
                mem.submit_failures += 1
                return

        mem.submits += 1
        logger.info("SUBMIT: ответ %s", answer)
        changed, after = await self._wait_task_change(errors_before)
        if changed:
            self._submitted = True
            self._tasks_done += 1
            logger.info("✅ ЗАДАНИЕ ОТПРАВЛЕНО (всего: %d)", self._tasks_done)
            self._record_training(mode, accepted=True, answer=answer)
            mem.add(ActionType.SUBMIT, button, result="✓ отправлено, задание сменилось")
            return

        mem.submit_failures += 1
        errors = [t for t in (after.notice_texts("error", "warning") if after else []) if t not in errors_before]
        if not errors and after is not None:
            # то же уведомление о неверном ответе от прошлой попытки ещё на экране — «новым» оно не выглядит
            errors = [t for t in after.notice_texts("error", "warning") if _is_wrong_answer(t)]
        hints = after.notice_texts("hint") if after else []
        if any(_is_wrong_answer(t) for t in errors):
            self._wrong_total += 1
            mem.wrong_answers.append(answer)
            explained = [t if _WRONG_RE.search(t) else
                         f"Ответ НЕВЕРНЫЙ — платформа его не приняла (её сообщение: «{t}»; ответ был полным, "
                         "значит, неверен выбор в каком-то из вопросов)" for t in errors]
            mem.feedback = explained + [f"Подсказка платформы: {h}" for h in hints]
            logger.warning("❌ Платформа: неверный ответ (%s)%s", answer,
                           f"; подсказка: {hints[0][:300]}" if hints else "")
            # разбор ошибки уйдёт в базу знаний, когда станет известен итог задания
            mem.case = mem.case or mem.observation or f"задание «{state.task_preview[:70]}»"
            mem.hints += [h for h in hints if h not in mem.hints]
            self._record_training(mode, accepted=False, answer=answer)
            mem.add(ActionType.SUBMIT, button, result="✗ платформа: НЕВЕРНЫЙ ОТВЕТ — прочитай подсказку и исправь ответ")
            return

        result = "✗ задание не сменилось"
        if errors:
            result += "; сообщения: " + " | ".join(errors)
            mem.feedback = errors
        dialog = [e.label() for e in (after.visible_elements if after else []) if e.container == "dialog"][:4]
        if dialog:
            result += "; открыт диалог: " + ", ".join(f"«{d}»" for d in dialog)
            if not errors and any(_label_matches(d, START_BUTTON_TEXTS) for d in dialog):
                # последнее задание тренировки: вместо следующего задания — окно «…экзамен / Начать»
                self._record_training(mode, accepted=True, answer=answer)
        logger.warning("SUBMIT: %s", result)
        mem.add(ActionType.SUBMIT, button, result=result)

    async def _wait_task_change(self, errors_before: set[str]) -> tuple[bool, Optional[PageState]]:
        """Поллинг до SUBMIT_WAIT: сменилось задание (True) или платформа ответила
        сообщением об ошибке на том же задании (False, снимок с сообщением)."""
        identity = self._identity
        started = time.monotonic()
        deadline = started + SUBMIT_WAIT
        hard_deadline = started + SUBMIT_WAIT + 30        # «вечный» лоадер не вешает агента
        last: Optional[PageState] = None
        missing = 0
        while time.monotonic() < min(deadline, hard_deadline):
            await asyncio.sleep(0.6)
            frame = await self._browser.find_target_frame()
            if frame is None:
                missing += 1
                if missing >= 2:          # фрейм задания ушёл — задание принято
                    return True, None
                continue
            try:
                last = await DomParser(frame).parse(quiet=True)
            except PlaywrightError:
                continue                  # фрейм перезагружается
            if last.loading:
                deadline = max(deadline, time.monotonic() + 1.0)   # идёт отправка — ждём дольше
                continue
            if self._is_orders_list(last) or self._is_order_done(last):
                return True, last
            if identity is None or not identity.same_task(TaskIdentity.of(last), submitted=True):
                return True, last
            fresh = [t for t in last.notice_texts("error", "warning") if t not in errors_before]
            if fresh:
                await asyncio.sleep(0.8)  # подсказка обычно появляется вместе с «Неверный ответ»
                try:
                    last = await DomParser(frame).parse(quiet=True)
                except PlaywrightError:
                    pass
                return False, last
        return False, last

    # ------------------------------------------------------------------
    # Локальный флоу, бюджет, капча
    # ------------------------------------------------------------------

    async def _maybe_click_start(self, frame: Frame, state: PageState) -> bool:
        """«Начать/ОК/Понятно» — без LLM, но только на заставке или в диалоге.

        v2 искал эти слова на КАЖДОМ шаге до LLM, подстрокой и по любому элементу:
        вариант ответа «Хорошо» или кнопка «Далее» до ответа нажимались бесконечно.
        """
        visible = state.visible_elements
        enabled_buttons = [
            e for e in visible
            if e.kind == ElementKind.BUTTON and not e.is_disabled and not e.occluded and not is_denied_button(e)
        ]
        dialog_buttons = [b for b in enabled_buttons if b.container == "dialog"]
        working = [
            e for e in visible
            if e.kind in (ElementKind.OPTION, ElementKind.FOLDER, ElementKind.INPUT, ElementKind.DROPDOWN)
            and not e.is_disabled and not e.occluded and e.container not in ("dialog", "popup")
        ]
        if working and not dialog_buttons:
            return False

        def start_like(b: ParsedElement) -> bool:
            return _label_matches(b.text, START_BUTTON_TEXTS)

        pool = dialog_buttons or enabled_buttons
        if not dialog_buttons and any(not start_like(b) for b in pool):
            return False       # на экране есть и другие кнопки (ответы «Да/Нет»?) — решает LLM
        button: Optional[ParsedElement] = None
        for text in START_BUTTON_TEXTS:
            button = next(
                (b for b in pool if normalize_text(b.text) == text or normalize_text(b.text).startswith(text + " ")),
                None,
            )
            if button is not None:
                break
        if button is None:
            return False
        if self._memory.start_clicks[state.state_hash] >= 2:
            logger.warning("Кнопка «%s» уже нажималась без эффекта — решение передаю LLM", button.label())
            return False

        self._memory.start_clicks[state.state_hash] += 1
        logger.info("Локальный флоу: нажимаю «%s»", button.label())
        outcome = await self._browser.click_element(frame, button)
        self._memory.add(ActionType.CLICK, button, note="(авто)",
                         result="нажата" if outcome.ok else f"✗ {outcome.detail}")
        await self._browser.wait_settle(frame)
        return outcome.ok

    async def _handle_budget_exhausted(self, frame: Frame, state: PageState) -> StepResult:
        selected = TaskMemory.selected_options(state)
        unanswered = self._unanswered_groups(state)
        if selected and unanswered:
            return await self._wait_for_human(
                f"бюджет задания ({MAX_STEPS_PER_TASK} шагов) исчерпан, а ответ выбран не во всех вопросах "
                f"страницы (без ответа: {unanswered}) — неполный ответ не отправляю")
        if selected and self._memory.submit_failures == 0:
            logger.warning(
                "Бюджет задания исчерпан — отправляю текущий выбор: %s",
                ", ".join(f"«{e.label()}»" for e in selected),
            )
            await self._do_submit(frame, state, None)
            return StepResult.ACTED
        return await self._wait_for_human(f"бюджет задания ({MAX_STEPS_PER_TASK} шагов) исчерпан, ответ не найден")

    @staticmethod
    def _unanswered_groups(state: PageState) -> int:
        """Сколько групп radio на странице без выбранного варианта (Ozon: до 10 вопросов на странице)."""
        groups: dict[str, bool] = {}
        for e in state.visible_elements:
            if e.kind == ElementKind.OPTION and e.choice_type == "radio" and e.group and e.container == "":
                groups[e.group] = groups.get(e.group, False) or e.is_selected
        return sum(1 for answered in groups.values() if not answered)

    @staticmethod
    def _first_unanswered_options(state: PageState) -> list[ParsedElement]:
        """Первый вариант каждой группы radio без ответа — чтобы модель нашла пропущенный вопрос."""
        answered = {e.group for e in state.visible_elements if e.group and e.is_selected}
        first: dict[str, ParsedElement] = {}
        for e in state.visible_elements:
            if (e.kind == ElementKind.OPTION and e.choice_type == "radio" and e.group and e.container == ""
                    and e.group not in answered):
                first.setdefault(e.group, e)
        return list(first.values())

    async def _wait_for_human(self, reason: str) -> StepResult:
        """Агент сам не справится: сказать почему и ждать, пока человек не сменит задание."""
        logger.error("%s. Нужна помощь человека: решите задание сами — жду смены задания до %.0f с",
                     reason[:1].upper() + reason[1:], MAX_IDLE_SECONDS)
        identity = self._identity
        deadline = time.monotonic() + MAX_IDLE_SECONDS
        while time.monotonic() < deadline:
            await asyncio.sleep(5.0)
            if self._browser.is_closed():
                return StepResult.STOP
            current = await self._browser.find_target_frame()
            if current is None:
                continue
            try:
                snapshot = await DomParser(current).parse(quiet=True)
            except PlaywrightError:
                continue
            if identity is None or not identity.same_task(TaskIdentity.of(snapshot), submitted=True):
                logger.info("Задание сменилось — продолжаю")
                return StepResult.IDLE
        return StepResult.STOP

    async def _wait_captcha_solved(self, poll: float = 5.0) -> None:
        """Ждём ручного решения капчи. v2 ждал бесконечно (while True) и проверял
        слово «капча» в тексте страницы — задание про капчу вешало агента навсегда."""
        logger.warning("КАПЧА! Решите её вручную в окне браузера (жду до %.0f с)…", CAPTCHA_TIMEOUT)
        deadline = time.monotonic() + CAPTCHA_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(poll)
            if self._browser.is_closed():
                return
            if await self._browser.page_has_captcha():
                continue
            frame = await self._browser.find_target_frame()
            if frame is None:
                return                    # фрейм грузится — основной цикл подождёт сам
            try:
                state = await DomParser(frame).parse(quiet=True)
            except PlaywrightError:
                continue
            if not state.has_captcha:
                logger.info("Капча решена, продолжаем")
                return
        logger.error("Капча не решена за %.0f с — 5 минут не реагирую на её признаки", CAPTCHA_TIMEOUT)
        self._captcha_suppressed_until = time.monotonic() + 300

    def _log_idle(self, message: str) -> None:
        if time.monotonic() - self._last_idle_log > 30:
            self._last_idle_log = time.monotonic()
            logger.warning(message)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(Agent().run())
