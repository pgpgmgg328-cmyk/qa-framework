"""dom_parser.py v2 — умный парсер DOM с распознаванием папок, опций и полей ввода."""

from __future__ import annotations

import logging
from typing import Optional

from playwright.async_api import Frame

from config import MAX_ELEMENTS
from models import ElementKind, FolderState, PageState, ParsedElement

logger = logging.getLogger("twork.dom_parser")


# ---------------------------------------------------------------------------
# JS-скрипт: вытащивает все интерактивные элементы одним запросом
# ---------------------------------------------------------------------------

_JS_COLLECT = """
() => {
    // Селекторы для Taiga UI / Angular платформы
    const SELECTORS = [
        'button',
        'a[href]',
        '[role="option"]',
        '[role="menuitem"]',
        '[role="treeitem"]',
        '[role="combobox"]',
        '[role="listbox"]',
        'tui-select',
        'tui-combo-box',
        '.t-select',
        'tui-radio-labeled',
        'tui-checkbox-labeled',
        'label.t-item',
        '.tui-tree-item__content',
        '.tui-tree-item',
        'div[class*="child__header"]',
        'button[class*="child__expand"]',
        'button[class*="expand"]',
        '[class*="tree-item"]',
        '[class*="category-item"]',
        '[class*="list-item"]',
        'input:not([type="hidden"])',
        'textarea',
        '[contenteditable="true"]',
    ].join(',');

    const MAX = 300;
    const seen = new Set();
    const result = [];

    // Вспомогатель: чистый видимый текст
    function getText(el) {
        // Для деревьев: возьмём только свой текстовый узел (не дочерние элементы)
        let t = '';
        for (const node of el.childNodes) {
            if (node.nodeType === Node.TEXT_NODE) {
                t += node.textContent;
            }
        }
        t = t.trim();
        if (!t) {
            // фоллбэк: полный innerText без иконок и метки
            t = (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 200);
        }
        return t;
    }

    // Вспомогатель: определить глубину в дереве
    function getDepth(el) {
        let d = 0;
        let cur = el.parentElement;
        while (cur) {
            const cls = cur.className || '';
            if (
                cur.tagName === 'TUI-TREE' ||
                cls.includes('tui-tree') ||
                cls.includes('tree-item') ||
                cls.includes('child__')
            ) d++;
            cur = cur.parentElement;
        }
        return Math.min(d, 6);
    }

    // Вспомогатель: является ли элемент папкой (деревовидный узел)
    function isFolder(el) {
        const cls = el.className || '';
        const tag = el.tagName.toLowerCase();
        // 1. Прямой aria
        if (el.hasAttribute('aria-expanded')) return true;
        // 2. Классы Taiga UI expand
        if (cls.includes('expand') || cls.includes('child__header') || cls.includes('tree-item__content')) return true;
        // 3. Тег tui-tree
        if (tag === 'tui-tree' || tag === 'tui-tree-item') return true;
        // 4. Есть ли внутри стрелка / expand-кнопка
        const inner = el.querySelector('[class*="expand"], [class*="arrow"], tui-svg, .t-icon');
        if (inner) return true;
        // 5. Role="group" / "tree"
        const role = el.getAttribute('role') || '';
        if (role === 'group' || role === 'tree') return true;
        return false;
    }

    // Вспомогатель: открыта ли папка
    function isFolderOpen(el) {
        if (el.getAttribute('aria-expanded') === 'true') return true;
        const cls = el.className || '';
        return cls.includes('_expanded') || cls.includes('-expanded') || cls.includes('--open');
    }

    // Вспомогатель: выбран ли элемент
    function isSelected(el) {
        if (el.checked === true) return true;
        if (el.getAttribute('aria-selected') === 'true') return true;
        if (el.getAttribute('aria-checked') === 'true') return true;
        const cls = el.className || '';
        return [
            '_checked', '_active', '_selected', '-selected', '-active',
            '--selected', '--active', 'selected', 'active'
        ].some(c => cls.includes(c));
    }

    // Вспомогатель: заблокирован ли элемент
    function isDisabled(el) {
        if (el.hasAttribute('disabled')) return true;
        if (el.getAttribute('aria-disabled') === 'true') return true;
        const cls = el.className || '';
        return cls.includes('disabled') || cls.includes('_disabled');
    }

    // Вспомогатель: видим ли элемент
    function isVisible(el) {
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return false;
        const style = window.getComputedStyle(el);
        return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
    }

    const els = document.querySelectorAll(SELECTORS);
    let idx = 0;

    for (const el of els) {
        if (idx >= MAX) break;

        // Исключаем дубликаты
        if (seen.has(el)) continue;
        seen.add(el);

        // Исключаем невидимые
        if (!isVisible(el)) continue;

        const tag    = el.tagName.toLowerCase();
        const cls    = el.className || '';
        const role   = el.getAttribute('role') || '';
        const inType = el.getAttribute('type') || '';
        const text   = getText(el);
        const ph     = el.getAttribute('placeholder') || '';

        // Определяем тип
        let kind = 'OTHER';
        let folderOpen = null;

        if (tag === 'input' || tag === 'textarea' || el.getAttribute('contenteditable') === 'true') {
            kind = 'INPUT';
        } else if (isFolder(el)) {
            kind = 'FOLDER';
            folderOpen = isFolderOpen(el);
        } else if (
            role === 'option' || role === 'menuitem' ||
            tag === 'tui-radio-labeled' || tag === 'tui-checkbox-labeled' ||
            cls.includes('t-item') || cls.includes('list-item') ||
            inType === 'radio' || inType === 'checkbox'
        ) {
            kind = 'OPTION';
        } else if (tag === 'button' || tag === 'a') {
            kind = 'BUTTON';
        }

        result.push({
            tag,
            cls: cls.slice(0, 120),
            role,
            kind,
            folderOpen,
            text,
            placeholder: ph,
            selected: isSelected(el),
            disabled: isDisabled(el),
            depth: getDepth(el),
            inputType: inType,
        });

        idx++;
    }

    return result;
}
"""


class DomParser:
    """Асинхронный парсер DOM целевого фрейма."""

    def __init__(self, frame: Frame) -> None:
        self._frame = frame

    async def parse(self) -> PageState:
        """Собрать весь PageState за один вызов."""
        logger.debug("Парсинг DOM…")

        task_text  = await self._get_task_text()
        hint_text  = await self._get_hint_text()
        elements   = await self._collect_elements()
        task_id    = await self.get_task_identifier()
        has_captcha = await self._detect_captcha()

        state = PageState(
            task_text=task_text,
            hint_text=hint_text,
            elements=elements,
            task_identifier=task_id,
            has_captcha=has_captcha,
        )
        logger.info(
            "Парсинг завершён: elements=%d, task_id=%r, captcha=%s",
            len(elements), task_id[:60], has_captcha,
        )
        return state

    async def get_task_identifier(self) -> str:
        """Уникальный идентификатор задания (не меняется при раскрытии папок).

        1. src первой картинки задания (если есть)
        2. Первый абзац текста задания
        """
        # 1. src первой картинки
        try:
            src: str = await self._frame.eval_on_selector("img", "el => el.src || ''")
            if src and src.startswith("http"):
                return src
        except Exception:
            pass

        # 2. Первый абзац текста задания
        for sel in ("[class*='task'] p", "[class*='question'] p", "p"):
            try:
                text: str = await self._frame.eval_on_selector(
                    sel, "el => el.innerText.trim()"
                )
                if text:
                    return text[:120]
            except Exception:
                continue

        return ""

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    async def _get_task_text(self) -> str:
        candidates = [
            "[class*='question']",
            "[class*='task-text']",
            "[class*='task__text']",
            "[class*='scenario']",
            "[class*='description']",
            "h1", "h2", "h3",
        ]
        parts: list[str] = []
        for sel in candidates:
            try:
                texts: list[str] = await self._frame.eval_on_selector_all(
                    sel,
                    "els => els.map(e => (e.innerText || '').trim()).filter(Boolean)",
                )
                parts.extend(texts)
            except Exception:
                continue
        result = "\n".join(dict.fromkeys(parts))
        logger.debug("task_text %d симв.: %r", len(result), result[:80])
        return result

    async def _get_hint_text(self) -> str:
        candidates = [
            "[class*='hint']",
            "[class*='tip']",
            "[class*='tooltip']",
            "[class*='note']",
        ]
        parts: list[str] = []
        for sel in candidates:
            try:
                texts: list[str] = await self._frame.eval_on_selector_all(
                    sel,
                    "els => els.map(e => (e.innerText || '').trim()).filter(Boolean)",
                )
                parts.extend(texts)
            except Exception:
                continue
        return "\n".join(dict.fromkeys(parts))

    async def _collect_elements(self) -> list[ParsedElement]:
        """Запустить JS и преобразовать результат в ParsedElement."""
        raw_list: list[dict] = await self._frame.evaluate(_JS_COLLECT)
        elements: list[ParsedElement] = []

        for idx, raw in enumerate(raw_list):
            if len(elements) >= MAX_ELEMENTS:
                break

            # Определяем kind
            kind_str = raw.get("kind", "OTHER")
            try:
                kind = ElementKind(kind_str)
            except ValueError:
                kind = ElementKind.OTHER

            # Определяем folder_state
            folder_open = raw.get("folderOpen")
            if kind == ElementKind.FOLDER and folder_open is not None:
                folder_state = FolderState.OPEN if folder_open else FolderState.CLOSED
            else:
                folder_state = FolderState.NA

            el = ParsedElement(
                index=idx,
                kind=kind,
                folder_state=folder_state,
                tag=raw.get("tag", ""),
                text=raw.get("text", ""),
                is_selected=bool(raw.get("selected", False)),
                is_disabled=bool(raw.get("disabled", False)),
                depth=int(raw.get("depth", 0)),
                placeholder=raw.get("placeholder", ""),
            )
            elements.append(el)

        folders_open   = sum(1 for e in elements if e.kind == ElementKind.FOLDER and e.folder_state == FolderState.OPEN)
        folders_closed = sum(1 for e in elements if e.kind == ElementKind.FOLDER and e.folder_state == FolderState.CLOSED)
        options        = sum(1 for e in elements if e.kind == ElementKind.OPTION)
        logger.debug(
            "Итого: %d эл., папок откр.=%d закр.=%d опц.=%d",
            len(elements), folders_open, folders_closed, options,
        )
        return elements

    async def _detect_captcha(self) -> bool:
        keywords = ("captcha", "recaptcha", "hcaptcha", "капча")
        try:
            body: str = await self._frame.evaluate(
                "() => document.body.innerText.toLowerCase()"
            )
            if any(k in body for k in keywords):
                logger.warning("Обнаружена капча")
                return True
        except Exception:
            pass
        return False


def build_elements_prompt(elements: list[ParsedElement]) -> str:
    """Строковое представление всех элементов для LLM-промпта."""
    lines: list[str] = []
    for el in elements:
        if el.is_disabled:
            continue  # заблокированные не показываем
        lines.append(el.prompt_line())
    return "\n".join(lines)
