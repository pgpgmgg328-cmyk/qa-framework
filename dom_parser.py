"""dom_parser.py v4 — атомарный снимок DOM целевого фрейма.

v4: снимок содержит страницу задания ЦЕЛИКОМ (reader-view): заголовки, текст с
переносами, таблицы, ссылки с адресами, фото [ФОТО n], аудио [АУДИО n], сообщения
платформы и интерактивные элементы на своих местах. Модель видит вопрос рядом с
вариантами, подпись «Цвет» рядом со списком значений и координаты в ссылках на карту.
Исправлено по записи реальных заданий: ложная «загрузка» (скрытый global-loader),
Angular-класс ng-invalid как «ошибка», обрезка текста задания на 1500 символах,
пункты длинных выпадающих списков, «обрезанные» внешней обёрткой.

Что изменилось в v3 относительно v2 и почему:

1. Стабильные метки вместо «индекса в querySelectorAll».
   В v2 парсер нумеровал только ВИДИМЫЕ элементы, а BrowserController кликал
   `locator(другой_селектор).nth(index)` по ВСЕМ совпадениям (скрытые input'ы,
   display:none-кнопки, shadow DOM). Индексы расходились, и клик уходил в чужой
   элемент — это выглядело как «галлюцинация индексов у LLM». Теперь каждый
   элемент снимка помечается атрибутом data-agent-id="<поколение>-<номер>",
   и клик идёт строго по этой метке. Если Angular успел перерисовать узел,
   метки нет → клик не выполняется, агент делает новый снимок.

2. Один evaluate вместо ~18 последовательных вызовов: текст задания, элементы,
   отпечаток задания, капча и лоадеры читаются из ОДНОГО состояния DOM.

3. Вложенные совпадения сливаются в одну «строку»: контейнер ветки, заголовок,
   кнопка-стрелка и скрытый radio больше не превращаются в 3–4 дубля. Стрелка
   запоминается как toggle (data-agent-toggle), radio — как control.

4. Видимость: учитываются opacity предков, overflow:hidden-обрезка схлопнутых
   веток (tui-expand/аккордеон), checkVisibility(); открытые shadow root обходятся.

5. Состояния: классы проверяются по токенам (в v2 'inactive'.includes('active')
   давал ложное «✓Выбран»), selected берётся из вложенного input.checked,
   раскрытость папки — из aria-expanded, классов и структуры DOM (видимые дети).

6. Отпечаток задания (task_identifier) — хэш «крупнейшая картинка + текст
   вопроса + путь URL» без текста дерева: не меняется при раскрытии папок и не
   залипает на логотипе (в v2 брался src ПЕРВОГО <img>).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import re
from collections import Counter
from typing import Optional
from urllib.parse import parse_qs, urlsplit

from playwright.async_api import Frame

from config import FINISH_DENY_SUBSTRINGS, MAX_ELEMENTS, READER_MAX_CHARS
from models import (
    ElementKind,
    FolderState,
    MediaAudio,
    MediaImage,
    Notice,
    PageState,
    ParsedElement,
    normalize_text,
)

logger = logging.getLogger("twork.dom_parser")


# ---------------------------------------------------------------------------
# Селекторы
# ---------------------------------------------------------------------------

# Кандидаты в интерактивные элементы. Перебор намеренно широкий: вложенные
# совпадения потом сливаются в одну строку, невидимые отбрасываются.
INTERACTIVE_SELECTORS: tuple[str, ...] = (
    # Нативные элементы и ARIA-роли
    "button", "a[href]", "summary", "select", "textarea",
    "input:not([type='hidden'])",
    "[contenteditable='true']", "[contenteditable='']",
    "[role='button']", "[role='link']", "[role='tab']",
    "[role='option']", "[role='menuitem']", "[role='menuitemradio']", "[role='menuitemcheckbox']",
    "[role='treeitem']", "[role='radio']", "[role='checkbox']", "[role='switch']",
    "[role='combobox']", "[role='textbox']", "[role='searchbox']", "[role='spinbutton']",
    "[aria-expanded]", "[aria-haspopup]:not([aria-haspopup='false'])",
    # Taiga UI
    "tui-select", "tui-combo-box", "tui-multi-select",
    "tui-radio-labeled", "tui-checkbox-labeled", "tui-radio-block", "tui-checkbox-block",
    "tui-tree-item", "tui-tree-item-content", "[tuiOption]",
    ".t-select", "label.t-item", ".tui-tree-item__content", ".tui-tree-item",
    "label:has(input[type='radio'])", "label:has(input[type='checkbox'])",
    # Классы, найденные на T-Work (перенесены из v2)
    "div[class*='child__header']", "button[class*='child__expand']", "button[class*='expand']",
    "[class*='tree-item']", "[class*='category-item']", "[class*='list-item']",
)
INTERACTIVE_SELECTOR: str = ", ".join(INTERACTIVE_SELECTORS)

# Где искать текст задания (в порядке приоритета). Текст вложенных
# интерактивных строк вырезается, поэтому описания категорий сюда не попадут.
TASK_TEXT_SELECTORS: tuple[str, ...] = (
    "[class*='question']", "[class*='task-text']", "[class*='task__text']",
    "[class*='task-title']", "[class*='task__title']", "[class*='scenario']",
    "[class*='description']", "[class*='instruction']", "h1", "h2", "h3",
)
# Подсказки. В v2 здесь был "[class*='tip']", совпадавший с 'multiple'/'tooltip'.
HINT_SELECTORS: tuple[str, ...] = (
    "[class*='hint']", "[class*='note']", "[class*='help']", "[role='note']",
)

# Сколько элементов максимум размечает JS (показ LLM ограничен MAX_ELEMENTS)
_JS_MAX_ELEMENTS = 1500

_GENERATION = itertools.count(1)

# Таймеры и обратный отсчёт не должны менять отпечаток задания:
# «04:59», «1:02:03», «59 сек», «осталось 3 мин»
_TIMER_RE = re.compile(
    r"(?:осталось|таймер|время)\s*:?\s*"          # подпись таймера (удаляется целиком)
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"
    r"|\b\d+\s*(?:сек(?:унд[аы]?)?|мин(?:ут[аы]?)?|ч(?:ас(?:а|ов)?)?|sec|min)\b\.?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# JS: весь снимок за один вызов
# ---------------------------------------------------------------------------

_JS_SNAPSHOT = r"""
(args) => {
    const GEN = String(args.gen);
    const MAX = args.max || 1500;
    const A_ID = 'data-agent-id', A_TOGGLE = 'data-agent-toggle',
          A_SELECT = 'data-agent-select', A_IMG = 'data-agent-img', A_MEDIA = 'data-agent-media';

    // ------------------------------------------------------------ утилиты
    const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim();
    const MEANINGFUL = /[\p{L}\p{N}]/u;             // есть буква или цифра
    const meaningful = (s) => MEANINGFUL.test(s || '');
    // getAttribute('class'), а не className: у SVG className — объект
    // SVGAnimatedString, и cls.includes(...) в v2 ронял весь парсинг.
    const clsOf = (el) => (el.getAttribute && el.getAttribute('class')) || '';
    const tokensOf = (el) => clsOf(el).split(/\s+/).filter(Boolean);
    const hasToken = (el, re) => tokensOf(el).some((t) => re.test(t));
    const tagOf = (el) => String(el.localName || el.tagName || '').toLowerCase();
    const attr = (el, name) => (el.getAttribute ? el.getAttribute(name) : null);
    const roleOf = (el) => String(attr(el, 'role') || '').toLowerCase();
    const flatParent = (el) => el.assignedSlot || el.parentElement
        || (el.parentNode && el.parentNode.host) || null;
    const styleCache = new Map();
    const styleOf = (el) => {
        let s = styleCache.get(el);
        if (!s) { s = getComputedStyle(el); styleCache.set(el, s); }
        return s;
    };
    const contains = (a, b) => { for (let x = b; x; x = flatParent(x)) if (x === a) return true; return false; };
    const vw = window.innerWidth, vh = window.innerHeight;

    // ------------------------------------ 0. все элементы (+ открытые shadow root)
    const ALL = [];
    const collect = (root) => {
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
        for (let n = walker.nextNode(); n; n = walker.nextNode()) {
            ALL.push(n);
            if (n.shadowRoot) collect(n.shadowRoot);
        }
    };
    collect(document);
    // метки прошлого снимка больше не действительны
    for (const el of ALL) {
        if (el.hasAttribute(A_ID)) el.removeAttribute(A_ID);
        if (el.hasAttribute(A_TOGGLE)) el.removeAttribute(A_TOGGLE);
        if (el.hasAttribute(A_SELECT)) el.removeAttribute(A_SELECT);
        if (el.hasAttribute(A_IMG)) el.removeAttribute(A_IMG);
        if (el.hasAttribute(A_MEDIA)) el.removeAttribute(A_MEDIA);
    }

    // ------------------------------------------------------------ видимость
    const VIS = { checkOpacity: true, checkVisibilityCSS: true, opacityProperty: true, visibilityProperty: true };
    const rendered = (el) => {
        if (typeof el.checkVisibility === 'function') return el.checkVisibility(VIS);
        const s = styleOf(el);
        return s.display !== 'none' && s.visibility === 'visible' && parseFloat(s.opacity || '1') > 0;
    };
    const clipCache = new Map();
    const visibleRect = (el) => {
        const r = el.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) return null;       // v2: '&&' пропускал схлопнутые по высоте
        if (!rendered(el)) return null;                     // opacity/visibility с учётом предков
        // probe — что должно пересекаться с областью очередного предка. Выше прокручиваемого
        // контейнера важна видимость самого контейнера: пункт длинного списка ниже края
        // доступен прокруткой, хотя внешняя обёртка с overflow:hidden его «обрезает»
        // (в записи из 20 значений списка «Цвет» терялись 5 нижних).
        let probe = r;
        let fixed = styleOf(el).position === 'fixed';
        for (let a = flatParent(el); a && !fixed; a = flatParent(a)) {
            if (a === document.body || a === document.documentElement) break;
            const s = styleOf(a);
            if (s.position === 'fixed') fixed = true;
            if (s.overflowX === 'visible' && s.overflowY === 'visible') continue;
            let c = clipCache.get(a);
            if (!c) {
                const scrollY = /(auto|scroll|overlay)/.test(s.overflowY) && a.scrollHeight > a.clientHeight + 1;
                const scrollX = /(auto|scroll|overlay)/.test(s.overflowX) && a.scrollWidth > a.clientWidth + 1;
                c = { rect: a.getBoundingClientRect(), scrollable: scrollY || scrollX };
                clipCache.set(a, c);
            }
            if (c.rect.width < 1 || c.rect.height < 1) return null;   // схлопнутая ветка / аккордеон
            if (c.scrollable) { probe = c.rect; continue; }            // до элемента можно доскроллить
            const ix = Math.min(probe.right, c.rect.right) - Math.max(probe.left, c.rect.left);
            const iy = Math.min(probe.bottom, c.rect.bottom) - Math.max(probe.top, c.rect.top);
            if (ix < 1 || iy < 1) return null;                        // обрезан overflow:hidden
        }
        return r;
    };

    // ------------------------------------------------- 1. кандидаты
    let selectorOk = true;
    try { document.createElement('div').matches(args.selectors); } catch (e) { selectorOk = false; }
    const matchesSel = (el) => {
        if (!selectorOk) return false;
        try { return el.matches(args.selectors); } catch (e) { return false; }
    };
    // Кастомные строки без ролей/классов: верхний элемент с cursor:pointer
    const pointerCandidate = (el) => {
        if (!(el instanceof HTMLElement)) return false;
        const tag = tagOf(el);
        if (tag === 'html' || tag === 'body' || tag === 'iframe') return false;
        if (el.childElementCount > 12) return false;
        if (styleOf(el).cursor !== 'pointer') return false;
        const p = flatParent(el);
        return !(p && p.nodeType === 1 && styleOf(p).cursor === 'pointer');
    };

    const cands = [];
    const info = new Map();
    let hiddenCount = 0;
    for (const el of ALL) {
        const bySelector = matchesSel(el);
        const byPointer = !bySelector && args.pointer && pointerCandidate(el);
        if (!bySelector && !byPointer) continue;
        const r = visibleRect(el);
        if (!r) { if (bySelector) hiddenCount++; continue; }
        // большие «кликабельные карточки» — не строки
        if (byPointer && (r.height > 150 || r.width * r.height > 0.3 * vw * vh)) continue;
        cands.push(el);
        info.set(el, { el, r, parent: null, toggle: null, control: null,
                       hasKeptDesc: false, firstKeptDesc: null });
        if (cands.length >= MAX * 4) break;
    }
    const candSet = new Set(cands);
    for (const el of cands) {
        let p = flatParent(el);
        while (p && !candSet.has(p)) p = flatParent(p);
        info.get(el).parent = p || null;
    }

    // ------------------------------------------------- 2. текст и подписи
    const SKIP_TEXT = new Set(['script', 'style', 'noscript', 'template', 'svg',
                               'input', 'textarea', 'select', 'iframe', 'object']);
    const flatChildren = (node) => {
        if (node.shadowRoot) return node.shadowRoot.childNodes;
        if (tagOf(node) === 'slot') {
            const assigned = node.assignedNodes({ flatten: true });
            if (assigned.length) return assigned;
        }
        return node.childNodes;
    };
    // Видимый текст узла БЕЗ текста вложенных строк (exclude) — в v2 контейнер
    // ветки получал склейку текстов всех детей, и автокоррекция по подстроке
    // уводила клик в центр контейнера.
    const textOf = (root, exclude, limit) => {
        let out = '';
        const stack = [];
        const push = (node, block) => {
            if (block) stack.push(' ');
            const ch = flatChildren(node);
            for (let i = ch.length - 1; i >= 0; i--) stack.push(ch[i]);
            if (block) stack.push(' ');
        };
        push(root, false);
        while (stack.length && out.length < limit * 2) {
            const n = stack.pop();
            if (typeof n === 'string') { out += n; continue; }
            if (n.nodeType === 3) { out += n.nodeValue; continue; }
            if (n.nodeType !== 1) continue;
            if (exclude && exclude.has(n)) { out += ' '; continue; }
            const tag = tagOf(n);
            if (tag === 'br') { out += ' '; continue; }
            if (SKIP_TEXT.has(tag)) continue;
            // мелкие aria-hidden-узлы — декор: «зеркало» значения в tui-textarea, иконки-лигатуры.
            // Крупные не пропускаем: при открытом диалоге aria-hidden вешают на всё приложение.
            if (attr(n, 'aria-hidden') === 'true' && n.childElementCount <= 3) continue;
            const s = styleOf(n);
            if (s.display === 'none' || s.visibility === 'hidden' || s.visibility === 'collapse') continue;
            push(n, !(s.display.startsWith('inline') || s.display === 'contents'));
        }
        return norm(out).slice(0, limit);
    };
    const ariaText = (el) => {
        const ids = attr(el, 'aria-labelledby');
        if (ids) {
            const root = el.getRootNode();
            const t = ids.split(/\s+/).map((id) => {
                const x = root.getElementById ? root.getElementById(id) : document.getElementById(id);
                return x ? textOf(x, null, 200) : '';
            }).join(' ');
            if (meaningful(t)) return norm(t);
        }
        const al = attr(el, 'aria-label');
        return meaningful(al) ? norm(al) : '';
    };
    const rowLabel = (el) => {
        let t = textOf(el, candSet, 400);
        if (meaningful(t)) return t;
        t = ariaText(el) || norm(attr(el, 'title'));
        if (meaningful(t)) return t.slice(0, 200);
        const img = el.querySelector && el.querySelector('img[alt]');
        if (img && meaningful(img.alt)) return norm(img.alt).slice(0, 200);
        if (tagOf(el) === 'input' && meaningful(el.value)) return norm(el.value).slice(0, 200);
        return '';
    };
    const FIELD_WRAP = 'tui-input, tui-textfield, tui-input-number, tui-textarea, tui-combo-box, '
                     + 'tui-select, tui-input-date, tui-input-tag, label';
    const inputLabel = (el) => {
        let t = ariaText(el);
        if (!t && el.labels && el.labels.length) t = textOf(el.labels[0], null, 200);
        if (!meaningful(t)) t = norm(attr(el, 'placeholder'));
        if (!meaningful(t)) t = norm(attr(el, 'title'));
        if (!meaningful(t) && el.closest) {
            // Taiga: подпись поля живёт в обёртке tui-input/tui-textfield. В обёртке
            // tui-textarea есть и «зеркало» введённого текста — его из подписи вырезаем,
            // иначе подпись (и ключ памяти) менялась бы с каждым введённым символом.
            const wrap = el.closest(FIELD_WRAP);
            if (wrap) {
                const ph = wrap.querySelector('[automation-id*="placeholder" i], .t-placeholder, [class*="placeholder" i]');
                let w = ph ? textOf(ph, candSet, 200) : '';
                if (!meaningful(w)) w = textOf(wrap, candSet, 200);
                const v = norm(el.value || '');
                if (v) { const i = w.indexOf(v.slice(0, 24)); if (i >= 0) w = norm(w.slice(0, i)); }
                if (meaningful(w)) t = w;
            }
        }
        return meaningful(t) ? t.slice(0, 200) : '';
    };
    const contextLabel = (el) => {
        const own = ariaText(el) || norm(attr(el, 'title'));
        if (meaningful(own)) return own.slice(0, 200);
        for (let a = flatParent(el), n = 0; a && n < 3; a = flatParent(a), n++) {
            if (a === document.body) break;
            const t = textOf(a, candSet, 120);
            if (meaningful(t)) return t;
        }
        return '';
    };
    // tui-select без видимого текста: подпись — placeholder/значение внутреннего input
    const innerField = (el) => (el.querySelector ? el.querySelector('input, textarea, [role="textbox"]') : null);
    const dropdownLabel = (el) => {
        const t = rowLabel(el);
        if (meaningful(t)) return t;
        const field = innerField(el);
        if (field) {
            const f = inputLabel(field);
            if (meaningful(f)) return f;
        }
        return contextLabel(el);
    };

    // ------------------------------------------------- 3. семантика
    const NON_TEXT_INPUTS = new Set(['radio', 'checkbox', 'button', 'submit', 'reset',
                                     'image', 'hidden', 'file', 'range', 'color']);
    const inputTypeOf = (el) => String(attr(el, 'type') || 'text').toLowerCase();
    const isTextInput = (el) => {
        const tag = tagOf(el), role = roleOf(el);
        if (tag === 'textarea' || tag === 'select') return true;
        if (tag === 'input') return !NON_TEXT_INPUTS.has(inputTypeOf(el));
        if (attr(el, 'contenteditable') !== null && el.isContentEditable) {
            const p = flatParent(el);
            return !(p && p.isContentEditable);          // только корень редактора
        }
        return role === 'textbox' || role === 'searchbox' || role === 'spinbutton';
    };
    const CHOICE_ROLES = new Set(['radio', 'checkbox', 'switch', 'menuitemradio', 'menuitemcheckbox']);
    const isChoiceControl = (el) => {
        if (tagOf(el) === 'input') { const t = inputTypeOf(el); return t === 'radio' || t === 'checkbox'; }
        return CHOICE_ROLES.has(roleOf(el));
    };
    const choiceTypeOf = (el) => {
        if (tagOf(el) === 'input') { const t = inputTypeOf(el); return (t === 'radio' || t === 'checkbox') ? t : ''; }
        const role = roleOf(el);
        if (role === 'radio' || role === 'menuitemradio') return 'radio';
        if (role === 'checkbox' || role === 'switch' || role === 'menuitemcheckbox') return 'checkbox';
        return '';
    };
    const OPTION_TAGS = new Set(['tui-radio-labeled', 'tui-checkbox-labeled', 'tui-radio-block', 'tui-checkbox-block']);
    const hasOptionSemantics = (el) => {
        if (isChoiceControl(el)) return true;
        const tag = tagOf(el);
        if (roleOf(el) === 'option' || OPTION_TAGS.has(tag) || el.hasAttribute('tuioption')) return true;
        return tag === 'label' && !!el.querySelector('input[type="radio"], input[type="checkbox"]');
    };
    const isDropdownTrigger = (el) => {
        const tag = tagOf(el), role = roleOf(el);
        if (tag === 'tui-select' || tag === 'tui-multi-select' || tag === 'tui-combo-box') return true;
        if (hasToken(el, /^t-select$/)) return true;
        if (role === 'combobox' && tag !== 'input') return true;
        const hp = String(attr(el, 'aria-haspopup') || '').toLowerCase();
        return !!hp && hp !== 'false' && hp !== 'dialog';
    };
    const BUTTON_ROLES = new Set(['button', 'link', 'tab', 'menuitem']);
    const isButtonish = (el) => {
        const tag = tagOf(el);
        if (tag === 'button' || tag === 'a' || tag === 'summary') return true;
        if (tag === 'input') return ['button', 'submit', 'reset', 'image'].includes(inputTypeOf(el));
        return BUTTON_ROLES.has(roleOf(el));
    };
    // В v2 ЛЮБАЯ иконка внутри (tui-svg, .t-icon) делала элемент папкой —
    // включая чекбоксы Taiga с галочкой и кнопки с иконкой. Теперь стрелкой
    // считается только элемент с явными признаками раскрытия.
    const TOGGLE_RE = /(expand|collaps|toggle|chevron|caret|arrow|twisty|disclosure)/i;
    const TOGGLE_ICON = '[class*="chevron" i], [class*="arrow" i], [class*="caret" i], [class*="expand" i], '
                      + '[class*="toggle" i], [src*="chevron" i], [src*="arrow" i], [icon*="chevron" i], [icon*="arrow" i]';
    const looksLikeToggle = (el) => {
        if (el.hasAttribute('aria-expanded') && !isDropdownTrigger(el)) return true;
        if (hasToken(el, TOGGLE_RE)) return true;
        try { return !!(el.querySelector && el.querySelector(TOGGLE_ICON)); } catch (e) { return false; }
    };

    // ------------------------- 4. слияние вложенных совпадений в «строки»
    for (const el of cands) {
        const rec = info.get(el);
        rec.textInput = isTextInput(el);
        rec.choice = isChoiceControl(el);
        rec.dropdown = !rec.textInput && isDropdownTrigger(el);
        rec.visibleText = rec.textInput ? '' : textOf(el, candSet, 400);
        rec.label = rec.textInput ? inputLabel(el)
                  : rec.dropdown ? dropdownLabel(el)
                  : (meaningful(rec.visibleText) ? rec.visibleText : rowLabel(el));
    }
    const keptFlag = new Set();
    const passUp = (rec, parent) => {
        if (rec.toggle && !parent.toggle) parent.toggle = rec.toggle;
        if (rec.control && !parent.control) parent.control = rec.control;
    };
    // обратный порядок документа = потомки обрабатываются раньше предков
    for (let i = cands.length - 1; i >= 0; i--) {
        const el = cands[i];
        const rec = info.get(el);
        const parent = rec.parent ? info.get(rec.parent) : null;
        const visible = meaningful(rec.visibleText);
        let keep = false;
        if (rec.textInput && el.readOnly && parent && parent.dropdown) {
            keep = false;                         // readonly-input внутри tui-select — часть списка
        } else if (!visible && !rec.textInput && !rec.dropdown && !rec.hasKeptDesc && parent
                   && (rec.choice || looksLikeToggle(el))) {
            // иконка без видимого текста внутри строки: стрелка → toggle, radio → control.
            // aria-label вроде «Развернуть» не делает её отдельным элементом.
            if (rec.choice) { if (!parent.control) parent.control = el; }
            else if (!parent.toggle) parent.toggle = el;
            passUp(rec, parent);
        } else if (rec.textInput || visible || (!rec.hasKeptDesc && meaningful(rec.label))) {
            keep = true;
        } else if (rec.hasKeptDesc) {
            // обёртка без своего видимого текста (контейнер ветки): не показываем, но её
            // стрелку/переключатель отдаём первой строке внутри — заголовку ветки
            const head = rec.firstKeptDesc ? info.get(rec.firstKeptDesc) : null;
            if (head) passUp(rec, head);
        } else if (parent) {
            passUp(rec, parent);                  // прочая безымянная иконка — просто часть строки
        } else if (rec.choice || rec.dropdown || looksLikeToggle(el) || isButtonish(el)) {
            // одиночная иконка без строки-родителя: подпись — у ближайшего предка с текстом
            rec.label = contextLabel(el);
            keep = meaningful(rec.label);
            if (keep && looksLikeToggle(el)) rec.selfToggle = true;
        }
        if (keep) {
            keptFlag.add(el);
            for (let a = rec.parent; a; a = info.get(a).parent) {
                const ar = info.get(a);
                ar.hasKeptDesc = true;
                ar.firstKeptDesc = el;                // последним запишется самый ранний в документе
            }
        }
    }
    let kept = cands.filter((el) => keptFlag.has(el));
    const totalKept = kept.length;
    if (kept.length > MAX) {
        // Сначала кнопки/поля/списки — кнопка «Завершить» обычно в самом конце документа
        // и при обрезке «по порядку» пропадала бы первой. Остальное — строки по порядку.
        const important = new Set(kept.filter((el) => {
            const r = info.get(el);
            return r.textInput || r.dropdown || (isButtonish(el) && !hasOptionSemantics(el));
        }));
        let budget = MAX - important.size;
        kept = kept.filter((el) => important.has(el) || budget-- > 0);
    }
    const keptSet = new Set(kept);
    const indexOf = new Map(kept.map((el, i) => [el, i]));

    const keptCount = new Map();   // предок → сколько строк внутри
    const firstKept = new Map();   // предок → первая строка внутри (в порядке документа)
    for (const el of kept) {
        for (let a = flatParent(el), n = 0; a && n < 60; a = flatParent(a), n++) {
            keptCount.set(a, (keptCount.get(a) || 0) + 1);
            if (!firstKept.has(a)) firstKept.set(a, el);
        }
    }

    // --------------------------------------- 5. структура дерева
    const STATE_TOKEN = /(^|[_-])(open|opened|expanded|collapsed|closed|active|selected|checked|disabled|focused|focus|hover|hovered|pressed|visible|hidden|first|last|odd|even|loading|\d+)$/i;
    const sigCache = new Map();
    // «Форма» элемента без state-модификаторов и служебных классов Angular
    const signature = (el) => {
        let s = sigCache.get(el);
        if (s === undefined) {
            const toks = tokensOf(el).filter((c) => !/^_?ng-/.test(c) && !STATE_TOKEN.test(c)).sort().join('.');
            s = tagOf(el) + '|' + toks + '|' + roleOf(el);
            sigCache.set(el, s);
        }
        return s;
    };
    const preKind = new Map();
    for (const el of kept) {
        const rec = info.get(el);
        let k = 'row';
        if (rec.textInput) k = 'input';
        else if (isDropdownTrigger(el)) k = 'dropdown';
        else if (isButtonish(el) && !hasOptionSemantics(el) && !rec.toggle && !rec.selfToggle
                 && !el.hasAttribute('aria-expanded')) k = 'button';
        preKind.set(el, k);
    }
    const rowCount = new Map();
    for (const el of kept) {
        if (preKind.get(el) !== 'row') continue;
        for (let a = flatParent(el), n = 0; a && n < 60; a = flatParent(a), n++) {
            rowCount.set(a, (rowCount.get(a) || 0) + 1);
        }
    }
    const rowsIn = (x) => (rowCount.get(x) || 0) + (keptSet.has(x) && preKind.get(x) === 'row' ? 1 : 0);
    const hasSameSigSibling = (a) => {
        const p = a.parentElement;
        if (!p) return false;
        const sig = signature(a);
        for (const s of p.children) {
            if (s !== a && signature(s) === sig && rowsIn(s) > 0) return true;
        }
        return false;
    };
    // Сколько видимых строк в «контейнере детей», идущем за заголовком ветки.
    // Работает для <div.child>[header][children]</div>, tui-tree-item, li>ul и т.п.
    const childRowsOf = (el) => {
        let x = el;
        for (let lvl = 0; lvl < 4; lvl++) {
            const A = flatParent(x);
            if (!A || A === document.body || firstKept.get(A) !== el) return 0;
            const sigX = signature(x);
            let count = 0, listLevel = false;
            for (const ch of A.children) {
                if (ch === x) continue;
                const inside = rowsIn(ch);
                if (!inside) continue;
                if (keptSet.has(ch) || signature(ch) === sigX) { listLevel = true; continue; }
                count += inside;
            }
            if (listLevel) return 0;          // x — элемент плоского списка, а не заголовок
            if (count > 0) return count;
            x = A;
        }
        return 0;
    };
    // Собственные контейнеры строки: предки, в которых она — первая строка
    const ownContainers = (el) => {
        const out = [];
        for (let a = flatParent(el), n = 0; a && n < 3; a = flatParent(a), n++) {
            if (a === document.body || firstKept.get(a) !== el) break;
            out.push(a);
            if (hasSameSigSibling(a)) break;
        }
        return out;
    };
    const EXPANDED_RE = /(^|[_-])(expanded|opened|open|unfolded)$/i;
    const COLLAPSED_RE = /(^|[_-])(collapsed|closed|folded)$/i;
    const expandedFromAttrs = (el, rec, containers) => {
        const probes = [el];
        if (rec.toggle) probes.push(rec.toggle);
        probes.push(...containers);
        for (const x of probes) {
            const v = attr(x, 'aria-expanded');
            if (v === 'true') return true;
            if (v === 'false') return false;
        }
        for (const x of probes) {
            if (hasToken(x, EXPANDED_RE)) return true;
            if (hasToken(x, COLLAPSED_RE)) return false;
        }
        return null;
    };

    const FOLDER_TOKEN_RE = /(^|[_-])(expandable|has-?children|folder|branch|collapsible|parent)$/i;
    const ROW_TOKEN_RE = /(tree-?item|category-?item|list-?item|^t-item$|child__header|menu-?item)/i;
    for (const el of kept) {
        const rec = info.get(el);
        rec.containers = ownContainers(el);
        const pk = preKind.get(el);
        if (pk === 'input') { rec.kind = 'INPUT'; continue; }
        if (pk === 'dropdown') {
            rec.kind = 'DROPDOWN';
            rec.expanded = attr(el, 'aria-expanded') === 'true'
                || !!(el.querySelector && el.querySelector('[aria-expanded="true"]'));
            continue;
        }
        const exp = expandedFromAttrs(el, rec, rec.containers);
        const childRows = childRowsOf(el);
        const optionish = hasOptionSemantics(el) || !!rec.control;
        const folderish = !!rec.toggle || !!rec.selfToggle || exp !== null || childRows > 0
            || hasToken(el, FOLDER_TOKEN_RE);
        if (folderish && pk === 'row') {
            rec.kind = 'FOLDER';
            rec.expanded = exp === true || childRows > 0;   // видимые дети важнее устаревшего aria
            rec.selectable = optionish;
        } else if (optionish) rec.kind = 'OPTION';
        else if (isButtonish(el)) rec.kind = 'BUTTON';
        else if (hasToken(el, ROW_TOKEN_RE) || roleOf(el) === 'treeitem') rec.kind = 'OPTION';
        else rec.kind = 'OTHER';
    }

    // Путь (родительские папки) и глубина
    const levelOf = (el) => {
        for (let a = el, n = 0; a && n < 3; a = flatParent(a), n++) {
            if (n > 0 && firstKept.get(a) !== el) break;
            const v = parseInt(attr(a, 'aria-level'), 10);
            if (v > 0) return v;
        }
        return 0;
    };
    const ancestryOf = (el) => {
        const heads = [];
        let c = el;
        for (let A = flatParent(el), guard = 0; A && guard < 60; c = A, A = flatParent(A), guard++) {
            if (A === document.body || A === document.documentElement) break;
            const sigC = signature(c);
            for (let s = c.previousElementSibling, n = 0; s && n < 3; s = s.previousElementSibling, n++) {
                const kc = keptSet.has(s) ? 1 + (keptCount.get(s) || 0) : (keptCount.get(s) || 0);
                if (!kc) continue;                               // пустой/служебный сосед
                if (signature(s) === sigC) break;                // соседний элемент того же списка
                const head = keptSet.has(s) ? s : firstKept.get(s);
                if (head && head !== el && info.get(head).kind === 'FOLDER' && kc <= 3) heads.unshift(head);
                break;
            }
        }
        return heads;
    };
    if (kept.some((el) => levelOf(el) > 0)) {
        const stack = [];
        for (const el of kept) {
            const rec = info.get(el);
            const lvl = levelOf(el);
            if (!lvl) { rec.path = []; rec.depth = 0; continue; }
            stack.length = Math.min(stack.length, lvl - 1);
            rec.path = stack.slice(0, lvl - 1).filter(Boolean).map((h) => info.get(h).label);
            rec.depth = lvl - 1;
            if (rec.kind === 'FOLDER') stack[lvl - 1] = el;
        }
    } else {
        for (const el of kept) {
            const rec = info.get(el);
            rec.path = ancestryOf(el).map((h) => info.get(h).label);
            rec.depth = rec.path.length;
        }
    }

    // --------------------------------------------- 6. состояния
    const nearestKept = (x) => { for (let a = x; a; a = flatParent(a)) if (keptSet.has(a)) return a; return null; };
    const ownQuery = (el, sel) => {
        let list = [];
        try { list = el.querySelectorAll(sel); } catch (e) { return null; }
        for (const x of list) if (nearestKept(x) === el) return x;
        if (el.shadowRoot) { try { const x = el.shadowRoot.querySelector(sel); if (x) return x; } catch (e) {} }
        return null;
    };
    const CHOICE_SEL = 'input[type="radio"], input[type="checkbox"], [role="radio"], [role="checkbox"], [role="switch"]';
    const choiceState = (x) => {
        if (tagOf(x) === 'input') return !!x.checked;
        const v = attr(x, 'aria-checked') || attr(x, 'aria-selected') || attr(x, 'aria-pressed');
        if (v === 'true' || v === 'mixed') return true;
        if (v === 'false') return false;
        return null;
    };
    // Токены, а не подстроки: 'inactive', 'interactive', 'unselected' больше не «выбраны»
    const SELECTED_RE = /(^|[_-])(selected|checked|chosen|picked|active)$/i;
    const selectedOf = (el, rec) => {
        if (tagOf(el) === 'input' && isChoiceControl(el)) return !!el.checked;
        for (const a of ['aria-checked', 'aria-selected', 'aria-pressed']) {
            const v = attr(el, a);
            if (v === 'true' || v === 'mixed') return true;
            if (v === 'false') return false;
        }
        const ctrl = rec.control || ownQuery(el, CHOICE_SEL);     // скрытый input внутри label/tui-radio
        if (ctrl) { const st = choiceState(ctrl); if (st !== null) return st; }
        for (const c of [el, ...(rec.containers || [])]) if (hasToken(c, SELECTED_RE)) return true;
        return false;
    };
    const DISABLED_RE = /(^|[_-])disabled$/i;
    const disabledOf = (el, rec) => {
        try { if (el.matches(':disabled')) return true; } catch (e) {}
        if (el.closest && el.closest('[aria-disabled="true"], [inert]')) return true;
        if (hasToken(el, DISABLED_RE)) return true;
        if (rec.textInput && el.readOnly === true) return true;
        const ctrl = rec.control || ownQuery(el, 'input[type="radio"], input[type="checkbox"]');
        return !!(ctrl && ctrl.disabled);
    };
    const DIALOG_SEL = '[role="dialog"], [role="alertdialog"], dialog[open], [aria-modal="true"], tui-dialog';
    const POPUP_SEL = 'tui-dropdown, tui-data-list, [role="listbox"], [role="menu"], .cdk-overlay-pane';
    const containerOf = (el) => {
        if (!el.closest) return '';
        if (el.closest('tui-alerts, tui-alert, [class*="toast" i]')) return 'toast';
        if (el.closest(DIALOG_SEL)) return 'dialog';
        const p = el.closest(POPUP_SEL);
        if (p) {
            if (p.closest('tui-dropdown, .cdk-overlay-pane')) return 'popup';
            const pos = styleOf(p).position;
            if (pos === 'absolute' || pos === 'fixed') return 'popup';
        }
        return '';
    };
    const deepHit = (x, y) => {
        let hit = document.elementFromPoint(x, y);
        for (let i = 0; hit && hit.shadowRoot && i < 10; i++) {
            const inner = hit.shadowRoot.elementFromPoint(x, y);
            if (!inner || inner === hit) break;
            hit = inner;
        }
        return hit;
    };
    const occluderOf = (el, r) => {
        const x = r.left + r.width / 2, y = r.top + r.height / 2;
        if (x < 0 || y < 0 || x >= vw || y >= vh) return null;
        const hit = deepHit(x, y);
        if (!hit || contains(el, hit) || contains(hit, el)) return null;
        return hit;
    };
    const describe = (x) => {
        const t = textOf(x, null, 80);
        const c = tokensOf(x).slice(0, 2).join('.');
        return tagOf(x) + (c ? '.' + c : '') + (t ? ' «' + t + '»' : '');
    };

    // --------------------------------------------- 7. служебные элементы и подписи
    // Плееры аудио/видео: свои кнопки и ползунки агенту не нужны (запись он расшифровывает
    // и доигрывает сам) — в тексте страницы плеер заменяется строкой [АУДИО n].
    const mediaEls = ALL.filter((n) => { const t = tagOf(n); return t === 'audio' || t === 'video'; });
    const widgetRootOf = (m) => {
        let root = m;
        for (let a = flatParent(m), n = 0; a && n < 6; a = flatParent(a), n++) {
            if (a === document.body || a === document.documentElement) break;
            const r = a.getBoundingClientRect();
            if (r.height > 160) break;
            if (mediaEls.some((o) => o !== m && contains(a, o))) break;
            root = a;
        }
        return root;
    };
    const mediaRootOf = new Map();                  // корень виджета → <audio>/<video>
    for (const m of mediaEls) mediaRootOf.set(widgetRootOf(m), m);
    const inMediaWidget = (el) => {
        for (const root of mediaRootOf.keys()) if (root !== el && contains(root, el)) return true;
        return false;
    };
    // Переключатели карусели фото («1» … «20», стрелки): фото агент скачивает сам
    const PAGER_RE = /(^|[_-])(pagination|pager|carousel|slider|swiper|gallery|dots|bullets)([_-]|$)/i;
    const isPagerControl = (el, label) => {
        if (!(label === '' || /^\d{1,3}$/.test(label) || /^[‹›<>←→«»]$/.test(label))) return false;
        for (let a = el, n = 0; a && n < 6; a = flatParent(a), n++) {
            const t = tagOf(a);
            if (t === 'tui-pagination' || t === 'tui-carousel' || hasToken(a, PAGER_RE)) return true;
        }
        return false;
    };
    // Подпись поля — ближайший текст перед ним («Цвет» перед списком «Выберите значение»,
    // «Ниже укажите полный адрес…» перед полем ввода)
    const FIELD_HOST = 'tui-input, tui-textfield, tui-input-number, tui-textarea, tui-input-date, '
                     + 'tui-input-tag, tui-select, tui-combo-box, tui-multi-select';
    const captionOf = (el) => {
        // поиск начинается снаружи обёртки поля: внутри неё — декор и «зеркало» значения
        const start = (el.closest && el.closest(FIELD_HOST)) || el;
        for (let a = start, n = 0; a && n < 4; a = flatParent(a), n++) {
            if (a === document.body || a === document.documentElement) break;
            let sib = a.previousElementSibling;
            for (let k = 0; sib && k < 3; sib = sib.previousElementSibling, k++) {
                if (keptSet.has(sib) || (keptCount.get(sib) || 0) > 0) return '';
                if (attr(sib, 'aria-hidden') === 'true') continue;
                const st = styleOf(sib);
                if (st.display === 'none' || st.visibility === 'hidden') continue;
                const t = textOf(sib, keptSet, 200);
                if (meaningful(t)) return t.length <= 160 ? t : '';
            }
        }
        return '';
    };
    const PLACEHOLDER_RE = /^(выберите|выбрать|не выбран|select|choose|укажите|—|-)/i;

    // --------------------------------------------- 8. разметка и результат
    const elements = [];
    const occluders = new Map();
    kept.forEach((el, i) => {
        const rec = info.get(el);
        const uid = GEN + '-' + i;
        el.setAttribute(A_ID, uid);
        let hasToggle = false, hasSelect = false;
        if (rec.toggle && rec.toggle !== el && rec.toggle.isConnected) {
            rec.toggle.setAttribute(A_TOGGLE, uid); hasToggle = true;
        }
        if (rec.control && rec.control !== el && visibleRect(rec.control)) {
            rec.control.setAttribute(A_SELECT, uid); hasSelect = true;
        }
        const r = rec.r;
        const inViewport = r.bottom > 0 && r.right > 0 && r.top < vh && r.left < vw;
        const occ = inViewport ? occluderOf(el, r) : null;
        if (occ) occluders.set(occ, (occluders.get(occ) || 0) + 1);
        const tag = tagOf(el);
        const kind = rec.kind;
        let text = rec.textInput ? '' : rec.label;
        let placeholder = rec.textInput ? rec.label : '';
        let inputType = '', value = '', options = [], caption = '';
        if (kind === 'INPUT') {
            inputType = tag === 'textarea' ? 'textarea' : tag === 'select' ? 'select'
                : tag === 'input' ? inputTypeOf(el) : (el.isContentEditable ? 'contenteditable' : (roleOf(el) || 'text'));
            if (tag === 'select') {
                options = Array.from(el.options || []).map((o) => norm(o.textContent)).filter(Boolean).slice(0, 60);
                const so = el.selectedOptions && el.selectedOptions[0];
                value = so ? norm(so.textContent) : '';
            } else if (tag === 'input' || tag === 'textarea') {
                value = inputType === 'password' ? (el.value ? '***' : '') : String(el.value || '');
            } else {
                value = norm(el.innerText);
            }
            caption = captionOf(el);
        } else if (kind === 'DROPDOWN') {
            // Подпись списка стабильна («Цвет»), а видимый текст — это выбранное значение
            // или заглушка «Выберите значение»: ключ памяти не должен меняться при выборе.
            const shown = rec.label;
            const field = innerField(el);
            const ph = field ? norm(attr(field, 'placeholder')) : '';
            const fieldVal = field && typeof field.value === 'string' ? norm(field.value) : '';
            const isPh = !meaningful(shown) || shown === ph || PLACEHOLDER_RE.test(shown);
            caption = captionOf(el);
            if (caption) {
                text = caption;
                value = fieldVal || (isPh ? '' : shown);
                placeholder = isPh ? shown : ph;
            } else {
                value = fieldVal;
            }
        }
        let choiceType = '';
        if (kind === 'OPTION' || kind === 'FOLDER') {
            choiceType = choiceTypeOf(el) || (rec.control ? choiceTypeOf(rec.control) : '');
            if (!choiceType) {
                const c = ownQuery(el, 'input[type="radio"], input[type="checkbox"]');
                if (c) choiceType = inputTypeOf(c);
            }
        }
        let href = '';
        const link = tag === 'a' ? el : (el.querySelector ? el.querySelector('a[href]') : null);
        if (link && nearestKept(link) === el && /^https?:/i.test(String(link.href || ''))) href = String(link.href);
        const aux = inMediaWidget(el) || isPagerControl(el, norm(text));
        elements.push({
            uid, tag, kind, text, placeholder,
            value: String(value).slice(0, 2000), inputType, choiceType, options,
            selected: kind !== 'INPUT' && kind !== 'DROPDOWN' ? selectedOf(el, rec) : false,
            disabled: disabledOf(el, rec),
            expanded: (kind === 'FOLDER' || kind === 'DROPDOWN') ? !!rec.expanded : null,
            selectable: !!rec.selectable, hasToggle, hasSelect,
            depth: rec.depth || 0, path: rec.path || [],
            container: containerOf(el), inViewport, occluded: !!occ,
            caption, href, aux,
        });
    });
    let overlay = '';
    let topOcc = null, topN = 0;
    for (const [o, n] of occluders) if (n > topN) { topOcc = o; topN = n; }
    if (topOcc && topN >= 2) overlay = describe(topOcc);

    // --------------------------------------------- 9. медиа: фото и аудио
    const images = [], imageIndex = new Map();
    const IMG_SKIP_RE = /(^|[_-])(icon|logo|avatar|emoji|badge|flag|spinner|loader)([_-]|$)/i;
    for (const n of ALL) {
        if (tagOf(n) !== 'img' || images.length >= 120) continue;
        const src = String(n.currentSrc || n.src || '');
        if (!src || /^data:image\/svg|\.svg(\?|#|$)/i.test(src)) continue;
        if (hasToken(n, IMG_SKIP_RE)) continue;
        // скрытые слайды карусели нужны (overflow-обрезка не в счёт), display:none — нет
        if (n.checkVisibility && !n.checkVisibility({ checkVisibilityCSS: true, visibilityProperty: true })) continue;
        const r = n.getBoundingClientRect();
        const nw = n.naturalWidth || 0, nh = n.naturalHeight || 0;
        if (!((nw >= 64 && nh >= 64) || (r.width >= 48 && r.height >= 48))) continue;
        const k = images.length + 1;
        const uid = GEN + '-i' + k;
        n.setAttribute(A_MEDIA, uid);
        imageIndex.set(n, k);
        images.push({
            n: k, uid, src, alt: norm(attr(n, 'alt')).slice(0, 120),
            loaded: !(n.complete && nw === 0), width: nw, height: nh,
        });
    }
    const audios = [], audioIndex = new Map();
    const PLAY_RE = /(play|pause|воспроизв|пауз)/i;
    for (const [root, m] of mediaRootOf) {
        const rr = root.getBoundingClientRect();
        const shown = (rr.width >= 1 && rr.height >= 1 && rendered(root)) || (m.controls && rendered(m));
        if (!shown) continue;
        let src = String(m.currentSrc || m.src || '');
        if (!src) {
            const so = m.querySelector('source[src]');
            if (so) src = String(so.src || '');
        }
        const k = audios.length + 1;
        const uid = GEN + '-a' + k;
        m.setAttribute(A_MEDIA, uid);
        audioIndex.set(root, k);
        let playUid = '';
        if (root !== m && root.querySelectorAll) {
            for (const b of root.querySelectorAll('button, [role="button"]')) {
                const label = [attr(b, 'title'), attr(b, 'aria-label'), attr(b, 'automation-id'), clsOf(b)].join(' ');
                if (PLAY_RE.test(label)) { playUid = uid + 'p'; b.setAttribute(A_MEDIA, playUid); break; }
            }
        }
        const dur = Number.isFinite(m.duration) ? m.duration : 0;
        audios.push({
            n: k, uid, playUid, src, duration: dur, current: m.currentTime || 0,
            paused: !!m.paused, ended: !!m.ended || (dur > 0 && m.currentTime >= dur - 0.25),
            kind: tagOf(m),
        });
    }

    // --------------------------------------------- 10. сообщения платформы
    // Уведомления Taiga (tui-notification/tui-alert), role=alert, ошибки полей.
    // v3 искал [class*="invalid"] — это совпадало с Angular-классом ng-invalid у любого
    // незаполненного поля, и подпись «Цвет» попадала в «сообщения страницы».
    const NOTICE_SEL = 'tui-notification, tui-alert, tui-error, [role="alert"], [role="status"], '
        + '[class*="notification" i], [class*="toast" i], [class*="alert" i], [class*="error" i], '
        + '[class*="warning" i], [class*="hint" i]';
    const APPEARANCE = { error: 'error', negative: 'error', danger: 'error', warning: 'warning',
                         success: 'success', positive: 'success', info: 'hint', neutral: 'info', action: 'hint' };
    const noticeTokens = (el) => tokensOf(el).filter((t) => !/^_?ng-/.test(t) && !/^_/.test(t));
    const noticeKind = (el) => {
        const ap = String(attr(el, 'data-appearance') || attr(el, 'appearance') || '').toLowerCase();
        for (const key of Object.keys(APPEARANCE)) if (ap.includes(key)) return APPEARANCE[key];
        const toks = noticeTokens(el).join(' ');
        if (/(error|invalid|danger|negative|fail)/i.test(toks)) return 'error';
        if (/warning/i.test(toks)) return 'warning';
        if (/success|positive/i.test(toks)) return 'success';
        if (/(hint|tip|help|notification|info)/i.test(toks)) return 'hint';
        const role = roleOf(el), tag = tagOf(el);
        if (role === 'alert' || tag === 'tui-error') return 'error';
        return 'info';
    };
    const noticeMatches = (el) => {
        const tag = tagOf(el), role = roleOf(el);
        if (tag === 'tui-notification' || tag === 'tui-alert' || tag === 'tui-error') return true;
        if (role === 'alert' || role === 'status') return true;
        return noticeTokens(el).some((t) => /(^|[_-])(notification|toast|alert|error|warning|hint)([_-]|$)/i.test(t));
    };
    const notices = [], noticeIndex = new Map();
    try {
        for (const n of document.querySelectorAll(NOTICE_SEL)) {
            if (notices.length >= 12) break;
            if (!noticeMatches(n) || keptSet.has(n)) continue;
            let inside = false;
            for (const x of noticeIndex.keys()) if (contains(x, n)) { inside = true; break; }
            if (inside) continue;
            if ((keptCount.get(n) || 0) > 2) continue;            // обёртка формы, а не сообщение
            const r = n.getBoundingClientRect();
            if (r.width < 1 || r.height < 1) continue;
            if (n.checkVisibility && !n.checkVisibility(VIS)) continue;
            if (n.closest && n.closest('tui-hint, [role="tooltip"]')) continue;
            const t = textOf(n, keptSet, 800);
            if (t.length < 3) continue;
            const where = (n.closest && n.closest('tui-alerts, tui-alert, [class*="toast" i]')) ? 'toast'
                : (n.closest && n.closest(DIALOG_SEL)) ? 'dialog' : 'inline';
            noticeIndex.set(n, notices.length);
            notices.push({ kind: noticeKind(n), text: t, where });
        }
    } catch (e) {}

    // --------------------------------------------- 11. reader-view страницы
    // Весь видимый текст по порядку документа: заголовки, абзацы (с переносами <br>),
    // таблицы, фото, аудио и интерактивные элементы НА СВОИХ МЕСТАХ — вопрос стоит
    // строкой выше своих вариантов, подпись «Цвет» — строкой выше списка.
    const sinks = { main: [], dialog: [], popup: [], toast: [] };
    let sink = 'main', cur = '', budget = 60000;
    const pushLine = (line) => { if (budget > 0) { sinks[sink].push(line); budget -= line.length; } };
    const flush = () => {
        const t = cur.replace(/[ \t ​]+/g, ' ').trim();
        if (t) pushLine(t);
        cur = '';
    };
    const SKIP_TAGS = new Set(['script', 'style', 'noscript', 'template', 'svg', 'head', 'meta', 'link',
                               'title', 'object', 'embed', 'canvas', 'select', 'option', 'input',
                               'textarea', 'button', 'audio', 'video', 'source', 'track', 'map']);
    const BLOCK_RE = /^(block|flex|grid|table|list-item|flow-root|table-row|table-caption|table-row-group|table-header-group|table-footer-group)/;
    const isBlock = (st) => BLOCK_RE.test(st.display);
    const HEAD_RE = /(^|__|-)(title|header|heading|subtitle)(__|-|$)|^tui-text_h\d$/i;
    const headingLevel = (el) => {
        const t = tagOf(el);
        const m = /^h([1-6])$/.exec(t);
        if (m) return +m[1];
        if (roleOf(el) === 'heading') return parseInt(attr(el, 'aria-level'), 10) || 2;
        if (!tokensOf(el).some((c) => HEAD_RE.test(c))) return 0;
        if ((keptCount.get(el) || 0) > 0) return 0;
        for (const ch of el.children) if (isBlock(styleOf(ch))) return 0;
        const txt = textOf(el, keptSet, 200);
        return (txt && txt.length <= 150) ? 4 : 0;         // «заголовок по классу» — младший уровень
    };
    const tableLines = (tbl) => {
        const out = [];
        for (const row of tbl.querySelectorAll('tr')) {
            if (row.closest('table') !== tbl) continue;
            const cells = [];
            for (const c of row.children) {
                const tc = tagOf(c);
                if (tc !== 'td' && tc !== 'th') continue;
                cells.push(textOf(c, keptSet, 600) || '—');
            }
            if (cells.some(meaningful)) out.push('| ' + cells.join(' | ') + ' |');
            if (out.length >= 300) break;
        }
        return out;
    };
    const DIALOG_ROOT = (el) => { try { return el.matches(DIALOG_SEL); } catch (e) { return false; } };
    const POPUP_ROOT = (el) => {
        const t = tagOf(el);
        if (t === 'tui-dropdown' || hasToken(el, /^cdk-overlay-pane$/)) return true;
        const role = roleOf(el);
        if (role === 'listbox' || role === 'menu') {
            const pos = styleOf(el).position;
            return pos === 'absolute' || pos === 'fixed';
        }
        return false;
    };
    let muted = 0;
    const walk = (node) => {
        if (budget <= 0) return;
        if (node.nodeType === 3) { if (!muted) cur += node.nodeValue; return; }
        if (node.nodeType !== 1 && node.nodeType !== 11) return;
        if (node.nodeType === 11) { for (const ch of node.childNodes) walk(ch); return; }
        const el = node, tag = tagOf(el);
        if (keptSet.has(el)) { flush(); pushLine('\u0000E' + elIndexOf.get(el) + '\u0000'); return; }
        if (attr(el, 'aria-hidden') === 'true' && el.childElementCount <= 3 && !imageIndex.has(el)) return;
        if (noticeIndex.has(el)) {
            flush();
            const k = noticeIndex.get(el);
            const prev = sink;
            if (notices[k].where === 'toast') sink = 'toast';
            pushLine('\u0001N' + k + '\u0001');
            sink = prev;
            return;
        }
        if (audioIndex.has(el)) { flush(); pushLine('\u0000A' + audioIndex.get(el) + '\u0000'); return; }
        if (tag === 'img') { if (imageIndex.has(el)) cur += ' \u0000I' + imageIndex.get(el) + '\u0000 '; return; }
        if (tag === 'br') { flush(); return; }
        if (tag === 'hr') { flush(); return; }
        if (tag === 'iframe') {
            const r = el.getBoundingClientRect();
            if (r.width >= 100 && r.height >= 100) { flush(); pushLine('[встроенная страница]'); }
            return;
        }
        if (SKIP_TAGS.has(tag)) return;
        // всплывающие подсказки при наведении — временные, в текст страницы не входят
        if (tag === 'tui-hints' || tag === 'tui-hint' || roleOf(el) === 'tooltip') return;
        const st = styleOf(el);
        if (st.display === 'none' || st.visibility === 'hidden' || st.visibility === 'collapse') return;
        if (st.display !== 'contents') {
            const r = el.getBoundingClientRect();
            const clips = st.overflowX !== 'visible' || st.overflowY !== 'visible';
            if ((r.width < 1 || r.height < 1) && clips) return;      // схлопнутая ветка/аккордеон
        }
        const level = headingLevel(el);
        if (level) {
            flush();
            const t = textOf(el, keptSet, 300);
            if (meaningful(t)) pushLine('#'.repeat(Math.min(level, 4)) + ' ' + t);
            return;
        }
        if (tag === 'table' && !(keptCount.get(el) || 0)) {
            flush();
            for (const line of tableLines(el)) pushLine(line);
            return;
        }
        let switched = null;
        if (sink === 'main' && DIALOG_ROOT(el) && rendered(el)) switched = 'dialog';
        else if (sink === 'main' && POPUP_ROOT(el)) switched = 'popup';
        const prev = sink;
        const block = isBlock(st) || switched;
        if (block) flush();
        if (switched) sink = switched;
        if (tag === 'li') cur += '• ';
        if (st.display === 'table-cell') cur += ' ';
        const mute = fieldWraps.has(el);
        if (mute) muted++;
        for (const ch of flatChildren(el)) walk(ch);
        if (mute) muted--;
        if (st.display === 'table-cell') cur += ' ';
        if (block) flush();
        sink = prev;
    };
    const elIndexOf = new Map(kept.map((el, i) => [el, i]));
    // Обёртки полей ввода: их текст — это подпись и «зеркало» значения, которые уже есть
    // в строке элемента; в тексте страницы они дублировали бы введённое значение
    const fieldWraps = new Set();
    for (const el of kept) {
        const k = info.get(el).kind;
        if (k !== 'INPUT' || !el.closest) continue;
        const w = el.closest('tui-input, tui-textfield, tui-input-number, tui-textarea, tui-input-date, tui-input-tag');
        if (w && !keptSet.has(w)) fieldWraps.add(w);
    }
    try { if (document.body) walk(document.body); flush(); } catch (e) { sinks.main.push('[ошибка чтения страницы: ' + e + ']'); }

    // --------------------------------------------- 12. капча, загрузка
    // Капча — только видимый iframe виджета. По тексту/классам не ищем: задание
    // ПРО капчу или подпись «protected by reCAPTCHA» вешали v2 навсегда.
    let captcha = false;
    try {
        for (const n of document.querySelectorAll('iframe')) {
            if (!/captcha|turnstile|challenges\.cloudflare/i.test(n.src || '')) continue;
            const r = n.getBoundingClientRect();
            if (r.width < 100 || r.height < 70) continue;          // невидимые бейджи не считаем
            if (n.checkVisibility && !n.checkVisibility(VIS)) continue;
            captcha = true; break;
        }
    } catch (e) {}

    // Лоадер считается, только если его крутилка реально видна. В v3 хватало класса
    // _loading — а у T-Work global-loader с data-visible="false" лежит в DOM всегда,
    // и агент на каждом шаге «ждал загрузку». Мелкие лоадеры (галерея фото) не блокируют.
    let loading = false, dialogLoading = false, localLoading = 0;
    try {
        const LOADER_RE = /(^|[_-])(spinner|loader|loading|preloader|skeleton)($|[_-])/i;
        const sel = '[aria-busy="true"], [class*="spin" i], [class*="load" i], [class*="skeleton" i], '
                  + '[role="progressbar"], tui-loader';
        const seen = [];
        for (const n of document.querySelectorAll(sel)) {
            const tag = tagOf(n);
            if (tag === 'tui-loader') { if (!hasToken(n, /^_loading$/)) continue; }
            else if (attr(n, 'aria-busy') !== 'true' && roleOf(n) !== 'progressbar' && !hasToken(n, LOADER_RE)) continue;
            if (seen.some((x) => contains(x, n))) continue;
            if ((keptCount.get(n) || 0) > 0 && tag !== 'tui-loader') continue;   // обёртка с контентом
            const spin = tag === 'tui-loader' ? (n.querySelector('.t-loader') || null) : n;
            if (!spin) continue;
            const r = spin.getBoundingClientRect();
            if (r.width < 4 || r.height < 4) continue;
            if (spin.checkVisibility && !spin.checkVisibility(VIS)) continue;
            if (n.checkVisibility && !n.checkVisibility(VIS)) continue;
            if (r.bottom < 0 || r.right < 0 || r.top > vh || r.left > vw) continue;
            seen.push(n);
            const box = n.getBoundingClientRect();
            if (n.closest && n.closest(DIALOG_SEL)) { dialogLoading = true; continue; }
            const pos = styleOf(n).position;
            if (box.width * box.height >= 0.25 * vw * vh || pos === 'fixed' || /global/i.test(tag)) loading = true;
            else localLoading++;
        }
    } catch (e) {}

    let dialogOpen = false, dialogFrames = 0;
    const boxOf = (d) => {
        // у кастомных элементов (tui-dialog) без display:block собственный bbox бывает нулевым
        let r = d.getBoundingClientRect();
        if (r.width >= 100 && r.height >= 60) return r;
        for (const ch of d.children) {
            const cr = ch.getBoundingClientRect();
            if (cr.width * cr.height > r.width * r.height) r = cr;
        }
        return r;
    };
    try {
        for (const d of document.querySelectorAll(DIALOG_SEL)) {
            if (d.closest('tui-alerts, tui-alert')) continue;
            const r = boxOf(d);
            if (r.width < 100 || r.height < 60 || !rendered(d)) continue;
            dialogOpen = true;
            for (const f of d.querySelectorAll('iframe')) {
                const fr = f.getBoundingClientRect();
                if (fr.width >= 100 && fr.height >= 100) dialogFrames++;
            }
        }
    } catch (e) {}

    // Главная картинка — для совместимости с LLM_VISION=image/frame и отпечатком v3
    let image = null, bestArea = 0;
    for (const n of ALL) {
        const tag = tagOf(n);
        if (tag !== 'img' && tag !== 'canvas' && tag !== 'video') continue;
        const r = n.getBoundingClientRect();
        if (r.width < 48 || r.height < 48) continue;              // иконки и логотипы-миниатюры
        if (nearestKept(n)) continue;
        if (n.checkVisibility && !n.checkVisibility(VIS)) continue;
        if (r.width * r.height > bestArea) { bestArea = r.width * r.height; image = n; }
    }
    let imageSrc = '';
    if (image) {
        image.setAttribute(A_IMG, GEN);
        const t = tagOf(image);
        imageSrc = t === 'img' ? String(image.currentSrc || image.src || '')
                               : t + ':' + Math.round(image.width) + 'x' + Math.round(image.height);
        if (imageSrc.startsWith('data:')) imageSrc = 'data:' + imageSrc.length + ':' + imageSrc.slice(-64);
    }

    // --------------------------------------------- 13. прокручиваемые списки
    const scrollables = [];
    const seenScroll = new Set();
    for (const el of kept) {
        if (scrollables.length >= 5) break;
        for (let a = flatParent(el), n = 0; a && n < 40; a = flatParent(a), n++) {
            if (a === document.body || a === document.documentElement || seenScroll.has(a)) break;
            const s = styleOf(a);
            if (/(auto|scroll|overlay)/.test(s.overflowY) && a.scrollHeight > a.clientHeight + 8) {
                seenScroll.add(a);
                let first = -1, last = -1;
                kept.forEach((k, i) => { if (contains(a, k)) { if (first < 0) first = i; last = i; } });
                scrollables.push({
                    first, last,
                    below: Math.max(0, Math.round(a.scrollHeight - a.scrollTop - a.clientHeight)),
                    above: Math.round(a.scrollTop),
                    virtual: tagOf(a) === 'cdk-virtual-scroll-viewport' || hasToken(a, /virtual/i),
                });
                break;
            }
        }
    }

    // Строка JSON вместо объекта: структурная сериализация Playwright на 1500 строках
    // стоит ~400 мс, JSON.stringify + json.loads — ~60 мс
    return JSON.stringify({
        gen: GEN, url: location.href,
        elements, total: totalKept, hidden: hiddenCount,
        reader: sinks.main, dialog: sinks.dialog, popup: sinks.popup, toast: sinks.toast,
        notices, images, audios,
        captcha, loading, dialogLoading, localLoading, dialogOpen, dialogFrames,
        overlay, imageSrc, scrollables,
    });
}
"""


# ---------------------------------------------------------------------------
# Python-обёртка
# ---------------------------------------------------------------------------

# Плейсхолдеры reader-view (ставит JS): элемент, фото, аудио, уведомление
_PH_RE = re.compile(r"\x00([EIA])(\d+)\x00|\x01N(\d+)\x01")
_ONLY_PHOTOS_RE = re.compile(r"^(?:\s*\[ФОТО \d+(?: не загрузилось)?\]\s*)+$")
# Заголовки, которые есть у любого задания — не описывают вид задания
_GENERIC_HEADINGS = {"выполните задание", "задание", "инструкция", "подробная инструкция"}


def _clean_basis(text: str) -> str:
    return normalize_text(_TIMER_RE.sub("", text))


class DomParser:
    """Асинхронный парсер DOM целевого фрейма (один evaluate на снимок)."""

    def __init__(self, frame: Frame, *, max_elements: int = MAX_ELEMENTS) -> None:
        self._frame = frame
        self._max_elements = max_elements

    async def parse(self, *, quiet: bool = False) -> PageState:
        """Собрать атомарный PageState и разметить элементы метками data-agent-id.

        quiet=True — сводка в DEBUG (для поллинга, чтобы не засорять лог)."""
        generation = f"{next(_GENERATION):x}"
        payload: str = await self._frame.evaluate(
            _JS_SNAPSHOT,
            {
                "gen": generation,
                "max": _JS_MAX_ELEMENTS,
                "selectors": INTERACTIVE_SELECTOR,
                "pointer": True,
            },
        )
        raw: dict = json.loads(payload)

        elements = self._build_elements(raw.get("elements") or [])
        notices = [
            Notice(kind=str(n.get("kind") or "info"), text=str(n.get("text") or ""),
                   where=str(n.get("where") or "inline"))
            for n in raw.get("notices") or []
        ]
        images = [MediaImage(**{k: v for k, v in img.items() if k in MediaImage.model_fields})
                  for img in raw.get("images") or []]
        audios = [
            MediaAudio(n=int(a.get("n", 0)), uid=str(a.get("uid", "")), play_uid=str(a.get("playUid", "")),
                       src=str(a.get("src", "")), duration=float(a.get("duration") or 0),
                       current=float(a.get("current") or 0), paused=bool(a.get("paused", True)),
                       ended=bool(a.get("ended", False)))
            for a in raw.get("audios") or []
        ]
        reader = [str(line) for line in raw.get("reader") or []]
        dialog = [str(line) for line in raw.get("dialog") or []]
        popup = [str(line) for line in raw.get("popup") or []]
        toast = [str(line) for line in raw.get("toast") or []]

        task_text = _plain_text(reader)
        headings = [line for line in reader if line.startswith("#")]
        hint_text = "\n".join(n.text for n in notices if n.kind == "hint")
        alerts = [n.text for n in notices if n.kind in ("error", "warning") or n.where == "toast"]
        image_src = str(raw.get("imageSrc") or "")
        url = str(raw.get("url") or "")
        task_id, preview, content_hash, loose_hash = self._fingerprint(
            url, reader, elements, images, audios, headings,
        )
        pool_key, pool_title, pool_signature = self._pool(headings, elements, task_text)
        headings = [h.lstrip("#").strip() for h in headings]

        state = PageState(
            task_text=task_text,
            hint_text=hint_text,
            elements=elements,
            task_identifier=task_id,
            task_preview=preview,
            state_hash=self._state_hash(elements, alerts + [n.text for n in notices]),
            has_captcha=bool(raw.get("captcha")),
            loading=bool(raw.get("loading")),
            alerts=alerts,
            overlay_text=str(raw.get("overlay") or ""),
            scroll_hints=self._scroll_hints(raw.get("scrollables") or []),
            image_src=image_src,
            generation=generation,
            frame_url=url,
            total_elements=int(raw.get("total") or len(elements)),
            hidden_elements=int(raw.get("hidden") or 0),
            reader=reader,
            dialog_lines=dialog,
            popup_lines=popup,
            toast_lines=toast,
            notices=notices,
            images=images,
            audios=audios,
            headings=headings,
            pool_key=pool_key,
            pool_title=pool_title,
            pool_signature=pool_signature,
            dialog_open=bool(raw.get("dialogOpen")),
            dialog_loading=bool(raw.get("dialogLoading")),
            dialog_frames=int(raw.get("dialogFrames") or 0),
            local_loading=int(raw.get("localLoading") or 0),
            content_hash=content_hash,
            loose_hash=loose_hash,
            media_srcs=[i.src for i in images] + [a.src for a in audios],
        )
        folders = [e for e in elements if e.kind == ElementKind.FOLDER]
        logger.log(
            logging.DEBUG if quiet else logging.INFO,
            "Снимок #%s: элементов=%d (папок откр./закр.=%d/%d, опций=%d, скрытых=%d), фото=%d, аудио=%d, "
            "сообщений=%d, task_id=%s, loading=%s%s",
            generation, len(elements),
            sum(1 for f in folders if f.folder_state == FolderState.OPEN),
            sum(1 for f in folders if f.folder_state == FolderState.CLOSED),
            sum(1 for e in elements if e.kind == ElementKind.OPTION),
            state.hidden_elements, len(images), len(audios), len(notices), task_id, state.loading,
            ", диалог" if state.dialog_open else "",
        )
        return state

    async def get_task_identifier(self) -> str:
        """Совместимость с v2: отпечаток текущего задания."""
        return (await self.parse()).task_identifier

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    @staticmethod
    def _build_elements(raw_list: list[dict]) -> list[ParsedElement]:
        elements: list[ParsedElement] = []
        for idx, raw in enumerate(raw_list):
            try:
                kind = ElementKind(raw.get("kind", "OTHER"))
            except ValueError:
                kind = ElementKind.OTHER
            expanded = raw.get("expanded")
            if kind in (ElementKind.FOLDER, ElementKind.DROPDOWN) and expanded is not None:
                folder_state = FolderState.OPEN if expanded else FolderState.CLOSED
            else:
                folder_state = FolderState.NA
            elements.append(ParsedElement(
                index=idx,
                uid=str(raw.get("uid", "")),
                kind=kind,
                folder_state=folder_state,
                tag=str(raw.get("tag", "")),
                text=str(raw.get("text", "")),
                is_selected=bool(raw.get("selected", False)),
                is_disabled=bool(raw.get("disabled", False)),
                depth=int(raw.get("depth", 0) or 0),
                path=[str(p) for p in raw.get("path") or []],
                placeholder=str(raw.get("placeholder", "")),
                value=str(raw.get("value", "")),
                input_type=str(raw.get("inputType", "")),
                choice_type=str(raw.get("choiceType", "")),
                options=[str(o) for o in raw.get("options") or []],
                container=str(raw.get("container", "")),
                selectable=bool(raw.get("selectable", False)),
                has_toggle=bool(raw.get("hasToggle", False)),
                has_select=bool(raw.get("hasSelect", False)),
                in_viewport=bool(raw.get("inViewport", True)),
                occluded=bool(raw.get("occluded", False)),
                caption=str(raw.get("caption", "")),
                href=str(raw.get("href", "")),
                aux=bool(raw.get("aux", False)),
            ))
        DomParser._assign_keys(elements)
        return elements

    @staticmethod
    def _assign_keys(elements: list[ParsedElement]) -> None:
        """Стабильный ключ «группа|путь|текст[#n]» — память агента привязана к нему,
        а не к индексу, который сдвигается при каждом раскрытии папки."""
        seen: Counter[str] = Counter()
        for el in elements:
            if el.kind in (ElementKind.INPUT, ElementKind.DROPDOWN):
                group = "input"
            elif el.kind == ElementKind.BUTTON:
                group = "button"
            else:
                group = "row"
            label = el.text or el.placeholder or el.caption
            if el.kind == ElementKind.INPUT and el.caption:
                label = f"{el.caption}|{label}"
            base = f"{group}|{' › '.join(el.path)}|{normalize_text(label)}"
            n = seen[base]
            seen[base] += 1
            el.key = base if n == 0 else f"{base}#{n}"

    @staticmethod
    def _fingerprint(
        url: str,
        reader: list[str],
        elements: list[ParsedElement],
        images: list[MediaImage],
        audios: list[MediaAudio],
        headings: list[str],
    ) -> tuple[str, str, str, str]:
        """Отпечаток задания: содержимое страницы БЕЗ сообщений платформы, таймеров,
        подсказок при наведении, состояний и значений полей. «Неверный ответ» и подсказка
        после отправки отпечаток не меняют (в v3 меняли — и память о неверном ответе стиралась).

        Возвращает (task_id, превью, хэш текста, хэш текста без цифр). Хэши текста нужны
        агенту, чтобы отличать новое задание от догрузившихся фото (TaskIdentity)."""
        by_index = {e.index: e for e in elements}
        parts: list[str] = []
        for line in reader:
            def repl(m: re.Match) -> str:
                kind, num, notice = m.group(1), m.group(2), m.group(3)
                if notice is not None or kind != "E":
                    return " "
                el = by_index.get(int(num))
                # строки раскрытых веток (depth > 0) появляются и исчезают при раскрытии —
                # в отпечаток не входят, иначе каждое «open» сбрасывало бы память задания
                if el is None or el.aux or el.depth > 0 or el.container in ("dialog", "popup", "toast"):
                    return " "
                return f" {el.kind.value}:{el.text or el.placeholder or el.caption} "
            parts.append(_PH_RE.sub(repl, line))
        text = f"{urlsplit(url).path}|{_clean_basis(chr(10).join(parts))[:6000]}"
        content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        loose_hash = hashlib.sha1(re.sub(r"\d+", "", text).encode("utf-8")).hexdigest()[:16]
        # первое фото/аудио различает задания с одинаковым текстом («оцените фото»)
        media = (images[0].src if images else "") + "|" + (audios[0].src if audios else "")
        digest = hashlib.sha1(f"{text}|{media}".encode("utf-8")).hexdigest()[:16]
        titled = [t for t in (_heading_text(h) for h in _content_headings(headings))
                  if normalize_text(t) not in _GENERIC_HEADINGS]
        plain = _plain_text(reader)
        preview = (titled[0] if titled else (plain.strip().splitlines() or [""])[0])[:80]
        return digest, preview, content_hash, loose_hash

    @staticmethod
    def _pool(headings: list[str], elements: list[ParsedElement], task_text: str) -> tuple[str, str, list[str]]:
        """«Вид задания»: структура формы без данных — заголовки, варианты radio/checkbox,
        подписи полей. Одинаков у всех заданий одного пула → общая инструкция и уроки."""
        content = _content_headings(headings)
        heads = sorted({re.sub(r"\d+", "", normalize_text(_heading_text(h)))
                        for h in content if 2 < len(_heading_text(h)) <= 80})
        fields = sorted({
            re.sub(r"\d+", "", normalize_text(e.caption or e.placeholder or e.text))
            for e in elements
            if not e.aux and e.container not in ("dialog", "popup") and (
                (e.kind == ElementKind.OPTION and e.choice_type)
                or e.kind in (ElementKind.INPUT, ElementKind.DROPDOWN)
            )
        })
        signature = "|".join(heads) + "#" + "|".join(fields[:80])
        key = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12] if (heads or fields) else ""
        parts = [f"h:{h}" for h in heads] + [f"f:{f}" for f in fields[:80]]
        titled = [(len(h) - len(h.lstrip("#")), _heading_text(h)) for h in content]
        titled = [(lvl, t) for lvl, t in titled if normalize_text(t) not in _GENERIC_HEADINGS and len(t) <= 80]
        major = [t for lvl, t in titled if lvl <= 3]
        if major or titled:
            return key, (major or [t for _, t in titled])[0], parts
        # заголовка нет — первая строка текста после панели режима
        lines = [ln for ln in task_text.splitlines() if ln.strip() and not ln.startswith("#")]
        start = next((i for i, h in enumerate(task_text.splitlines()) if h.startswith("#")
                      and len(h) - len(h.lstrip("#")) <= 3), None)
        if start is not None:
            after = [ln for ln in task_text.splitlines()[start + 1:] if ln.strip() and not ln.startswith("#")]
            lines = after or lines
        return key, (lines[0][:60] if lines else ""), parts

    @staticmethod
    def _state_hash(elements: list[ParsedElement], alerts: list[str]) -> str:
        h = hashlib.sha1()
        for el in elements:
            h.update(
                f"{el.key}|{el.kind.value}|{el.folder_state.value}|{int(el.is_selected)}|"
                f"{int(el.is_disabled)}|{el.value}\n".encode("utf-8")
            )
        for alert in alerts:
            h.update(alert.encode("utf-8"))
        return h.hexdigest()[:16]

    @staticmethod
    def _scroll_hints(scrollables: list[dict]) -> list[str]:
        hints: list[str] = []
        for s in scrollables:
            below, above = int(s.get("below", 0)), int(s.get("above", 0))
            if below < 20 and above < 20:
                continue
            span = f"[{s.get('first')}]–[{s.get('last')}]"
            kind = ("виртуальный список: элементы подгружаются при прокрутке" if s.get("virtual")
                    else "все загруженные элементы уже перечислены")
            hints.append(f"Список {span} прокручивается: ниже ещё ~{below}px, выше ~{above}px ({kind}).")
        return hints


_PANEL_HEADINGS = re.compile(r"^(тренировка|экзамен|обучение|цена задания|задание)\b", re.IGNORECASE)


def _heading_text(line: str) -> str:
    return line.lstrip("#").strip()


def _content_headings(headings: list[str]) -> list[str]:
    """Заголовки содержимого: начиная с первого настоящего заголовка (h1–h3), без
    панели режима («Тренировка», «Экзамен», «Цена задания») — иначе у тренировки и
    обычных заданий одного вида получались разные «виды заданий»."""
    first = next((i for i, h in enumerate(headings) if len(h) - len(h.lstrip("#")) <= 3), 0)
    return [h for h in headings[first:] if not _PANEL_HEADINGS.match(_heading_text(h))]


def _plain_text(lines: list[str]) -> str:
    """Текст страницы без элементов, фото и уведомлений (для логов, отпечатка, превью)."""
    out = []
    for line in lines:
        text = _PH_RE.sub(" ", line).strip()
        if text:
            out.append(re.sub(r"\s{2,}", " ", text))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Представление для LLM
# ---------------------------------------------------------------------------

def is_denied_button(el: ParsedElement) -> bool:
    """Кнопка из стоп-списка FINISH_DENY_SUBSTRINGS («Завершить смену», «Выйти» …):
    LLM её не видит, агент её не нажимает. Строки дерева (OPTION/FOLDER) не затрагиваются —
    категория «Выходная обувь» остаётся доступной."""
    if el.kind not in (ElementKind.BUTTON, ElementKind.OTHER):
        return False
    label = normalize_text(el.text)
    return any(deny in label for deny in FINISH_DENY_SUBSTRINGS)


def select_for_prompt(
    elements: list[ParsedElement], limit: int = MAX_ELEMENTS,
) -> tuple[list[ParsedElement], int]:
    """Выбрать элементы для промпта. В v2 обрезка шла по порядку документа,
    и при большом раскрытом дереве из списка пропадала кнопка «Завершить».
    Теперь кнопки/поля/выбранные/папки сохраняются всегда, а кнопки из
    стоп-списка и служебные элементы (плеер, карусель) не показываются вовсе."""
    visible = [
        e for e in elements
        if not (e.is_disabled and e.kind == ElementKind.OTHER) and not is_denied_button(e)
        and not e.aux and e.container != "toast"
    ]
    if len(visible) <= limit:
        return visible, 0

    def must_keep(e: ParsedElement) -> bool:
        return (
            e.kind in (ElementKind.BUTTON, ElementKind.INPUT, ElementKind.DROPDOWN, ElementKind.FOLDER)
            or e.is_selected or e.container in ("dialog", "popup")
        )

    chosen = {e.index for e in visible if must_keep(e)}
    for e in visible:                       # остальное — в порядке документа, OTHER в последнюю очередь
        if len(chosen) >= limit:
            break
        if e.kind != ElementKind.OTHER:
            chosen.add(e.index)
    for e in visible:
        if len(chosen) >= limit:
            break
        chosen.add(e.index)
    shown = [e for e in visible if e.index in chosen]
    return shown, len(visible) - len(shown)


def build_elements_prompt(elements: list[ParsedElement], limit: int = MAX_ELEMENTS) -> str:
    """Список элементов для LLM-промпта (формат v3; reader-view — render_page)."""
    shown, omitted = select_for_prompt(elements, limit)
    label_counts = Counter(normalize_text(e.text) for e in elements if e.text)
    lines = [
        e.prompt_line(show_path=label_counts[normalize_text(e.text)] > 1) for e in shown
    ]
    if omitted:
        lines.append(f"… ещё {omitted} элементов не показано (лимит {limit}).")
    return "\n".join(lines) if lines else "(интерактивных элементов нет)"


_URL_IN_TEXT = re.compile(r"https?://[^\s<>«»\"']+")


def _coords_from_url(url: str) -> Optional[tuple[float, float]]:
    """Координаты (широта, долгота) из ссылки на карту: Google (ll=/q=/@lat,lon) —
    «широта,долгота»; Яндекс (ll=/pt=) — «долгота,широта»."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = parts.netloc.lower()
    yandex = "yandex" in host or host.endswith("ya.ru")
    query = parse_qs(parts.query)
    pair = None
    for key in ("ll", "pt", "q", "query", "center"):
        if key in query:
            pair = query[key][0]
            break
    if pair is None:
        m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", parts.path)
        pair = f"{m.group(1)},{m.group(2)}" if m else None
    if not pair:
        return None
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", pair)
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    lat, lon = (b, a) if yandex else (a, b)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def _distance_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*p1, *p2))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def geo_facts(state: PageState) -> list[str]:
    """Расстояния между точками из ссылок на карты («Открыть карту» у двух карточек) —
    модели не нужно сравнивать координаты в уме."""
    points: list[tuple[str, tuple[float, float]]] = []
    seen: set[str] = set()
    for el in state.elements:
        if el.href and not el.aux and el.href not in seen:
            c = _coords_from_url(el.href)
            if c:
                seen.add(el.href)
                points.append((f"[{el.index}] «{el.label()}»", c))
    for url in _URL_IN_TEXT.findall(_plain_text(state.reader)):
        if url not in seen:
            c = _coords_from_url(url)
            if c:
                seen.add(url)
                points.append((url[:60], c))
    facts: list[str] = []
    points = points[:4]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            (la, pa), (lb, pb) = points[i], points[j]
            d = _distance_m(pa, pb)
            dist = f"≈ {d:.0f} м" if d < 1000 else f"≈ {d / 1000:.1f} км"
            facts.append(f"Точки на карте {la} ({pa[0]:.5f}, {pa[1]:.5f}) и {lb} ({pb[0]:.5f}, {pb[1]:.5f}): "
                         f"расстояние {dist}")
    return facts


def media_facts(state: PageState) -> list[str]:
    """Одинаковые файлы фото под разными номерами (одно фото в обеих карточках)."""
    by_src: dict[str, list[int]] = {}
    for img in state.images:
        by_src.setdefault(img.src, []).append(img.n)
    return [f"ФОТО {', '.join(map(str, ns))} — один и тот же файл (одинаковый адрес)"
            for ns in by_src.values() if len(ns) > 1]


def photo_groups(state: PageState) -> list[list[int]]:
    """Блоки фото в порядке страницы: подряд идущие [ФОТО n] без текста и элементов между
    ними (галерея, карусель одной карточки). Коллажи не смешивают фото разных блоков."""
    groups: list[list[int]] = []
    current: list[int] = []
    for line in state.reader:
        found = [int(m.group(2)) for m in _PH_RE.finditer(line) if m.group(1) == "I"]
        rest = _PH_RE.sub(lambda m: "" if m.group(1) == "I" else "#", line).strip()
        if found and not rest:
            current.extend(found)
            continue
        if current:
            groups.append(current)
            current = []
        if found:
            groups.append(found)
    if current:
        groups.append(current)
    known = {n for g in groups for n in g}
    missing = [i.n for i in state.images if i.n not in known]   # фото вне основного текста
    if missing:
        groups.append(missing)
    return groups


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def audio_label(audio: MediaAudio) -> str:
    if audio.ended:
        status = "прослушана до конца"
    elif audio.paused:
        status = f"на паузе, {_fmt_time(audio.current)}"
    else:
        status = f"играет, {_fmt_time(audio.current)}"
    length = f"{_fmt_time(audio.duration)}, " if audio.duration else ""
    return f"[АУДИО {audio.n}: {length}{status}]"


def _photo_runs(line: str) -> str:
    """«[ФОТО 1] [ФОТО 2] [ФОТО 3]» → «[ФОТО 1–3]» (подряд идущие номера)."""
    nums = [int(n) for n in re.findall(r"\[ФОТО (\d+)\]", line)]
    broken = set(int(n) for n in re.findall(r"\[ФОТО (\d+) не загрузилось\]", line))
    nums = sorted(set(nums) | broken)
    if not nums:
        return line
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:  # type: ignore[list-item]
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}–{prev}")
        if n is not None:
            start = prev = n
    note = f" (не загрузились: {', '.join(map(str, sorted(broken)))})" if broken else ""
    return f"[ФОТО {', '.join(parts)}]{note}"


def render_page(state: PageState, *, limit: int = READER_MAX_CHARS) -> tuple[str, list[ParsedElement]]:
    """Reader-view для LLM: страница по порядку, элементы — на своих местах.

    Возвращает текст и список показанных элементов (для проверки номеров)."""
    shown, omitted = select_for_prompt(state.elements)
    shown_by_index = {e.index: e for e in shown}
    label_counts = Counter(normalize_text(e.text) for e in state.elements if e.text)
    images = {i.n: i for i in state.images}
    audios = {a.n: a for a in state.audios}
    placed: set[int] = set()

    def expand(lines: list[str]) -> list[str]:
        out: list[str] = []
        photos: list[str] = []            # подряд идущие строки из одних фото → одна строка

        def flush_photos() -> None:
            if photos:
                out.append(_photo_runs(" ".join(photos)))
                photos.clear()

        for line in lines:
            element_line = False

            def repl(m: re.Match) -> str:
                nonlocal element_line
                kind, num, notice = m.group(1), m.group(2), m.group(3)
                if notice is not None:
                    k = int(notice)
                    return state.notices[k].render() if k < len(state.notices) else ""
                n = int(num)
                if kind == "E":
                    el = shown_by_index.get(n)
                    if el is None:
                        return ""
                    placed.add(n)
                    element_line = True
                    above = normalize_text(out[-1]) if out else ""
                    return el.prompt_line(
                        show_path=label_counts[normalize_text(el.text)] > 1,
                        show_caption=not el.caption or normalize_text(el.caption) != above,
                    )
                if kind == "I":
                    img = images.get(n)
                    return f"[ФОТО {n}{'' if img is None or img.loaded else ' не загрузилось'}]"
                audio = audios.get(n)
                return audio_label(audio) if audio else f"[АУДИО {n}]"

            text = _PH_RE.sub(repl, line)
            lead = re.match(r"^ *", text).group(0) if element_line else ""   # отступ дерева
            text = lead + re.sub(r"[ \t]{2,}", " ", text.strip())
            if not text.strip():
                continue
            if not element_line and _ONLY_PHOTOS_RE.match(text):
                photos.append(text)
                continue
            flush_photos()
            if "[ФОТО" in text:
                text = re.sub(r"(?:\[ФОТО \d+(?: не загрузилось)?\]\s*){2,}",
                              lambda m: _photo_runs(m.group(0)) + " ", text).rstrip()
            out.append(text)
        flush_photos()
        return out

    main = expand(state.reader)
    dialog = expand(state.dialog_lines)
    popup = expand(state.popup_lines)
    toasts = expand(state.toast_lines)
    rest = [e for e in shown if e.index not in placed]

    main = _fit(main, limit)
    parts = main or ["(текст страницы не обнаружен)"]
    if rest:
        parts += ["", "Прочие элементы:"] + [
            e.prompt_line(show_path=label_counts[normalize_text(e.text)] > 1) for e in rest
        ]
    if omitted:
        parts.append(f"… ещё {omitted} элементов не показано (лимит {MAX_ELEMENTS}).")
    if dialog:
        parts += ["", "═══ ОТКРЫТЫЙ ДИАЛОГ (поверх страницы) ═══"] + _fit(dialog, limit // 2)
    if popup:
        parts += ["", "═══ ОТКРЫТЫЙ ВЫПАДАЮЩИЙ СПИСОК ═══"] + _fit(popup, limit // 3)
    if toasts:
        parts += ["", "═══ ВСПЛЫВАЮЩИЕ УВЕДОМЛЕНИЯ ═══"] + toasts[:10]
    return "\n".join(parts), shown


def _fit(lines: list[str], limit: int) -> list[str]:
    """Уложить текст в лимит, сохранив строки элементов, заголовки и уведомления."""
    total = sum(len(line) + 1 for line in lines)
    if total <= limit:
        return lines
    essential = re.compile(r"^\s*(\[\d+\]|#|‼|⚠|💡|✓|ℹ|\[АУДИО|\[ФОТО)")
    out: list[str] = []
    used = sum(len(line) + 1 for line in lines if essential.match(line))
    budget = max(limit - used, limit // 4)
    cut = False
    for line in lines:
        if essential.match(line):
            out.append(line)
            continue
        if budget <= 0:
            cut = True
            continue
        piece = line if len(line) <= budget else line[:budget] + " …"
        budget -= len(piece) + 1
        out.append(piece)
    if cut:
        out.append("[… часть текста страницы не поместилась в лимит …]")
    return out
