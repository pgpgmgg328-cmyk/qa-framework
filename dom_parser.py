"""dom_parser.py v3 — атомарный снимок DOM целевого фрейма.

Что изменилось относительно v2 и почему:

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
import re
from collections import Counter
from urllib.parse import urlsplit

from playwright.async_api import Frame

from config import MAX_ELEMENTS
from models import ElementKind, FolderState, PageState, ParsedElement, normalize_text

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
          A_SELECT = 'data-agent-select', A_IMG = 'data-agent-img';

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
            if (c.scrollable) continue;                               // до элемента можно доскроллить
            const ix = Math.min(r.right, c.rect.right) - Math.max(r.left, c.rect.left);
            const iy = Math.min(r.bottom, c.rect.bottom) - Math.max(r.top, c.rect.top);
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
        let t = textOf(el, candSet, 200);
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
            // Taiga: подпись поля живёт в обёртке tui-input/tui-textfield
            const wrap = el.closest(FIELD_WRAP);
            if (wrap) { const w = textOf(wrap, candSet, 120); if (meaningful(w)) t = w; }
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
        rec.visibleText = rec.textInput ? '' : textOf(el, candSet, 200);
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

    // --------------------------------------------- 7. разметка и результат
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
        let inputType = '', value = '', options = [];
        if (kind === 'INPUT') {
            inputType = tag === 'textarea' ? 'textarea' : tag === 'select' ? 'select'
                : tag === 'input' ? inputTypeOf(el) : (el.isContentEditable ? 'contenteditable' : (roleOf(el) || 'text'));
            if (tag === 'select') {
                options = Array.from(el.options || []).map((o) => norm(o.textContent)).filter(Boolean).slice(0, 40);
                const so = el.selectedOptions && el.selectedOptions[0];
                value = so ? norm(so.textContent) : '';
            } else if (tag === 'input' || tag === 'textarea') {
                value = inputType === 'password' ? (el.value ? '***' : '') : String(el.value || '');
            } else {
                value = norm(el.innerText);
            }
        } else if (kind === 'DROPDOWN') {
            const field = innerField(el);                    // выбранное значение tui-select
            if (field && typeof field.value === 'string') value = field.value;
        }
        let choiceType = '';
        if (kind === 'OPTION' || kind === 'FOLDER') {
            choiceType = choiceTypeOf(el) || (rec.control ? choiceTypeOf(rec.control) : '');
            if (!choiceType) {
                const c = ownQuery(el, 'input[type="radio"], input[type="checkbox"]');
                if (c) choiceType = inputTypeOf(c);
            }
        }
        elements.push({
            uid, tag, kind,
            text: rec.textInput ? '' : rec.label,
            placeholder: rec.textInput ? rec.label : '',
            value: String(value).slice(0, 300), inputType, choiceType, options,
            selected: kind !== 'INPUT' && kind !== 'DROPDOWN' ? selectedOf(el, rec) : false,
            disabled: disabledOf(el, rec),
            expanded: (kind === 'FOLDER' || kind === 'DROPDOWN') ? !!rec.expanded : null,
            selectable: !!rec.selectable, hasToggle, hasSelect,
            depth: rec.depth || 0, path: rec.path || [],
            container: containerOf(el), inViewport, occluded: !!occ,
        });
    });
    let overlay = '';
    let topOcc = null, topN = 0;
    for (const [o, n] of occluders) if (n > topN) { topOcc = o; topN = n; }
    if (topOcc && topN >= 2) overlay = describe(topOcc);

    // --------------------------------------------- 8. текст задания и подсказки
    const insideKept = (x) => nearestKept(x) !== null;
    const collectTexts = (selectors, limit) => {
        const out = [], taken = [];
        for (const sel of selectors) {
            let nodes = [];
            try { nodes = document.querySelectorAll(sel); } catch (e) { continue; }
            for (const n of nodes) {
                if (taken.some((t) => t.contains(n) || n.contains(t))) continue;
                if (insideKept(n)) continue;
                const r = n.getBoundingClientRect();
                if (r.width < 1 || r.height < 1) continue;
                if (n.checkVisibility && !n.checkVisibility(VIS)) continue;
                const t = textOf(n, keptSet, limit);
                if (!meaningful(t)) continue;
                out.push(t); taken.push(n);
            }
        }
        return out;
    };
    let taskTexts = collectTexts(args.taskSelectors, 1500);
    const taskFromSelectors = taskTexts.length > 0;
    if (!taskFromSelectors && document.body) {
        const t = textOf(document.body, keptSet, 1500);          // фоллбэк: весь неинтерактивный текст
        if (meaningful(t)) taskTexts = [t];
    }
    const hintTexts = collectTexts(args.hintSelectors, 800)
        .filter((h) => !taskTexts.some((t) => t.includes(h)));

    const alerts = [];
    try {
        const ALERT_SEL = '[role="alert"], [aria-live="assertive"], tui-error, tui-notification, tui-alert, '
            + '[class*="error" i], [class*="invalid" i], [class*="warning" i], [class*="toast" i], [class*="notification" i]';
        for (const n of document.querySelectorAll(ALERT_SEL)) {
            if (alerts.length >= 5) break;
            const r = n.getBoundingClientRect();
            if (r.width < 1 || r.height < 1) continue;
            if (n.checkVisibility && !n.checkVisibility(VIS)) continue;
            const t = textOf(n, keptSet, 300);
            if (t.length < 3 || t.length >= 300) continue;
            if (alerts.some((a) => a.includes(t) || t.includes(a))) continue;
            alerts.push(t);
        }
    } catch (e) {}

    // --------------------------------------------- 9. капча, загрузка, картинка
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

    let loading = false;
    try {
        const LOADER_RE = /(^|[_-])(spinner|loader|loading|preloader|skeleton)($|[_-])/i;
        const sel = '[aria-busy="true"], [class*="spin" i], [class*="load" i], [class*="skeleton" i], [role="progressbar"], tui-loader';
        for (const n of document.querySelectorAll(sel)) {
            const tag = tagOf(n);
            if (tag === 'tui-loader') { if (hasToken(n, /^_loading$/)) { loading = true; break; } continue; }
            if (attr(n, 'aria-busy') !== 'true' && roleOf(n) !== 'progressbar' && !hasToken(n, LOADER_RE)) continue;
            if ((keptCount.get(n) || 0) > 0) continue;            // обёртка с контентом — не спиннер
            const r = n.getBoundingClientRect();
            if (r.width < 4 || r.height < 4) continue;
            if (n.checkVisibility && !n.checkVisibility(VIS)) continue;
            loading = true; break;
        }
    } catch (e) {}

    let image = null, bestArea = 0;
    for (const n of ALL) {
        const tag = tagOf(n);
        if (tag !== 'img' && tag !== 'canvas' && tag !== 'video') continue;
        const r = n.getBoundingClientRect();
        if (r.width < 48 || r.height < 48) continue;              // иконки и логотипы-миниатюры
        if (insideKept(n)) continue;
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

    // --------------------------------------------- 10. прокручиваемые списки
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
        taskTexts, taskFromSelectors, hintTexts, alerts,
        captcha, loading, overlay, imageSrc, scrollables,
    });
}
"""


# ---------------------------------------------------------------------------
# Python-обёртка
# ---------------------------------------------------------------------------

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
                "taskSelectors": list(TASK_TEXT_SELECTORS),
                "hintSelectors": list(HINT_SELECTORS),
                "pointer": True,
            },
        )
        raw: dict = json.loads(payload)

        elements = self._build_elements(raw.get("elements") or [])
        task_text = "\n".join(raw.get("taskTexts") or [])
        hint_text = "\n".join(raw.get("hintTexts") or [])
        alerts = [str(a) for a in raw.get("alerts") or []]
        image_src = str(raw.get("imageSrc") or "")
        task_id, preview = self._fingerprint(
            raw.get("url", ""), image_src, task_text, bool(raw.get("taskFromSelectors")),
        )

        state = PageState(
            task_text=task_text,
            hint_text=hint_text,
            elements=elements,
            task_identifier=task_id,
            task_preview=preview,
            state_hash=self._state_hash(elements, alerts),
            has_captcha=bool(raw.get("captcha")),
            loading=bool(raw.get("loading")),
            alerts=alerts,
            overlay_text=str(raw.get("overlay") or ""),
            scroll_hints=self._scroll_hints(raw.get("scrollables") or []),
            image_src=image_src,
            generation=generation,
            frame_url=str(raw.get("url") or ""),
            total_elements=int(raw.get("total") or len(elements)),
            hidden_elements=int(raw.get("hidden") or 0),
        )
        folders = [e for e in elements if e.kind == ElementKind.FOLDER]
        logger.log(
            logging.DEBUG if quiet else logging.INFO,
            "Снимок #%s: элементов=%d (папок откр./закр.=%d/%d, опций=%d, скрытых совпадений=%d), "
            "task_id=%s, captcha=%s, loading=%s",
            generation, len(elements),
            sum(1 for f in folders if f.folder_state == FolderState.OPEN),
            sum(1 for f in folders if f.folder_state == FolderState.CLOSED),
            sum(1 for e in elements if e.kind == ElementKind.OPTION),
            state.hidden_elements, task_id, state.has_captcha, state.loading,
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
            base = f"{group}|{' › '.join(el.path)}|{normalize_text(el.text or el.placeholder)}"
            n = seen[base]
            seen[base] += 1
            el.key = base if n == 0 else f"{base}#{n}"

    @staticmethod
    def _fingerprint(url: str, image_src: str, task_text: str, from_selectors: bool) -> tuple[str, str]:
        """Отпечаток задания: не зависит от состояния дерева, таймеров и логотипов."""
        text = _TIMER_RE.sub("", task_text)
        if not from_selectors:
            # фоллбэк-текст всей страницы содержит счётчики «Задание 5 из 100» — цифры убираем
            text = re.sub(r"\d+", "", text)
        basis = f"{urlsplit(url).path}|{image_src}|{normalize_text(text)[:600]}"
        digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
        preview = (task_text.strip().splitlines() or [""])[0][:80]
        return digest, preview

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


# ---------------------------------------------------------------------------
# Представление для LLM
# ---------------------------------------------------------------------------

def select_for_prompt(
    elements: list[ParsedElement], limit: int = MAX_ELEMENTS,
) -> tuple[list[ParsedElement], int]:
    """Выбрать элементы для промпта. В v2 обрезка шла по порядку документа,
    и при большом раскрытом дереве из списка пропадала кнопка «Завершить».
    Теперь кнопки/поля/выбранные/папки сохраняются всегда."""
    visible = [e for e in elements if not (e.is_disabled and e.kind == ElementKind.OTHER)]
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
    """Строковое представление элементов для LLM-промпта."""
    shown, omitted = select_for_prompt(elements, limit)
    label_counts = Counter(normalize_text(e.text) for e in elements if e.text)
    lines = [
        e.prompt_line(show_path=label_counts[normalize_text(e.text)] > 1) for e in shown
    ]
    if omitted:
        lines.append(f"… ещё {omitted} элементов не показано (лимит {limit}).")
    return "\n".join(lines) if lines else "(интерактивных элементов нет)"
