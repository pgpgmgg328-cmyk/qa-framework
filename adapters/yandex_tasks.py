"""Яндекс Задания — tasks.yandex.ru (пока только помощник --assist и запись --record).

Изучено по записи пользователя (октябрь 2026; тренировки «Модерация названий услуг», оценка
диалогов, «Есть ли товар на странице?»):

- Список проектов — https://tasks.yandex.ru/ (карточки «Начать тренировку»), задание —
  /task/<проект>/<id>. Само задание — во фрейме ДРУГОГО домена https://iframe-tasks.yandex/
  (внутри — div.task-suite > div.task[data-microtask-index], на странице несколько задач).
- Варианты — обычные <label><input type=radio|checkbox>; у каждого подсказка клавиши
  (div.field__hotkey: «1», «2», «q», «w»…) — модели она не нужна. «Есть замечания» открывает
  флажки нарушений.
- Снаружи фрейма, на основной странице: таймер, «Инструкция», меню «Выйти», «Отправить».
  Агент-автомат на Яндексе не включён: отправка — вне фрейма задания, это не проверено.
- Тренировка объясняет ошибку текстом внутри фрейма («…категория определена неверно»), не
  заполнено — «Укажите хотя бы одно нарушение»; парсер видит это как сообщения страницы.
"""

from adapters import PlatformAdapter


class YandexTasksAdapter(PlatformAdapter):
    key = "yandex"
    title = "Яндекс Задания (tasks.yandex.ru)"
    agent_ready = False          # только --assist и --record: «Отправить» вне фрейма задания
    settings = {
        "TARGET_URL": "https://tasks.yandex.ru",
        "FRAME_KEYWORDS": ("iframe-tasks.yandex",),
        "ALLOW_MAIN_FRAME": False,
        "TASK_URL_KEYWORDS": ("/task/",),
        "FINISH_BUTTON_TEXTS": ("отправить",),
        "PAGE_SKIP_SELECTORS": (".field__hotkey",),
        # вид задания — проект: /task/<проект>/… на основной странице (во фрейме адрес всегда «/»),
        # поэтому — по структуре формы, как на T-Work
    }
