"""Сквозной прогон Agent на фикстуре: реальный браузер + сценарная «LLM».

Проверяет весь конвейер v3: снимок → решение → автокоррекция → клик по метке →
проверка эффекта → submit с проверкой смены задания → сброс памяти.
"""

from __future__ import annotations

import agent as agent_module
from agent import Agent, StepResult
from browser_controller import BrowserController
from models import ActionType, DecisionContext, ElementKind, LLMDecision, PageState
from tests.helpers import install_routes, run


class ScriptedLLM:
    """Ведёт себя как аккуратная модель, но с типичными «ошибками» LLM:
    сдвинутый индекс и скопированная целиком строка вместо текста."""

    def __init__(self) -> None:
        self.calls: list[tuple[PageState, DecisionContext]] = []

    async def decide(self, state: PageState, context: DecisionContext) -> LLMDecision:
        self.calls.append((state, context))
        smartphone = "смартфон" in state.task_text.lower()
        want, folder = ("Смартфоны", "Электроника") if smartphone else ("Куртки", "Одежда")
        chosen = [e for e in state.elements if e.text == want and e.is_selected]
        if chosen:
            return LLMDecision(reasoning="ответ выбран", action=ActionType.SUBMIT)
        option = next((e for e in state.elements if e.text == want and e.kind == ElementKind.OPTION), None)
        if option is not None:
            return LLMDecision(reasoning="вариант виден", action=ActionType.CLICK,
                               target_index=option.index + 1, target_text=want)   # индекс «съехал»
        node = next(e for e in state.elements if e.text == folder)
        return LLMDecision(reasoning="раскрываю ветку", action=ActionType.OPEN,
                           target_index=node.index, target_text=f"[{node.index}] {node.prompt_line()}")


def test_agent_solves_two_tasks_and_never_touches_end_shift():
    async def scenario():
        llm = ScriptedLLM()
        agent = Agent(browser=BrowserController(on_context=install_routes), llm=llm)
        results = []
        async with agent._browser:
            for _ in range(20):
                results.append(await agent._step())
                if agent._tasks_done >= 2:
                    break
            frame = await agent._browser.find_target_frame()
            log = await frame.evaluate("() => window.__log")

        assert agent._tasks_done == 2
        assert log == ["toggle:Электроника", "submit:phones", "toggle:Одежда", "submit:jackets"]
        assert "END-SHIFT" not in log                     # «Завершить смену» отфильтрована
        assert results.count(StepResult.ACTED) == 6       # по 3 шага на задание: open, click, submit
        # на втором шаге модель видит фактический результат первого действия
        _, second_ctx = llm.calls[1]
        assert any("✓ раскрыто, появилось 3" in line for line in second_ctx.history)
        # после смены задания память сброшена: история второго задания начинается заново
        _, fourth_ctx = llm.calls[3]
        assert fourth_ctx.history == [] and fourth_ctx.step_in_task == 1
    run(scenario())


def test_start_button_is_clicked_locally_without_llm(monkeypatch):
    monkeypatch.setattr("browser_controller.TARGET_URL", "https://t-work.test/index.html?task=intro")

    async def scenario():
        llm = ScriptedLLM()
        agent = Agent(browser=BrowserController(on_context=install_routes), llm=llm)
        async with agent._browser:
            result = StepResult.IDLE
            for _ in range(10):
                result = await agent._step()
                if result == StepResult.ACTED:
                    break
            frame = await agent._browser.find_target_frame()
            overlay_left = await frame.evaluate("() => !!document.querySelector('.overlay')")
        assert result == StepResult.ACTED and not overlay_left
        assert llm.calls == []
    run(scenario())


def test_run_loop_stops_at_max_steps(monkeypatch):
    monkeypatch.setattr(agent_module, "MAX_STEPS", 2)

    async def scenario():
        llm = ScriptedLLM()
        agent = Agent(browser=BrowserController(on_context=install_routes), llm=llm)
        await agent.run()                                  # должен сам закрыть браузер
        assert len(llm.calls) == 2 and agent._browser.is_closed()
    run(scenario())
