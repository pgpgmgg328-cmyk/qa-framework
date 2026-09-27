"""LLMConnector против фейкового OpenAI-совместимого сервера (без сети и ключей)."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import openrouter_connector as oc
from models import ActionType, DecisionContext, ElementKind, PageState, ParsedElement, VisionImage
from tests.helpers import run

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ["NO_PROXY"] += ",127.0.0.1,localhost"

VALID = {
    "goal": "категория", "observation": "[0] «Электроника» закрыта", "reasoning": "раскрываю",
    "action": "open", "target_index": 0, "target_text": "Электроника",
    "type_text": None, "scroll_direction": None, "confidence": 0.9,
}


def completion(content: str) -> tuple[int, dict]:
    return 200, {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "gpt-4o",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def bad_request(message: str) -> tuple[int, dict]:
    return 400, {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}}


@contextmanager
def fake_openai(responses: list[tuple[int, dict]], models: Optional[list[dict]] = None):
    """POST — ответы по очереди из responses; GET /models — список моделей (у OpenRouter — с ценами)."""
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if models is None or not self.path.endswith("/models"):
                self.send_error(404)
                return
            data = json.dumps({"object": "list", "data": models}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            if "json" in (self.headers.get("Content-Type") or ""):
                body = json.loads(raw)
            else:                                   # multipart: audio/transcriptions
                body = {"_raw": raw}
            body["_path"] = self.path
            requests.append(body)
            status, payload = responses.pop(0)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # тишина в выводе pytest
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()


def state() -> PageState:
    return PageState(task_text="Выберите категорию", elements=[
        ParsedElement(index=0, uid="1-0", key="row||электроника", text="Электроника", kind=ElementKind.FOLDER),
    ])


def use_model(monkeypatch, *models: str) -> None:
    """LLM_MODEL=модель или лестница «дешёвая,сильная» (как в .env)."""
    monkeypatch.setattr(oc, "LLM_MODEL", models[0])
    monkeypatch.setattr(oc, "LLM_MODELS", tuple(models))


def make_connector(monkeypatch, base_url: str) -> oc.LLMConnector:
    monkeypatch.setattr(oc, "OPENAI_BASE_URL", base_url)
    monkeypatch.setattr(oc, "LLM_MAX_RETRIES", 0)
    return oc.LLMConnector()


def test_structured_outputs_fallback_to_json_object(monkeypatch):
    responses = [
        bad_request("Invalid parameter: 'response_format' of type 'json_schema' is not supported with this model."),
        completion(json.dumps(VALID)),
    ]
    with fake_openai(responses) as (url, requests):
        connector = make_connector(monkeypatch, url)
        decision = run(connector.decide(state(), DecisionContext()))
    assert decision.action == ActionType.OPEN and decision.target_text == "Электроника"
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[0]["response_format"]["json_schema"]["strict"] is True
    assert requests[1]["response_format"] == {"type": "json_object"}


def test_reasoning_model_parameters_are_adapted(monkeypatch):
    """Незнакомая модель: параметры подстраиваются по ошибкам 400."""
    use_model(monkeypatch, "some-new-model")
    responses = [
        bad_request("Unsupported parameter: 'max_tokens' is not supported with this model. "
                    "Use 'max_completion_tokens' instead."),
        bad_request("Unsupported value: 'temperature' does not support 0.0 with this model."),
        completion(json.dumps(VALID)),
    ]
    with fake_openai(responses) as (url, requests):
        connector = make_connector(monkeypatch, url)
        decision = run(connector.decide(state(), DecisionContext()))
    assert decision.action == ActionType.OPEN
    assert "max_tokens" in requests[0] and "max_completion_tokens" in requests[1]
    assert "temperature" in requests[1] and "temperature" not in requests[2]
    assert all("reasoning_effort" not in r for r in requests)


def test_gpt5_models_get_their_parameters_at_once(monkeypatch):
    """gpt-5.x: сразу max_completion_tokens и минимальное «обдумывание»; значение, которое модель
    не принимает, заменяется следующим (none → minimal)."""
    use_model(monkeypatch, "gpt-5-mini")
    responses = [
        bad_request("Unsupported value: 'reasoning_effort' does not support 'none' with this model. "
                    "Supported values are: 'minimal', 'low', 'medium', and 'high'."),
        completion(json.dumps(VALID)),
        completion(json.dumps(VALID)),
    ]
    with fake_openai(responses) as (url, requests):
        connector = make_connector(monkeypatch, url)
        run(connector.decide(state(), DecisionContext()))
        run(connector.decide(state(), DecisionContext()))
    assert "max_completion_tokens" in requests[0] and "max_tokens" not in requests[0]
    assert [r["reasoning_effort"] for r in requests] == ["none", "minimal", "minimal"]


def test_invalid_json_gets_one_repair_round(monkeypatch):
    responses = [completion('{"action": "dance"}'), completion(json.dumps(VALID))]
    with fake_openai(responses) as (url, requests):
        connector = make_connector(monkeypatch, url)
        decision = run(connector.decide(state(), DecisionContext()))
    assert decision.action == ActionType.OPEN
    repair = requests[1]["messages"]
    assert repair[-2]["role"] == "assistant" and "не прошёл проверку" in repair[-1]["content"]


def test_images_are_sent_with_captions_before_the_page(monkeypatch):
    """Неизменное внутри задания (знания, фото) — в начале сообщения, страница и история — в
    конце: так повторные вызовы по заданию попадают в кэш OpenAI."""
    photos = [VisionImage(caption="ФОТО 1–4", b64="BBBB"), VisionImage(caption="ФОТО 5", b64="CCCC")]
    ctx = DecisionContext(images=photos, image_b64="AAAA", knowledge="Правило: сравнивай адреса.",
                          history=["1. click «Да» → ✓ выбран"])
    with fake_openai([completion(json.dumps(VALID))]) as (url, requests):
        connector = make_connector(monkeypatch, url)
        run(connector.decide(state(), ctx))
    content = requests[0]["messages"][1]["content"]
    kinds = [part["type"] for part in content]
    assert kinds == ["text", "text", "image_url", "text", "image_url", "text", "text", "image_url"]
    head, page = content[0]["text"], content[5]["text"]
    assert head.startswith("═══ ЗНАНИЯ О ВИДЕ ЗАДАНИЙ ═══") and "ФОТО 1–4; ФОТО 5" in head
    assert "СТРАНИЦА ЗАДАНИЯ" not in head and "ИСТОРИЯ" not in head
    assert page.startswith("═══ СТРАНИЦА ЗАДАНИЯ ═══") and "═══ ИСТОРИЯ" in page
    assert content[1]["text"] == "ФОТО 1–4:" and content[2]["image_url"]["url"].endswith("BBBB")
    assert content[4]["image_url"]["url"].endswith("CCCC")
    assert content[7]["image_url"]["url"].startswith("data:image/jpeg;base64,AAAA")   # скриншот фрейма


def test_token_usage_is_counted(monkeypatch):
    use_model(monkeypatch, "gpt-4o")
    body = completion(json.dumps(VALID))
    body[1]["usage"] = {"prompt_tokens": 4000, "completion_tokens": 150, "total_tokens": 4150,
                        "prompt_tokens_details": {"cached_tokens": 2048}}
    with fake_openai([body, body]) as (url, _):
        connector = make_connector(monkeypatch, url)
        run(connector.decide(state(), DecisionContext()))
        before = connector.usage.snapshot()
        run(connector.decide(state(), DecisionContext()))
    assert (connector.usage.calls, connector.usage.prompt, connector.usage.cached, connector.usage.completion) \
        == (2, 8000, 4096, 300)
    assert connector.usage.since(before).render() == \
        "вызовов модели 1, токенов: вход 4000 (из кэша 2048), выход 150, ≈ $0.0089"


def test_cost_in_your_currency(monkeypatch):
    """Цены из личного кабинета прокси (LLM_PRICE) — стоимость в логе в рублях."""
    monkeypatch.setattr(oc, "LLM_PRICE", "225, 22.5, 1350")
    monkeypatch.setattr(oc, "LLM_PRICE_CURRENCY", "₽")
    body = completion(json.dumps(VALID))
    body[1]["usage"] = {"prompt_tokens": 4000, "completion_tokens": 150, "total_tokens": 4150,
                        "prompt_tokens_details": {"cached_tokens": 2048}}
    with fake_openai([body]) as (url, _):
        connector = make_connector(monkeypatch, url)
        run(connector.decide(state(), DecisionContext()))
    assert connector.usage.render().endswith(", ≈ 0.69 ₽")
    assert oc.parse_price("0,75; 0,075; 4,5") == (0.75, 0.075, 4.5) and oc.parse_price("1,2") == (1.0, 1.0, 2.0)
    assert oc.model_price("openai/gpt-4o-2024-08-06") is None               # свои цены — только для LLM_MODEL
    monkeypatch.setattr(oc, "LLM_PRICE", "")
    assert oc.model_price("openai/gpt-4o-2024-08-06") == (2.50, 1.25, 10.00)
    assert oc.model_price("gpt-5.4-mini") == (0.75, 0.075, 4.50) and oc.model_price("gpt-9") is None


def usage_body(prompt: int, completion_tokens: int = 20) -> tuple[int, dict]:
    status, body = completion(json.dumps(VALID))
    body["usage"] = {"prompt_tokens": prompt, "completion_tokens": completion_tokens,
                     "total_tokens": prompt + completion_tokens}
    return status, body


def test_model_check_reports_photo_cost(monkeypatch, caplog):
    use_model(monkeypatch, "gpt-4o")
    monkeypatch.setattr(oc, "LLM_VISION", "auto")
    with fake_openai([usage_body(120), usage_body(375)]) as (url, requests):
        connector = make_connector(monkeypatch, url)
        with caplog.at_level("INFO", logger="twork.llm"):
            run(connector.check_model())
    assert requests[1]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "Модель gpt-4o: доступна" in caplog.text and "фото 512×512 — 255 токенов на вход" in caplog.text
    assert "очень дорого" not in caplog.text


def test_model_check_warns_about_expensive_photos(monkeypatch, caplog):
    """Модель, которая берёт за картинку десятки тысяч токенов, видна до начала работы."""
    use_model(monkeypatch, "gpt-5.4-mini")
    monkeypatch.setattr(oc, "LLM_VISION", "auto")
    with fake_openai([usage_body(120), usage_body(16_500)]) as (url, _):
        connector = make_connector(monkeypatch, url)
        with caplog.at_level("INFO", logger="twork.llm"):
            run(connector.check_model())
    assert "очень дорого: 16380 токенов" in caplog.text


def test_model_check_stops_on_unknown_model(monkeypatch):
    use_model(monkeypatch, "gpt-9-turbo")
    not_found = (404, {"error": {"message": "The model `gpt-9-turbo` does not exist or you do not have access "
                                            "to it.", "type": "invalid_request_error", "code": "model_not_found"}})
    with fake_openai([not_found]) as (url, _):
        connector = make_connector(monkeypatch, url)
        try:
            run(connector.check_model())
        except oc.ModelUnavailable as exc:
            assert "gpt-9-turbo" in str(exc) and "LLM_MODEL" in str(exc)
        else:
            raise AssertionError("недоступная модель не остановила запуск")


def test_retired_ladder_model_is_dropped_at_start(monkeypatch, caplog):
    """Модель лестницы сняли с прокси — агент не останавливается, а работает остальными."""
    use_model(monkeypatch, "google/gemini-2.5-flash-lite", "google/gemini-3.1-flash-lite")
    monkeypatch.setattr(oc, "LLM_VISION", "off")
    retired = (404, {"error": {"message": "No endpoints found for google/gemini-2.5-flash-lite.", "code": 404}})
    with fake_openai([retired, usage_body(120), completion(json.dumps(VALID))]) as (url, requests):
        connector = make_connector(monkeypatch, url)
        with caplog.at_level("INFO", logger="twork.llm"):
            run(connector.check_model())
        decision = run(connector.decide(state(), DecisionContext()))
    assert connector.models == ["google/gemini-3.1-flash-lite"] and "Убираю её из лестницы" in caplog.text
    assert decision.action == ActionType.OPEN and requests[-1]["model"] == "google/gemini-3.1-flash-lite"


def test_unsure_submit_is_rechecked_by_stronger_model(monkeypatch):
    """Дешёвая модель не уверена в ответе, который отправляет, — решение за сильной моделью."""
    use_model(monkeypatch, "gpt-5.4-nano")
    monkeypatch.setattr(oc, "LLM_CHECK_MODEL", "gpt-5.4")
    unsure = {"observation": "", "plan": "", "reasoning": "не уверен", "confidence": 0.4, "actions": [
        {"action": "click", "target_index": 0, "target_text": "Электроника", "value": None},
        {"action": "submit", "target_index": None, "target_text": None, "value": None}]}
    sure = {**unsure, "reasoning": "проверил", "confidence": 0.9, "actions": [
        {"action": "open", "target_index": 0, "target_text": "Электроника", "value": None}]}
    confident_submit = {**unsure, "confidence": 0.95}
    responses = [completion(json.dumps(unsure)), completion(json.dumps(sure)), completion(json.dumps(confident_submit))]
    with fake_openai(responses) as (url, requests):
        connector = make_connector(monkeypatch, url)
        first = run(connector.decide(state(), DecisionContext()))
        second = run(connector.decide(state(), DecisionContext()))
    assert [r["model"] for r in requests] == ["gpt-5.4-nano", "gpt-5.4", "gpt-5.4-nano"]
    assert first.action == ActionType.OPEN and first.reasoning == "проверил"
    assert second.confidence == 0.95 and len(second.steps()) == 2      # уверенный ответ — без перепроверки


def test_missing_pinned_model_falls_back_to_main_model(monkeypatch):
    """Модель из «Заметок» с опечаткой: задание решает основная модель, запрос к недоступной
    больше не повторяется."""
    use_model(monkeypatch, "gpt-5.4-nano", "gpt-5.4-mini")
    missing = (404, {"error": {"message": "The model `gpt-5.4-mimi` does not exist", "type": "invalid_request_error",
                               "code": "model_not_found"}})
    responses = [missing, completion(json.dumps(VALID)), completion(json.dumps(VALID))]
    with fake_openai(responses) as (url, requests):
        connector = make_connector(monkeypatch, url)
        first = run(connector.decide(state(), DecisionContext(model="gpt-5.4-mimi")))
        second = run(connector.decide(state(), DecisionContext(model="gpt-5.4-mimi")))
    assert first.action == second.action == ActionType.OPEN
    assert [r["model"] for r in requests] == ["gpt-5.4-mimi", "gpt-5.4-nano", "gpt-5.4-nano"]


def test_empty_balance_is_reported_plainly_and_agent_waits(monkeypatch, caplog):
    unpaid = (402, {"error": {"message": "Insufficient credits. Add more using https://openrouter.ai/settings/credits",
                              "code": 402}})
    with fake_openai([unpaid, unpaid]) as (url, _):
        connector = make_connector(monkeypatch, url)
        with caplog.at_level("INFO", logger="twork.llm"):
            first = run(connector.decide(state(), DecisionContext()))
            second = run(connector.decide(state(), DecisionContext()))
    assert first.action == second.action == ActionType.SKIP
    assert caplog.text.count("💳 Прокси отказал в оплате") == 1          # не чаще раза в 10 минут


def test_transcription_falls_back_to_next_model(monkeypatch):
    monkeypatch.setattr(oc, "TRANSCRIBE_MODELS", ("gpt-4o-transcribe", "whisper-1"))
    verbose = {"text": "Алло. Слушаю.", "language": "russian", "duration": 4.2, "segments": [
        {"id": 0, "start": 0.0, "end": 1.4, "text": " Алло."},
        {"id": 1, "start": 1.6, "end": 4.2, "text": " Слушаю."},
    ]}
    responses = [bad_request("Model gpt-4o-transcribe is not available"), (200, verbose)]
    with fake_openai(responses) as (url, requests):
        connector = make_connector(monkeypatch, url)
        text = run(connector.transcribe(b"ID3fake", "audio.mp3", "audio/mpeg"))
    assert text == "[0:00–0:01] Алло.\n[0:01–0:04] Слушаю."
    assert [r["_path"] for r in requests] == ["/v1/audio/transcriptions"] * 2
    assert b'name="model"' in requests[0]["_raw"] and b"gpt-4o-transcribe" in requests[0]["_raw"]
    assert b"verbose_json" in requests[1]["_raw"] and b'name="language"' in requests[1]["_raw"]
    assert "gpt-4o-transcribe" in connector._dead_transcribers   # больше не запрашивается


def test_transcription_without_timestamps_falls_back_to_plain_text(monkeypatch):
    """Прокси (OpenRouter) не отдаёт таймкоды — расшифровка простым текстом той же моделью."""
    monkeypatch.setattr(oc, "TRANSCRIBE_MODELS", ("openai/whisper-1",))
    responses = [bad_request("response_format 'verbose_json' is not supported"), (200, {"text": "Алло. Слушаю."})]
    with fake_openai(responses) as (url, requests):
        connector = make_connector(monkeypatch, url)
        text = run(connector.transcribe(b"ID3fake", "audio.mp3", "audio/mpeg"))
    assert text == "Алло. Слушаю."
    assert b"verbose_json" in requests[0]["_raw"] and b"openai/whisper-1" in requests[0]["_raw"]
    assert b"verbose_json" not in requests[1]["_raw"] and b"timestamp_granularities" not in requests[1]["_raw"]
    assert not connector._dead_transcribers


def test_unknown_bad_request_retries_with_plain_json_then_skips(monkeypatch):
    """Непонятная ошибка 400 у не-OpenAI модели (так выглядит отказ от строгой схемы) — ещё одна
    попытка с обычным JSON; снова ошибка — шаг пропускается. У моделей OpenAI — сразу пропуск."""
    use_model(monkeypatch, "qwen/qwen3-vl-32b-instruct")
    responses = [bad_request("Something else went wrong"), bad_request("Something else went wrong")]
    with fake_openai(responses) as (url, requests):
        connector = make_connector(monkeypatch, url)
        decision = run(connector.decide(state(), DecisionContext()))
    assert decision.action == ActionType.SKIP
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[1]["response_format"] == {"type": "json_object"} and len(requests) == 2

    use_model(monkeypatch, "openai/gpt-5.4-mini")
    with fake_openai([bad_request("Something else went wrong")]) as (url, requests):
        connector = make_connector(monkeypatch, url)
        assert run(connector.decide(state(), DecisionContext())).action == ActionType.SKIP
    assert len(requests) == 1


def test_gemini_schema_error_falls_back_to_plain_json(monkeypatch):
    use_model(monkeypatch, "google/gemini-2.5-flash-lite")
    monkeypatch.setattr(oc, "OPENROUTER", True)
    responses = [bad_request("Provider returned error: Invalid JSON payload received. Unknown name "
                             "\"additionalProperties\" at 'generation_config.response_schema'"),
                 completion(json.dumps(VALID)), completion(json.dumps(VALID))]
    with fake_openai(responses) as (url, requests):
        connector = make_connector(monkeypatch, url)
        first = run(connector.decide(state(), DecisionContext()))
        second = run(connector.decide(state(), DecisionContext()))
    assert first.action == second.action == ActionType.OPEN
    assert [r["response_format"]["type"] for r in requests] == ["json_schema", "json_object", "json_object"]


def test_openrouter_prices_reasoning_and_model_per_task_type(monkeypatch, caplog):
    """OpenRouter: цены — с его списка моделей, «обдумывание» выключено его параметром reasoning,
    стоимость — точная из ответа, решение — моделью, которую агент выбрал для вида заданий."""
    monkeypatch.setattr(oc, "OPENROUTER", True)
    monkeypatch.setattr(oc, "LIVE_PRICES", {})
    monkeypatch.setattr(oc, "LLM_VISION", "auto")
    use_model(monkeypatch, "google/gemini-2.5-flash-lite", "openai/gpt-5.4-mini")
    models = [
        {"id": "google/gemini-2.5-flash-lite", "object": "model", "created": 0, "owned_by": "google",
         "pricing": {"prompt": "0.0000001", "completion": "0.0000004", "input_cache_read": "0.000000025"}},
        {"id": "openai/gpt-5.4-mini", "object": "model", "created": 0, "owned_by": "openai",
         "pricing": {"prompt": "0.00000075", "completion": "0.0000045", "input_cache_read": "0.000000075"}},
    ]
    reported = usage_body(4000, 150)
    reported[1]["usage"]["cost"] = 0.0021
    responses = [usage_body(120), usage_body(378), usage_body(120), usage_body(427),
                 usage_body(4000, 150), reported]
    with fake_openai(responses, models=models) as (url, requests):
        connector = make_connector(monkeypatch, url)
        with caplog.at_level("INFO", logger="twork.llm"):
            run(connector.check_model())
        before = connector.usage.snapshot()
        decision = run(connector.decide(state(), DecisionContext(model="openai/gpt-5.4-mini")))
        estimated = connector.usage.since(before)
        run(connector.decide(state(), DecisionContext(model="openai/gpt-5.4-mini")))
    assert decision.action == ActionType.OPEN
    assert oc.LIVE_PRICES["google/gemini-2.5-flash-lite"] == (0.1, 0.025, 0.4)
    assert "Модель google/gemini-2.5-flash-lite: доступна" in caplog.text
    assert "фото 512×512 — 258 токенов на вход (≈ $0.000026)" in caplog.text
    assert "цена за 1 млн токенов: вход $0.1, выход $0.4" in caplog.text
    assert "Модель openai/gpt-5.4-mini: доступна" in caplog.text and "вход $0.75, выход $4.5" in caplog.text
    assert [r["model"] for r in requests] == ["google/gemini-2.5-flash-lite"] * 2 + ["openai/gpt-5.4-mini"] * 4
    for body in requests:
        assert body["reasoning"] == {"effort": "none"} and "reasoning_effort" not in body
        assert body["usage"] == {"include": True}
    assert "max_tokens" in requests[0] and "max_completion_tokens" in requests[-1]
    assert estimated.render() == "вызовов модели 1, токенов: вход 4000, выход 150, ≈ $0.0037"
    assert abs(connector.usage.since(before).cost - estimated.cost - 0.0021) < 1e-9   # из ответа OpenRouter
