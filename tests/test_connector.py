"""LLMConnector против фейкового OpenAI-совместимого сервера (без сети и ключей)."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
def fake_openai(responses: list[tuple[int, dict]]):
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
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
    assert connector.usage.since(before).render() == "вызовов модели 1, токенов: вход 4000 (из кэша 2048), выход 150"


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


def test_api_error_becomes_skip(monkeypatch):
    with fake_openai([bad_request("Something else went wrong")]) as (url, _):
        connector = make_connector(monkeypatch, url)
        decision = run(connector.decide(state(), DecisionContext()))
    assert decision.action == ActionType.SKIP
