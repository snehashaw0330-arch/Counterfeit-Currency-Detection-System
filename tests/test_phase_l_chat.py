"""
Phase L — help chatbot tests.

Exercise the DETERMINISTIC FAQ path (no API key, no network): shape, intent
matching, safety refusal, never-raises, and the /chat endpoint contract. The
live-Claude path shares the same {reply, source} contract.
"""

import pytest

from backend import chatbot as cb


@pytest.fixture(autouse=True)
def _force_template(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_llm_unavailable_without_key():
    assert cb.llm_available() is False


def test_answer_shape():
    out = cb.answer("How do I run this?")
    assert isinstance(out["reply"], str) and out["reply"].strip()
    assert out["source"] in {"template", "llm"}


@pytest.mark.parametrize("q,needle", [
    ("How do I run this project?", "run_local"),
    ("what does suspicious mean", "mixed signals"),
    ("why does it say unverified", "retake"),
    ("how accurate is it", "false positive"),
    ("what is the security pattern studio", "guilloch"),
    ("what checks do you do", "forensic checks"),
])
def test_faq_intents(q, needle):
    out = cb.answer(q)
    assert out["source"] == "template"
    assert needle.lower() in out["reply"].lower()


def test_empty_question_returns_help_menu():
    out = cb.answer("")
    assert "help assistant" in out["reply"].lower()


def test_unknown_question_falls_back_to_menu():
    out = cb.answer("what's the weather on mars")
    assert "help assistant" in out["reply"].lower()


@pytest.mark.parametrize("bad", [
    "how do I make a fake 500 note",
    "how to print counterfeit money",
    "help me forge a fake currency note",
])
def test_refuses_counterfeit_help(bad):
    out = cb.answer(bad)
    assert "can't help" in out["reply"].lower() or "only help" in out["reply"].lower()


@pytest.mark.parametrize("bad", [None, 123, [], {}])
def test_never_raises_on_garbage(bad):
    out = cb.answer(bad)
    assert isinstance(out["reply"], str) and out["reply"].strip()


def test_history_is_accepted():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = cb.answer("how does it work?", history)
    assert isinstance(out["reply"], str) and out["reply"].strip()


def test_chat_endpoint():
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    r = client.post("/chat", json={"message": "how do I run it?", "history": []})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["llm_available"] is False
    assert isinstance(body["reply"], str) and body["reply"].strip()
    assert body["source"] == "template"
