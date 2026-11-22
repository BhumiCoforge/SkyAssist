import pytest
from langchain_core.messages import AIMessage

import agent
from tests.fakes import ScriptedLLM, call

SESSION = "session-cite-1"


@pytest.fixture
def llm(monkeypatch):
    fake = ScriptedLLM()
    monkeypatch.setattr(agent, "_agent_llm", fake)
    return fake


def test_chat_marks_which_policy_sources_the_answer_cited(db, llm, monkeypatch):
    fake_sources = [
        {"n": 1, "file": "baggage.md", "section": "Cabin baggage", "score": 0.5, "snippet": "7 kg"},
        {"n": 2, "file": "baggage.md", "section": "Restricted items", "score": 0.3, "snippet": "power banks"},
    ]
    monkeypatch.setattr(agent, "retrieve_policy",
                        lambda question, k=None: ("[1] cabin 7 kg\n\n[2] restricted items", fake_sources))
    llm.script = [call("policy_lookup", dict(question="cabin bag weight")),
                  AIMessage("Cabin bags can weigh up to 7 kg [1].")]
    r = agent.chat(SESSION, "how heavy can my cabin bag be?")
    assert r["tools"] == ["policy_lookup"]
    assert [s["cited"] for s in r["sources"]] == [True, False]
    assert r["sources"][0]["section"] == "Cabin baggage"


def test_chat_without_policy_lookup_returns_no_sources(db, llm):
    llm.script = [AIMessage("Hello! How can I help?")]
    r = agent.chat(SESSION, "hi")
    assert r["sources"] == []