import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent
import services as svc
from db import Booking, PendingAction, SessionLocal
from tests.conftest import make_flight
from tests.fakes import ScriptedLLM, call

S1, S2 = "session-one-1", "session-two-2"


@pytest.fixture
def llm(monkeypatch):
    fake = ScriptedLLM()
    monkeypatch.setattr(agent, "_agent_llm", fake)
    return fake


def test_search_flow_returns_flight_cards_and_saves_memory(db, llm):
    make_flight(db, flight_no="SA101")
    llm.script = [call("search_flights", dict(origin="Delhi", destination="Mumbai")),
                  AIMessage("I found SA101 for you.")]
    r = agent.chat(S1, "flights delhi to mumbai")
    assert r["tools"] == ["search_flights"] and r["flights"][0]["flight_no"] == "SA101"
    assert r["answer"] == "I found SA101 for you."          # SA101 was in a tool result: no warning
    hist = agent.load_history(S1)
    assert [type(m).__name__ for m in hist] == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert hist[1].tool_calls[0]["name"] == "search_flights"   # tool calls survive the DB round trip


def test_memory_is_per_session_and_reused(db, llm):
    llm.script = [AIMessage("Hello!"), AIMessage("Sure.")]
    agent.chat(S1, "my name is Asha")
    agent.chat(S1, "what is my name?")
    second_call = llm.seen[1]
    assert any(isinstance(m, HumanMessage) and "Asha" in m.content for m in second_call)
    assert agent.load_history(S2) == []


def test_failed_turn_saves_nothing(db, llm):
    llm.script = []                      # model "crashes" (pop from empty list)
    with pytest.raises(IndexError):
        agent.chat(S1, "hello")
    assert agent.load_history(S1) == []


def test_booking_needs_confirmation_then_creates_booking(db, llm):
    f = make_flight(db, flight_no="SA101")
    llm.script = [call("request_booking", dict(flight_id=f.id, passenger_names=["Asha Rao", "Ravi Rao"],
                                               email="asha@example.com")),
                  AIMessage("Please review and press Confirm.")]
    r = agent.chat(S1, "book it for two")
    assert len(r["pending"]) == 1 and "SA101" in r["pending"][0]["summary"]
    assert db.query(Booking).count() == 0                     # nothing booked yet
    pid = r["pending"][0]["id"]

    assert agent.confirm(pid, S2)["ok"] is False              # another session cannot confirm it
    assert db.query(Booking).count() == 0
    out = agent.confirm(pid, S1)
    assert out["ok"] and "Booking confirmed" in out["message"]
    assert db.query(Booking).count() == 1
    assert agent.confirm(pid, S1)["ok"] is False              # cannot be confirmed twice
    assert db.query(Booking).count() == 1


def test_pending_survives_restart_and_expires(db, llm, monkeypatch):
    f = make_flight(db)
    llm.script = [call("request_booking", dict(flight_id=f.id, passenger_names=["Asha Rao"],
                                               email="asha@example.com")), AIMessage("Confirm?")]
    pid = agent.chat(S1, "book")["pending"][0]["id"]
    with SessionLocal() as s2:                                # a "new process" reads it from the DB
        assert s2.get(PendingAction, pid) is not None
    monkeypatch.setattr(agent.settings, "confirm_ttl_seconds", -1)   # pretend 10+ minutes passed
    out = agent.confirm(pid, S1)
    assert out["ok"] is False and "expired" in out["message"]
    assert db.query(Booking).count() == 0


def test_decline_discards_action(db, llm):
    f = make_flight(db)
    llm.script = [call("request_booking", dict(flight_id=f.id, passenger_names=["Asha Rao"],
                                               email="asha@example.com")), AIMessage("Confirm?")]
    pid = agent.chat(S1, "book")["pending"][0]["id"]
    assert agent.decline(pid, S1)["ok"]
    assert agent.confirm(pid, S1)["ok"] is False
    assert db.query(Booking).count() == 0


def test_cancellation_flow_and_ownership(db, llm):
    f = make_flight(db)
    b = svc.create_booking(db, f.id, "Asha Rao", "asha@example.com", "", [dict(name="Asha Rao")])
    # wrong email: the tool reports an error and nothing is queued
    llm.script = [call("request_cancellation", dict(booking_id=b["id"], email="thief@example.com")),
                  AIMessage("I couldn't find that booking.")]
    r = agent.chat(S1, "cancel booking")
    assert r["pending"] == []
    tool_msg = [m for m in agent.load_history(S1) if isinstance(m, ToolMessage)][0]
    assert "No booking found" in tool_msg.content
    # right email
    llm.script = [call("request_cancellation", dict(booking_id=b["id"], email="asha@example.com")),
                  AIMessage("Please confirm.")]
    r = agent.chat(S1, "cancel booking, my email is asha@example.com")
    assert len(r["pending"]) == 1 and "Refund" in r["pending"][0]["summary"]
    assert agent.confirm(r["pending"][0]["id"], S1)["ok"]
    assert db.get(Booking, b["id"]).status == "CANCELLED"


def test_get_booking_tool_requires_matching_email(db):
    f = make_flight(db)
    b = svc.create_booking(db, f.id, "Asha Rao", "asha@example.com", "", [dict(name="Asha Rao")])
    ok = agent.TOOLS["get_booking"].invoke(dict(booking_id=b["id"], email="asha@example.com"))
    assert ok["id"] == b["id"]
    with pytest.raises(svc.BookingError):
        agent.TOOLS["get_booking"].invoke(dict(booking_id=b["id"], email="other@example.com"))


def test_invalid_tool_arguments_are_reported_to_the_model(db, llm):
    llm.script = [call("search_flights", dict(date="tomorrow")), AIMessage("Which date exactly?")]
    r = agent.chat(S1, "flights tomorrow")
    tool_msg = [m for m in agent.load_history(S1) if isinstance(m, ToolMessage)][0]
    assert tool_msg.content.startswith("Tool error")
    assert r["flights"] == []


def test_invented_flight_number_is_flagged(db, llm):
    make_flight(db, flight_no="SA101")
    llm.script = [call("search_flights", {}), AIMessage("Take SA101 or the cheap ZZ999.")]
    r = agent.chat(S1, "flights")
    assert "couldn't verify ZZ999" in r["answer"] and "SA101 in our" not in r["answer"]


def test_gives_up_politely_after_max_steps(db, llm, monkeypatch):
    monkeypatch.setattr(agent.settings, "agent_max_steps", 2)
    llm.script = [call("search_flights", {}, "a"), call("search_flights", {}, "b")]
    r = agent.chat(S1, "loop forever")
    assert "couldn't finish" in r["answer"]
    # memory stays valid: every tool call has its tool result
    hist = agent.load_history(S1)
    assert isinstance(hist[-1], ToolMessage)


def test_history_trimming_never_starts_on_a_tool_result(db, monkeypatch):
    monkeypatch.setattr(agent.settings, "history_max_tokens", 1000)
    msgs = []
    for i in range(30):
        msgs += [HumanMessage(f"question {i} " + "x" * 400),
                 call("search_flights", {}, f"id{i}"), ToolMessage("[]", tool_call_id=f"id{i}"),
                 AIMessage("answer")]
    trimmed = agent.for_model(msgs)
    assert 0 < len(trimmed) < len(msgs)
    assert isinstance(trimmed[0], HumanMessage)