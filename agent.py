"""Modules 3-5: LangChain agent. The LLM routes each request to tools (flight search,
policy RAG, booking lookup) and write actions wait for customer confirmation."""
import json
import uuid
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import services as svc
from db import SessionLocal
from rag import get_llm, retrieve_policy, text_of

PENDING: dict[str, dict] = {}   # actions waiting for customer confirmation
HISTORY: dict[str, list] = {}   # conversation memory per session

SYSTEM = """You are SkyAssist, a friendly airline support and booking assistant. Today is {today}.
- Use search_flights for any flight request. Convert dates like "20 December" to YYYY-MM-DD (use the next upcoming occurrence). Prices are in INR. Never invent flights.
- Use policy_lookup for baggage, refund, cancellation, check-in and delay questions, answer ONLY from its results and cite sources like [1]. If it does not contain the answer, say you don't know.
- A message can need several tools (e.g. flights AND baggage rules): call all that are needed.
- "the second flight" refers to the second flight in your most recent search results; use its id.
- To book, you need the flight id, every passenger's full name and a contact email. Ask for what is missing, then call request_booking. To cancel, call request_cancellation.
- request_booking and request_cancellation only queue the action; tell the customer to review the summary and press Confirm. Never say a booking is done until it is confirmed.
- Flight cards are shown to the customer automatically, so only summarize flights briefly. Keep replies short."""


@tool
def search_flights(origin: str = "", destination: str = "", date: str = "", max_price: float = 0,
                   passengers: int = 1) -> list:
    """Search upcoming flights. date is YYYY-MM-DD (empty = any date). Leave unused fields empty."""
    with SessionLocal() as db:
        return svc.search_flights(db, origin, destination, date, max_price, passengers)


@tool
def policy_lookup(question: str) -> str:
    """Look up airline policy documents (baggage, refunds, cancellation, check-in, delays)."""
    return retrieve_policy(question)[0]


@tool
def get_booking(booking_id: int) -> dict:
    """Check the status and details of a booking."""
    with SessionLocal() as db:
        return svc.booking_dict(svc.get_booking(db, booking_id))


@tool
def request_booking(flight_id: int, passenger_names: list[str], email: str, phone: str = "") -> str:
    """Queue a booking for customer confirmation. Needs flight id, all passenger names and an email."""
    with SessionLocal() as db:
        f = svc.check_bookable(db, flight_id, len(passenger_names))
        total = f.price * len(passenger_names)
        summary = (f"Book {f.flight_no} {f.origin} to {f.destination}, departing "
                   f"{f.departure:%d %b %Y %H:%M}\nPassengers: {', '.join(passenger_names)}\n"
                   f"Total: INR {total:,.0f}\nContact: {email}")
    pid = uuid.uuid4().hex[:8]
    PENDING[pid] = dict(action="book", summary=summary, flight_id=flight_id,
                        passenger_names=passenger_names, email=email, phone=phone)
    return f"Queued (id {pid}). Total INR {total:,.0f}. Waiting for the customer to confirm."


@tool
def request_cancellation(booking_id: int) -> str:
    """Queue a booking cancellation for customer confirmation."""
    with SessionLocal() as db:
        b = svc.get_booking(db, booking_id)
        if b.status == "CANCELLED":
            return "That booking is already cancelled."
        q = svc.refund_quote(b)
        summary = (f"Cancel booking #{b.id} ({b.flight.flight_no}, {b.flight.departure:%d %b %Y %H:%M})\n"
                   f"Refund: INR {q['refund']:,.0f} ({q['rule']})")
    pid = uuid.uuid4().hex[:8]
    PENDING[pid] = dict(action="cancel", summary=summary, booking_id=booking_id)
    return f"Queued (id {pid}). {summary}. Waiting for the customer to confirm."


TOOLS = {t.name: t for t in (search_flights, policy_lookup, get_booking, request_booking, request_cancellation)}
_agent_llm = None


def agent_llm():
    global _agent_llm
    if _agent_llm is None:
        _agent_llm = get_llm().bind_tools(list(TOOLS.values()))
    return _agent_llm


def _trim(hist: list):
    while len(hist) > 40:
        hist.pop(0)
    while hist and not isinstance(hist[0], HumanMessage):
        hist.pop(0)


def chat(session_id: str, message: str) -> dict:
    hist = HISTORY.setdefault(session_id, [])
    hist.append(HumanMessage(message))
    _trim(hist)
    system = SystemMessage(SYSTEM.format(today=date.today().strftime("%A, %d %B %Y")))
    before, flights, sources, tools_used = set(PENDING), [], [], []
    ai = None
    for _ in range(6):
        ai = agent_llm().invoke([system] + hist)
        hist.append(ai)
        if not ai.tool_calls:
            break
        for tc in ai.tool_calls:
            tools_used.append(tc["name"])
            try:
                if tc["name"] == "policy_lookup":
                    out, sources = retrieve_policy(tc["args"].get("question", message))
                else:
                    out = TOOLS[tc["name"]].invoke(tc["args"])
                    if tc["name"] == "search_flights":
                        flights = out
            except svc.BookingError as e:
                out = f"Error: {e.message}"
            except Exception as e:
                out = f"Tool error: {e}"
            content = json.dumps(out, default=str) if isinstance(out, (list, dict)) else str(out)
            hist.append(ToolMessage(content, tool_call_id=tc["id"]))
    answer = text_of(ai.content).strip() or "Done."
    pending = []
    for k, v in PENDING.items():
        if k not in before:
            v["session_id"] = session_id
            pending.append(dict(id=k, action=v["action"], summary=v["summary"]))
    return dict(answer=answer, flights=flights, sources=sources, pending=pending, tools=tools_used)


def confirm(pid: str) -> dict:
    a = PENDING.pop(pid, None)
    if not a:
        return dict(ok=False, message="This action has expired or was already handled.")
    try:
        with SessionLocal() as db:
            if a["action"] == "book":
                names = a["passenger_names"]
                r = svc.create_booking(db, a["flight_id"], names[0], a["email"], a["phone"],
                                       [dict(name=n) for n in names])
                msg = (f"Booking confirmed! Booking ID #{r['id']} on {r['flight']['flight_no']}, "
                       f"total INR {r['total_price']:,.0f}.")
            else:
                r = svc.cancel_booking(db, a["booking_id"])
                msg = f"Booking #{a['booking_id']} cancelled. Refund: INR {r['refund']:,.0f}."
        ok = True
    except svc.BookingError as e:
        ok, msg = False, e.message
    h = HISTORY.get(a.get("session_id"))   # let the agent know the outcome
    if h:
        h.append(AIMessage(f"[Customer confirmed] {msg}"))
    return dict(ok=ok, message=msg)


def decline(pid: str) -> dict:
    PENDING.pop(pid, None)
    return dict(ok=True, message="Okay, I've cancelled that request.")
