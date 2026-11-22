"""Chat, RAG and agent endpoints. The AI modules are imported lazily so the REST API works
even when no API key or embedding model is available."""
import logging

from fastapi import APIRouter, Depends, HTTPException

from deps import require_admin
from schemas import AskIn, ChatIn, SessionIn

log = logging.getLogger("skyassist.chat")
router = APIRouter(tags=["AI assistant"])


@router.post("/chat")
def chat_endpoint(body: ChatIn):
    from agent import chat

    try:
        return chat(body.session_id, body.message)
    except Exception:
        log.exception("chat failed")   # full error goes to the log, not to the customer
        raise HTTPException(500, "The assistant is unavailable right now. Please try again.")


@router.post("/chat/confirm/{pid}")
def chat_confirm(pid: str, body: SessionIn):
    from agent import confirm

    return confirm(pid, body.session_id)


@router.post("/chat/decline/{pid}")
def chat_decline(pid: str, body: SessionIn):
    from agent import decline

    return decline(pid, body.session_id)


@router.post("/rag/ask")
def rag_ask(body: AskIn):
    """Plain policy Q&A (retriever + LCEL chain), without the booking agent."""
    from rag import answer_policy

    try:
        return answer_policy(body.question)
    except Exception:
        log.exception("rag failed")
        raise HTTPException(500, "Policy search is unavailable right now.")


@router.post("/rag/rebuild", dependencies=[Depends(require_admin)])
def rebuild_index():
    from rag import build_index

    return {"chunks_indexed": build_index()}