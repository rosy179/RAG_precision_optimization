import json
import time
from datetime import datetime

import openai
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db, ChatSession, Document, Message, SessionLocal, User
from backend.api.auth import get_current_user
from backend.services.user_rag import get_service, openai_error_detail

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    # Names of documents attached in the input bar for this message,
    # echoed back so the UI can render them inside the user bubble.
    attachments: list[str] = []
    # When set, replace the content of this assistant message instead of
    # appending a new user+assistant pair (Regenerate button).
    regenerate_message_id: str | None = None
    # Source picker: session doc ids to search (None = all), and whether
    # the shared knowledge base participates in this question.
    include_doc_ids: list[str] | None = None
    use_global_kb: bool = True


class FeedbackRequest(BaseModel):
    rating: str | None = None  # "up" | "down" | None clears the rating


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/sessions")
def create_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = ChatSession(user_id=current_user.id, title="New Chat")
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"session_id": session.id, "title": session.title, "created_at": session.created_at.isoformat()}


@router.get("/sessions")
def list_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return [{"id": s.id, "title": s.title, "created_at": s.created_at.isoformat()} for s in sessions]


@router.get("/sessions/{session_id}/messages")
def get_messages(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id, ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(404, "Session not found")
    return [
        {
            "id":          m.id,
            "role":        m.role,
            "content":     m.content,
            "sources":     json.loads(m.sources_json or "[]"),
            "attachments": json.loads(m.attachments_json or "[]"),
            "feedback":    m.feedback,
            "created_at":  m.created_at.isoformat(),
        }
        for m in session.messages
    ]


@router.post("/sessions/{session_id}/chat")
def chat(
    session_id: str,
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id, ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(404, "Session not found")

    rag = get_service(current_user.id)
    try:
        result = rag.query(req.question, req.history, session_id=session_id,
                           include_doc_ids=req.include_doc_ids,
                           use_global_kb=req.use_global_kb)
    except openai.APIError as e:
        raise HTTPException(502, openai_error_detail(e)) from e

    # Save user message
    user_msg = Message(
        session_id=session_id,
        role="user",
        content=req.question,
        attachments_json=json.dumps(req.attachments, ensure_ascii=False),
    )
    db.add(user_msg)

    # Save assistant message
    ai_msg = Message(
        session_id=session_id,
        role="assistant",
        content=result["answer"],
        sources_json=json.dumps(result["sources"], ensure_ascii=False),
    )
    db.add(ai_msg)

    # Auto-title the session after first message
    if session.title == "New Chat":
        session.title = req.question[:60] + ("…" if len(req.question) > 60 else "")

    db.commit()

    return {
        "answer":     result["answer"],
        "reasoning":  result.get("reasoning", ""),
        "sources":    result["sources"],
        "latency_ms": result.get("latency_ms", 0),
        "from_cache": result.get("from_cache", False),
        "complexity": result.get("complexity", "medium"),
    }


@router.post("/sessions/{session_id}/chat/stream")
def chat_stream(
    session_id: str,
    req: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """SSE variant of /chat: `meta` (sources) → `delta`* → `done` | `error`."""
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id, ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(404, "Session not found")

    if req.regenerate_message_id:
        target = db.query(Message).filter(
            Message.id == req.regenerate_message_id,
            Message.session_id == session_id,
            Message.role == "assistant",
        ).first()
        if not target:
            raise HTTPException(404, "Message to regenerate not found")

    rag = get_service(current_user.id)

    def persist(db2: Session, answer: str, sources: list[dict]) -> dict:
        """Save the exchange; returns the ids the UI needs for feedback."""
        ids: dict = {}
        if req.regenerate_message_id:
            msg = db2.query(Message).filter(Message.id == req.regenerate_message_id).first()
            if msg:
                msg.content = answer
                msg.sources_json = json.dumps(sources, ensure_ascii=False)
                msg.feedback = None  # old rating applied to the old answer
                msg.created_at = datetime.utcnow()
                ids["message_id"] = msg.id
        else:
            user_msg = Message(
                session_id=session_id,
                role="user",
                content=req.question,
                attachments_json=json.dumps(req.attachments, ensure_ascii=False),
            )
            db2.add(user_msg)
            ai_msg = Message(
                session_id=session_id,
                role="assistant",
                content=answer,
                sources_json=json.dumps(sources, ensure_ascii=False),
            )
            db2.add(ai_msg)
            sess = db2.query(ChatSession).filter(ChatSession.id == session_id).first()
            if sess and sess.title == "New Chat":
                sess.title = req.question[:60] + ("…" if len(req.question) > 60 else "")
                ids["session_title"] = sess.title
            db2.flush()
            ids["user_message_id"] = user_msg.id
            ids["message_id"] = ai_msg.id
        db2.commit()
        return ids

    def gen():
        # The request-scoped db session may be torn down once streaming
        # starts, so persistence uses its own session.
        db2 = SessionLocal()
        t0 = time.time()
        parts: list[str] = []
        sources: list[dict] = []
        try:
            try:
                ret = rag.retrieve(req.question, req.history, session_id=session_id,
                                   include_doc_ids=req.include_doc_ids,
                                   use_global_kb=req.use_global_kb)
            except openai.APIError as e:
                yield _sse("error", {"detail": openai_error_detail(e)})
                return
            sources = ret["sources"]
            yield _sse("meta", {"sources": sources, "complexity": ret["complexity"]})

            if ret["empty"]:
                parts.append(ret["notice"])
                yield _sse("delta", {"text": ret["notice"]})
            else:
                try:
                    for delta in rag.generate_stream(req.question, ret["top_chunks"],
                                                     req.history):
                        parts.append(delta)
                        yield _sse("delta", {"text": delta})
                except openai.APIError as e:
                    yield _sse("error", {"detail": openai_error_detail(e)})
                    if not parts:
                        return  # nothing to save — same as old non-stream behavior

            ids = persist(db2, "".join(parts).strip(), sources)
            yield _sse("done", {
                "latency_ms": int((time.time() - t0) * 1000),
                **ids,
            })
        except GeneratorExit:
            # Client aborted (Stop button / closed tab): keep the partial
            # answer so the conversation stays coherent on reload.
            if parts:
                persist(db2, "".join(parts).strip(), sources)
            raise
        finally:
            db2.close()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/messages/{message_id}/feedback")
def set_feedback(
    message_id: str,
    req: FeedbackRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if req.rating not in ("up", "down", None):
        raise HTTPException(422, "rating must be 'up', 'down' or null")
    msg = (
        db.query(Message)
        .join(ChatSession, Message.session_id == ChatSession.id)
        .filter(Message.id == message_id, ChatSession.user_id == current_user.id)
        .first()
    )
    if not msg:
        raise HTTPException(404, "Message not found")
    msg.feedback = req.rating
    db.commit()
    return {"success": True, "feedback": msg.feedback}


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id, ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(404, "Session not found")

    # Documents are session-scoped: remove this session's docs from the
    # vector store and registry too, so they don't linger as orphans.
    docs = db.query(Document).filter(
        Document.session_id == session_id, Document.user_id == current_user.id
    ).all()
    if docs:
        rag = get_service(current_user.id)
        for doc in docs:
            rag.remove_document(doc.id)
            db.delete(doc)

    db.delete(session)
    db.commit()
    return {"success": True}
