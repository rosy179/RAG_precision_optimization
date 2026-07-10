import json
from datetime import datetime

import openai
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db, ChatSession, Message, User
from backend.api.auth import get_current_user
from backend.services.user_rag import get_service, openai_error_detail

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


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
            "id":         m.id,
            "role":       m.role,
            "content":    m.content,
            "sources":    json.loads(m.sources_json or "[]"),
            "created_at": m.created_at.isoformat(),
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
        result = rag.query(req.question, req.history)
    except openai.APIError as e:
        raise HTTPException(502, openai_error_detail(e)) from e

    # Save user message
    user_msg = Message(session_id=session_id, role="user", content=req.question)
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
    db.delete(session)
    db.commit()
    return {"success": True}
