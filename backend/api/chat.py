import json
import time
from datetime import datetime

import openai
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import (
    get_db, ChatSession, Document, Message, QueryLog, SessionLocal, User,
)
from backend.api.auth import get_current_user
from backend.services.user_rag import (
    get_service, generate_session_title, is_summary_question, openai_error_detail,
    wants_prev_transform,
)

router = APIRouter()

# gpt-4o-mini pricing (USD per token) for dashboard cost estimates
_PRICE_IN  = 0.15 / 1_000_000
_PRICE_OUT = 0.60 / 1_000_000


def _log_query(db2: Session, user_id: str, session_id: str, complexity: str,
               n_hops: int, latency_ms: int, answer: str, n_sources: int,
               status: str, usage: dict | None = None):
    """Best-effort per-query log for the monitoring dashboard.

    Real token counts (accumulated from OpenAI usage across every LLM call
    of the request) are used when available; otherwise falls back to rough
    estimates — good enough for cost trends, not billing.
    """
    try:
        if usage and usage.get("real"):
            tokens_in, tokens_out = usage.get("in", 0), usage.get("out", 0)
        else:
            tokens_in = 400 + n_sources * 500 + n_hops * 1500
            tokens_out = max(1, len(answer) // 4) + n_hops * 60
        db2.add(QueryLog(
            user_id=user_id, session_id=session_id, complexity=complexity,
            multihop=n_hops, latency_ms=latency_ms,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost_usd=tokens_in * _PRICE_IN + tokens_out * _PRICE_OUT,
            status=status,
        ))
        db2.commit()
    except Exception:
        db2.rollback()


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


def _save_suggestions(db2: Session, message_id: str | None,
                      questions: list[str]) -> None:
    """Attach the post-done suggestion chips to the persisted message so
    they survive a reload (rendered on the conversation's last answer)."""
    if not message_id:
        return
    try:
        msg = db2.query(Message).filter(Message.id == message_id).first()
        if msg:
            msg.suggestions_json = json.dumps(questions, ensure_ascii=False)
            db2.commit()
    except Exception:
        db2.rollback()


def _attached_doc_ids(db: Session, user_id: str, session_id: str,
                      names: list[str]) -> list[str] | None:
    """Resolve attachment names sent with the message to session doc ids so
    retrieval can treat them as the question's subject."""
    if not names:
        return None
    rows = db.query(Document).filter(
        Document.user_id == user_id,
        Document.session_id == session_id,
        Document.name.in_(names),
    ).all()
    return [d.id for d in rows] or None


def _question_with_attachments(question: str, attachments: list[str],
                               attached_ids: list[str] | None) -> str:
    """Deictic questions ("dự án này", "file này") sent WITH attachments refer
    to those attachments — spell that out for retrieval/generation so recent
    history (about older documents) can't hijack the subject. The stored user
    message keeps the original question."""
    if not attached_ids or not attachments:
        return question
    names = ", ".join(attachments)
    return (f"{question}\n(Tài liệu đính kèm theo câu hỏi này: {names} — "
            f'các cụm "tài liệu/dự án/bài viết/file này" chỉ tài liệu đính kèm đó.)')


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
            "steps":       json.loads(m.steps_json or "[]"),
            "suggestions": json.loads(m.suggestions_json or "[]"),
            "created_at":  m.created_at.isoformat(),
        }
        for m in session.messages
    ]


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
    # Resolve before streaming starts — the request-scoped db session may be
    # torn down once the generator is running.
    attached_ids = _attached_doc_ids(db, current_user.id, session_id, req.attachments)
    gen_question = _question_with_attachments(req.question, req.attachments, attached_ids)

    def persist(db2: Session, answer: str, sources: list[dict],
                steps: list[dict] | None = None) -> dict:
        """Save the exchange; returns the ids the UI needs for feedback."""
        ids: dict = {}
        steps_json = json.dumps(steps or [], ensure_ascii=False)
        if req.regenerate_message_id:
            msg = db2.query(Message).filter(Message.id == req.regenerate_message_id).first()
            if msg:
                msg.content = answer
                msg.sources_json = json.dumps(sources, ensure_ascii=False)
                msg.feedback = None  # old rating applied to the old answer
                msg.steps_json = steps_json
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
                steps_json=steps_json,
            )
            db2.add(ai_msg)
            sess = db2.query(ChatSession).filter(ChatSession.id == session_id).first()
            if sess and sess.title == "New Chat":
                sess.title = generate_session_title(req.question, answer)
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
        steps: list[dict] = []
        complexity = "medium"
        n_hops = 0
        retrieval_empty = False
        # True once the exchange is saved — a disconnect during the
        # post-done extras must not persist it a second time.
        persisted = False
        # Real token usage accumulated across every LLM call of this request
        usage: dict = {"in": 0, "out": 0, "real": False}

        def log(status: str):
            _log_query(db2, current_user.id, session_id, complexity, n_hops,
                       int((time.time() - t0) * 1000), "".join(parts),
                       len(sources), status, usage)

        try:
            # Answer cache: a history/attachment-free question already
            # answered in this same scope is replayed from memory —
            # sources + reasoning steps restored, answer streamed in chunks.
            cacheable = (not req.history and not req.attachments
                         and not req.regenerate_message_id)
            if cacheable:
                hit = rag.cache_lookup(req.question, session_id,
                                       req.include_doc_ids, req.use_global_kb)
                if hit:
                    complexity = hit["complexity"]
                    sources = hit["sources"]
                    usage["real"] = True  # served from cache: 0 tokens spent
                    yield _sse("meta", {"sources": sources,
                                        "complexity": complexity,
                                        "from_cache": True})
                    for s in hit["steps"]:
                        steps.append(s)
                        yield _sse("step", s)
                    answer = hit["answer"]
                    for i in range(0, len(answer), 64):
                        parts.append(answer[i:i + 64])
                        yield _sse("delta", {"text": answer[i:i + 64]})
                    ids = persist(db2, answer, sources, steps)
                    persisted = True
                    log("success")
                    yield _sse("done", {
                        "latency_ms": int((time.time() - t0) * 1000),
                        "from_cache": True,
                        **ids,
                    })
                    if hit.get("suggestions"):
                        yield _sse("suggestions", {"questions": hit["suggestions"]})
                        _save_suggestions(db2, ids.get("message_id"),
                                          hit["suggestions"])
                    return

            if wants_prev_transform(req.question, req.history):
                # "Dịch bản tóm tắt đó sang tiếng Nhật"-type requests re-render
                # the PREVIOUS answer. Retrieval can't help (no chunk contains
                # the answer) and its grounded-only prompt makes the model
                # refuse — transform directly, carrying the citations over.
                complexity = "transform"
                prev = next(str(m["content"]) for m in reversed(req.history)
                            if m.get("role") == "assistant" and m.get("content"))
                prev_msg = (db2.query(Message)
                            .filter(Message.session_id == session_id,
                                    Message.role == "assistant")
                            .order_by(Message.created_at.desc())
                            .first())
                sources = json.loads(prev_msg.sources_json or "[]") if prev_msg else []
                yield _sse("meta", {"sources": sources, "complexity": complexity})
                try:
                    for delta in rag.transform_stream(req.question, prev,
                                                      usage=usage):
                        parts.append(delta)
                        yield _sse("delta", {"text": delta})
                except openai.APIError as e:
                    yield _sse("error", {"detail": openai_error_detail(e)})
                    if not parts:
                        log("error")
                        return
                ids = persist(db2, "".join(parts).strip(), sources, steps)
                persisted = True
                log("success")
                yield _sse("done", {
                    "latency_ms": int((time.time() - t0) * 1000),
                    **ids,
                })
                return

            # Layer 1 — Adaptive router: medium/complex questions get one
            # extra LLM call deciding whether they need chained retrieval
            # (bridging entity discovered in hop 1 feeds hop 2's query).
            # Bridging questions often read as "medium" to the heuristic
            # ("Who received X for the work that enabled Y?"), so gating on
            # complex-only would never fire. Simple lookups skip the router
            # entirely — zero added latency.
            from adaptive_rag import classify_heuristic
            complexity = classify_heuristic(req.question)
            sub_questions: list[str] = []
            # Summarize-type questions never need chained retrieval — they
            # go straight to the document digest path inside retrieve().
            if (complexity in ("medium", "complex")
                    and not is_summary_question(req.question)
                    and rag.has_any_source(
                        session_id, req.include_doc_ids, req.use_global_kb)):
                sub_questions = rag.route_multihop(req.question, usage=usage)

            if sub_questions:
                complexity = "multihop"
                n_hops = len(sub_questions)
                yield _sse("meta", {"sources": [], "complexity": "multihop",
                                    "n_hops": n_hops})
                try:
                    for kind, payload in rag.multihop_events(
                            gen_question, sub_questions, req.history,
                            session_id=session_id,
                            include_doc_ids=req.include_doc_ids,
                            use_global_kb=req.use_global_kb,
                            attached_doc_ids=attached_ids,
                            usage=usage):
                        if kind == "step":
                            steps.append(payload)
                            yield _sse("step", payload)
                        elif kind == "sources":
                            sources = payload["sources"]
                            yield _sse("meta", {"sources": sources,
                                                "complexity": "multihop",
                                                "n_hops": n_hops})
                        else:  # delta
                            parts.append(payload)
                            yield _sse("delta", {"text": payload})
                except openai.APIError as e:
                    yield _sse("error", {"detail": openai_error_detail(e)})
                    if not parts:
                        log("error")
                        return
            else:
                try:
                    ret = rag.retrieve(gen_question, req.history, session_id=session_id,
                                       include_doc_ids=req.include_doc_ids,
                                       use_global_kb=req.use_global_kb,
                                       attached_doc_ids=attached_ids,
                                       usage=usage)
                except openai.APIError as e:
                    yield _sse("error", {"detail": openai_error_detail(e)})
                    log("error")
                    return
                sources = ret["sources"]
                complexity = ret["complexity"]
                yield _sse("meta", {"sources": sources, "complexity": complexity})

                if ret["empty"]:
                    retrieval_empty = True
                    parts.append(ret["notice"])
                    yield _sse("delta", {"text": ret["notice"]})
                else:
                    try:
                        for delta in rag.generate_stream(gen_question, ret["top_chunks"],
                                                         req.history, usage=usage):
                            parts.append(delta)
                            yield _sse("delta", {"text": delta})
                    except openai.APIError as e:
                        yield _sse("error", {"detail": openai_error_detail(e)})
                        if not parts:
                            log("error")
                            return  # nothing to save — same as old behavior

            ids = persist(db2, "".join(parts).strip(), sources, steps)
            persisted = True
            log("success")
            yield _sse("done", {
                "latency_ms": int((time.time() - t0) * 1000),
                **ids,
            })
            # Post-done extras — suggestion chips, then the answer cache —
            # run AFTER done so they never delay the answer (the extra LLM /
            # embedding calls would add ~1s each). A client disconnect here
            # only skips the extras; the exchange is already saved above.
            suggestions: list[str] = []
            if parts and sources and not retrieval_empty:
                suggestions = rag.suggest_questions(req.question,
                                                    "".join(parts), sources)
                if suggestions:
                    yield _sse("suggestions", {"questions": suggestions})
                    _save_suggestions(db2, ids.get("message_id"), suggestions)
            if cacheable and parts and sources and not retrieval_empty:
                try:
                    rag.cache_store(req.question, session_id,
                                    req.include_doc_ids, req.use_global_kb,
                                    "".join(parts).strip(), sources,
                                    complexity, steps, suggestions)
                except Exception:
                    pass
        except GeneratorExit:
            # Client aborted (Stop button / closed tab): keep the partial
            # answer so the conversation stays coherent on reload. Once
            # `persisted` is set the exchange is already saved — the client
            # merely left during the post-done extras.
            if parts and not persisted:
                persist(db2, "".join(parts).strip(), sources, steps)
                log("aborted")
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
