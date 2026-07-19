"""Monitoring dashboard: per-query logs aggregated into latency/cost/usage
stats. Read access follows the same admin rule as KB management."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db, ChatSession, Message, QueryLog, User
from backend.api.auth import get_current_user
from backend.api.knowledge import can_manage_kb

router = APIRouter()

# Downvoted answers a human marked for the offline eval set — the online
# feedback → offline Ragas loop (TASKLIST D4). Ground truth is filled in
# later by a reviewer; here we capture the question + the bad answer.
EVAL_QUEUE_PATH = Path(__file__).parent.parent.parent / "data" / "feedback_eval_queue.json"


def _load_eval_queue() -> list[dict]:
    if not EVAL_QUEUE_PATH.exists():
        return []
    try:
        return json.loads(EVAL_QUEUE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_eval_queue(items: list[dict]) -> None:
    EVAL_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_QUEUE_PATH.write_text(json.dumps(items, ensure_ascii=False, indent=2),
                               encoding="utf-8")


def _pct(sorted_vals: list, p: int) -> int:
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


@router.get("/stats")
def get_stats(
    days: int = 14,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not can_manage_kb(current_user):
        raise HTTPException(403, "Chỉ quản trị viên mới xem được thống kê hệ thống")
    days = max(1, min(days, 90))
    since = datetime.utcnow() - timedelta(days=days)

    logs = (
        db.query(QueryLog)
        .filter(QueryLog.created_at >= since)
        .order_by(QueryLog.created_at.asc())
        .all()
    )

    ok_lat = sorted(l.latency_ms for l in logs if l.status == "success")
    errors = sum(1 for l in logs if l.status == "error")
    aborted = sum(1 for l in logs if l.status == "aborted")

    by_complexity: dict[str, int] = {}
    per_day: dict[str, dict] = {}
    for l in logs:
        by_complexity[l.complexity] = by_complexity.get(l.complexity, 0) + 1
        day = l.created_at.strftime("%Y-%m-%d")
        bucket = per_day.setdefault(day, {"date": day, "queries": 0, "cost_usd": 0.0,
                                          "multihop": 0})
        bucket["queries"] += 1
        bucket["cost_usd"] += l.cost_usd or 0.0
        if l.multihop:
            bucket["multihop"] += 1

    # Fill missing days so the chart has a continuous axis
    series = []
    for i in range(days - 1, -1, -1):
        day = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        series.append(per_day.get(day, {"date": day, "queries": 0,
                                        "cost_usd": 0.0, "multihop": 0}))
    for b in series:
        b["cost_usd"] = round(b["cost_usd"], 6)

    fb = (
        db.query(Message.feedback)
        .filter(Message.feedback.isnot(None), Message.created_at >= since)
        .all()
    )
    fb_up = sum(1 for (f,) in fb if f == "up")
    fb_down = sum(1 for (f,) in fb if f == "down")

    return {
        "window_days": days,
        "n_queries": len(logs),
        "errors": errors,
        "aborted": aborted,
        "error_rate": round(errors / len(logs), 4) if logs else 0.0,
        "latency_ms": {
            "avg": round(sum(ok_lat) / len(ok_lat)) if ok_lat else 0,
            "p50": _pct(ok_lat, 50),
            "p95": _pct(ok_lat, 95),
            "p99": _pct(ok_lat, 99),
        },
        "total_cost_usd": round(sum(l.cost_usd or 0.0 for l in logs), 6),
        "total_tokens": sum((l.tokens_in or 0) + (l.tokens_out or 0) for l in logs),
        "by_complexity": by_complexity,
        "multihop_queries": sum(1 for l in logs if l.multihop),
        "feedback": {"up": fb_up, "down": fb_down},
        "per_day": series,
    }


def _question_for(db: Session, msg: Message) -> str:
    """The user question that produced an assistant answer: the last user
    message in the same session created no later than the answer."""
    q = (db.query(Message)
         .filter(Message.session_id == msg.session_id,
                 Message.role == "user",
                 Message.created_at <= msg.created_at)
         .order_by(Message.created_at.desc())
         .first())
    return q.content if q else ""


@router.get("/downvoted")
def list_downvoted(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Answers users marked 👎 — the raw material for closing the quality
    loop (review → add to eval set). Admin-only, same rule as stats."""
    if not can_manage_kb(current_user):
        raise HTTPException(403, "Chỉ quản trị viên mới xem được mục này")
    days = max(1, min(days, 365))
    since = datetime.utcnow() - timedelta(days=days)
    msgs = (
        db.query(Message)
        .filter(Message.role == "assistant", Message.feedback == "down",
                Message.created_at >= since)
        .order_by(Message.created_at.desc())
        .limit(200)
        .all()
    )
    queued = {i["message_id"] for i in _load_eval_queue()}
    return {
        "items": [{
            "message_id": m.id,
            "session_id": m.session_id,
            "question":   _question_for(db, m),
            "answer":     m.content,
            "sources":    json.loads(m.sources_json or "[]"),
            "created_at": m.created_at.isoformat(),
            "in_eval_queue": m.id in queued,
        } for m in msgs],
    }


class EvalQueueAdd(BaseModel):
    message_id: str
    ground_truth: str | None = None  # optional reviewer-provided correct answer


@router.post("/eval-queue")
def add_to_eval_queue(
    body: EvalQueueAdd,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Append a downvoted answer to the offline eval queue (TASKLIST D4).
    Idempotent per message_id."""
    if not can_manage_kb(current_user):
        raise HTTPException(403, "Chỉ quản trị viên mới thao tác được")
    msg = (
        db.query(Message)
        .join(ChatSession, Message.session_id == ChatSession.id)
        .filter(Message.id == body.message_id, Message.role == "assistant")
        .first()
    )
    if not msg:
        raise HTTPException(404, "Message not found")

    items = _load_eval_queue()
    if any(i["message_id"] == msg.id for i in items):
        return {"success": True, "already_queued": True, "total": len(items)}
    items.append({
        "message_id":   msg.id,
        "question":     _question_for(db, msg),
        "bad_answer":   msg.content,
        "ground_truth": (body.ground_truth or "").strip(),
        "sources":      [s.get("title", "") for s in json.loads(msg.sources_json or "[]")],
        "added_at":     datetime.utcnow().isoformat(),
        "added_by":     current_user.email,
    })
    _save_eval_queue(items)
    return {"success": True, "already_queued": False, "total": len(items)}
