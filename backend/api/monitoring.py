"""Monitoring dashboard: per-query logs aggregated into latency/cost/usage
stats. Read access follows the same admin rule as KB management."""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db, Message, QueryLog, User
from backend.api.auth import get_current_user
from backend.api.knowledge import can_manage_kb

router = APIRouter()


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
