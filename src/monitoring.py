#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/monitoring.py
-----------------
Lightweight monitoring and logging for RAG pipelines in production.

Tracks per-query:
  - Latency (ms)
  - Token usage
  - API cost
  - Success / fallback / error
  - Query + answer (truncated for privacy)

Aggregates:
  - P50 / P95 / P99 latency
  - Error rate
  - Total API cost
  - Rolling window stats

Usage:
    from monitoring import RAGMonitor
    monitor = RAGMonitor(log_dir="logs")
    monitor.log_query(question, answer, latency_ms=230, tokens=450, cost=0.00008)
    monitor.print_summary()
    monitor.save_report("results/monitoring_report.json")
"""

import json
import time
import statistics
from datetime import datetime
from pathlib import Path


# ── Per-query record ──────────────────────────────────────

def _make_record(question: str, answer: str, latency_ms: float,
                 tokens: int, cost_usd: float, status: str = "success",
                 pipeline: str = "", error: str = "") -> dict:
    return {
        "ts":           datetime.now().isoformat(),
        "pipeline":     pipeline,
        "question":     question[:120],
        "answer":       answer[:200],
        "latency_ms":   round(latency_ms, 1),
        "tokens":       tokens,
        "cost_usd":     round(cost_usd, 8),
        "status":       status,
        "error":        error,
    }


# ── Percentile helper ─────────────────────────────────────

def _pct(data: list, p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


# ── Main Monitor ──────────────────────────────────────────

class RAGMonitor:
    """
    In-memory query monitor with optional disk logging.

    Parameters
    ----------
    log_dir : str
        Directory to write per-session log files (JSONL format).
    pipeline : str
        Name of the RAG pipeline being monitored.
    alert_latency_ms : float
        Log a warning if any query exceeds this latency.
    alert_error_rate : float
        Log a warning if rolling error rate exceeds this threshold (0-1).
    """

    def __init__(self, log_dir: str = "logs", pipeline: str = "rag",
                 alert_latency_ms: float = 5000.0, alert_error_rate: float = 0.1):
        self._log_dir         = Path(log_dir)
        self._pipeline        = pipeline
        self._alert_lat_ms    = alert_latency_ms
        self._alert_err_rate  = alert_error_rate
        self._records: list   = []
        self._log_file        = None
        self._start_time      = time.time()
        self._setup_log_file()

    def _setup_log_file(self):
        self._log_dir.mkdir(parents=True, exist_ok=True)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = self._log_dir / f"{self._pipeline}_{session_id}.jsonl"

    def log_query(self, question: str, answer: str,
                  latency_ms: float, tokens: int = 0,
                  cost_usd: float = 0.0, status: str = "success",
                  error: str = "") -> dict:
        """Record a single query execution."""
        record = _make_record(
            question, answer, latency_ms, tokens, cost_usd,
            status=status, pipeline=self._pipeline, error=error,
        )
        self._records.append(record)

        # Disk write (append JSONL)
        with open(self._log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Anomaly alerts
        if latency_ms > self._alert_lat_ms:
            print(f"  [MONITOR ALERT] High latency: {latency_ms:.0f}ms for '{question[:50]}'")
        if self.error_rate > self._alert_err_rate and len(self._records) >= 10:
            print(f"  [MONITOR ALERT] Error rate {self.error_rate:.1%} exceeds threshold")

        return record

    # ── Aggregates ────────────────────────────────────────

    @property
    def n_queries(self) -> int:
        return len(self._records)

    @property
    def error_rate(self) -> float:
        if not self._records:
            return 0.0
        errors = sum(1 for r in self._records if r["status"] != "success")
        return errors / len(self._records)

    @property
    def total_cost_usd(self) -> float:
        return sum(r["cost_usd"] for r in self._records)

    @property
    def total_tokens(self) -> int:
        return sum(r["tokens"] for r in self._records)

    def latency_stats(self) -> dict:
        lats = [r["latency_ms"] for r in self._records if r["status"] == "success"]
        if not lats:
            return {"avg": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        return {
            "avg": round(statistics.mean(lats), 1),
            "p50": _pct(lats, 50),
            "p95": _pct(lats, 95),
            "p99": _pct(lats, 99),
            "min": round(min(lats), 1),
            "max": round(max(lats), 1),
        }

    def summary(self) -> dict:
        session_s = round(time.time() - self._start_time, 1)
        lat = self.latency_stats()
        errors   = sum(1 for r in self._records if r["status"] != "success")
        fallbacks = sum(1 for r in self._records if r["status"] == "fallback")
        return {
            "pipeline":        self._pipeline,
            "session_seconds": session_s,
            "n_queries":       self.n_queries,
            "errors":          errors,
            "fallbacks":       fallbacks,
            "error_rate":      round(self.error_rate, 4),
            "latency_ms":      lat,
            "total_tokens":    self.total_tokens,
            "total_cost_usd":  round(self.total_cost_usd, 6),
            "avg_cost_usd":    round(self.total_cost_usd / max(1, self.n_queries), 8),
            "qps":             round(self.n_queries / max(1, session_s), 2),
        }

    def print_summary(self):
        s = self.summary()
        lat = s["latency_ms"]
        print("\n  === RAG Monitor Summary ===")
        print(f"  Pipeline    : {s['pipeline']}")
        print(f"  Queries     : {s['n_queries']}  (errors={s['errors']}, "
              f"fallbacks={s['fallbacks']}, error_rate={s['error_rate']:.1%})")
        print(f"  Latency     : avg={lat['avg']}ms  P50={lat['p50']}ms  "
              f"P95={lat['p95']}ms  P99={lat['p99']}ms")
        print(f"  Tokens      : {s['total_tokens']} total  "
              f"({s['total_tokens'] // max(1, s['n_queries'])} avg/query)")
        print(f"  Cost        : ${s['total_cost_usd']:.6f} total  "
              f"(${s['avg_cost_usd']:.8f} avg/query)")
        print(f"  Throughput  : {s['qps']} queries/sec")
        print(f"  Log file    : {self._log_file}")

    def save_report(self, out_path: str = "results/monitoring_report.json"):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated": datetime.now().isoformat(),
            "summary":   self.summary(),
            "records":   self._records[-100:],  # last 100 queries
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return out

    def recent_errors(self, n: int = 5) -> list:
        return [r for r in reversed(self._records) if r["status"] != "success"][:n]


# ── Context manager integration ───────────────────────────

class MonitoredQuery:
    """
    Context manager to time and log a query automatically.

    Usage:
        with MonitoredQuery(monitor, question, tokens=450, cost=0.0001) as mq:
            result = rag.query(question)
        mq.log(answer=result["answer"])
    """

    def __init__(self, monitor: RAGMonitor, question: str,
                 tokens: int = 0, cost_usd: float = 0.0):
        self._monitor = monitor
        self._question = question
        self._tokens = tokens
        self._cost = cost_usd
        self._start = None
        self._elapsed_ms = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._elapsed_ms = (time.perf_counter() - self._start) * 1000
        return False  # don't suppress exceptions

    def log(self, answer: str = "", status: str = "success", error: str = ""):
        self._monitor.log_query(
            self._question, answer,
            latency_ms=self._elapsed_ms,
            tokens=self._tokens,
            cost_usd=self._cost,
            status=status,
            error=error,
        )
