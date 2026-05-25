#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/resilience.py
-----------------
Production-grade resilience wrapper for RAG pipelines.

Features:
  - Retry with exponential backoff (handles transient API failures)
  - Per-query timeout
  - Graceful fallback: cache → mock response
  - Rate limit detection and wait

Usage:
    from resilience import ResilientRAG
    rag = ResilientRAG(my_cot_rag, max_retries=3, timeout=15.0)
    result = rag.query("What is RAG?")
"""

import time
import threading
from typing import Any


# ── Exponential backoff ───────────────────────────────────

def _backoff_wait(attempt: int, base: float = 1.0, cap: float = 30.0):
    """Wait 2^attempt * base seconds, capped at cap."""
    wait = min(cap, base * (2 ** attempt))
    time.sleep(wait)


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg or "too many requests" in msg


# ── Timeout via thread ────────────────────────────────────

class _TimeoutError(Exception):
    pass


def _call_with_timeout(fn, args, kwargs, timeout_s: float):
    """Run fn(*args, **kwargs) in a thread; raise _TimeoutError if it exceeds timeout_s."""
    result_holder = [None]
    exc_holder    = [None]

    def target():
        try:
            result_holder[0] = fn(*args, **kwargs)
        except Exception as e:
            exc_holder[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        raise _TimeoutError(f"Query timed out after {timeout_s}s")
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]


# ── Mock fallback ─────────────────────────────────────────

def _mock_response(question: str, reason: str = "API unavailable") -> dict:
    return {
        "answer":   f"[FALLBACK] Could not generate answer: {reason}. Question: {question[:80]}",
        "context":  [],
        "fallback": True,
        "reason":   reason,
    }


# ── Main class ────────────────────────────────────────────

class ResilientRAG:
    """
    Wraps any RAG pipeline with production-grade resilience.

    Parameters
    ----------
    rag_pipeline : object
        Any object with a .query(question, verbose=False) method.
    max_retries : int
        Number of retry attempts on transient failures (default 3).
    timeout : float
        Per-query timeout in seconds (default 15.0).
    cache : RAGCache or None
        If provided, successful answers are cached; failed queries try cache first.
    backoff_base : float
        Base delay for exponential backoff in seconds (default 1.0).
    """

    def __init__(self, rag_pipeline, max_retries: int = 3,
                 timeout: float = 15.0, cache=None, backoff_base: float = 1.0):
        self._rag          = rag_pipeline
        self.max_retries   = max_retries
        self.timeout       = timeout
        self._cache        = cache
        self._backoff_base = backoff_base
        self._stats        = {"success": 0, "cache_hit": 0, "fallback": 0, "error": 0}

    def build(self):
        """Delegate build to the underlying pipeline."""
        self._rag.build()
        return self

    def query(self, question: str, verbose: bool = False) -> dict:
        """
        Query with retry, timeout, cache fallback, and mock fallback.

        Priority order on failure:
          1. Retry up to max_retries with exponential backoff
          2. Return cached response if available
          3. Return mock response
        """
        pipeline_name = type(self._rag).__name__

        # Try cache first if available
        if self._cache:
            cached = self._cache.responses.get_response(question, pipeline=pipeline_name)
            if cached:
                self._stats["cache_hit"] += 1
                if verbose:
                    print(f"  [Cache HIT] {question[:50]}")
                return {"answer": cached["answer"], "context": cached["context"],
                        "from_cache": True}

        last_exc = None
        for attempt in range(self.max_retries):
            try:
                if verbose and attempt > 0:
                    print(f"  [Retry {attempt}] {question[:50]}")

                result = _call_with_timeout(
                    self._rag.query,
                    args=(question,),
                    kwargs={"verbose": verbose},
                    timeout_s=self.timeout,
                )

                # Success
                self._stats["success"] += 1
                if self._cache:
                    self._cache.responses.set_response(
                        question, result["answer"],
                        pipeline=pipeline_name,
                        context=result.get("context", []),
                    )
                return result

            except _TimeoutError as e:
                last_exc = e
                if verbose:
                    print(f"  [TIMEOUT] attempt {attempt+1}: {e}")
                # Don't backoff on timeout — just retry immediately once then fallback
                if attempt >= 1:
                    break

            except Exception as e:
                last_exc = e
                if _is_rate_limit(e):
                    wait = min(60.0, self._backoff_base * (2 ** (attempt + 2)))
                    if verbose:
                        print(f"  [RATE LIMIT] Waiting {wait:.0f}s before retry...")
                    time.sleep(wait)
                elif attempt < self.max_retries - 1:
                    _backoff_wait(attempt, self._backoff_base)

        # Fallback: try cache
        if self._cache:
            cached = self._cache.responses.get_response(question, pipeline=pipeline_name)
            if cached:
                self._stats["cache_hit"] += 1
                return {"answer": cached["answer"], "context": cached["context"],
                        "from_cache": True, "fallback_reason": str(last_exc)}

        # Final fallback: mock
        self._stats["fallback"] += 1
        return _mock_response(question, reason=str(last_exc))

    @property
    def stats(self) -> dict:
        total = sum(self._stats.values())
        return {
            **self._stats,
            "total":       total,
            "success_rate": round(self._stats["success"] / total, 3) if total else 0,
        }
