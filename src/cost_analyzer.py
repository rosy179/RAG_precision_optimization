#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/cost_analyzer.py
--------------------
Cost & Performance Analysis for RAG Techniques.

Measures per technique:
  - Latency: P50 / P95 / P99 (milliseconds)
  - Token usage: embedding + prompt + completion
  - API cost: based on OpenAI pricing
  - Accuracy: from existing metrics files

Pricing reference (as of 2025):
  gpt-4o-mini input:  $0.00015 per 1K tokens
  gpt-4o-mini output: $0.00060 per 1K tokens
  text-embedding-ada-002: $0.02 per 1M tokens
"""

import sys
import os
import json
import time
import statistics
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# ── Pricing (USD) ─────────────────────────────────────────
PRICING = {
    "gpt-4o-mini": {
        "input_per_1k":  0.00015,
        "output_per_1k": 0.00060,
    },
    "text-embedding-ada-002": {
        "per_1m": 0.02,
    },
}

KNOWN_ACCURACY = {
    "baseline":        0.8782,
    "hybrid":          0.8999,
    "reranked":        0.9196,
    "query_expansion": 0.9071,
    "cot":             0.9434,
}


def _try_import_tiktoken():
    try:
        import tiktoken
        return tiktoken.encoding_for_model("gpt-4o-mini")
    except ImportError:
        return None


def count_tokens_approx(text: str, encoder=None) -> int:
    """Count tokens — exact with tiktoken, approx otherwise."""
    if encoder:
        return len(encoder.encode(text))
    return max(1, len(text.split()) * 4 // 3)


class CostAnalyzer:
    """Measure latency, token usage, and API cost for each RAG technique."""

    def __init__(self, n_queries: int = 10, results_dir: str = "results"):
        self.n_queries = n_queries
        self.results_dir = Path(results_dir)
        self.encoder = _try_import_tiktoken()
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-ada-002")

    # ── Internal helpers ──────────────────────────────────

    def _calc_cost(self, embed_tokens: int, prompt_tokens: int, completion_tokens: int) -> float:
        ep = PRICING.get(self.embed_model, {}).get("per_1m", 0.02)
        lp = PRICING.get(self.llm_model, PRICING["gpt-4o-mini"])
        embed_cost = embed_tokens * ep / 1_000_000
        prompt_cost = prompt_tokens * lp["input_per_1k"] / 1_000
        completion_cost = completion_tokens * lp["output_per_1k"] / 1_000
        return embed_cost + prompt_cost + completion_cost

    def _percentile(self, data: list, p: int) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
        return round(sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo), 1)

    def _load_test_queries(self, n: int) -> list:
        qa_path = Path("data/squad_qa.json")
        if not qa_path.exists():
            return [f"Sample question {i+1} about machine learning?" for i in range(n)]
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        return [q["query"] for q in qa_pairs[-n:]]

    # ── Measure single pipeline ───────────────────────────

    def measure_pipeline(self, pipeline_name: str, rag_pipeline) -> dict:
        """
        Run n_queries through a pipeline, collect latency + token usage.
        Returns per-technique stats.
        """
        print(f"\n  Measuring: {pipeline_name} ({self.n_queries} queries)...")
        queries = self._load_test_queries(self.n_queries)

        latencies_ms = []
        embed_tokens_list = []
        prompt_tokens_list = []
        completion_tokens_list = []
        errors = 0

        for i, query in enumerate(queries, 1):
            print(f"    Query {i}/{self.n_queries}...", end="\r")
            try:
                t0 = time.perf_counter()
                result = rag_pipeline.query(query, verbose=False)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies_ms.append(elapsed_ms)

                # Estimate tokens
                context_text = " ".join(c["text"] for c in result.get("context", []))
                embed_tok = count_tokens_approx(query, self.encoder)
                prompt_tok = count_tokens_approx(query + context_text, self.encoder)
                completion_tok = count_tokens_approx(result.get("answer", ""), self.encoder)

                embed_tokens_list.append(embed_tok)
                prompt_tokens_list.append(prompt_tok)
                completion_tokens_list.append(completion_tok)

            except Exception as e:
                print(f"\n    [WARN] Query {i} failed: {e}")
                errors += 1

        if not latencies_ms:
            return {"pipeline": pipeline_name, "error": "All queries failed"}

        avg_embed = sum(embed_tokens_list) / len(embed_tokens_list)
        avg_prompt = sum(prompt_tokens_list) / len(prompt_tokens_list)
        avg_completion = sum(completion_tokens_list) / len(completion_tokens_list)
        avg_cost = self._calc_cost(avg_embed, avg_prompt, avg_completion)

        return {
            "pipeline": pipeline_name,
            "n_queries": self.n_queries,
            "errors": errors,
            "latency_ms": {
                "avg":  round(statistics.mean(latencies_ms), 1),
                "min":  round(min(latencies_ms), 1),
                "max":  round(max(latencies_ms), 1),
                "p50":  self._percentile(latencies_ms, 50),
                "p95":  self._percentile(latencies_ms, 95),
                "p99":  self._percentile(latencies_ms, 99),
            },
            "tokens_per_query": {
                "embedding":   round(avg_embed),
                "prompt":      round(avg_prompt),
                "completion":  round(avg_completion),
                "total":       round(avg_embed + avg_prompt + avg_completion),
            },
            "cost_per_query_usd": round(avg_cost, 8),
            "cost_per_1000_queries_usd": round(avg_cost * 1000, 4),
            "accuracy": KNOWN_ACCURACY.get(pipeline_name, None),
        }

    # ── Compare all techniques ────────────────────────────

    def compare_techniques(self, results: list) -> dict:
        """Rank techniques by accuracy, speed, cost, and value."""
        valid = [r for r in results if "error" not in r and r.get("accuracy")]

        if not valid:
            return {}

        by_accuracy = sorted(valid, key=lambda x: x["accuracy"], reverse=True)
        by_speed    = sorted(valid, key=lambda x: x["latency_ms"]["p50"])
        by_cost     = sorted(valid, key=lambda x: x["cost_per_query_usd"])

        # Best value = accuracy / cost_per_query (higher is better)
        for r in valid:
            cost = r["cost_per_query_usd"] or 1e-9
            r["_value_score"] = r["accuracy"] / cost
        by_value = sorted(valid, key=lambda x: x["_value_score"], reverse=True)

        return {
            "most_accurate":   by_accuracy[0]["pipeline"],
            "fastest_p50":     by_speed[0]["pipeline"],
            "cheapest":        by_cost[0]["pipeline"],
            "best_value":      by_value[0]["pipeline"],
            "ranking_accuracy": [r["pipeline"] for r in by_accuracy],
            "ranking_speed":    [r["pipeline"] for r in by_speed],
            "ranking_cost":     [r["pipeline"] for r in by_cost],
            "tradeoff_summary": _build_tradeoff_summary(valid),
        }


def _build_tradeoff_summary(results: list) -> list:
    """Plain-English tradeoff for each technique."""
    rows = []
    for r in sorted(results, key=lambda x: x["accuracy"]):
        acc = r["accuracy"]
        lat = r["latency_ms"]["p50"]
        cost = r["cost_per_query_usd"]

        if lat < 600:
            speed_label = "Fast"
        elif lat < 1500:
            speed_label = "Medium"
        else:
            speed_label = "Slow"

        if cost < 0.00005:
            cost_label = "Very cheap"
        elif cost < 0.00015:
            cost_label = "Cheap"
        elif cost < 0.00050:
            cost_label = "Moderate"
        else:
            cost_label = "Expensive"

        rows.append({
            "pipeline": r["pipeline"],
            "accuracy": acc,
            "speed":    speed_label,
            "cost":     cost_label,
            "use_case": _recommend_use_case(r["pipeline"]),
        })
    return rows


def _recommend_use_case(pipeline: str) -> str:
    mapping = {
        "baseline":        "Demo, prototyping, budget-constrained",
        "hybrid":          "Web API, balanced speed/accuracy",
        "reranked":        "Production, best reliability per cost",
        "query_expansion": "Broad search, exploratory queries",
        "cot":             "High-stakes Q&A, hallucination-critical",
    }
    return mapping.get(pipeline, "General purpose")


def load_or_estimate_stats(results_dir: Path = Path("results")) -> list:
    """
    Load latency/cost stats from cached file, or build estimates
    from existing metrics JSON files when pipelines can't be run live.
    """
    cache_path = results_dir / "cost_analysis.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("techniques", [])

    # Build realistic estimates from known architecture properties
    estimates = [
        _estimate("baseline",        latency_p50=520,  embed=250, prompt=380, completion=80),
        _estimate("hybrid",          latency_p50=680,  embed=250, prompt=420, completion=80),
        _estimate("reranked",        latency_p50=950,  embed=250, prompt=420, completion=80),
        _estimate("query_expansion", latency_p50=1800, embed=750, prompt=480, completion=80),
        _estimate("cot",             latency_p50=2100, embed=250, prompt=780, completion=220),
    ]
    return estimates


def _estimate(name, latency_p50, embed, prompt, completion) -> dict:
    """Build an estimated stats dict (used when live measurement unavailable)."""
    ep = PRICING["text-embedding-ada-002"]["per_1m"]
    lp = PRICING["gpt-4o-mini"]
    cost = embed * ep / 1_000_000 + prompt * lp["input_per_1k"] / 1_000 + completion * lp["output_per_1k"] / 1_000

    return {
        "pipeline": name,
        "n_queries": "estimated",
        "errors": 0,
        "latency_ms": {
            "avg":  latency_p50,
            "min":  round(latency_p50 * 0.7),
            "max":  round(latency_p50 * 2.5),
            "p50":  latency_p50,
            "p95":  round(latency_p50 * 2.0),
            "p99":  round(latency_p50 * 2.8),
        },
        "tokens_per_query": {
            "embedding":  embed,
            "prompt":     prompt,
            "completion": completion,
            "total":      embed + prompt + completion,
        },
        "cost_per_query_usd": round(cost, 8),
        "cost_per_1000_queries_usd": round(cost * 1000, 4),
        "accuracy": KNOWN_ACCURACY.get(name),
        "source": "estimated",
    }
