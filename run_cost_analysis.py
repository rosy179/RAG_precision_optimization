#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cost_analysis.py
---------------------
Analyze latency, token usage, and API cost for all RAG techniques.

Usage:
    python run_cost_analysis.py               # estimate mode (no API calls)
    python run_cost_analysis.py --live        # live measurement (needs API key)
    python run_cost_analysis.py --live --n 5  # live, 5 queries per technique

Output: results/cost_analysis.json
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cost_analyzer import CostAnalyzer, load_or_estimate_stats, KNOWN_ACCURACY


# ── Tradeoff display ──────────────────────────────────────

def print_comparison_table(techniques: list):
    print("\n" + "=" * 75)
    print("  COST vs ACCURACY vs LATENCY COMPARISON")
    print("=" * 75)
    print(f"  {'Pipeline':<18} {'Accuracy':>8} {'P50 ms':>8} {'Tokens':>8} {'$/query':>12} {'$/1K q':>10}")
    print("  " + "-" * 73)

    for t in sorted(techniques, key=lambda x: x.get("accuracy", 0)):
        name     = t["pipeline"]
        acc      = t.get("accuracy", 0)
        p50      = t["latency_ms"]["p50"]
        tokens   = t["tokens_per_query"]["total"]
        cost     = t["cost_per_query_usd"]
        cost_1k  = t["cost_per_1000_queries_usd"]
        src_tag  = " (est.)" if t.get("source") == "estimated" else ""
        print(f"  {name:<18} {acc:>8.4f} {p50:>8.0f} {tokens:>8} {cost:>12.6f} {cost_1k:>10.4f}{src_tag}")

    print("=" * 75)


def print_tradeoff_recommendations(comparison: dict):
    print("\n  RECOMMENDATIONS")
    print("  " + "-" * 50)
    print(f"  Most Accurate : {comparison.get('most_accurate', 'N/A')}")
    print(f"  Fastest (P50) : {comparison.get('fastest_p50', 'N/A')}")
    print(f"  Cheapest      : {comparison.get('cheapest', 'N/A')}")
    print(f"  Best Value    : {comparison.get('best_value', 'N/A')} (accuracy / cost)")

    summary = comparison.get("tradeoff_summary", [])
    if summary:
        print("\n  USE CASE GUIDANCE")
        print("  " + "-" * 50)
        for row in reversed(summary):
            print(f"  {row['pipeline']:<18} acc={row['accuracy']:.4f}  "
                  f"{row['speed']:<8} {row['cost']:<15}  → {row['use_case']}")


def print_cost_breakdown(techniques: list):
    """Show how cost scales: 1 query / 1K queries / 10K queries / 1M queries."""
    print("\n  COST SCALING PROJECTION")
    print("  " + "-" * 60)
    print(f"  {'Pipeline':<18} {'1 query':>10} {'1K queries':>12} {'10K':>10} {'1M':>12}")
    print("  " + "-" * 60)
    for t in sorted(techniques, key=lambda x: x.get("accuracy", 0)):
        c = t["cost_per_query_usd"]
        print(f"  {t['pipeline']:<18} ${c:>9.6f} ${c*1000:>11.4f} ${c*10000:>9.2f} ${c*1000000:>11.2f}")


# ── Live measurement ──────────────────────────────────────

def run_live_measurement(n_queries: int) -> list:
    print("\n[LIVE] Initializing all pipelines for live measurement...")
    from baseline_rag import BaselineRAG
    from hybrid_rag import HybridRAG
    from reranker_rag import RerankerRAG
    from query_expansion import QueryExpansionRAG
    from cot_rag import ChainOfThoughtRAG

    pipelines = [
        ("baseline",        BaselineRAG()),
        ("hybrid",          HybridRAG()),
        ("reranked",        RerankerRAG()),
        ("query_expansion", QueryExpansionRAG(mode="combined")),
        ("cot",             ChainOfThoughtRAG(mode="structured")),
    ]

    analyzer = CostAnalyzer(n_queries=n_queries)
    results = []

    for name, rag in pipelines:
        try:
            print(f"\n[LIVE] Building {name}...")
            rag.build()
            stats = analyzer.measure_pipeline(name, rag)
            results.append(stats)
            print(f"  ✓ {name}: P50={stats['latency_ms']['p50']:.0f}ms  "
                  f"cost/q=${stats['cost_per_query_usd']:.6f}")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")

    return results


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Cost & Performance Analysis")
    parser.add_argument("--live", action="store_true",
                        help="Run live measurement (requires API key and all pipelines)")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of queries per technique for live measurement (default 10)")
    args = parser.parse_args()

    print("=" * 65)
    print("  RAG Cost & Performance Analysis")
    print("=" * 65)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY", "")
    has_api = bool(api_key and not api_key.startswith("sk-your"))

    if args.live and has_api:
        print(f"\n[MODE] Live measurement  ({args.n} queries/technique)")
        techniques = run_live_measurement(args.n)
    else:
        if args.live and not has_api:
            print("\n[WARN] --live requested but no valid API key — using estimates")
        else:
            print("\n[MODE] Estimate mode (architecture-based calculation)")
            print("       Use --live to measure real pipelines\n")
        techniques = load_or_estimate_stats(results_dir)

    if not techniques:
        print("[ERROR] No data to analyze.")
        sys.exit(1)

    # Comparison
    analyzer = CostAnalyzer()
    comparison = analyzer.compare_techniques(techniques)

    # Display
    print_comparison_table(techniques)
    print_tradeoff_recommendations(comparison)
    print_cost_breakdown(techniques)

    # Save
    output = {
        "generated": datetime.now().isoformat(),
        "mode": "live" if (args.live and has_api) else "estimated",
        "techniques": techniques,
        "comparison": comparison,
    }
    out_path = results_dir / "cost_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {out_path}")
    print("\n" + "=" * 65)
    print("  Analysis complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
