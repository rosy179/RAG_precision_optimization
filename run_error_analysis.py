#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_error_analysis.py
----------------------
Analyze failure modes across all RAG techniques.

Reads existing test result files (results/*_test_results.json) and produces:
  - Category breakdown: correct / incomplete / low_relevance / hallucination
  - Difficulty breakdown: simple / medium / hard query accuracy
  - Cross-technique comparison

Usage:
    python run_error_analysis.py
    python run_error_analysis.py --pipeline cot   # single pipeline
    python run_error_analysis.py --verbose        # show example failures

Output: results/error_analysis.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from error_analyzer import ErrorAnalyzer


# ── Display helpers ───────────────────────────────────────

CATEGORY_LABELS = {
    "correct":       "Correct      ✓",
    "incomplete":    "Incomplete   ~",
    "low_relevance": "Low Relevance ○",
    "hallucination": "Hallucination ✗",
}

DIFFICULTY_ICONS = {"simple": "🟢", "medium": "🟡", "hard": "🔴"}


def print_pipeline_report(report: dict, verbose: bool = False):
    name = report["pipeline"]
    n    = report["n_analyzed"]
    f1   = report["avg_f1"]

    print(f"\n  ── {name.upper()} (n={n}, avg F1={f1:.4f}) ──")

    print("  Error categories:")
    for cat, label in CATEGORY_LABELS.items():
        info = report["category_breakdown"].get(cat, {})
        count = info.get("count", 0)
        pct   = info.get("percentage", 0)
        bar   = "█" * int(pct / 5)
        print(f"    {label:<22} {count:>3}  ({pct:>5.1f}%)  {bar}")

    print("  Difficulty breakdown:")
    for diff in ("simple", "medium", "hard"):
        info = report["difficulty_breakdown"].get(diff, {})
        acc  = info.get("accuracy", 0)
        cnt  = info.get("count", 0)
        icon = DIFFICULTY_ICONS.get(diff, "")
        print(f"    {icon} {diff:<8}  count={cnt:>3}  accuracy={acc:.4f}")

    if verbose:
        failures = [s for s in report.get("per_sample", []) if s["category"] != "correct"]
        if failures:
            print(f"\n  Failure examples (first 3):")
            for s in failures[:3]:
                print(f"    [{s['category']}] {s['question'][:70]}")
                print(f"      Expected: {s['ground_truth'][:60]}")
                print(f"      Got:      {s['answer'][:60]}")
                print(f"      F1: {s['f1_score']:.4f}  Difficulty: {s['difficulty']}")


def print_comparison(comparison: dict):
    print("\n" + "=" * 70)
    print("  CROSS-TECHNIQUE ERROR COMPARISON")
    print("=" * 70)
    print(f"  {'Pipeline':<18} {'F1':>6} {'Correct%':>9} {'Halluc%':>9} {'LowRel%':>9} {'Incompl%':>9}")
    print("  " + "-" * 68)
    for name, stats in sorted(comparison.items(), key=lambda x: -x[1]["avg_f1"]):
        print(f"  {name:<18} {stats['avg_f1']:>6.4f} {stats['correct_pct']:>9.1f} "
              f"{stats['hallucination_pct']:>9.1f} {stats['low_relevance_pct']:>9.1f} "
              f"{stats['incomplete_pct']:>9.1f}")
    print("=" * 70)


def print_key_insights(comparison: dict, reports: list):
    print("\n  KEY INSIGHTS")
    print("  " + "-" * 50)

    # Technique with lowest hallucination
    by_halluc = sorted(comparison.items(), key=lambda x: x[1]["hallucination_pct"])
    print(f"  Lowest hallucination : {by_halluc[0][0]} ({by_halluc[0][1]['hallucination_pct']:.1f}%)")

    # Technique with highest correct %
    by_correct = sorted(comparison.items(), key=lambda x: -x[1]["correct_pct"])
    print(f"  Highest correct rate : {by_correct[0][0]} ({by_correct[0][1]['correct_pct']:.1f}%)")

    # Hard query accuracy
    print("\n  Hard query accuracy per technique:")
    for report in sorted(reports, key=lambda r: r["pipeline"]):
        hard = report["difficulty_breakdown"].get("hard", {})
        acc  = hard.get("accuracy", 0)
        cnt  = hard.get("count", 0)
        print(f"    {report['pipeline']:<18}  {acc:.4f}  (n={cnt})")


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Error Analysis")
    parser.add_argument("--pipeline", type=str, default=None,
                        help="Analyze a single pipeline only (e.g. cot, hybrid)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Show example failure cases per pipeline")
    args = parser.parse_args()

    print("=" * 65)
    print("  RAG Error Analysis")
    print("=" * 65)

    results_dir = Path("results")
    analyzer    = ErrorAnalyzer(qa_path="data/squad_qa.json", n_test=30)

    # Load test results
    all_results = analyzer.load_test_results(results_dir)

    if not all_results:
        print("[ERROR] No test result files found in results/")
        print("        Run evaluation scripts first:")
        print("        python run_cot_eval.py")
        sys.exit(1)

    if args.pipeline:
        if args.pipeline not in all_results:
            print(f"[ERROR] Pipeline '{args.pipeline}' not found.")
            print(f"        Available: {list(all_results.keys())}")
            sys.exit(1)
        all_results = {args.pipeline: all_results[args.pipeline]}

    # Analyze each pipeline
    reports = []
    for name, results in all_results.items():
        print(f"\n[Analyzing] {name} ({len(results)} samples)...")
        report = analyzer.analyze_results(results, pipeline_name=name)
        reports.append(report)
        print_pipeline_report(report, verbose=args.verbose)

    # Cross-technique comparison
    comparison = {}
    if len(reports) > 1:
        comparison = analyzer.compare_pipelines(reports)
        print_comparison(comparison)
        print_key_insights(comparison, reports)

    # Save
    output = {
        "generated":  datetime.now().isoformat(),
        "n_pipelines": len(reports),
        "reports":    [{k: v for k, v in r.items() if k != "per_sample"} for r in reports],
        "per_sample_detail": {r["pipeline"]: r.get("per_sample", []) for r in reports},
        "comparison": comparison,
    }

    out_path = results_dir / "error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {out_path}")
    print("\n" + "=" * 65)
    print("  Error analysis complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
