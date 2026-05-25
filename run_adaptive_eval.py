#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_adaptive_eval.py
---------------------
Day 18-20: Evaluate the Adaptive Retrieval pipeline.
Runs 30 SQUAD test samples and compares against all previous techniques.

Usage:
    python run_adaptive_eval.py                       # heuristic classifier (default)
    python run_adaptive_eval.py --classify llm        # LLM classifier (+1 API call/query)
    python run_adaptive_eval.py --show-tiers          # show complexity distribution only
"""

import sys
import json
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adaptive_rag import AdaptiveRAG, classify_query
from evaluation import (
    build_eval_dataset,
    run_ragas_evaluation,
    save_results,
    display_results,
    compare_runs,
)


def show_tier_distribution(qa_path: str = "data/squad_qa.json",
                            n_samples: int = 30,
                            method: str = "heuristic"):
    """Print the complexity distribution for the test set."""
    import json
    with open(qa_path, encoding="utf-8") as f:
        pairs = json.load(f)
    test = pairs[-n_samples:]

    counts = {"simple": 0, "medium": 0, "complex": 0}
    print(f"\n{'='*55}")
    print(f"  QUERY COMPLEXITY DISTRIBUTION ({method})")
    print(f"{'='*55}")
    for qa in test:
        tier, cfg = classify_query(qa["query"], method=method)
        counts[tier] += 1
        print(f"  [{tier:<8}] {qa['query'][:65]}")
    print(f"\n  Summary: simple={counts['simple']} "
          f"/ medium={counts['medium']} "
          f"/ complex={counts['complex']}")
    print(f"{'='*55}")


def main(classify_method: str = "heuristic", show_tiers: bool = False):
    if show_tiers:
        show_tier_distribution(method=classify_method)
        return

    print("=" * 65)
    print(f"  Adaptive RAG Evaluation -- Day 18-20 "
          f"(classify={classify_method})")
    print("=" * 65)

    # 1. Build pipeline
    print("\n[STEP 1] Initializing AdaptiveRAG pipeline...")
    rag = AdaptiveRAG(classify_method=classify_method)
    rag.build()

    # 2. Build evaluation dataset
    print("\n[STEP 2] Building evaluation dataset (30 samples)...")
    eval_data = build_eval_dataset(
        rag,
        qa_path="data/squad_qa.json",
        n_samples=30,
    )

    # 3. Ragas evaluation
    print("\n[STEP 3] Running Ragas evaluation...")
    run_name = f"adaptive_{classify_method}"
    metrics = run_ragas_evaluation(eval_data, run_name=run_name)

    # 4. Display + save
    display_results(metrics)
    save_results(metrics, out_dir="results")

    # 5. Show complexity distribution
    show_tier_distribution(method=classify_method)

    # 6. Compare all techniques
    print("\n[STEP 6] Comparing all techniques...")
    compare_runs("results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classify",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="Complexity classification method (default: heuristic)"
    )
    parser.add_argument(
        "--show-tiers",
        action="store_true",
        help="Just show complexity distribution, don't run evaluation"
    )
    args = parser.parse_args()
    main(classify_method=args.classify, show_tiers=args.show_tiers)
