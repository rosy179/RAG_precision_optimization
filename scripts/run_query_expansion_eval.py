#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_query_expansion_eval.py
----------------------------
Day 15-17: Evaluate the Query Expansion pipeline (Multi-Query + HyDE).
Runs 30 SQUAD test samples in "combined" mode and compares
against baseline, hybrid, and reranked results.

Usage:
    python run_query_expansion_eval.py
    python run_query_expansion_eval.py --mode multi_query
    python run_query_expansion_eval.py --mode hyde
    python run_query_expansion_eval.py --mode combined   (default)
"""

import sys
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from query_expansion import QueryExpansionRAG
from evaluation import (
    build_eval_dataset,
    run_ragas_evaluation,
    save_results,
    display_results,
    compare_runs,
)


def main(mode: str = "combined"):
    print("=" * 65)
    print(f"  Query Expansion RAG Evaluation -- Day 15-17  (mode={mode})")
    print("=" * 65)

    # 1. Build pipeline
    print("\n[STEP 1] Initializing QueryExpansionRAG pipeline...")
    rag = QueryExpansionRAG(mode=mode, n_queries=3)
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
    run_name = f"query_expansion_{mode}"
    metrics = run_ragas_evaluation(eval_data, run_name=run_name)

    # 4. Save and display
    display_results(metrics)
    save_results(metrics, out_dir="results")

    # 5. Compare all techniques
    print("\n[STEP 5] Comparing all techniques...")
    compare_runs("results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["multi_query", "hyde", "combined"],
        default="combined",
        help="Query expansion mode (default: combined)",
    )
    args = parser.parse_args()
    main(mode=args.mode)
