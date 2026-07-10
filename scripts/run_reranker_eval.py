#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_reranker_eval.py
---------------------
Day 11-13: Evaluate the Reranker RAG pipeline.
Runs evaluation on 30 SQUAD test samples and compares
against baseline and hybrid results.

Usage:
    python run_reranker_eval.py
"""

import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add src/ to path so modules resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reranker_rag import RerankerRAG
from evaluation import (
    build_eval_dataset,
    run_ragas_evaluation,
    save_results,
    display_results,
    compare_runs,
)


def main():
    print("=" * 60)
    print("  Reranker RAG Evaluation -- Day 11-13")
    print("=" * 60)

    # 1. Build reranker pipeline
    print("\n[STEP 1] Initializing RerankerRAG pipeline...")
    rag = RerankerRAG()
    rag.build()

    # 2. Build evaluation dataset (30 SQUAD test samples)
    print("\n[STEP 2] Building evaluation dataset...")
    eval_data = build_eval_dataset(
        rag,
        qa_path="data/squad_qa.json",
        n_samples=30,
    )

    # 3. Run Ragas evaluation
    print("\n[STEP 3] Running Ragas evaluation...")
    metrics = run_ragas_evaluation(eval_data, run_name="reranked")

    # 4. Save and display results
    display_results(metrics)
    save_results(metrics, out_dir="results")

    # 5. Compare all techniques side by side
    print("\n[STEP 5] Comparing all techniques...")
    compare_runs("results")


if __name__ == "__main__":
    main()
