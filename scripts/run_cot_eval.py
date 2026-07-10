#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cot_eval.py
----------------
Day 22-24: Evaluate the Chain-of-Thought RAG pipeline.
Runs 30 SQUAD test samples and compares against all previous techniques.

Usage:
    python run_cot_eval.py                    # structured mode (default)
    python run_cot_eval.py --mode simple
    python run_cot_eval.py --mode citation
    python run_cot_eval.py --mode structured
"""

import sys
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cot_rag import ChainOfThoughtRAG
from evaluation import (
    build_eval_dataset,
    run_ragas_evaluation,
    save_results,
    display_results,
    compare_runs,
)


def main(mode: str = "structured"):
    print("=" * 65)
    print(f"  Chain-of-Thought RAG Evaluation -- Day 22-24  (mode={mode})")
    print("=" * 65)

    # 1. Build pipeline
    print("\n[STEP 1] Initializing ChainOfThoughtRAG pipeline...")
    rag = ChainOfThoughtRAG(mode=mode)
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
    run_name = f"cot_{mode}"
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
        choices=["structured", "simple", "citation"],
        default="structured",
        help="CoT generation mode (default: structured)",
    )
    args = parser.parse_args()
    main(mode=args.mode)
