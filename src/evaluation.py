#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/evaluation.py
-----------------
Day 3-4: Ragas Evaluation Framework
Evaluate RAG pipeline with 4 core metrics:
  - Faithfulness      : Is the answer grounded in the retrieved context?
  - Answer Relevance  : Is the answer relevant to the question?
  - Context Precision : Are retrieved chunks actually useful?
  - Context Recall    : Does context contain all info needed for the answer?
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()


# ── Build Evaluation Dataset ──────────────────────────────
def build_eval_dataset(rag_pipeline, qa_path="data/squad_qa.json", n_samples=30):
    """
    Run RAG on test QA pairs and collect inputs/outputs for Ragas.
    Uses first n_samples from SQUAD as test set.
    """
    print(f"[1/3] Building evaluation dataset ({n_samples} samples)...")

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    # Use last 30% as test set (train=0-104, test=105-149)
    test_pairs = qa_pairs[-n_samples:]

    eval_data = {
        "question":          [],
        "answer":            [],
        "contexts":          [],
        "ground_truth":      [],
    }

    for i, qa in enumerate(test_pairs):
        print(f"  Processing {i+1}/{n_samples}...", end="\r")
        result = rag_pipeline.query(qa["query"], verbose=False)

        eval_data["question"].append(qa["query"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append([c["text"] for c in result["context"]])
        eval_data["ground_truth"].append(qa["answer"])

    print(f"\n  Built eval dataset with {n_samples} samples")
    return eval_data


# ── Run Ragas Evaluation ──────────────────────────────────
def run_ragas_evaluation(eval_data: dict, run_name="baseline") -> dict:
    """Run Ragas metrics on the evaluation dataset."""
    print("\n[2/3] Running Ragas evaluation...")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        print("  [WARN] No API key — returning mock metrics for structure testing")
        return mock_metrics(run_name)

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        embed_model = os.getenv("EMBED_MODEL", "text-embedding-ada-002")

        # Explicitly initialize and wrap LangChain models for Ragas
        langchain_llm = ChatOpenAI(model=llm_model, api_key=api_key)
        langchain_embeddings = OpenAIEmbeddings(model=embed_model, api_key=api_key)

        evaluator_llm = LangchainLLMWrapper(langchain_llm)
        evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

        dataset = Dataset.from_dict(eval_data)

        scores = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        def safe_mean(values):
            clean_values = [v for v in values if v is not None]
            if not clean_values:
                return 0.0
            return sum(clean_values) / len(clean_values)

        results = {
            "run_name":          run_name,
            "timestamp":         datetime.now().isoformat(),
            "n_samples":         len(eval_data["question"]),
            "faithfulness":      round(safe_mean(scores["faithfulness"]), 4),
            "answer_relevancy":  round(safe_mean(scores["answer_relevancy"]), 4),
            "context_precision": round(safe_mean(scores["context_precision"]), 4),
            "context_recall":    round(safe_mean(scores["context_recall"]), 4),
        }
        results["avg_score"] = round(sum([
            results["faithfulness"],
            results["answer_relevancy"],
            results["context_precision"],
            results["context_recall"],
        ]) / 4, 4)
        return results

    except ImportError:
        print("  [ERROR] Run: pip install ragas")
        return mock_metrics(run_name)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [ERROR] Ragas evaluation failed: {e}")
        return mock_metrics(run_name)


def mock_metrics(run_name: str) -> dict:
    """Return typical baseline metrics for structure testing."""
    return {
        "run_name":          run_name,
        "timestamp":         datetime.now().isoformat(),
        "n_samples":         30,
        "faithfulness":      0.68,
        "answer_relevancy":  0.65,
        "context_precision": 0.60,
        "context_recall":    0.55,
        "avg_score":         0.62,
        "note":              "MOCK - run with real OPENAI_API_KEY for actual scores",
    }


# ── Save & Display Results ────────────────────────────────
def save_results(metrics: dict, out_dir="results"):
    """Save evaluation results to JSON."""
    out = Path(out_dir) / f"{metrics['run_name']}_metrics.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [SAVED] {out}")
    return out


def display_results(metrics: dict):
    """Print a formatted results table."""
    print("\n" + "=" * 55)
    print(f"  EVALUATION RESULTS: {metrics['run_name'].upper()}")
    print("=" * 55)
    print(f"  Samples evaluated  : {metrics['n_samples']}")
    print(f"  Timestamp          : {metrics['timestamp'][:19]}")
    print()
    print(f"  Faithfulness       : {metrics['faithfulness']:.4f}")
    print(f"  Answer Relevancy   : {metrics['answer_relevancy']:.4f}")
    print(f"  Context Precision  : {metrics['context_precision']:.4f}")
    print(f"  Context Recall     : {metrics['context_recall']:.4f}")
    print(f"  {'─'*35}")
    print(f"  Average Score      : {metrics['avg_score']:.4f}")

    if metrics.get("note"):
        print(f"\n  NOTE: {metrics['note']}")
    print("=" * 55)


# ── Compare Runs ──────────────────────────────────────────
def compare_runs(results_dir="results"):
    """Load all saved metrics and print comparison table."""
    result_files = sorted(Path(results_dir).glob("*_metrics.json"))
    if not result_files:
        print("No results found in ./results/")
        return

    runs = []
    for f in result_files:
        with open(f) as fp:
            runs.append(json.load(fp))

    # Sort by avg_score descending
    runs.sort(key=lambda x: x.get("avg_score", 0), reverse=True)

    print("\n" + "=" * 75)
    print("  TECHNIQUE COMPARISON TABLE")
    print("=" * 75)
    print(f"  {'Run':<25} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Recall':>7} {'AVG':>7}")
    print(f"  {'─'*25} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

    for r in runs:
        print(
            f"  {r['run_name']:<25} "
            f"{r['faithfulness']:>7.4f} "
            f"{r['answer_relevancy']:>7.4f} "
            f"{r['context_precision']:>7.4f} "
            f"{r['context_recall']:>7.4f} "
            f"{r['avg_score']:>7.4f}"
        )
    print("=" * 75)


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    from baseline_rag import BaselineRAG

    print("=" * 55)
    print("  RAG Evaluation Framework -- Day 3-4")
    print("=" * 55)

    # Build pipeline
    rag = BaselineRAG().build()

    # Build eval dataset
    eval_data = build_eval_dataset(rag, n_samples=30)

    # Run evaluation
    metrics = run_ragas_evaluation(eval_data, run_name="baseline")

    # Display + save
    display_results(metrics)
    save_results(metrics)

    # Show comparison (just baseline for now)
    compare_runs()
