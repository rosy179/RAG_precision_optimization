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
        # Judge embeddings (answer_relevancy) are PINNED, independent of the
        # pipeline's EMBED_MODEL: that may be a local model the OpenAI API
        # rejects, and swapping the judge embedder changes the cosine scale
        # (ada-002 ≈0.95 vs 3-small ≈0.84 on the same paraphrase pair),
        # which would make scores incomparable with every earlier run in
        # results/. Override only deliberately, via RAGAS_EMBED_MODEL.
        embed_model = os.getenv("RAGAS_EMBED_MODEL", "text-embedding-ada-002")

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
            # Ragas marks failed rows as NaN (not None) — drop both, else a
            # single failed row poisons the whole aggregate.
            clean_values = [v for v in values if v is not None and v == v]
            return sum(clean_values) / len(clean_values) if clean_values else 0.0

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
    from hybrid_rag import HybridRAG
    from reranker_rag import RerankerRAG

    print("=" * 55)
    print("  RAG Evaluation Framework -- Day 3-4 / 8-10 / 11-13")
    print("=" * 55)

    # 1. Run Baseline Evaluation
    print("\n" + "="*55)
    print("  1. RUNNING BASELINE RAG EVALUATION")
    print("="*55)
    baseline_rag = BaselineRAG().build()
    baseline_eval_data = build_eval_dataset(baseline_rag, n_samples=30)
    baseline_metrics = run_ragas_evaluation(baseline_eval_data, run_name="baseline")
    display_results(baseline_metrics)
    save_results(baseline_metrics)

    # 2. Run Hybrid Evaluation
    print("\n" + "="*55)
    print("  2. RUNNING HYBRID RAG EVALUATION")
    print("="*55)
    hybrid_rag = HybridRAG().build()
    hybrid_eval_data = build_eval_dataset(hybrid_rag, n_samples=30)
    hybrid_metrics = run_ragas_evaluation(hybrid_eval_data, run_name="hybrid")
    display_results(hybrid_metrics)
    save_results(hybrid_metrics)

    # 3. Run Reranker Evaluation
    print("\n" + "="*55)
    print("  3. RUNNING RERANKER RAG EVALUATION")
    print("="*55)
    reranker_rag = RerankerRAG().build()
    reranker_eval_data = build_eval_dataset(reranker_rag, n_samples=30)
    reranker_metrics = run_ragas_evaluation(reranker_eval_data, run_name="reranked")
    display_results(reranker_metrics)
    save_results(reranker_metrics)

    # 4. Run Query Expansion Evaluation
    print("\n" + "="*55)
    print("  4. RUNNING QUERY EXPANSION RAG EVALUATION")
    print("="*55)
    from query_expansion import QueryExpansionRAG
    qe_rag = QueryExpansionRAG(mode="combined", n_queries=3).build()
    qe_eval_data = build_eval_dataset(qe_rag, n_samples=30)
    qe_metrics = run_ragas_evaluation(qe_eval_data, run_name="query_expansion_combined")
    display_results(qe_metrics)
    save_results(qe_metrics)

    # 5. Run Chain-of-Thought Evaluation
    print("\n" + "="*55)
    print("  5. RUNNING CHAIN-OF-THOUGHT RAG EVALUATION")
    print("="*55)
    from cot_rag import ChainOfThoughtRAG
    cot_rag = ChainOfThoughtRAG(mode="structured").build()
    cot_eval_data = build_eval_dataset(cot_rag, n_samples=30)
    cot_metrics = run_ragas_evaluation(cot_eval_data, run_name="cot_structured")
    display_results(cot_metrics)
    save_results(cot_metrics)

    # 6. Show Comparison
    compare_runs()
