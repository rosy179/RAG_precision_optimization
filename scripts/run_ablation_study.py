#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation_study.py
----------------------
Ablation study: test technique combinations to find the optimal setup.

Tests which components contribute most when combined or skipped.

Configurations tested:
  (A) Already measured (loaded from results/):
      baseline, hybrid, reranked, query_expansion, cot

  (B) New combinations:
      hybrid_cot          — Hybrid retrieval + CoT generation (skip reranker)
      reranker_only_cot   — Baseline → Reranker → CoT (skip hybrid BM25)
      full_stack          — Hybrid + Reranker + CoT (= current best)

Usage:
    python run_ablation_study.py             # use mock scores (no API key)
    python run_ablation_study.py --live      # run live evaluation

Output: results/ablation_study.json
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Known individual results ──────────────────────────────

KNOWN_SCORES = {
    "baseline":        {"faithfulness": 0.8389, "answer_relevancy": 0.8405,
                        "context_precision": 0.9000, "context_recall": 0.9333, "avg": 0.8782},
    "hybrid":          {"faithfulness": 0.8833, "answer_relevancy": 0.8717,
                        "context_precision": 0.8778, "context_recall": 0.9667, "avg": 0.8999},
    "reranked":        {"faithfulness": 0.8389, "answer_relevancy": 0.8755,
                        "context_precision": 0.9639, "context_recall": 1.0000, "avg": 0.9196},
    "query_expansion": {"faithfulness": 0.8222, "answer_relevancy": 0.8423,
                        "context_precision": 0.9639, "context_recall": 1.0000, "avg": 0.9071},
    "cot":             {"faithfulness": 0.9000, "answer_relevancy": 0.9097,
                        "context_precision": 0.9639, "context_recall": 1.0000, "avg": 0.9434},
}


# ── Mock evaluation for estimate mode ────────────────────

def estimate_combo_score(components: list) -> dict:
    """
    Estimate combined score using a simple additive model.
    Each component adds a portion of its individual gain over baseline.
    This is a rough estimate — run --live for real numbers.
    """
    base = KNOWN_SCORES["baseline"].copy()

    gains = {
        "hybrid":          {k: KNOWN_SCORES["hybrid"][k] - KNOWN_SCORES["baseline"][k]
                            for k in base},
        "reranker":        {k: KNOWN_SCORES["reranked"][k] - KNOWN_SCORES["hybrid"][k]
                            for k in base},
        "query_expansion": {k: KNOWN_SCORES["query_expansion"][k] - KNOWN_SCORES["reranked"][k]
                            for k in base},
        "cot":             {k: KNOWN_SCORES["cot"][k] - KNOWN_SCORES["reranked"][k]
                            for k in base},
    }

    result = base.copy()
    for comp in components:
        g = gains.get(comp, {})
        for metric in result:
            # Diminishing returns: each extra technique contributes 85% of its marginal gain
            result[metric] = min(1.0, result[metric] + g.get(metric, 0) * 0.85)

    result["avg"] = round(
        (result["faithfulness"] + result["answer_relevancy"] +
         result["context_precision"] + result["context_recall"]) / 4, 4
    )
    for k in result:
        result[k] = round(result[k], 4)
    return result


# ── Build and evaluate a live combination ────────────────

def evaluate_live(pipeline_name: str, rag_pipeline, n_samples: int = 30) -> dict:
    """Build + evaluate a live pipeline. Returns metrics dict."""
    from evaluation import build_eval_dataset, run_ragas_evaluation
    print(f"\n  [LIVE] Evaluating: {pipeline_name}...")
    rag_pipeline.build()
    eval_data = build_eval_dataset(rag_pipeline, qa_path="data/squad_qa.json",
                                   n_samples=n_samples)
    metrics = run_ragas_evaluation(eval_data, run_name=pipeline_name)
    return metrics


def run_live_combinations(n_samples: int) -> list:
    from baseline_rag import BaselineRAG
    from hybrid_rag import HybridRAG
    from reranker_rag import RerankerRAG
    from cot_rag import ChainOfThoughtRAG

    # Custom combination: Hybrid retrieval + CoT, skipping CrossEncoder reranker
    class HybridCoTRAG:
        """HybridRAG retrieval + CoT generation (no reranker)."""

        def build(self):
            self._hybrid = HybridRAG()
            self._hybrid.build()
            self.mode = "structured"
            return self

        def query(self, question, verbose=False):
            # Get hybrid results (no reranking)
            hybrid_result = self._hybrid.query(question, verbose=verbose)
            context = hybrid_result["context"]
            # Apply CoT generation
            from cot_rag import generate_cot_answer, _build_context_string
            ranked_ctx = [{"rank": i+1, **c} for i, c in enumerate(context)]
            answer, _ = generate_cot_answer(question, ranked_ctx, mode=self.mode)
            return {"answer": answer, "context": context}

    class BaselineRerankerCoTRAG:
        """Baseline semantic retrieval + Reranker + CoT (no hybrid BM25)."""

        def build(self):
            # RerankerRAG already builds baseline semantic + reranker
            self._reranker = RerankerRAG()
            self._reranker.build()
            self.mode = "structured"
            return self

        def query(self, question, verbose=False):
            reranked = self._reranker.query(question, verbose=verbose)
            from cot_rag import generate_cot_answer
            answer, _ = generate_cot_answer(question, reranked["context"], mode=self.mode)
            return {"answer": answer, "context": reranked["context"]}

    combos = [
        ("hybrid_cot",           HybridCoTRAG()),
        ("reranker_only_cot",    BaselineRerankerCoTRAG()),
        ("full_stack",           ChainOfThoughtRAG(mode="structured")),  # = cot
    ]

    results = []
    for name, rag in combos:
        try:
            metrics = evaluate_live(name, rag, n_samples)
            results.append({"name": name, "metrics": metrics, "source": "live"})
        except Exception as e:
            print(f"  [WARN] {name} failed: {e}")

    return results


# ── Display ───────────────────────────────────────────────

def print_ablation_table(all_configs: list):
    print("\n" + "=" * 78)
    print("  ABLATION STUDY — TECHNIQUE COMBINATIONS")
    print("=" * 78)
    print(f"  {'Config':<22} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Recall':>7} {'AVG':>7}  Source")
    print("  " + "-" * 76)

    for cfg in sorted(all_configs, key=lambda x: x["metrics"].get("avg", 0)):
        m    = cfg["metrics"]
        name = cfg["name"]
        src  = cfg.get("source", "known")
        tag  = "" if src == "live" else " (est.)" if src == "estimated" else ""
        print(f"  {name:<22} {m.get('faithfulness',0):>7.4f} {m.get('answer_relevancy',0):>7.4f} "
              f"{m.get('context_precision',0):>7.4f} {m.get('context_recall',0):>7.4f} "
              f"{m.get('avg',0):>7.4f}{tag}")

    print("=" * 78)


def print_insights(all_configs: list):
    print("\n  KEY FINDINGS")
    print("  " + "-" * 50)
    sorted_cfgs = sorted(all_configs, key=lambda x: -x["metrics"].get("avg", 0))
    best = sorted_cfgs[0]
    worst = sorted_cfgs[-1]
    baseline = next((c for c in all_configs if c["name"] == "baseline"), None)

    print(f"  Best overall       : {best['name']} (avg={best['metrics']['avg']:.4f})")
    print(f"  Worst              : {worst['name']} (avg={worst['metrics']['avg']:.4f})")

    if baseline:
        gain = best["metrics"]["avg"] - baseline["metrics"]["avg"]
        print(f"  Total gain vs base : +{gain:.4f} ({gain*100:.2f}%)")

    # Find if hybrid+CoT without reranker is competitive
    hc = next((c for c in all_configs if c["name"] == "hybrid_cot"), None)
    rc = next((c for c in all_configs if c["name"] == "reranked"), None)
    if hc and rc:
        diff = hc["metrics"]["avg"] - rc["metrics"]["avg"]
        arrow = ">" if diff > 0 else "<"
        print(f"  hybrid+CoT vs reranked: {arrow} ({diff:+.4f}) "
              f"— {'CoT compensates for reranker' if diff > 0 else 'reranker adds value beyond CoT'}")


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Ablation Study")
    parser.add_argument("--live", action="store_true",
                        help="Run live evaluation (needs API key)")
    parser.add_argument("--n", type=int, default=30,
                        help="Number of test samples for live eval (default 30)")
    args = parser.parse_args()

    print("=" * 65)
    print("  RAG Ablation Study")
    print("=" * 65)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY", "")
    has_api = bool(api_key and not api_key.startswith("sk-your"))

    # 1. Load known individual scores
    all_configs = [
        {"name": name, "metrics": metrics, "source": "known",
         "components": [name]}
        for name, metrics in KNOWN_SCORES.items()
    ]

    # 2. Add combination estimates or live results
    if args.live and has_api:
        print("\n[MODE] Live evaluation")
        live_results = run_live_combinations(args.n)
        all_configs.extend(live_results)
    else:
        if args.live and not has_api:
            print("\n[WARN] --live requested but no valid API key — using estimates")
        else:
            print("\n[MODE] Estimate mode (no API calls)")

        combos = [
            ("hybrid_cot",        ["hybrid", "cot"],                   "estimated"),
            ("reranker_only_cot", ["reranker", "cot"],                  "estimated"),
            ("full_stack",        ["hybrid", "reranker", "cot"],        "estimated"),
            ("all_techniques",    ["hybrid", "reranker", "query_expansion", "cot"], "estimated"),
        ]
        for name, components, src in combos:
            estimated = estimate_combo_score(components)
            all_configs.append({"name": name, "metrics": estimated,
                                 "source": src, "components": components})

    # Display
    print_ablation_table(all_configs)
    print_insights(all_configs)

    # Save
    output = {
        "generated":   datetime.now().isoformat(),
        "mode":        "live" if (args.live and has_api) else "estimated",
        "configs":     all_configs,
        "recommendation": sorted(all_configs, key=lambda x: -x["metrics"].get("avg", 0))[0]["name"],
    }
    out_path = results_dir / "ablation_study.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[SAVED] {out_path}")
    print("\n" + "=" * 65)
    print("  Ablation study complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
