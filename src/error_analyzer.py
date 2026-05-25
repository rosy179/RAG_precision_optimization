#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/error_analyzer.py
---------------------
Error Analysis for RAG pipelines.

Analyzes test results to classify failure modes:
  - hallucination   : answer contains facts NOT found in retrieved context
  - low_relevance   : retrieved context doesn't cover the answer
  - incomplete      : answer is missing key facts from ground truth
  - correct         : answer matches ground truth well enough

Difficulty classification (heuristic):
  - simple  : single-word or short answers, factual lookups
  - medium  : 3-10 word answers, require some reasoning
  - hard    : long answers, multi-hop, or domain-specific
"""

import sys
import re
import json
import string
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Text utilities ────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def _word_set(text: str) -> set:
    return set(_normalize(text).split())


def _token_overlap(a: str, b: str) -> float:
    """F1-style token overlap between two strings."""
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return 0.0
    common = wa & wb
    if not common:
        return 0.0
    precision = len(common) / len(wa)
    recall    = len(common) / len(wb)
    return 2 * precision * recall / (precision + recall)


def _answer_in_context(answer: str, context_chunks: list) -> bool:
    """Check if key answer terms appear somewhere in the retrieved context."""
    combined_context = " ".join(c.get("text", "") for c in context_chunks)
    answer_words = _word_set(answer)
    # Remove stopwords for the check
    stopwords = {"the", "a", "an", "is", "was", "are", "were", "of", "in",
                 "to", "and", "or", "it", "its", "this", "that", "with", "for"}
    key_words = answer_words - stopwords
    if not key_words:
        return True  # can't determine, assume ok
    context_words = _word_set(combined_context)
    return len(key_words & context_words) / len(key_words) > 0.5


# ── Difficulty classification ─────────────────────────────

def classify_difficulty(question: str, ground_truth: str) -> str:
    """
    Heuristic difficulty classification based on answer length and question complexity.
    simple  → factual, short answer (1-3 words)
    medium  → moderate answer (4-15 words)
    hard    → long answer or multi-part question (16+ words)
    """
    answer_words = len(ground_truth.split())
    question_lower = question.lower()

    # Hard: long answers or complex question patterns
    if answer_words >= 16:
        return "hard"

    multi_hop_patterns = ["how many", "why", "explain", "describe", "what are the",
                          "compare", "relationship", "difference between"]
    if any(p in question_lower for p in multi_hop_patterns):
        if answer_words >= 8:
            return "hard"
        return "medium"

    # Simple: very short answers (who/what/when questions with short answers)
    if answer_words <= 3:
        return "simple"

    return "medium"


# ── Error classification ──────────────────────────────────

def classify_error(question: str, answer: str, ground_truth: str,
                   context_chunks: list) -> dict:
    """
    Classify a single Q&A result into an error category.

    Returns:
        {
          "category": "correct" | "incomplete" | "low_relevance" | "hallucination",
          "f1_score": float,
          "answer_in_context": bool,
          "details": str
        }
    """
    f1 = _token_overlap(answer, ground_truth)
    answer_in_ctx = _answer_in_context(answer, context_chunks)

    if f1 >= 0.6:
        return {
            "category":         "correct",
            "f1_score":         round(f1, 4),
            "answer_in_context": answer_in_ctx,
            "details":          "Answer sufficiently matches ground truth",
        }

    if not answer_in_ctx:
        # The answer mentions things not found in context → hallucination risk
        gt_in_ctx = _answer_in_context(ground_truth, context_chunks)
        if gt_in_ctx:
            # Ground truth IS in context but answer went elsewhere → hallucination
            return {
                "category":         "hallucination",
                "f1_score":         round(f1, 4),
                "answer_in_context": False,
                "details":          "Answer not grounded in context; ground truth IS in context",
            }
        else:
            # Neither answer nor ground truth in context → retrieval missed it
            return {
                "category":         "low_relevance",
                "f1_score":         round(f1, 4),
                "answer_in_context": False,
                "details":          "Retrieved context does not contain relevant information",
            }

    # Answer is in context but doesn't match well → incomplete
    return {
        "category":         "incomplete",
        "f1_score":         round(f1, 4),
        "answer_in_context": True,
        "details":          "Answer is grounded but missing key facts from ground truth",
    }


# ── Main Analyzer ─────────────────────────────────────────

class ErrorAnalyzer:
    """Analyze RAG test results for failure patterns."""

    def __init__(self, qa_path: str = "data/squad_qa.json", n_test: int = 30):
        self.qa_path = Path(qa_path)
        self.n_test = n_test
        self._ground_truths: dict = {}

    def _load_ground_truths(self):
        if self._ground_truths:
            return
        if not self.qa_path.exists():
            return
        with open(self.qa_path, "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)
        test_pairs = qa_pairs[-self.n_test:]
        self._ground_truths = {q["query"]: q["answer"] for q in test_pairs}

    def analyze_results(self, test_results: list, pipeline_name: str) -> dict:
        """
        Analyze a list of test result dicts (from results/*_test_results.json).
        Returns a structured error report.
        """
        self._load_ground_truths()

        by_difficulty = defaultdict(lambda: {"correct": 0, "total": 0, "samples": []})
        by_category   = defaultdict(list)
        per_sample    = []

        for item in test_results:
            question    = item.get("question", "")
            answer      = item.get("answer", "")
            context     = item.get("context", [])
            gt          = self._ground_truths.get(question, "")

            if not gt:
                continue

            difficulty = classify_difficulty(question, gt)
            result     = classify_error(question, answer, gt, context)

            sample_record = {
                "question":   question,
                "answer":     answer,
                "ground_truth": gt,
                "difficulty": difficulty,
                **result,
            }
            per_sample.append(sample_record)
            by_difficulty[difficulty]["total"] += 1
            by_category[result["category"]].append(sample_record)

            if result["category"] == "correct":
                by_difficulty[difficulty]["correct"] += 1

        # Aggregate
        n_total = len(per_sample)
        category_summary = {}
        for cat in ("correct", "incomplete", "low_relevance", "hallucination"):
            count = len(by_category[cat])
            category_summary[cat] = {
                "count":      count,
                "percentage": round(100 * count / n_total, 1) if n_total else 0,
                "examples":   [s["question"][:80] for s in by_category[cat][:3]],
            }

        difficulty_summary = {}
        for diff in ("simple", "medium", "hard"):
            d = by_difficulty[diff]
            total = d["total"]
            correct = d["correct"]
            difficulty_summary[diff] = {
                "count":    total,
                "accuracy": round(correct / total, 4) if total else 0,
            }

        avg_f1 = (sum(s["f1_score"] for s in per_sample) / n_total) if n_total else 0

        return {
            "pipeline":           pipeline_name,
            "n_analyzed":         n_total,
            "avg_f1":             round(avg_f1, 4),
            "category_breakdown": category_summary,
            "difficulty_breakdown": difficulty_summary,
            "per_sample":         per_sample,
        }

    def compare_pipelines(self, reports: list) -> dict:
        """Compare error profiles across techniques."""
        comparison = {}
        for report in reports:
            name = report["pipeline"]
            comparison[name] = {
                "avg_f1":          report["avg_f1"],
                "correct_pct":     report["category_breakdown"].get("correct", {}).get("percentage", 0),
                "hallucination_pct": report["category_breakdown"].get("hallucination", {}).get("percentage", 0),
                "low_relevance_pct": report["category_breakdown"].get("low_relevance", {}).get("percentage", 0),
                "incomplete_pct":  report["category_breakdown"].get("incomplete", {}).get("percentage", 0),
            }
        return comparison

    def load_test_results(self, results_dir: Path = Path("results")) -> dict:
        """Load all available test result files."""
        files = {
            "baseline":        "baseline_test_results.json",
            "hybrid":          "hybrid_test_results.json",
            "reranked":        "reranker_test_results.json",
            "query_expansion": "query_expansion_test_results.json",
            "cot":             "cot_test_results.json",
        }
        loaded = {}
        for name, fname in files.items():
            path = results_dir / fname
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    loaded[name] = json.load(f)
        return loaded
