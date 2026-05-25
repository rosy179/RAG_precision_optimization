#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/adaptive_rag.py
--------------------
Day 18-20: Adaptive Retrieval — Dynamic top_k and retrieve_depth

Problem this solves:
  Fixed top_k=3 is a compromise:
    - Too many for simple factual questions ("When was X born?") → noise in context
    - Too few for complex multi-part questions ("Explain + compare + analyze...") → missing info

Solution:
  Classify query complexity → route to a different retrieval config.

  Simple  (factual, short):  retrieve_depth=20, top_k=3, no query expansion
  Medium  (1-2 sub-concepts): retrieve_depth=20, top_k=4
  Complex (multi-hop, why/how/compare): retrieve_depth=30, top_k=5 + Multi-Query expansion

Classification:
  Heuristic (default, no API call, fast):
    - Keyword signals: "why", "how", "compare", "explain", "analyze" → complex
    - Short simple question words: "who", "when", "where", "what is" → simple
    - Word count: <8 words → bias simple; >20 words → bias complex

  LLM (accurate, costs 1 extra API call):
    - Ask GPT-4o-mini to score complexity 1-3
    - Use when heuristic is ambiguous

Architecture:
  Built on ChainOfThoughtRAG (best stack: Hybrid + CrossEncoder + CoT)
  Only the retrieval config changes — same generation layer
"""

import sys
import os
import re
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────
LLM_MODEL  = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K      = int(os.getenv("TOP_K", 3))

# ── Complexity Tiers ──────────────────────────────────────
TIERS = {
    "simple": {
        "top_k":          3,
        "retrieve_depth": 20,
        "use_expansion":  False,
        "label":          "simple (factual)",
    },
    "medium": {
        "top_k":          4,
        "retrieve_depth": 20,
        "use_expansion":  False,
        "label":          "medium (conceptual)",
    },
    "complex": {
        "top_k":          5,
        "retrieve_depth": 30,
        "use_expansion":  True,   # uses Multi-Query
        "label":          "complex (multi-hop / analytical)",
    },
}

# Signals for heuristic classifier
COMPLEX_SIGNALS   = {"why", "compare", "explain", "difference", "differences",
                      "contrast", "analyze", "analyse", "relationship", "impact",
                      "implications", "advantages", "disadvantages", "pros", "cons",
                      "elaborate", "discuss", "describe in detail",
                      "how does", "how do", "how would", "how can", "how did"}
# "how many/much/long" are factual → NOT complex; bare "how" removed intentionally
SIMPLE_SIGNALS    = {"who", "when", "where", "which", "what year", "what date",
                     "how many", "how much", "how long", "is", "are", "was", "were",
                     "does", "did", "name", "list"}


# ── Query Complexity Classifier ───────────────────────────

def _signal_count(signals: set, text: str) -> int:
    """
    Count how many phrases from signals appear in text.
    Multi-word phrases: substring match.  Single words: whole-word match only
    (avoids "cons" matching inside "constructed", "discuss" inside "discussion", etc.)
    """
    count = 0
    for ph in signals:
        if " " in ph:          # multi-word phrase → substring OK
            count += ph in text
        else:                  # single word → require word boundary
            count += bool(re.search(r'\b' + re.escape(ph) + r'\b', text))
    return count


def classify_heuristic(query: str) -> str:
    """
    Fast heuristic classifier — no API call.
    Returns 'simple', 'medium', or 'complex'.
    """
    q = query.lower().strip()
    words = q.split()
    word_count = len(words)

    complex_hits = _signal_count(COMPLEX_SIGNALS, q)
    simple_hits  = _signal_count(SIMPLE_SIGNALS,  q)

    # Explicit complex triggers
    if complex_hits >= 2:
        return "complex"
    if "and" in words and word_count > 15:
        return "complex"   # long conjunctive question
    if complex_hits == 1 and word_count > 12:
        return "complex"

    # Explicit simple triggers
    if simple_hits >= 1 and complex_hits == 0 and word_count <= 12:
        return "simple"

    # Word-count tiebreak
    if word_count <= 8:
        return "simple"
    if word_count > 20:
        return "complex"
    return "medium"


def classify_llm(query: str) -> str:
    """
    LLM-based classifier — accurate, costs 1 API call.
    Returns 'simple', 'medium', or 'complex'.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return classify_heuristic(query)   # graceful fallback

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    prompt = f"""Classify the complexity of this question into exactly one category.

Categories:
  simple  — factual, single answer, short (e.g. "Who invented RAG?", "When was X founded?")
  medium  — 1-2 concepts, moderate context needed (e.g. "What is BM25?", "How does RAG work?")
  complex — multi-hop, requires comparison/analysis, multiple sub-parts
            (e.g. "Compare BM25 vs semantic search and explain when to use each")

Question: {query}

Reply with ONLY one word: simple, medium, or complex."""

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        label = resp.choices[0].message.content.strip().lower()
        return label if label in ("simple", "medium", "complex") else classify_heuristic(query)
    except Exception:
        return classify_heuristic(query)


def classify_query(query: str, method: str = "heuristic") -> tuple:
    """
    Classify query and return (tier_name, tier_config).
    method: 'heuristic' (fast) or 'llm' (accurate)
    """
    tier_name = classify_heuristic(query) if method == "heuristic" else classify_llm(query)
    return tier_name, TIERS[tier_name]


# ── Main Class ────────────────────────────────────────────

class AdaptiveRAG:
    """
    Adaptive RAG: routes each query to a different retrieval config
    based on estimated complexity.

    Built on ChainOfThoughtRAG (best retrieval + generation stack).
    Only the retrieval parameters (top_k, retrieve_depth) change per query.

    classify_method:
      "heuristic" (default) — keyword + word-count rules, no extra API call
      "llm"                 — GPT-4o-mini classifies, +1 API call per query
    """

    def __init__(self, classify_method: str = "heuristic"):
        self.classify_method = classify_method
        self._cot_rag = None
        self._qe_rag  = None   # used for complex queries with expansion

    def build(self):
        src_dir = Path(__file__).parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from cot_rag import ChainOfThoughtRAG
        print("[Adaptive] Building base CoT pipeline (simple + medium queries)...")
        self._cot_rag = ChainOfThoughtRAG(mode="structured")
        self._cot_rag.build()

        # Complex queries also use Multi-Query expansion
        from query_expansion import QueryExpansionRAG
        print("[Adaptive] Building Query Expansion pipeline (complex queries)...")
        self._qe_rag = QueryExpansionRAG(mode="multi_query", n_queries=3)
        # Reuse the already-built reranker from CoT to avoid double-loading
        self._qe_rag._reranker_rag = self._cot_rag._reranker_rag
        print("[Adaptive] QueryExpansionRAG attached to existing reranker (no re-index)")

        from cot_rag import generate_cot_answer
        self._generate_cot = generate_cot_answer

        print(f"\n[Adaptive] Pipeline ready! classify_method='{self.classify_method}'")
        print("           simple  → top_k=3, depth=20, no expansion")
        print("           medium  → top_k=4, depth=20, no expansion")
        print("           complex → top_k=5, depth=30, multi_query expansion")
        return self

    def query(self, question: str, verbose: bool = True) -> dict:
        if self._cot_rag is None:
            raise RuntimeError("Call .build() first")

        # 1. Classify complexity
        tier_name, tier = classify_query(question, method=self.classify_method)
        top_k          = tier["top_k"]
        retrieve_depth = tier["retrieve_depth"]
        use_expansion  = tier["use_expansion"]

        if verbose:
            print(f"\n{'='*65}")
            print(f"Q: {question}")
            print(f"{'='*65}")
            print(f"[Adaptive] Complexity: {tier['label']} "
                  f"→ top_k={top_k}, depth={retrieve_depth}, "
                  f"expansion={'ON' if use_expansion else 'OFF'}")

        # 2. Retrieve — route by tier
        if use_expansion:
            # Complex: Multi-Query expansion → merge → CrossEncoder rerank
            context = self._qe_rag.retrieve_expanded(
                question, top_k=top_k, retrieve_depth=retrieve_depth
            )
        else:
            # Simple / Medium: standard hybrid + rerank
            context = self._cot_rag._reranker_rag.retrieve_and_rerank(
                question, top_k=top_k, retrieve_depth=retrieve_depth
            )

        # 3. Generate with CoT
        from cot_rag import generate_cot_answer, LLM_MODEL
        final_answer, full_reasoning = generate_cot_answer(
            question, context, model=LLM_MODEL, mode="structured"
        )

        result = {
            "question":    question,
            "answer":      final_answer,
            "reasoning":   full_reasoning,
            "context":     context,
            "top_k":       top_k,
            "complexity":  tier_name,
            "tier_config": tier["label"],
            "expansion":   use_expansion,
        }

        if verbose:
            print(f"\nA: {final_answer}")
            print(f"\n--- Top {len(context)} passages (adaptive tier={tier_name}) ---")
            for c in context:
                rer = c.get("reranker_score", "N/A")
                rer_str = f"{rer:.3f}" if isinstance(rer, float) else str(rer)
                print(f"  [{c['rank']}] {c['title']} (reranker={rer_str})")
                print(f"       {c['text'][:110]}...")

        return result


# ── Quick Test ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Adaptive RAG Pipeline -- Day 18-20")
    print("=" * 65)

    rag = AdaptiveRAG(classify_method="heuristic").build()

    # Mix of simple, medium, complex questions
    test_questions = [
        # Simple — factual, short
        "Who invented the Transformer architecture?",
        "What year was BERT released?",
        # Medium — conceptual
        "What is Retrieval Augmented Generation?",
        "How does dense passage retrieval work?",
        # Complex — multi-hop, analytical
        "How does RAG compare to fine-tuning and what are the trade-offs for production systems?",
        "Why does HyDE improve retrieval for some query types but fail for factual questions?",
    ]

    results = []
    for q in test_questions:
        r = rag.query(q)
        results.append(r)

    out = Path("results/adaptive_test_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {out}")
    print("\nAdaptive RAG pipeline working!")
