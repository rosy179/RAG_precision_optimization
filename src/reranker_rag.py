#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/reranker_rag.py
--------------------
Day 11-13: Reranking with CrossEncoder
- Inherits Hybrid Search (BM25 + Semantic + RRF) from HybridRAG
- Retrieves a wider candidate pool (top 20) from hybrid search
- Applies CrossEncoder reranker to score all (query, passage) pairs
- Returns top K passages with the highest reranker score to LLM
- Expected improvement: Context Precision recover + overall +2-3%

CrossEncoder vs Bi-Encoder:
  Bi-Encoder (e.g. text-embedding-ada-002): encodes query & doc separately -> fast but less accurate
  CrossEncoder (e.g. ms-marco-MiniLM-L-6-v2): encodes (query, doc) pair jointly -> slower but much more accurate
  Strategy: Bi-Encoder for fast candidate retrieval, CrossEncoder for precise reranking.
"""

import sys
import os
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────
TOP_K          = int(os.getenv("TOP_K", 3))
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Number of candidates retrieved from hybrid search before reranking
RETRIEVE_DEPTH = int(os.getenv("RETRIEVE_DEPTH", 20))

# CrossEncoder model: fast local model (~85MB), no API key needed
# Alternatives: cross-encoder/ms-marco-MiniLM-L-12-v2 (larger, more accurate)
#               BAAI/bge-reranker-base (multilingual, high quality)
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerRAG:
    """
    Hybrid RAG + CrossEncoder Reranking.

    Pipeline:
      1. Hybrid retrieval (BM25 + Semantic + RRF) -> top RETRIEVE_DEPTH candidates
      2. CrossEncoder reranking: score each (query, passage) pair
      3. Select top_k highest-scored passages
      4. Generate answer with OpenAI LLM
    """

    def __init__(self, reranker_model: str = DEFAULT_RERANKER_MODEL):
        self.reranker_model = reranker_model
        self._hybrid = None   # HybridRAG instance (lazy-initialized in build)
        self.reranker = None  # CrossEncoder model

    def build(self):
        """Build the hybrid index and load the CrossEncoder reranker."""
        import sys
        src_dir = Path(__file__).parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from hybrid_rag import HybridRAG
        from sentence_transformers import CrossEncoder

        # 1-3. Build hybrid pipeline (ChromaDB + BM25)
        self._hybrid = HybridRAG()
        self._hybrid.build()

        # 4. Load CrossEncoder reranker (downloads model on first run, cached locally)
        print(f"\n[4.5/5] Loading CrossEncoder reranker: {self.reranker_model}")
        self.reranker = CrossEncoder(self.reranker_model)
        print(f"        Reranker loaded — ready to score (query, passage) pairs")

        print("\n[5/5] Reranker Pipeline ready!")
        return self

    def rerank(self, query: str, candidates: list, top_k: int = TOP_K) -> list:
        """
        Score each candidate passage against the query using CrossEncoder.
        Returns top_k candidates sorted by reranker_score descending.
        """
        if not candidates:
            return candidates

        # Build (query, passage) pairs for joint encoding
        pairs = [(query, c["text"]) for c in candidates]

        # CrossEncoder outputs a relevance score per pair (higher = more relevant)
        scores = self.reranker.predict(pairs)

        # Attach reranker scores and re-sort
        for i, candidate in enumerate(candidates):
            candidate["reranker_score"] = round(float(scores[i]), 6)

        reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)

        # Update rank field to reflect new order
        for i, chunk in enumerate(reranked[:top_k], 1):
            chunk["rank"] = i

        return reranked[:top_k]

    def retrieve_and_rerank(self, query: str, top_k: int = TOP_K,
                            retrieve_depth: int = RETRIEVE_DEPTH) -> list:
        """Fetch wide candidate pool from hybrid search, then rerank with CrossEncoder."""
        # Get more candidates than needed (wider net -> better reranking pool)
        candidates = self._hybrid.retrieve_hybrid(
            query, top_k=retrieve_depth, retrieve_depth=retrieve_depth
        )
        reranked = self.rerank(query, candidates, top_k=top_k)
        return reranked

    def query(self, question: str, top_k: int = TOP_K,
              retrieve_depth: int = RETRIEVE_DEPTH, verbose: bool = True) -> dict:
        """Run full pipeline: Hybrid Retrieval -> Rerank -> Generate."""
        if self._hybrid is None or self.reranker is None:
            raise RuntimeError("Call .build() first")

        # Step 1: Retrieve candidate pool from hybrid search
        context = self.retrieve_and_rerank(question, top_k=top_k,
                                           retrieve_depth=retrieve_depth)

        # Step 2: Generate answer using top-k reranked passages
        import sys
        src_dir = Path(__file__).parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from baseline_rag import generate_answer
        answer = generate_answer(question, context, model=LLM_MODEL)

        result = {
            "question": question,
            "answer":   answer,
            "context":  context,
            "top_k":    top_k,
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"Q: {question} (RERANKED)")
            print(f"{'='*60}")
            print(f"A: {answer}")
            print(f"\n--- Top {len(context)} reranked passages (from {retrieve_depth} candidates) ---")
            for c in context:
                rrf = c.get("rrf_score", "N/A")
                rer = c.get("reranker_score", "N/A")
                print(f"  [{c['rank']}] {c['title']}")
                print(f"      RRF={rrf}  |  CrossEncoder={rer}")
                print(f"      {c['text'][:120]}...")

        return result


# ── Quick Test ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Reranker RAG Pipeline -- Day 11-13")
    print("=" * 60)

    rag = RerankerRAG().build()

    test_questions = [
        "What is Retrieval Augmented Generation?",
        "How does dense passage retrieval work?",
        "What are the main components of a transformer model?",
        "What is the difference between BM25 and semantic search?",
        "How do vector databases store embeddings?",
    ]

    results = []
    for q in test_questions:
        r = rag.query(q)
        results.append(r)

    out = Path("results/reranker_test_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] Test results -> {out}")
    print("\nReranker RAG pipeline is working!")
