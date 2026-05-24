#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/query_expansion.py
-----------------------
Day 15-17: Query Expansion — Multi-Query + HyDE

Problem this solves:
  A single query might not capture all the angles needed to retrieve the right documents.
  "How does RAG work?" misses documents about "retrieval-augmented generation architecture".

Two techniques:
  1. Multi-Query: LLM rephrases the question N ways → retrieve for each → merge + dedup
  2. HyDE (Hypothetical Document Embeddings): LLM generates a *hypothetical answer* →
     embed that answer as the retrieval query (answer embeddings are closer to
     document embeddings than question embeddings are)

Combined pipeline:
  Original Query
      ↓ (Multi-Query expansion or HyDE)
  [Variant₁, Variant₂, ..., VariantN]
      ↓ Hybrid Retrieval for each variant (BM25 + Semantic + RRF)
  Merged + Deduplicated Candidates (up to ~60)
      ↓ CrossEncoder Reranking
  Top K passages
      ↓ LLM Generation
  Answer
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
RETRIEVE_DEPTH = int(os.getenv("RETRIEVE_DEPTH", 20))
N_QUERIES      = int(os.getenv("N_QUERIES", 3))   # Multi-query: number of variants to generate

MULTI_QUERY_PROMPT = """\
Generate {n} different ways to ask the following question. \
Vary the phrasing, synonyms, and focus while keeping the same meaning.
Return ONLY the rephrased questions, one per line. No numbering, no extra text.

Original question: {query}

Rephrased questions:"""

HYDE_PROMPT = """\
Write a short, factual paragraph (2-4 sentences) that directly answers the following question. \
Write as if it were an excerpt from a technical article or documentation.

Question: {query}

Answer paragraph:"""


# ── LLM Helpers ───────────────────────────────────────────
def _call_llm(prompt: str, model: str = LLM_MODEL, max_tokens: int = 300) -> str:
    """Call OpenAI and return the text response."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return ""   # caller handles the empty string (fallback to original query)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def expand_multi_query(query: str, n: int = N_QUERIES) -> list:
    """
    Generate N alternative phrasings of the query using LLM.
    Returns list: [original_query, variant1, variant2, ..., variantN]
    Falls back to [original_query] if API unavailable.
    """
    prompt = MULTI_QUERY_PROMPT.format(n=n, query=query)
    raw = _call_llm(prompt, max_tokens=200)

    if not raw:
        return [query]   # graceful fallback — still works, just no expansion

    variants = [line.strip() for line in raw.splitlines() if line.strip()]
    variants = variants[:n]   # cap at requested count
    return [query] + variants


def generate_hypothetical_doc(query: str) -> str:
    """
    Generate a hypothetical answer document for HyDE.
    Returns the hypothetical text, or the original query on failure.
    """
    prompt = HYDE_PROMPT.format(query=query)
    raw = _call_llm(prompt, max_tokens=250)
    return raw if raw else query


# ── Main Class ────────────────────────────────────────────
class QueryExpansionRAG:
    """
    Query Expansion pipeline: Multi-Query + HyDE, built on top of Reranker → Hybrid.

    mode options:
      "multi_query" — expand query into N variants, retrieve for each, merge
      "hyde"        — generate hypothetical answer, use it as retrieval query
      "combined"    — both: N variants + 1 HyDE doc, all merged
    """

    def __init__(self, mode: str = "combined",
                 n_queries: int = N_QUERIES,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.mode = mode
        self.n_queries = n_queries
        self.reranker_model = reranker_model
        self._reranker_rag = None   # RerankerRAG instance (provides hybrid + rerank)

    def build(self):
        """Build underlying RerankerRAG pipeline (ChromaDB + BM25 + CrossEncoder)."""
        src_dir = Path(__file__).parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from reranker_rag import RerankerRAG

        self._reranker_rag = RerankerRAG(reranker_model=self.reranker_model)
        self._reranker_rag.build()

        print(f"\n[QE] Query Expansion mode: '{self.mode}' | n_queries={self.n_queries}")
        print("[QE] Query Expansion Pipeline ready!")
        return self

    # ── Retrieval helpers ──────────────────────────────────

    def _retrieve_for_query(self, query: str, depth: int) -> list:
        """Run hybrid retrieval for a single query string."""
        return self._reranker_rag._hybrid.retrieve_hybrid(
            query, top_k=depth, retrieve_depth=depth
        )

    def _merge_candidates(self, candidate_lists: list) -> list:
        """
        Merge multiple candidate lists, deduplicating by chunk id.
        Earlier lists have priority for the chunk object (they may carry lower RRF scores
        but that doesn't matter — CrossEncoder will re-score everything).
        """
        seen = {}
        for candidates in candidate_lists:
            for chunk in candidates:
                if chunk["id"] not in seen:
                    seen[chunk["id"]] = chunk
        return list(seen.values())

    def retrieve_expanded(self, query: str,
                          top_k: int = TOP_K,
                          retrieve_depth: int = RETRIEVE_DEPTH) -> list:
        """
        Core method: expand query → retrieve → merge → rerank.
        Returns top_k reranked chunks.
        """
        candidate_lists = []

        # ── Step 1: Build retrieval queries ───────────────
        if self.mode in ("multi_query", "combined"):
            variants = expand_multi_query(query, n=self.n_queries)
            print(f"  [MQ] Generated {len(variants)} queries (original + {len(variants)-1} variants)")
            for i, v in enumerate(variants):
                label = "original" if i == 0 else f"variant {i}"
                print(f"       [{label}] {v}")
                # Divide depth across queries so total candidate pool stays bounded
                per_depth = max(5, retrieve_depth // len(variants))
                candidate_lists.append(self._retrieve_for_query(v, per_depth))

        if self.mode in ("hyde", "combined"):
            hyp_doc = generate_hypothetical_doc(query)
            print(f"  [HyDE] Hypothetical doc: {hyp_doc[:120]}...")
            candidate_lists.append(self._retrieve_for_query(hyp_doc, retrieve_depth))

        # Fallback: if both LLM calls failed (no API key), use original query
        if not candidate_lists:
            candidate_lists.append(self._retrieve_for_query(query, retrieve_depth))

        # ── Step 2: Merge + deduplicate ───────────────────
        merged = self._merge_candidates(candidate_lists)
        print(f"  [QE] Merged {len(merged)} unique candidates from {len(candidate_lists)} query variants")

        # ── Step 3: CrossEncoder rerank (score against original query) ─
        reranked = self._reranker_rag.rerank(query, merged, top_k=top_k)
        return reranked

    def query(self, question: str,
              top_k: int = TOP_K,
              retrieve_depth: int = RETRIEVE_DEPTH,
              verbose: bool = True) -> dict:
        """Full pipeline: expand → hybrid retrieve → rerank → generate."""
        if self._reranker_rag is None:
            raise RuntimeError("Call .build() first")

        if verbose:
            print(f"\n{'='*65}")
            print(f"Q: {question}")
            print(f"{'='*65}")

        context = self.retrieve_expanded(question, top_k=top_k, retrieve_depth=retrieve_depth)

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
            "mode":     self.mode,
        }

        if verbose:
            print(f"\nA: {answer}")
            print(f"\n--- Top {len(context)} passages (mode={self.mode}) ---")
            for c in context:
                rer = c.get("reranker_score", "N/A")
                print(f"  [{c['rank']}] {c['title']} (reranker={rer:.4f})" if isinstance(rer, float)
                      else f"  [{c['rank']}] {c['title']}")
                print(f"      {c['text'][:120]}...")

        return result


# ── Quick Test ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Query Expansion RAG Pipeline -- Day 15-17")
    print("=" * 65)

    rag = QueryExpansionRAG(mode="combined", n_queries=3).build()

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

    out = Path("results/query_expansion_test_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] Test results -> {out}")
    print("\nQuery Expansion RAG pipeline is working!")
