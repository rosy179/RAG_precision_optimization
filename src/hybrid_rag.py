#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/hybrid_rag.py
------------------
Day 8-10: Hybrid Search (BM25 + Semantic Search)
- Load documents from data/rag_dataset.json
- Chunk + index in local ChromaDB (Semantic)
- Chunk + index in BM25 (Lexical)
- Retrieve candidates from both, merge using Reciprocal Rank Fusion (RRF)
- Generate answer via OpenAI gpt-4o-mini
"""

import sys
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# ── Config ────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K         = int(os.getenv("TOP_K", 3))
EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
LLM_MODEL     = os.getenv("LLM_MODEL", "gpt-4o-mini")
DATA_PATH     = Path("data/rag_dataset.json")
DB_PATH       = Path("data/chroma_db")
COLLECTION    = "rag_baseline"  # We reuse the same DB collection for semantic queries


# ── Simple Tokenizer for BM25 ─────────────────────────────
def tokenize(text: str) -> list:
    """Lowercase and extract alphanumeric word tokens."""
    return re.findall(r'\w+', text.lower())


# ── RRF (Reciprocal Rank Fusion) ──────────────────────────
def reciprocal_rank_fusion(semantic_results: list, lexical_results: list, k=60, top_k=TOP_K) -> list:
    """
    Merge rankings from semantic and lexical searches using RRF.
    Each list should contain chunk dictionaries with an 'id' key.
    """
    rrf_scores = {}
    chunk_map = {}

    # Rank 1 to N for semantic
    for rank, chunk in enumerate(semantic_results, 1):
        chunk_id = chunk["id"]
        chunk_map[chunk_id] = chunk
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    # Rank 1 to N for lexical
    for rank, chunk in enumerate(lexical_results, 1):
        chunk_id = chunk["id"]
        chunk_map[chunk_id] = chunk
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    # Sort descending by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Reconstruct the top-k chunks with their updated ranks
    fused_chunks = []
    for i, chunk_id in enumerate(sorted_ids[:top_k], 1):
        orig = chunk_map[chunk_id]
        fused_chunks.append({
            "rank":      i,
            "id":        chunk_id,
            "text":      orig["text"],
            "title":     orig.get("title", ""),
            "source":    orig.get("source", ""),
            "rrf_score": round(rrf_scores[chunk_id], 6),
        })

    return fused_chunks


# ── Main Hybrid RAG Class ─────────────────────────────────
class HybridRAG:
    """End-to-end Hybrid RAG pipeline combining BM25 and ChromaDB."""

    def __init__(self):
        self.collection = None
        self.chunks = []
        self.bm25 = None

    def build(self):
        """Load documents, chunk, and index in ChromaDB & BM25."""
        from baseline_rag import load_documents, chunk_documents, build_vector_store
        from rank_bm25 import BM25Okapi

        # 1. Load and chunk documents
        docs = load_documents()
        self.chunks = chunk_documents(docs)

        # 2. Build ChromaDB index (Semantic)
        self.collection = build_vector_store(self.chunks)

        # 3. Build BM25 index (Lexical)
        print("[3.5/4] Building BM25 index...")
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"        Indexed {len(self.chunks)} chunks into BM25")

        print("\n[4/4] Hybrid Pipeline ready!")
        return self

    def retrieve_hybrid(self, query: str, top_k=TOP_K, retrieve_depth=20) -> list:
        """Retrieve using both ChromaDB and BM25, then fuse with RRF."""
        # 1. Semantic Retrieval (ChromaDB)
        # We perform standard query on the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=retrieve_depth,
        )
        semantic_list = []
        if results and results["documents"]:
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                # We need to find the correct chunk ID
                # ChromaDB stores the chunk ID in 'ids'
                chunk_id = results["ids"][0][i]
                semantic_list.append({
                    "id":       chunk_id,
                    "text":     doc,
                    "title":    meta.get("title", ""),
                    "source":   meta.get("source", ""),
                    "distance": dist,
                })

        # 2. Lexical Retrieval (BM25)
        tok_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tok_query)
        
        # Pair score and chunk, sort descending
        scored_chunks = sorted(
            zip(bm25_scores, self.chunks),
            key=lambda x: x[0],
            reverse=True
        )
        lexical_list = []
        for score, chunk in scored_chunks[:retrieve_depth]:
            if score > 0.0:  # Only retrieve chunks with some term matches
                lexical_list.append(chunk)

        # 3. RRF Rank Fusion
        fused_chunks = reciprocal_rank_fusion(semantic_list, lexical_list, k=60, top_k=top_k)
        return fused_chunks

    def query(self, question: str, top_k=TOP_K, verbose=True) -> dict:
        """Run full Hybrid RAG: retrieve (hybrid) + generate."""
        if not self.collection or not self.bm25:
            raise RuntimeError("Call .build() first")

        # Retrieve fused chunks
        context = self.retrieve_hybrid(question, top_k)
        
        # Generate answer (reuse baseline's generate_answer)
        from baseline_rag import generate_answer
        answer = generate_answer(question, context, model=LLM_MODEL)

        result = {
            "question": question,
            "answer":   answer,
            "context":  context,
            "top_k":    top_k,
        }

        if verbose:
            print(f"\n{'='*55}")
            print(f"Q: {question} (HYBRID)")
            print(f"{'='*55}")
            print(f"A: {answer}")
            print(f"\n--- Retrieved {len(context)} chunks (Fused by RRF) ---")
            for c in context:
                print(f"  [{c['rank']}] {c['title']} (RRF score={c['rrf_score']})")
                print(f"      {c['text'][:120]}...")

        return result


# ── Quick Test ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Hybrid RAG Pipeline -- Day 8-10")
    print("=" * 55)

    rag = HybridRAG().build()

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

    # Save results
    out = Path("results/hybrid_test_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] Test results -> {out}")
    print("\nHybrid RAG pipeline is working!")
