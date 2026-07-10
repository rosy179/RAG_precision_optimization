"""
Per-user Modular RAG pipeline (5 layers).

Imports helper functions from src/ directly — does NOT call .build()
on any existing RAG class (those hard-code JSON loading).
"""

import os
import re
import sys
import json
import math
import time
import hashlib
from pathlib import Path
from typing import Optional

# Add src/ to path so we can import helpers
_SRC = str(Path(__file__).parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv
load_dotenv()

DB_PATH     = Path(__file__).parent.parent.parent / "data" / "chroma_db_users"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
LLM_MODEL   = os.getenv("LLM_MODEL", "gpt-4o-mini")
RERANKER    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Shared reranker (loaded once at startup)
_reranker: Optional[CrossEncoder] = None
# Per-user RAG service instances
_instances: dict[str, "UserRAGService"] = {}


def warm_up():
    global _reranker
    print("[RAG] Loading CrossEncoder reranker...")
    try:
        _reranker = CrossEncoder(RERANKER)
        print("[RAG] Reranker ready.")
    except Exception as e:
        print(f"[RAG] Reranker load failed: {e}")


def get_service(user_id: str) -> "UserRAGService":
    if user_id not in _instances:
        _instances[user_id] = UserRAGService(user_id)
    return _instances[user_id]


def _tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


def _display_score(chunk: dict) -> float:
    """Normalize a chunk's relevance score to 0-1 for the UI badge.

    Cross-encoder rerank scores are raw, unbounded logits — squash them
    through a sigmoid. Semantic scores (1 - distance) can dip below 0
    depending on the distance metric, so clamp instead.
    """
    if "rerank_score" in chunk:
        return round(1.0 / (1.0 + math.exp(-chunk["rerank_score"])), 4)
    return round(max(0.0, min(1.0, chunk.get("score", 0.0))), 4)


def openai_error_detail(e: openai.APIError) -> str:
    """Translate an OpenAI SDK error into a user-facing Vietnamese message."""
    if isinstance(e, openai.RateLimitError):
        return "Tài khoản OpenAI đã hết quota. Vui lòng kiểm tra billing tại platform.openai.com."
    if isinstance(e, openai.AuthenticationError):
        return "OPENAI_API_KEY không hợp lệ. Vui lòng kiểm tra lại key trong file .env."
    return f"Lỗi kết nối OpenAI: {e.message if hasattr(e, 'message') else str(e)}"


class UserRAGService:
    def __init__(self, user_id: str):
        self._uid   = user_id
        self._api_key = os.getenv("OPENAI_API_KEY", "")

        DB_PATH.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(DB_PATH))

        if self._api_key and not self._api_key.startswith("sk-your"):
            embed_fn = OpenAIEmbeddingFunction(api_key=self._api_key, model_name=EMBED_MODEL)
        else:
            embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        self._chunk_col   = self._chroma.get_or_create_collection(
            f"u_{user_id[:8]}_chunks",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._summary_col = self._chroma.get_or_create_collection(
            f"u_{user_id[:8]}_summaries",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self._chunks_store: list[dict] = []  # in-memory for BM25
        self._bm25: Optional[BM25Okapi] = None
        self._doc_registry: list[dict] = []  # {id, name, type, chunk_count}

        # Restore from ChromaDB on init (handles server restarts)
        self._restore_from_chroma()

    # ── Ingestion ──────────────────────────────────────────

    def add_document(self, chunks: list[dict], doc_summary: str, doc_meta: dict) -> int:
        """Upsert chunks + summary into ChromaDB and rebuild BM25."""
        if not chunks:
            return 0

        doc_id = doc_meta["id"]

        # Upsert chunks into chunk collection
        self._chunk_col.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{
                "doc_id": doc_id,
                "title":  doc_meta["name"],
                "source": doc_meta.get("type", "unknown"),
                "chunk_index": i,
            } for i, c in enumerate(chunks)],
        )

        # Upsert doc summary into summary collection
        self._summary_col.upsert(
            ids=[doc_id],
            documents=[doc_summary],
            metadatas=[{"name": doc_meta["name"], "type": doc_meta.get("type", "unknown")}],
        )

        # Update in-memory BM25 store
        for c in chunks:
            self._chunks_store.append({"id": c["id"], "text": c["text"],
                                       "title": doc_meta["name"], "doc_id": doc_id})
        self._rebuild_bm25()

        # Update registry
        self._doc_registry.append({
            "id": doc_id,
            "name": doc_meta["name"],
            "type": doc_meta.get("type", "unknown"),
            "chunk_count": len(chunks),
        })

        return len(chunks)

    def remove_document(self, doc_id: str):
        """Remove all chunks for a document."""
        existing = self._chunk_col.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self._chunk_col.delete(ids=existing["ids"])
        try:
            self._summary_col.delete(ids=[doc_id])
        except Exception:
            pass
        self._chunks_store = [c for c in self._chunks_store if c.get("doc_id") != doc_id]
        self._doc_registry = [d for d in self._doc_registry if d["id"] != doc_id]
        self._rebuild_bm25()

    def get_documents(self) -> list[dict]:
        return self._doc_registry

    def chunk_count(self) -> int:
        return self._chunk_col.count()

    # ── Query ──────────────────────────────────────────────

    def query(self, question: str, history: list[dict] | None = None) -> dict:
        if self._chunk_col.count() == 0:
            return {
                "answer":  "Bạn chưa upload tài liệu nào. Hãy upload PDF, URL hoặc hình ảnh trước.",
                "sources": [],
                "reasoning": "",
                "latency_ms": 0,
                "from_cache": False,
            }

        t0 = time.time()
        history = history or []

        # Layer 1 — Adaptive Router
        from adaptive_rag import classify_heuristic
        complexity = classify_heuristic(question)

        # Layer 2 — Hierarchical: find relevant docs by summary
        top_k_docs = min(5, self._summary_col.count())
        relevant_doc_ids: list[str] = []
        if top_k_docs > 0:
            summary_res = self._summary_col.query(
                query_texts=[question], n_results=top_k_docs
            )
            relevant_doc_ids = summary_res["ids"][0] if summary_res["ids"] else []

        # Layer 3 — Advanced: Semantic + BM25 + RRF
        n_semantic = 20
        where_filter = {"doc_id": {"$in": relevant_doc_ids}} if relevant_doc_ids else None
        kwargs = {"query_texts": [question], "n_results": min(n_semantic, self._chunk_col.count())}
        if where_filter:
            kwargs["where"] = where_filter
        semantic_res = self._chunk_col.query(**kwargs)

        semantic_chunks = []
        if semantic_res["ids"] and semantic_res["ids"][0]:
            for i, chunk_id in enumerate(semantic_res["ids"][0]):
                meta = semantic_res["metadatas"][0][i] if semantic_res["metadatas"] else {}
                dist = semantic_res["distances"][0][i] if semantic_res["distances"] else 1.0
                semantic_chunks.append({
                    "id":     chunk_id,
                    "text":   semantic_res["documents"][0][i],
                    "title":  meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "score":  round(1.0 - float(dist), 4),
                })

        bm25_chunks = self._bm25_search(question, n=20)

        from hybrid_rag import reciprocal_rank_fusion
        top_k = 5 if complexity == "complex" else (4 if complexity == "medium" else 3)
        fused = reciprocal_rank_fusion(semantic_chunks, bm25_chunks, top_k=top_k * 4)

        # Rerank if not simple
        if complexity != "simple" and _reranker and len(fused) > 1:
            pairs = [(question, c["text"]) for c in fused]
            scores = _reranker.predict(pairs)
            for chunk, score in zip(fused, scores):
                chunk["rerank_score"] = float(score)
            fused.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        top_chunks = fused[:top_k]

        # Layer 4 — Iterative fallback: if best score < 0.5, HyDE re-query
        best_score = top_chunks[0].get("score", 1.0) if top_chunks else 0.0
        if best_score < 0.5 and self._chunk_col.count() > 0:
            from query_expansion import generate_hypothetical_doc
            hyde = generate_hypothetical_doc(question)
            if hyde and hyde != question:
                hyde_res = self._chunk_col.query(
                    query_texts=[hyde],
                    n_results=min(10, self._chunk_col.count()),
                )
                hyde_chunks = []
                if hyde_res["ids"] and hyde_res["ids"][0]:
                    for i, cid in enumerate(hyde_res["ids"][0]):
                        meta = hyde_res["metadatas"][0][i] if hyde_res["metadatas"] else {}
                        dist = hyde_res["distances"][0][i] if hyde_res["distances"] else 1.0
                        hyde_chunks.append({
                            "id": cid, "text": hyde_res["documents"][0][i],
                            "title": meta.get("title", ""), "source": meta.get("source", ""),
                            "doc_id": meta.get("doc_id", ""),
                            "score": round(1.0 - float(dist), 4),
                        })
                fused2 = reciprocal_rank_fusion(top_chunks, hyde_chunks, top_k=top_k)
                top_chunks = fused2[:top_k]

        # Add rank field for CoT context builder
        for i, c in enumerate(top_chunks, 1):
            c["rank"] = i

        # Layer 5 — CoT Generation
        from cot_rag import generate_cot_answer
        answer, reasoning = generate_cot_answer(question, top_chunks, mode="structured")

        latency = int((time.time() - t0) * 1000)

        sources = [{
            "rank":    c.get("rank", i + 1),
            "title":   c.get("title", "Unknown"),
            "snippet": c.get("text", "")[:300],
            "score":   _display_score(c),
        } for i, c in enumerate(top_chunks)]

        return {
            "answer":     answer,
            "reasoning":  reasoning,
            "sources":    sources,
            "latency_ms": latency,
            "from_cache": False,
            "complexity": complexity,
        }

    # ── Internals ──────────────────────────────────────────

    def _bm25_search(self, query: str, n: int = 20) -> list[dict]:
        if not self._bm25 or not self._chunks_store:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
        results = []
        for idx, score in ranked:
            if score > 0 and idx < len(self._chunks_store):
                c = self._chunks_store[idx]
                results.append({**c, "score": round(float(score), 4)})
        return results

    def _rebuild_bm25(self):
        if not self._chunks_store:
            self._bm25 = None
            return
        tokenized = [_tokenize(c["text"]) for c in self._chunks_store]
        self._bm25 = BM25Okapi(tokenized)

    def _restore_from_chroma(self):
        """Rebuild in-memory state from persisted ChromaDB on startup."""
        try:
            all_chunks = self._chunk_col.get(include=["documents", "metadatas"])
            if not all_chunks["ids"]:
                return
            chunk_by_doc: dict[str, dict] = {}
            for i, cid in enumerate(all_chunks["ids"]):
                meta = all_chunks["metadatas"][i] if all_chunks["metadatas"] else {}
                text = all_chunks["documents"][i] if all_chunks["documents"] else ""
                doc_id = meta.get("doc_id", "unknown")
                self._chunks_store.append({
                    "id": cid, "text": text,
                    "title": meta.get("title", ""), "doc_id": doc_id,
                })
                if doc_id not in chunk_by_doc:
                    chunk_by_doc[doc_id] = {"name": meta.get("title", ""), "type": meta.get("source", ""), "count": 0}
                chunk_by_doc[doc_id]["count"] += 1

            for doc_id, info in chunk_by_doc.items():
                self._doc_registry.append({
                    "id": doc_id, "name": info["name"],
                    "type": info["type"], "chunk_count": info["count"],
                })
            self._rebuild_bm25()
        except Exception as e:
            print(f"[RAG] Restore warning: {e}")
