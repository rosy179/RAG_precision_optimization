"""
Per-user Modular RAG pipeline (5 layers).

Imports helper functions from src/ directly — does NOT call .build()
on any existing RAG class (those hard-code JSON loading).
"""

import os
import re
import sys
import json
import time
import hashlib
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

# Add src/ to path so we can import helpers
_SRC = str(Path(__file__).parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import openai
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv
load_dotenv()

from backend.services import llm
from backend.services.embeddings import get_embedding_function
from backend.services.document_processor import contextual_text
from backend.services.global_kb import get_kb
from backend.services.prompts import (
    CHAT_SYSTEM_PROMPT, CONDENSE_PROMPT, GROUNDING_PROMPT, SUGGEST_PROMPT,
    TRANSFORM_SYSTEM_PROMPT,
)
from backend.services.intent import (
    is_summary_question, wants_prev_transform, lang_directive as _lang_directive,
    _SUMMARY_INTENT_RE, _DOC_REF_RE,
)
from backend.services.rag_helpers import (
    tokenize as _tokenize, add_usage as _add_usage, cosine as _cosine,
    merge_lexical as _merge_lexical, reattach_chunk_fields as _reattach_chunk_fields,
    chunks_to_sources as _chunks_to_sources, display_score as _display_score,
)
from backend.services.multihop import MultihopMixin

log = logging.getLogger("rag.user")

DB_PATH   = Path(__file__).parent.parent.parent / "data" / "chroma_db_users"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
# Multilingual cross-encoder: scores VI/JA/EN query-passage pairs directly,
# so retrieval no longer needs an English translation of the query.
# mmarco-mMiniLMv2 (~120MB) reranks 16 pairs in ~4s on this 4-thread CPU;
# BAAI/bge-reranker-v2-m3 is stronger but needs a GPU (57s/query on CPU) —
# switch via RERANKER_MODEL when deploying on GPU hardware.
RERANKER         = os.getenv("RERANKER_MODEL",
                             "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANKER_MAX_LEN = 512  # chunks are ~512 tokens; caps CPU latency per pair

# Answer cache: repeat questions in the same (session, source-filter, KB)
# scope are served from memory instead of re-running the pipeline.
ANSWER_CACHE_TTL_S = 3600
ANSWER_CACHE_MAX   = 50
ANSWER_CACHE_SIM   = 0.95  # cosine threshold for a semantic hit


# Shared reranker (loaded once at startup)
_reranker: Optional[CrossEncoder] = None
# Per-user RAG service instances, LRU-capped: each one keeps its user's
# chunk texts + BM25 token lists in memory, so idle users must be evicted.
# Eviction is safe — all state is restored from ChromaDB on next access.
MAX_INSTANCES = int(os.getenv("MAX_RAG_INSTANCES", "16"))
_instances: "OrderedDict[str, UserRAGService]" = OrderedDict()
_instances_lock = threading.Lock()


def warm_up():
    global _reranker
    log.info("Loading CrossEncoder reranker (%s)...", RERANKER)
    try:
        # Local cache first — HF Hub outages must not block startup.
        _reranker = CrossEncoder(RERANKER, max_length=RERANKER_MAX_LEN,
                                 local_files_only=True)
        log.info("Reranker ready (local cache).")
    except Exception:
        try:
            _reranker = CrossEncoder(RERANKER, max_length=RERANKER_MAX_LEN)
            log.info("Reranker ready.")
        except Exception as e:
            log.error("Reranker load failed: %s", e)


def get_service(user_id: str) -> "UserRAGService":
    with _instances_lock:
        svc = _instances.get(user_id)
        if svc is None:
            svc = UserRAGService(user_id)
            _instances[user_id] = svc
        else:
            _instances.move_to_end(user_id)
        while len(_instances) > MAX_INSTANCES:
            evicted, _ = _instances.popitem(last=False)
            log.info("LRU-evicted RAG service of user %s…", evicted[:8])
    return svc


def generate_session_title(question: str, answer: str = "") -> str:
    """Short LLM-generated conversation title; falls back to a truncated
    question in mock mode or on any API failure."""
    fallback = question[:60] + ("…" if len(question) > 60 else "")
    if llm.is_mock():
        return fallback
    try:
        client = llm.get_client()
        res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": (
                "Đặt tiêu đề thật ngắn gọn (tối đa 8 từ, không dấu ngoặc kép, "
                "không dấu chấm cuối) tóm tắt chủ đề đoạn hội thoại sau. "
                "Dùng đúng ngôn ngữ của câu hỏi.\n\n"
                f"Câu hỏi: {question[:300]}\n\nTrả lời: {answer[:300]}"
            )}],
            temperature=0.2,
            max_tokens=30,
        )
        title = (res.choices[0].message.content or "").strip().strip('"').strip()
        return title[:60] if title else fallback
    except Exception:
        return fallback


def openai_error_detail(e: openai.APIError) -> str:
    """Translate an OpenAI SDK error into a user-facing Vietnamese message."""
    if isinstance(e, openai.RateLimitError):
        return "Tài khoản OpenAI đã hết quota. Vui lòng kiểm tra billing tại platform.openai.com."
    if isinstance(e, openai.AuthenticationError):
        return "OPENAI_API_KEY không hợp lệ. Vui lòng kiểm tra lại key trong file .env."
    return f"Lỗi kết nối OpenAI: {e.message if hasattr(e, 'message') else str(e)}"


class UserRAGService(MultihopMixin):
    def __init__(self, user_id: str):
        self._uid = user_id

        DB_PATH.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(DB_PATH))

        # Shared factory — same vector space as the global KB. Also kept
        # for the answer cache's question-similarity checks.
        embed_fn = get_embedding_function()
        self._embed_fn = embed_fn

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
        self._doc_registry: list[dict] = []  # {id, name, type, session_id, chunk_count}

        # Guards the in-memory stores + BM25 cache: FastAPI runs sync
        # endpoints on a threadpool, so uploads and queries can overlap.
        self._lock = threading.Lock()
        # (session_id, doc-filter) → (BM25Okapi, pool). BM25 IDF statistics
        # depend on the exact chunk pool, so each filter combination gets its
        # own index; any document add/remove invalidates all entries.
        self._bm25_cache: OrderedDict[tuple, tuple] = OrderedDict()
        # Answer cache: entries carry a (doc-store, global-KB) version pair
        # and silently expire when either corpus changes.
        self._store_version = 0
        self._answer_cache: OrderedDict[str, dict] = OrderedDict()

        # Restore from ChromaDB on init (handles server restarts)
        self._restore_from_chroma()

    # ── Ingestion ──────────────────────────────────────────

    @staticmethod
    def _contextual_embeddings(chunks: list[dict]) -> Optional[list]:
        """Precomputed embeddings of the context-prefixed text, or None when
        no chunk has a context (let Chroma embed the plain `documents`)."""
        if not any(c.get("context") for c in chunks):
            return None
        return list(get_embedding_function()([contextual_text(c) for c in chunks]))

    def add_document(self, chunks: list[dict], doc_summary: str, doc_meta: dict) -> int:
        """Upsert chunks + summary into ChromaDB and rebuild BM25."""
        if not chunks:
            return 0

        doc_id = doc_meta["id"]
        session_id = doc_meta.get("session_id", "")
        # Persisted so "tài liệu mới nhất" stays correct across restarts
        # (the deictic digest fallback picks the most recently added doc).
        added_at = time.time()

        # Contextual Retrieval: embed the context-prefixed text but store the
        # original (see GlobalKBService for the full rationale). None → Chroma
        # embeds the plain documents.
        embeddings = self._contextual_embeddings(chunks)

        # Upsert chunks into chunk collection (page only exists for PDFs —
        # Chroma metadata rejects None values, so include it conditionally)
        self._chunk_col.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[{
                "doc_id": doc_id,
                "session_id": session_id,
                "title":  doc_meta["name"],
                "source": doc_meta.get("type", "unknown"),
                "chunk_index": i,
                "added_at": added_at,
                "context": c.get("context", ""),
                "heading": c.get("heading", ""),
                **({"page": c["page"]} if "page" in c else {}),
            } for i, c in enumerate(chunks)],
        )

        # Upsert doc summary into summary collection
        self._summary_col.upsert(
            ids=[doc_id],
            documents=[doc_summary],
            metadatas=[{"name": doc_meta["name"], "type": doc_meta.get("type", "unknown"),
                        "session_id": session_id}],
        )

        # Swap (never mutate) the in-memory stores so in-flight queries keep
        # a consistent snapshot; tokens are precomputed once for BM25.
        new_entries = [{"id": c["id"], "text": c["text"],
                        "title": doc_meta["name"], "doc_id": doc_id,
                        "session_id": session_id,
                        "heading": c.get("heading", ""),
                        "tokens": _tokenize(contextual_text(c)),
                        **({"page": c["page"]} if "page" in c else {})}
                       for c in chunks]
        with self._lock:
            self._chunks_store = self._chunks_store + new_entries
            self._doc_registry = self._doc_registry + [{
                "id": doc_id,
                "name": doc_meta["name"],
                "type": doc_meta.get("type", "unknown"),
                "session_id": session_id,
                "chunk_count": len(chunks),
                "added_at": added_at,
            }]
            self._bm25_cache.clear()
            self._store_version += 1

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
        with self._lock:
            self._chunks_store = [c for c in self._chunks_store if c.get("doc_id") != doc_id]
            self._doc_registry = [d for d in self._doc_registry if d["id"] != doc_id]
            self._bm25_cache.clear()
            self._store_version += 1

    def get_documents(self) -> list[dict]:
        return self._doc_registry

    def chunk_count(self) -> int:
        return self._chunk_col.count()

    def get_document_content(self, doc_id: str) -> dict | None:
        """Reconstruct a document's full text from its stored chunks for the
        viewer, with per-chunk char offsets so a cited chunk can be
        highlighted. Returns None if the doc has no chunks."""
        res = self._chunk_col.get(where={"doc_id": doc_id},
                                  include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        rows = sorted(
            zip(res["ids"], res["documents"], res["metadatas"]),
            key=lambda r: r[2].get("chunk_index", 0),
        )
        from backend.services.document_processor import reconstruct_from_chunks
        text, offsets = reconstruct_from_chunks([r[1] for r in rows])
        return {
            "text": text,
            "chunks": [{
                "id": rid, "start": start, "end": end,
                **({"page": meta["page"]} if "page" in meta else {}),
            } for (rid, _, meta), (start, end) in zip(rows, offsets)],
        }

    # ── Query ──────────────────────────────────────────────

    def retrieve(self, question: str, history: list[dict] | None = None,
                 session_id: str | None = None,
                 include_doc_ids: list[str] | None = None,
                 use_global_kb: bool = True,
                 attached_doc_ids: list[str] | None = None,
                 digest_ok: bool = True,
                 usage: dict | None = None) -> dict:
        """Run the full retrieval stack (Layers 1-4) without generation.

        Returns top_chunks + UI-ready sources so callers can either generate
        in one shot (eval scripts) or stream tokens (generate_stream).
        `usage` accumulates real token counts of internal LLM calls.

        include_doc_ids: session doc ids the user checked in the source
        picker (None = all session docs). use_global_kb=False excludes the
        shared KB for this question.

        attached_doc_ids: docs attached to THIS message — they are the
        question's subject, so retrieval is scoped to them entirely: a
        deictic question ("dự án này dùng công nghệ gì?") carries no topical
        signal, and without scoping an older session document with better
        keyword overlap wins the ranking. digest_ok=False disables the
        summarize digest path (multi-hop sub-queries must always run real
        retrieval).
        """
        # Retrieval draws from two pools: chunks uploaded in THIS session
        # (per-conversation scope, as before) plus the shared global
        # knowledge base available to every user and session.
        kb = get_kb()
        session_chunks = [c for c in self._chunks_store
                          if not session_id or c.get("session_id") == session_id]
        if include_doc_ids is not None:
            allowed = set(include_doc_ids)
            session_chunks = [c for c in session_chunks if c.get("doc_id") in allowed]

        # Docs attached to THIS message are its explicit subject — scope
        # retrieval to them so an older session document or the shared KB
        # can't outrank the document the user just attached. Falls through
        # to the normal pools when the attachments have no indexed chunks.
        attached_scope: set | None = None
        if attached_doc_ids:
            wanted = set(attached_doc_ids)
            scoped = [c for c in session_chunks if c.get("doc_id") in wanted]
            if scoped:
                session_chunks = scoped
                attached_scope = {c["doc_id"] for c in scoped}

        kb_enabled = use_global_kb and not kb.is_empty() and attached_scope is None
        if not session_chunks and not kb_enabled:
            filtered_out = include_doc_ids is not None or not use_global_kb
            return {
                "empty": True,
                "notice": (
                    "Tất cả nguồn tham khảo đang bị tắt. Hãy bật lại ít nhất một tài liệu "
                    "hoặc kho kiến thức chung trong bộ chọn nguồn rồi hỏi lại."
                    if filtered_out else
                    "Chưa có tài liệu nào — cả trong cuộc trò chuyện này lẫn kho "
                    "kiến thức chung. Hãy đính kèm PDF, dán URL hoặc tải ảnh/ghi âm "
                    "lên trước khi hỏi."
                ),
                "top_chunks": [],
                "sources": [],
                "complexity": "simple",
            }

        t0 = time.time()
        history = history or []
        session_filter = {"session_id": session_id} if session_id else None

        allowed_ids = set(include_doc_ids) if include_doc_ids is not None else None
        if attached_scope is not None:
            # Already respects the source picker: the scope was derived from
            # the include_doc_ids-filtered chunk pool above.
            allowed_ids = attached_scope
        session_docs = [d for d in self._doc_registry
                        if (not session_id or d.get("session_id") == session_id)
                        and (allowed_ids is None or d["id"] in allowed_ids)]

        # Summarize-type questions targeting an attached/referenced document
        # skip search entirely and read that document's chunks in order.
        if digest_ok:
            digest = self._digest_retrieve(question, session_docs, attached_doc_ids)
            if digest is not None:
                digest["retrieval_ms"] = int((time.time() - t0) * 1000)
                return digest

        # Condense follow-ups ("nói rõ hơn ý 2") into a standalone query via
        # history — generation still sees the original question + history.
        # Skipped when the message carries its own attachments: the question
        # is about THEM, and history (about earlier documents) would steer
        # the rewritten query back to the old topic. The whole retrieval
        # stack (BGE-M3 + multilingual reranker + multilingual heuristic)
        # works on the original language, so no English variant is needed.
        query = (question if attached_doc_ids
                 else self._condense_query(question, history, usage=usage))

        # Layer 1 — Adaptive Router
        from adaptive_rag import classify_heuristic
        complexity = classify_heuristic(query)

        # Layer 2 — Hierarchical: find relevant docs by summary, separately
        # per pool (session docs and global KB docs live in different
        # collections, so each needs its own doc-id shortlist).
        top_k_docs = min(5, len(session_docs))
        relevant_doc_ids: list[str] = []
        if top_k_docs > 0:
            kwargs = {"query_texts": [query], "n_results": top_k_docs}
            if session_filter:
                kwargs["where"] = session_filter
            summary_res = self._summary_col.query(**kwargs)
            if summary_res["ids"]:
                relevant_doc_ids = list(summary_res["ids"][0])
        if allowed_ids is not None:
            # The summary shortlist isn't checkbox-aware — constrain it, and
            # never fall through to an unfiltered chunk search.
            relevant_doc_ids = [d for d in relevant_doc_ids if d in allowed_ids] \
                or [d["id"] for d in session_docs]
        if attached_doc_ids:
            # Docs attached to this message are its most likely subject —
            # guarantee them a spot in the chunk-search shortlist.
            valid = {d["id"] for d in session_docs}
            boost = [i for i in attached_doc_ids if i in valid]
            relevant_doc_ids = list(dict.fromkeys(boost + relevant_doc_ids))
        kb_doc_ids = kb.search_summaries(query, top_k=5) if kb_enabled else []

        # Layer 3 — Advanced: Semantic + BM25 + RRF, over both pools
        semantic_chunks = []
        if session_chunks:
            filters = []
            if session_filter:
                filters.append(session_filter)
            if relevant_doc_ids:
                filters.append({"doc_id": {"$in": relevant_doc_ids}})
            where_filter = filters[0] if len(filters) == 1 else ({"$and": filters} if filters else None)

            kwargs = {"query_texts": [query], "n_results": min(20, len(session_chunks))}
            if where_filter:
                kwargs["where"] = where_filter
            semantic_res = self._chunk_col.query(**kwargs)

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
                        "scope":  "session",
                        "score":  round(1.0 - float(dist), 4),
                        **({"page": meta["page"]} if "page" in meta else {}),
                    })

        # Both collections use the same embedding model and cosine space,
        # so their (1 - distance) scores are directly comparable.
        kb_semantic = kb.search_semantic(query, n_results=20,
                                         doc_ids=kb_doc_ids or None) if kb_enabled else []
        semantic_chunks = sorted(semantic_chunks + kb_semantic,
                                 key=lambda c: c["score"], reverse=True)[:20]

        bm25_chunks = _merge_lexical(
            self._bm25_search(query, n=20, session_id=session_id,
                              allowed_doc_ids=allowed_ids),
            kb.search_bm25(query, n=20) if kb_enabled else [],
        )

        from hybrid_rag import reciprocal_rank_fusion
        # simple gets 4 (not 3): with competing near-answer chunks in the
        # corpus, the correct one must survive into the generation context.
        top_k = 5 if complexity == "complex" else 4
        fused = reciprocal_rank_fusion(semantic_chunks, bm25_chunks, top_k=top_k * 4)
        _reattach_chunk_fields(fused, bm25_chunks, semantic_chunks)

        # Rerank ALWAYS (the reranker is multilingual, so pairs are
        # scored in the query's original language). Skipping rerank
        # for "simple" questions saved ~0.5s but let near-duplicate facts
        # outrank the exact answer chunk — accuracy wins here.
        # The CE ordering is BLENDED with the fusion ordering (RRF over both
        # rank lists): when the corpus holds two competing near-answers, one
        # CE miss must not evict the chunk BM25+semantic both agreed on.
        if _reranker and len(fused) > 1:
            pairs = [(query, c["text"]) for c in fused]
            scores = _reranker.predict(pairs)
            ce_rank = {i: r for r, i in enumerate(
                sorted(range(len(fused)), key=lambda i: -float(scores[i])))}
            for i, chunk in enumerate(fused):
                chunk["rerank_score"] = float(scores[i])
                chunk["blend"] = 1.0 / (10 + i) + 1.0 / (10 + ce_rank[i])
            fused.sort(key=lambda c: c["blend"], reverse=True)

        top_chunks = fused[:top_k]

        # Layer 4 — Iterative fallback: HyDE re-query when nothing relevant
        # was found. Threshold tuned to BGE-M3's cosine scale (relevant
        # ~0.55-0.7, cross-lingual hits can dip to ~0.5, unrelated <0.45) —
        # the old 0.5 (ada-002 scale) would fire on good VI/JA retrievals.
        best_score = top_chunks[0].get("score", 1.0) if top_chunks else 0.0
        if best_score < 0.4:
            from query_expansion import generate_hypothetical_doc
            hyde = generate_hypothetical_doc(query)
            if hyde and hyde != query:
                hyde_chunks = []
                if session_chunks:
                    hyde_filters = []
                    if session_filter:
                        hyde_filters.append(session_filter)
                    if allowed_ids is not None:
                        hyde_filters.append({"doc_id": {"$in": list(allowed_ids)}})
                    hyde_kwargs = {"query_texts": [hyde],
                                   "n_results": min(10, len(session_chunks))}
                    if hyde_filters:
                        hyde_kwargs["where"] = (hyde_filters[0] if len(hyde_filters) == 1
                                                else {"$and": hyde_filters})
                    hyde_res = self._chunk_col.query(**hyde_kwargs)
                    if hyde_res["ids"] and hyde_res["ids"][0]:
                        for i, cid in enumerate(hyde_res["ids"][0]):
                            meta = hyde_res["metadatas"][0][i] if hyde_res["metadatas"] else {}
                            dist = hyde_res["distances"][0][i] if hyde_res["distances"] else 1.0
                            hyde_chunks.append({
                                "id": cid, "text": hyde_res["documents"][0][i],
                                "title": meta.get("title", ""), "source": meta.get("source", ""),
                                "doc_id": meta.get("doc_id", ""),
                                "scope": "session",
                                "score": round(1.0 - float(dist), 4),
                                **({"page": meta["page"]} if "page" in meta else {}),
                            })
                if kb_enabled:
                    hyde_chunks += kb.search_semantic(hyde, n_results=10)
                fused2 = reciprocal_rank_fusion(top_chunks, hyde_chunks, top_k=top_k)
                _reattach_chunk_fields(fused2, top_chunks, hyde_chunks)
                top_chunks = fused2[:top_k]

        # Add rank field for CoT context builder
        for i, c in enumerate(top_chunks, 1):
            c["rank"] = i

        return {
            "empty":        False,
            "notice":       None,
            "top_chunks":   top_chunks,
            "sources":      _chunks_to_sources(top_chunks),
            "complexity":   complexity,
            "retrieval_ms": int((time.time() - t0) * 1000),
        }

    def generate_stream(self, question: str, chunks: list[dict],
                        history: list[dict] | None = None,
                        usage: dict | None = None):
        """Yield answer text deltas (Layer 5, streaming variant).

        openai.APIError propagates to the caller so the endpoint can emit
        an SSE error event.
        """
        if llm.is_mock():
            preview = chunks[0]["text"][:200] if chunks else "No context"
            # Chunked mock so the UI streaming path is exercised without a key
            mock = f"[MOCK ANSWER] Based on context: '{preview}...'"
            for i in range(0, len(mock), 24):
                yield mock[i:i + 24]
            return

        client = llm.get_client()
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=self._chat_messages(question, chunks, history or []),
            temperature=0.2,
            max_tokens=1500,
            stream=True,
            stream_options={"include_usage": True},
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content
            _add_usage(usage, event)  # only the final chunk carries usage

    def transform_stream(self, question: str, prev_answer: str,
                         usage: dict | None = None):
        """Yield the previous answer re-rendered per the user's instruction
        (translate, shorten, reformat) — no retrieval involved."""
        if llm.is_mock():
            mock = f"[MOCK TRANSFORM] {question}: {prev_answer[:200]}"
            for i in range(0, len(mock), 24):
                yield mock[i:i + 24]
            return

        client = llm.get_client()
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": TRANSFORM_SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Câu trả lời trước:\n{prev_answer}\n\nYêu cầu: {question}"},
            ],
            temperature=0.2,
            max_tokens=2000,
            stream=True,
            stream_options={"include_usage": True},
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content
            _add_usage(usage, event)

    def suggest_questions(self, question: str, answer: str,
                          sources: list[dict],
                          usage: dict | None = None) -> list[str]:
        """Up to 3 follow-up questions for the click-to-ask chips.

        One cheap LLM call from the question + answer + source snippets;
        callers run it AFTER the done event so it never delays the answer.
        Returns [] on any failure — the feature degrades to nothing.
        """
        if not sources:
            return []
        if llm.is_mock():
            return ["Chủ đề này có những ứng dụng nào?",
                    "Có hạn chế hay nhược điểm gì không?",
                    "So sánh với các phương pháp tương tự?"]
        context = "\n".join(f"- {s.get('title', '')}: {s.get('snippet', '')}"
                            for s in sources[:5])
        try:
            client = llm.get_client()
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SUGGEST_PROMPT},
                    {"role": "user", "content":
                        f"Câu hỏi: {question}\n\nCâu trả lời: {answer[:1500]}\n\n"
                        f"Nguồn tài liệu:\n{context}"},
                ],
                temperature=0.7,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            _add_usage(usage, res)
            data = json.loads(res.choices[0].message.content or "{}")
            return [str(q).strip() for q in data.get("questions", [])
                    if str(q).strip()][:3]
        except Exception:
            return []

    def verify_grounding(self, answer: str, chunks: list[dict],
                         usage: dict | None = None) -> dict | None:
        """Check that every cited claim [n] in the answer is supported by
        source n (online Faithfulness). One cheap LLM call, run AFTER the
        answer is streamed. Returns {"total", "verified", "unsupported"[]},
        or None when there is nothing to verify (no citations) or on failure.
        """
        # Cheap gate: no citation markers → nothing to verify.
        if not chunks or not re.search(r"\[\d+\]", answer):
            return None
        if llm.is_mock():
            return {"total": 1, "verified": 1, "unsupported": []}  # mock: all good
        # Accept either full chunks (`text`) or UI source cards (`snippet`).
        context = "\n\n---\n\n".join(
            f"[Nguồn {c.get('rank', i + 1)}: {c.get('title', '')}]\n"
            f"{c.get('text') or c.get('snippet', '')}"
            for i, c in enumerate(chunks)
        )
        try:
            client = llm.get_client()
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": GROUNDING_PROMPT},
                    {"role": "user", "content":
                        f"Nguồn:\n{context}\n\nCâu trả lời:\n{answer}"},
                ],
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            _add_usage(usage, res)
            data = json.loads(res.choices[0].message.content or "{}")
            total = int(data.get("total", 0) or 0)
            unsupported = [str(u).strip() for u in data.get("unsupported", [])
                           if str(u).strip()]
            if total <= 0:
                return None
            # Guard against a model that lists more failures than claims.
            verified = max(0, total - len(unsupported))
            return {"total": total, "verified": verified, "unsupported": unsupported}
        except Exception:
            return None

    # ── Multi-Hop (Layer 6) ────────────────────────────────

    def has_any_source(self, session_id: str | None = None,
                       include_doc_ids: list[str] | None = None,
                       use_global_kb: bool = True) -> bool:
        """Cheap check whether retrieval has anything to search."""
        chunks = [c for c in self._chunks_store
                  if not session_id or c.get("session_id") == session_id]
        if include_doc_ids is not None:
            allowed = set(include_doc_ids)
            chunks = [c for c in chunks if c.get("doc_id") in allowed]
        return bool(chunks) or (use_global_kb and not get_kb().is_empty())


    # ── Internals ──────────────────────────────────────────

    def _digest_retrieve(self, question: str, session_docs: list[dict],
                         attached_doc_ids: list[str] | None) -> dict | None:
        """Direct-read path for "tóm tắt link này"-type questions.

        Such questions are topically empty, so similarity search returns
        noise. When the target document is identifiable — attached to this
        message, named in the question, or referenced deictically ("đường
        link đó") — answer from its chunks in document order instead.
        Returns None when this path does not apply.
        """
        if not session_docs or not _SUMMARY_INTENT_RE.search(question):
            return None

        targets: list[str] = []
        if attached_doc_ids:
            valid = {d["id"] for d in session_docs}
            targets = [i for i in attached_doc_ids if i in valid]
        if not targets:
            q = question.lower()
            # Doc named in the question ("tóm tắt file report.pdf")
            targets = [d["id"] for d in session_docs
                       if len(d["name"]) > 4
                       and d["name"].lower().rsplit(".", 1)[0] in q][:2]
        if not targets and _DOC_REF_RE.search(question):
            # "đường link đó" / "bài viết này" — most recently added doc.
            # added_at wins over registry position: after a restore the
            # registry order is not guaranteed chronological.
            newest = max(enumerate(session_docs),
                         key=lambda t: (t[1].get("added_at", 0.0), t[0]))[1]
            targets = [newest["id"]]
        if not targets:
            return None

        chunks = self._ordered_doc_chunks(targets)
        if not chunks:
            return None
        for i, c in enumerate(chunks, 1):
            c["rank"] = i
        return {
            "empty":      False,
            "notice":     None,
            "top_chunks": chunks,
            "sources":    _chunks_to_sources(chunks),
            "complexity": "simple",
        }

    def _ordered_doc_chunks(self, doc_ids: list[str], cap: int = 8) -> list[dict]:
        """A document's chunks in reading order, evenly sampled down to
        `cap` (first and last always kept) so long docs still fit the
        generation context."""
        def chunk_index(c: dict) -> int:
            m = re.search(r"_c(\d+)$", c["id"])
            return int(m.group(1)) if m else 0

        out: list[dict] = []
        for did in doc_ids:
            rows = sorted((c for c in self._chunks_store if c.get("doc_id") == did),
                          key=chunk_index)
            out.extend(rows)
        if len(out) > cap:
            picks = sorted({round(j * (len(out) - 1) / (cap - 1)) for j in range(cap)})
            out = [out[i] for i in picks]
        # score 1.0: chunks are read from the requested doc, not ranked guesses
        return [{**{k: v for k, v in c.items() if k != "tokens"},
                 "scope": "session", "score": 1.0} for c in out]

    def _condense_query(self, question: str, history: list[dict],
                        usage: dict | None = None) -> str:
        """Rewrite a history-dependent follow-up into a standalone search
        query in ONE LLM call. Returns the question unchanged when there is
        no history (nothing to condense), in mock mode, and on any failure.
        Generation still sees the original question + history."""
        if not history or llm.is_mock():
            return question
        convo = "\n".join(
            f"{'Người dùng' if m.get('role') == 'user' else 'Trợ lý'}: {str(m['content'])[:500]}"
            for m in history[-6:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        )
        try:
            client = llm.get_client()
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": CONDENSE_PROMPT},
                    {"role": "user", "content":
                        f"Lịch sử hội thoại:\n{convo or '(không có)'}\n\nCâu hỏi mới: {question}"},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            _add_usage(usage, res)
            return (res.choices[0].message.content or "").strip() or question
        except Exception:
            return question

    # ── Answer cache ───────────────────────────────────────

    def _cache_scope(self, session_id: str | None,
                     include_doc_ids: list[str] | None,
                     use_global_kb: bool) -> tuple:
        return (session_id or "",
                None if include_doc_ids is None else frozenset(include_doc_ids),
                bool(use_global_kb))

    def _cache_versions(self) -> tuple:
        """Doc-state fingerprint: any session-doc or global-KB change moves
        it, silently expiring every answer cached on the old corpus."""
        return (self._store_version, get_kb().version)

    def _embed_question(self, question: str) -> list | None:
        """Question embedding for cache similarity; None on any failure."""
        try:
            return list(self._embed_fn([question])[0])
        except Exception:
            return None

    def cache_lookup(self, question: str, session_id: str | None = None,
                     include_doc_ids: list[str] | None = None,
                     use_global_kb: bool = True) -> dict | None:
        """Cached final answer for a repeat question in the same scope.

        Exact normalized-text match first (free); otherwise embedding cosine
        ≥ ANSWER_CACHE_SIM against cached questions — one cheap embedding
        call, attempted only when same-scope candidates exist. Callers must
        only use this for history-free, attachment-free questions: follow-ups
        depend on conversation state the cache cannot see.
        """
        scope = self._cache_scope(session_id, include_doc_ids, use_global_kb)
        q_norm = " ".join(question.lower().split())
        ver = self._cache_versions()
        now = time.time()
        with self._lock:
            live = [e for e in self._answer_cache.values()
                    if e["scope"] == scope and e["ver"] == ver
                    and now - e["ts"] < ANSWER_CACHE_TTL_S]
            for e in live:
                if e["q_norm"] == q_norm:
                    return e
        candidates = [e for e in live if e.get("emb")]
        if not candidates:
            return None
        emb = self._embed_question(question)  # network call — outside the lock
        if not emb:
            return None
        sim, best = max(((_cosine(emb, e["emb"]), e) for e in candidates),
                        key=lambda t: t[0])
        return best if sim >= ANSWER_CACHE_SIM else None

    def cache_store(self, question: str, session_id: str | None,
                    include_doc_ids: list[str] | None, use_global_kb: bool,
                    answer: str, sources: list[dict], complexity: str,
                    steps: list[dict] | None = None,
                    suggestions: list[str] | None = None,
                    grounding: dict | None = None) -> None:
        scope = self._cache_scope(session_id, include_doc_ids, use_global_kb)
        q_norm = " ".join(question.lower().split())
        entry = {
            "scope": scope, "q_norm": q_norm, "ver": self._cache_versions(),
            "ts": time.time(), "emb": self._embed_question(question),
            "answer": answer, "sources": sources,
            "complexity": complexity, "steps": steps or [],
            "suggestions": suggestions or [], "grounding": grounding,
        }
        key = hashlib.sha1(repr((scope, q_norm)).encode()).hexdigest()
        with self._lock:
            self._answer_cache[key] = entry
            self._answer_cache.move_to_end(key)
            while len(self._answer_cache) > ANSWER_CACHE_MAX:
                self._answer_cache.popitem(last=False)

    def _chat_messages(self, question: str, chunks: list[dict],
                       history: list[dict]) -> list[dict]:
        """Build the LLM message list for chat generation.

        Includes recent conversation history so follow-up questions
        ("nó là gì?", "còn cách nào khác?") resolve correctly.
        """
        context_str = "\n\n---\n\n".join(
            f"[Nguồn {c.get('rank', i + 1)}: {c.get('title', 'Unknown')}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )

        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        for m in history[-6:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": str(m["content"])})
        messages.append({
            "role": "user",
            "content": (f"Ngữ cảnh:\n{context_str}\n\nCâu hỏi: {question}\n\n"
                        f"{_lang_directive(question)}"),
        })
        return messages

    def _generate_chat_answer(self, question: str, chunks: list[dict],
                              history: list[dict]) -> str:
        """Generate a structured Markdown answer for the chat UI in one shot."""
        if llm.is_mock():
            preview = chunks[0]["text"][:200] if chunks else "No context"
            return f"[MOCK ANSWER] Based on context: '{preview}...'"

        client = llm.get_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=self._chat_messages(question, chunks, history),
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()

    def _bm25_search(self, queries: str | list[str], n: int = 20,
                     session_id: str | None = None,
                     allowed_doc_ids: set[str] | None = None) -> list[dict]:
        if isinstance(queries, str):
            queries = [queries]
        bm25, pool = self._bm25_for(session_id, allowed_doc_ids)
        if bm25 is None:
            return []
        # Multi-language query variants: keep each chunk's best score
        # (an EN translation is what actually matches an EN corpus).
        merged = [0.0] * len(pool)
        for q in queries:
            for idx, score in enumerate(bm25.get_scores(_tokenize(q))):
                if score > merged[idx]:
                    merged[idx] = score
        ranked = sorted(enumerate(merged), key=lambda x: x[1], reverse=True)[:n]
        return [{**{k: v for k, v in pool[idx].items() if k != "tokens"},
                 "scope": "session", "score": round(float(score), 4)}
                for idx, score in ranked if score > 0]

    def _bm25_for(self, session_id: str | None,
                  allowed_doc_ids: set[str] | None) -> tuple:
        """Cached BM25 index for one (session, source-picker) chunk pool.

        Scoring stays within the exact pool so IDF weights are not skewed by
        documents from other conversations or unchecked sources — hence the
        filter is part of the cache key. Entries live until the next document
        add/remove (which clears the cache) and are LRU-capped.
        """
        key = (session_id or "",
               None if allowed_doc_ids is None else frozenset(allowed_doc_ids))
        with self._lock:
            entry = self._bm25_cache.get(key)
            if entry is not None:
                self._bm25_cache.move_to_end(key)
                return entry
            pool = [c for c in self._chunks_store
                    if (not session_id or c.get("session_id") == session_id)
                    and (allowed_doc_ids is None or c.get("doc_id") in allowed_doc_ids)]
            entry = (BM25Okapi([c["tokens"] for c in pool]), pool) if pool else (None, [])
            self._bm25_cache[key] = entry
            while len(self._bm25_cache) > 8:
                self._bm25_cache.popitem(last=False)
            return entry

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
                session_id = meta.get("session_id", "")
                # BM25 tokens follow the SAME contextual text used for the
                # stored embeddings (rebuilt from the persisted context).
                ctx_text = contextual_text({"text": text, "context": meta.get("context", "")})
                self._chunks_store.append({
                    "id": cid, "text": text,
                    "title": meta.get("title", ""), "doc_id": doc_id,
                    "session_id": session_id,
                    "heading": meta.get("heading", ""),
                    "tokens": _tokenize(ctx_text),
                    **({"page": meta["page"]} if "page" in meta else {}),
                })
                if doc_id not in chunk_by_doc:
                    chunk_by_doc[doc_id] = {"name": meta.get("title", ""), "type": meta.get("source", ""),
                                            "session_id": session_id, "count": 0,
                                            "added_at": float(meta.get("added_at", 0.0))}
                chunk_by_doc[doc_id]["count"] += 1

            # Chronological registry (docs indexed before added_at existed
            # sort first, keeping their relative Chroma order).
            for doc_id, info in sorted(chunk_by_doc.items(),
                                       key=lambda kv: kv[1]["added_at"]):
                self._doc_registry.append({
                    "id": doc_id, "name": info["name"],
                    "type": info["type"], "session_id": info["session_id"],
                    "chunk_count": info["count"],
                    "added_at": info["added_at"],
                })
        except Exception as e:
            log.warning("Restore warning: %s", e)
