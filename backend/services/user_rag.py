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

from backend.services.global_kb import get_kb

DB_PATH     = Path(__file__).parent.parent.parent / "data" / "chroma_db_users"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
LLM_MODEL   = os.getenv("LLM_MODEL", "gpt-4o-mini")
RERANKER    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# System prompt for the chat UI. Unlike the eval pipeline (cot_rag.py, tuned
# for short Ragas-scored answers), chat answers must be structured Markdown.
CHAT_SYSTEM_PROMPT = """\
Bạn là trợ lý tri thức IT, trả lời câu hỏi dựa trên tài liệu người dùng đã cung cấp.

NGUYÊN TẮC NỘI DUNG:
1. Chỉ dùng thông tin từ phần "Ngữ cảnh" và lịch sử hội thoại — tuyệt đối không bịa thêm ngoài hai nguồn đó.
2. Với yêu cầu trình bày lại nội dung đã trả lời trước đó (dịch sang ngôn ngữ khác, tóm tắt, rút gọn, đổi định dạng), hãy THỰC HIỆN dựa trên câu trả lời trước trong lịch sử hội thoại và ngữ cảnh — không được từ chối vì lý do "chỉ dùng ngữ cảnh".
3. Nếu ngữ cảnh không đủ để trả lời trọn vẹn, trả lời phần có thể và nói rõ phần nào còn thiếu thông tin.
4. Khách quan và chính xác: giữ nguyên số liệu, tên riêng, thuật ngữ trong tài liệu; giải thích ngắn gọn thuật ngữ khó.
5. Ngôn ngữ trả lời: nếu người dùng yêu cầu ngôn ngữ cụ thể (ví dụ "bằng tiếng Nhật") thì dùng đúng ngôn ngữ đó; nếu không, trả lời bằng ngôn ngữ của câu hỏi.

ĐỊNH DẠNG BẮT BUỘC (Markdown):
- Mở đầu bằng 1-2 câu trả lời thẳng vào ý chính của câu hỏi.
- Triển khai chi tiết bằng gạch đầu dòng "- ", mỗi ý một dòng, in đậm **từ khóa** ở đầu mỗi ý.
- Nếu câu trả lời có nhiều khía cạnh, chia mục bằng tiêu đề "### ".
- Dùng danh sách đánh số (1. 2. 3.) cho các bước hoặc quy trình.
- Dùng bảng Markdown khi cần so sánh từ 2 đối tượng trở lên.
- Không bao giờ dồn toàn bộ câu trả lời vào một dòng hay một đoạn văn duy nhất."""

# Rewrites follow-up questions ("dịch sang tiếng Nhật", "nói rõ hơn ý 2")
# into standalone retrieval queries using the conversation history.
CONDENSE_PROMPT = """\
Bạn nhận lịch sử hội thoại và một câu hỏi mới. Viết lại câu hỏi mới thành MỘT câu truy vấn tìm kiếm độc lập, nêu rõ chủ đề đang được nói tới, để tìm các đoạn tài liệu liên quan.
- Giữ nguyên ngôn ngữ chính của chủ đề trong hội thoại.
- Bỏ các yêu cầu về cách trình bày (ví dụ: "dịch sang tiếng Nhật", "tóm tắt lại", "viết ngắn hơn") — chỉ giữ chủ đề nội dung.
- Nếu câu hỏi mới đã độc lập và rõ ràng, trả lại nguyên văn.
Chỉ trả về câu truy vấn, không giải thích gì thêm."""

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


def _merge_lexical(session_hits: list[dict], global_hits: list[dict], n: int = 20) -> list[dict]:
    """Merge BM25 results from the session corpus and the global KB.

    Raw BM25 scores are not comparable across corpora (different IDF
    statistics), so normalize each list by its own max before merging.
    RRF downstream only consumes ranks, so this just needs to produce a
    fair combined ordering.
    """
    merged: list[dict] = []
    for hits in (session_hits, global_hits):
        if hits:
            mx = max(h["score"] for h in hits) or 1.0
            merged.extend({**h, "score": round(h["score"] / mx, 4)} for h in hits)
    merged.sort(key=lambda h: h["score"], reverse=True)
    return merged[:n]


def _reattach_chunk_fields(fused: list[dict], *source_lists: list[dict]):
    """RRF (hybrid_rag.reciprocal_rank_fusion) rebuilds chunk dicts and drops
    score/doc_id/scope — re-attach them so the HyDE low-score fallback and
    the per-source scope badge keep working."""
    by_id: dict[str, dict] = {}
    for chunks in source_lists:  # later lists win (semantic cosine score preferred)
        for c in chunks:
            by_id[c["id"]] = c
    for c in fused:
        orig = by_id.get(c["id"], {})
        for key in ("score", "doc_id", "scope"):
            if key in orig:
                c.setdefault(key, orig[key])


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
        self._doc_registry: list[dict] = []  # {id, name, type, session_id, chunk_count}

        # Restore from ChromaDB on init (handles server restarts)
        self._restore_from_chroma()

    # ── Ingestion ──────────────────────────────────────────

    def add_document(self, chunks: list[dict], doc_summary: str, doc_meta: dict) -> int:
        """Upsert chunks + summary into ChromaDB and rebuild BM25."""
        if not chunks:
            return 0

        doc_id = doc_meta["id"]
        session_id = doc_meta.get("session_id", "")

        # Upsert chunks into chunk collection
        self._chunk_col.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{
                "doc_id": doc_id,
                "session_id": session_id,
                "title":  doc_meta["name"],
                "source": doc_meta.get("type", "unknown"),
                "chunk_index": i,
            } for i, c in enumerate(chunks)],
        )

        # Upsert doc summary into summary collection
        self._summary_col.upsert(
            ids=[doc_id],
            documents=[doc_summary],
            metadatas=[{"name": doc_meta["name"], "type": doc_meta.get("type", "unknown"),
                        "session_id": session_id}],
        )

        # Update in-memory BM25 store
        for c in chunks:
            self._chunks_store.append({"id": c["id"], "text": c["text"],
                                       "title": doc_meta["name"], "doc_id": doc_id,
                                       "session_id": session_id})

        # Update registry
        self._doc_registry.append({
            "id": doc_id,
            "name": doc_meta["name"],
            "type": doc_meta.get("type", "unknown"),
            "session_id": session_id,
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

    def get_documents(self) -> list[dict]:
        return self._doc_registry

    def chunk_count(self) -> int:
        return self._chunk_col.count()

    # ── Query ──────────────────────────────────────────────

    def query(self, question: str, history: list[dict] | None = None,
              session_id: str | None = None) -> dict:
        # Retrieval draws from two pools: chunks uploaded in THIS session
        # (per-conversation scope, as before) plus the shared global
        # knowledge base available to every user and session.
        kb = get_kb()
        session_chunks = [c for c in self._chunks_store
                          if not session_id or c.get("session_id") == session_id]
        if not session_chunks and kb.is_empty():
            return {
                "answer": ("Chưa có tài liệu nào — cả trong cuộc trò chuyện này lẫn kho "
                           "kiến thức chung. Hãy đính kèm PDF, dán URL hoặc tải ảnh/ghi âm "
                           "lên trước khi hỏi."),
                "sources": [],
                "reasoning": "",
                "latency_ms": 0,
                "from_cache": False,
            }

        t0 = time.time()
        history = history or []
        session_filter = {"session_id": session_id} if session_id else None

        # Follow-ups ("dịch sang tiếng Nhật", "nói rõ hơn") carry no retrieval
        # signal on their own — condense with history into a standalone query.
        # Generation still sees the original question + history.
        search_query = self._condense_question(question, history)

        # Layer 1 — Adaptive Router
        from adaptive_rag import classify_heuristic
        complexity = classify_heuristic(search_query)

        # Layer 2 — Hierarchical: find relevant docs by summary, separately
        # per pool (session docs and global KB docs live in different
        # collections, so each needs its own doc-id shortlist).
        session_docs = [d for d in self._doc_registry
                        if not session_id or d.get("session_id") == session_id]
        top_k_docs = min(5, len(session_docs))
        relevant_doc_ids: list[str] = []
        if top_k_docs > 0:
            kwargs = {"query_texts": [search_query], "n_results": top_k_docs}
            if session_filter:
                kwargs["where"] = session_filter
            summary_res = self._summary_col.query(**kwargs)
            relevant_doc_ids = summary_res["ids"][0] if summary_res["ids"] else []
        kb_doc_ids = kb.search_summaries(search_query, top_k=5)

        # Layer 3 — Advanced: Semantic + BM25 + RRF, over both pools
        semantic_chunks = []
        if session_chunks:
            filters = []
            if session_filter:
                filters.append(session_filter)
            if relevant_doc_ids:
                filters.append({"doc_id": {"$in": relevant_doc_ids}})
            where_filter = filters[0] if len(filters) == 1 else ({"$and": filters} if filters else None)

            kwargs = {"query_texts": [search_query], "n_results": min(20, len(session_chunks))}
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
                    })

        # Both collections use the same embedding model and cosine space,
        # so their (1 - distance) scores are directly comparable.
        kb_semantic = kb.search_semantic(search_query, n_results=20,
                                         doc_ids=kb_doc_ids or None)
        semantic_chunks = sorted(semantic_chunks + kb_semantic,
                                 key=lambda c: c["score"], reverse=True)[:20]

        bm25_chunks = _merge_lexical(
            self._bm25_search(search_query, n=20, session_id=session_id),
            kb.search_bm25(search_query, n=20),
        )

        from hybrid_rag import reciprocal_rank_fusion
        top_k = 5 if complexity == "complex" else (4 if complexity == "medium" else 3)
        fused = reciprocal_rank_fusion(semantic_chunks, bm25_chunks, top_k=top_k * 4)
        _reattach_chunk_fields(fused, bm25_chunks, semantic_chunks)

        # Rerank if not simple
        if complexity != "simple" and _reranker and len(fused) > 1:
            pairs = [(search_query, c["text"]) for c in fused]
            scores = _reranker.predict(pairs)
            for chunk, score in zip(fused, scores):
                chunk["rerank_score"] = float(score)
            fused.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        top_chunks = fused[:top_k]

        # Layer 4 — Iterative fallback: if best score < 0.5, HyDE re-query
        best_score = top_chunks[0].get("score", 1.0) if top_chunks else 0.0
        if best_score < 0.5:
            from query_expansion import generate_hypothetical_doc
            hyde = generate_hypothetical_doc(search_query)
            if hyde and hyde != search_query:
                hyde_chunks = []
                if session_chunks:
                    hyde_kwargs = {"query_texts": [hyde],
                                   "n_results": min(10, len(session_chunks))}
                    if session_filter:
                        hyde_kwargs["where"] = session_filter
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
                            })
                hyde_chunks += kb.search_semantic(hyde, n_results=10)
                fused2 = reciprocal_rank_fusion(top_chunks, hyde_chunks, top_k=top_k)
                _reattach_chunk_fields(fused2, top_chunks, hyde_chunks)
                top_chunks = fused2[:top_k]

        # Add rank field for CoT context builder
        for i, c in enumerate(top_chunks, 1):
            c["rank"] = i

        # Layer 5 — Structured chat generation (grounded in top_chunks + history)
        answer = self._generate_chat_answer(question, top_chunks, history)
        reasoning = ""

        latency = int((time.time() - t0) * 1000)

        sources = [{
            "rank":    c.get("rank", i + 1),
            "title":   c.get("title", "Unknown"),
            "snippet": c.get("text", "")[:300],
            "score":   _display_score(c),
            "scope":   c.get("scope", "session"),  # "global" = shared KB
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

    def _condense_question(self, question: str, history: list[dict]) -> str:
        """Rewrite a follow-up question into a standalone retrieval query.

        First messages and mock mode skip the extra LLM call. Any failure
        falls back to the raw question so retrieval still runs.
        """
        if not history or not self._api_key or self._api_key.startswith("sk-your"):
            return question
        convo = "\n".join(
            f"{'Người dùng' if m.get('role') == 'user' else 'Trợ lý'}: {str(m['content'])[:500]}"
            for m in history[-6:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        )
        try:
            client = openai.OpenAI(api_key=self._api_key)
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": CONDENSE_PROMPT},
                    {"role": "user",
                     "content": f"Lịch sử hội thoại:\n{convo}\n\nCâu hỏi mới: {question}\n\nCâu truy vấn độc lập:"},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            rewritten = (res.choices[0].message.content or "").strip().strip('"')
            return rewritten or question
        except Exception:
            return question

    def _generate_chat_answer(self, question: str, chunks: list[dict],
                              history: list[dict]) -> str:
        """Generate a structured Markdown answer for the chat UI.

        Includes recent conversation history so follow-up questions
        ("nó là gì?", "còn cách nào khác?") resolve correctly.
        """
        if not self._api_key or self._api_key.startswith("sk-your"):
            preview = chunks[0]["text"][:200] if chunks else "No context"
            return f"[MOCK ANSWER] Based on context: '{preview}...'"

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
            "content": f"Ngữ cảnh:\n{context_str}\n\nCâu hỏi: {question}",
        })

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()

    def _bm25_search(self, query: str, n: int = 20,
                     session_id: str | None = None) -> list[dict]:
        # Score only within the session's chunks so IDF weights are not
        # skewed by documents from other conversations.
        pool = [c for c in self._chunks_store
                if not session_id or c.get("session_id") == session_id]
        if not pool:
            return []
        bm25 = BM25Okapi([_tokenize(c["text"]) for c in pool])
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
        return [{**pool[idx], "scope": "session", "score": round(float(score), 4)}
                for idx, score in ranked if score > 0]

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
                session_id = meta.get("session_id", "")
                self._chunks_store.append({
                    "id": cid, "text": text,
                    "title": meta.get("title", ""), "doc_id": doc_id,
                    "session_id": session_id,
                })
                if doc_id not in chunk_by_doc:
                    chunk_by_doc[doc_id] = {"name": meta.get("title", ""), "type": meta.get("source", ""),
                                            "session_id": session_id, "count": 0}
                chunk_by_doc[doc_id]["count"] += 1

            for doc_id, info in chunk_by_doc.items():
                self._doc_registry.append({
                    "id": doc_id, "name": info["name"],
                    "type": info["type"], "session_id": info["session_id"],
                    "chunk_count": info["count"],
                })
        except Exception as e:
            print(f"[RAG] Restore warning: {e}")
