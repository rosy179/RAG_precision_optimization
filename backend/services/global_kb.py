"""
Global shared knowledge base.

One ChromaDB collection pair shared by ALL users and sessions, next to the
per-user session-scoped collections in user_rag.py. Filled via the
/api/knowledge endpoints or scripts/ingest_knowledge.py; every chat query
merges retrieval from the session's own documents and this KB.
"""

import re
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi

from backend.services.embeddings import get_embedding_function

log = logging.getLogger("rag.kb")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "chroma_db_users"

_kb: Optional["GlobalKBService"] = None
_kb_lock = threading.Lock()


def get_kb() -> "GlobalKBService":
    global _kb
    if _kb is None:
        with _kb_lock:
            if _kb is None:
                _kb = GlobalKBService()
    return _kb


def _tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


class GlobalKBService:
    """Shared knowledge base with the same retrieval surface as a user's
    session documents: semantic search over chunks, cached BM25, and a
    summary collection for hierarchical doc routing.

    Unlike user documents there is no SQL row per document — the registry
    lives entirely in Chroma metadata and is restored on startup, so no
    schema migration is needed.
    """

    def __init__(self):
        DB_PATH.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(DB_PATH))

        # Shared factory — same vector space as the per-user collections.
        embed_fn = get_embedding_function()

        self._chunk_col = self._chroma.get_or_create_collection(
            "global_kb_chunks",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._summary_col = self._chroma.get_or_create_collection(
            "global_kb_summaries",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self._write_lock = threading.Lock()
        self._chunks_store: list[dict] = []
        self._doc_registry: list[dict] = []
        # Index and pool are kept as one tuple so a search never scores
        # against a pool the index wasn't built from. Unlike the per-session
        # BM25 (tiny, rebuilt per query), the KB corpus can be large, so the
        # index is rebuilt only on ingestion.
        self._bm25_state: tuple[Optional[BM25Okapi], list[dict]] = (None, [])
        # Bumped on every ingest/remove/reload — consumed by the per-user
        # answer caches to expire entries built on an older corpus.
        self._version = 0

        self._restore_from_chroma()
        log.info("Global knowledge base ready: %d docs, %d chunks.",
                 len(self._doc_registry), len(self._chunks_store))

    # ── Ingestion ──────────────────────────────────────────

    def add_document(self, chunks: list[dict], doc_summary: str, doc_meta: dict) -> int:
        if not chunks:
            return 0
        doc_id = doc_meta["id"]
        created_at = datetime.utcnow().isoformat()

        with self._write_lock:
            self._chunk_col.upsert(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[{
                    "doc_id": doc_id,
                    "title":  doc_meta["name"],
                    "source": doc_meta.get("type", "unknown"),
                    "chunk_index": i,
                    **({"page": c["page"]} if "page" in c else {}),
                } for i, c in enumerate(chunks)],
            )
            self._summary_col.upsert(
                ids=[doc_id],
                documents=[doc_summary],
                metadatas=[{"name": doc_meta["name"], "type": doc_meta.get("type", "unknown"),
                            "created_at": created_at}],
            )
            # Replace lists instead of mutating so in-flight searches keep
            # a consistent snapshot.
            self._chunks_store = self._chunks_store + [
                {"id": c["id"], "text": c["text"], "title": doc_meta["name"],
                 "doc_id": doc_id, "tokens": _tokenize(c["text"])}
                for c in chunks
            ]
            self._doc_registry = self._doc_registry + [{
                "id": doc_id,
                "name": doc_meta["name"],
                "type": doc_meta.get("type", "unknown"),
                "chunk_count": len(chunks),
                "created_at": created_at,
            }]
            self._rebuild_bm25()
            self._version += 1
        return len(chunks)

    def remove_document(self, doc_id: str):
        with self._write_lock:
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
            self._version += 1

    def get_documents(self) -> list[dict]:
        return self._doc_registry

    def get_document_content(self, doc_id: str) -> dict | None:
        """Reconstruct a KB document's full text + per-chunk offsets for the
        viewer (same contract as UserRAGService.get_document_content)."""
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

    def chunk_count(self) -> int:
        return len(self._chunks_store)

    # ── Chunk-level editing (admin precision tooling) ──────

    def list_chunks(self, doc_id: str) -> list[dict] | None:
        """Every parsed chunk of a document in reading order, for the admin
        chunk editor. Returns None if the document has no chunks."""
        res = self._chunk_col.get(where={"doc_id": doc_id},
                                  include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        rows = sorted(
            zip(res["ids"], res["documents"], res["metadatas"]),
            key=lambda r: r[2].get("chunk_index", 0),
        )
        return [{
            "id": cid,
            "text": text,
            "chunk_index": meta.get("chunk_index", 0),
            **({"page": meta["page"]} if "page" in meta else {}),
        } for cid, text, meta in rows]

    def update_chunk(self, doc_id: str, chunk_id: str, new_text: str) -> bool:
        """Replace a chunk's text (re-embeds it) and refresh the in-memory
        store + BM25. Returns False if the chunk doesn't belong to the doc."""
        new_text = new_text.strip()
        if not new_text:
            raise ValueError("Nội dung chunk không được để trống")
        existing = self._chunk_col.get(ids=[chunk_id], include=["metadatas"])
        if not existing["ids"]:
            return False
        meta = existing["metadatas"][0] if existing["metadatas"] else {}
        if meta.get("doc_id") != doc_id:
            return False
        with self._write_lock:
            # upsert with the same id + metadata re-embeds the new text
            self._chunk_col.upsert(ids=[chunk_id], documents=[new_text],
                                   metadatas=[meta])
            self._chunks_store = [
                {**c, "text": new_text, "tokens": _tokenize(new_text)}
                if c["id"] == chunk_id else c
                for c in self._chunks_store
            ]
            self._rebuild_bm25()
            self._version += 1
        return True

    def delete_chunk(self, doc_id: str, chunk_id: str) -> bool:
        """Remove one junk chunk. Returns False if it doesn't belong to the
        doc; keeps the document even if it becomes empty (delete the doc for
        that). Updates the registry chunk_count."""
        existing = self._chunk_col.get(ids=[chunk_id], include=["metadatas"])
        if not existing["ids"]:
            return False
        meta = existing["metadatas"][0] if existing["metadatas"] else {}
        if meta.get("doc_id") != doc_id:
            return False
        with self._write_lock:
            self._chunk_col.delete(ids=[chunk_id])
            self._chunks_store = [c for c in self._chunks_store if c["id"] != chunk_id]
            self._doc_registry = [
                {**d, "chunk_count": max(0, d["chunk_count"] - 1)}
                if d["id"] == doc_id else d
                for d in self._doc_registry
            ]
            self._rebuild_bm25()
            self._version += 1
        return True

    def is_empty(self) -> bool:
        return not self._chunks_store

    @property
    def version(self) -> int:
        return self._version

    def reload(self) -> dict:
        """Re-read the persisted Chroma collections into memory (registry +
        BM25 index). Lets bulk ingests done by scripts/ingest_knowledge.py in
        a separate process show up without restarting the server."""
        with self._write_lock:
            self._chunks_store = []
            self._doc_registry = []
            self._bm25_state = (None, [])
            self._restore_from_chroma()
            self._version += 1
        return {"documents": len(self._doc_registry), "chunks": len(self._chunks_store)}

    # ── Search (consumed by UserRAGService.query) ──────────

    def search_summaries(self, queries: str | list[str], top_k: int = 5) -> list[str]:
        """Doc ids of the KB documents most relevant to the query (multiple
        query variants — e.g. original + English translation — are unioned)."""
        if isinstance(queries, str):
            queries = [queries]
        n = min(top_k, len(self._doc_registry))
        if n == 0:
            return []
        res = self._summary_col.query(query_texts=queries, n_results=n)
        out: list[str] = []
        seen: set = set()
        for id_list in (res["ids"] or []):
            for did in id_list:
                if did not in seen:
                    seen.add(did)
                    out.append(did)
        return out

    def search_semantic(self, queries: str | list[str], n_results: int = 20,
                        doc_ids: Optional[list[str]] = None) -> list[dict]:
        if isinstance(queries, str):
            queries = [queries]
        n = min(n_results, len(self._chunks_store))
        if n == 0:
            return []
        kwargs = {"query_texts": queries, "n_results": n}
        if doc_ids:
            kwargs["where"] = {"doc_id": {"$in": doc_ids}}
        res = self._chunk_col.query(**kwargs)
        # A chunk found by several query variants keeps its best score
        best: dict[str, dict] = {}
        for qi in range(len(res["ids"] or [])):
            for i, cid in enumerate(res["ids"][qi]):
                meta = res["metadatas"][qi][i] if res["metadatas"] else {}
                dist = res["distances"][qi][i] if res["distances"] else 1.0
                score = round(1.0 - float(dist), 4)
                if cid in best and best[cid]["score"] >= score:
                    continue
                best[cid] = {
                    "id":     cid,
                    "text":   res["documents"][qi][i],
                    "title":  meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "scope":  "global",
                    "score":  score,
                    **({"page": meta["page"]} if "page" in meta else {}),
                }
        return sorted(best.values(), key=lambda c: c["score"], reverse=True)[:n]

    def search_bm25(self, queries: str | list[str], n: int = 20) -> list[dict]:
        if isinstance(queries, str):
            queries = [queries]
        bm25, pool = self._bm25_state
        if bm25 is None:
            return []
        merged = [0.0] * len(pool)
        for q in queries:
            for idx, score in enumerate(bm25.get_scores(_tokenize(q))):
                if score > merged[idx]:
                    merged[idx] = score
        ranked = sorted(enumerate(merged), key=lambda x: x[1], reverse=True)[:n]
        return [{"id": pool[idx]["id"], "text": pool[idx]["text"],
                 "title": pool[idx]["title"], "doc_id": pool[idx]["doc_id"],
                 "scope": "global", "score": round(float(score), 4)}
                for idx, score in ranked if score > 0]

    # ── Internals ──────────────────────────────────────────

    def _rebuild_bm25(self):
        pool = self._chunks_store
        if pool:
            self._bm25_state = (BM25Okapi([c["tokens"] for c in pool]), pool)
        else:
            self._bm25_state = (None, [])

    def _restore_from_chroma(self):
        """Rebuild in-memory state from persisted ChromaDB on startup."""
        try:
            all_chunks = self._chunk_col.get(include=["documents", "metadatas"])
            if not all_chunks["ids"]:
                return
            store: list[dict] = []
            by_doc: dict[str, dict] = {}
            for i, cid in enumerate(all_chunks["ids"]):
                meta = all_chunks["metadatas"][i] if all_chunks["metadatas"] else {}
                text = all_chunks["documents"][i] if all_chunks["documents"] else ""
                doc_id = meta.get("doc_id", "unknown")
                store.append({"id": cid, "text": text, "title": meta.get("title", ""),
                              "doc_id": doc_id, "tokens": _tokenize(text)})
                info = by_doc.setdefault(doc_id, {"name": meta.get("title", ""),
                                                  "type": meta.get("source", ""), "count": 0})
                info["count"] += 1

            created: dict[str, str] = {}
            try:
                summaries = self._summary_col.get(include=["metadatas"])
                for i, sid in enumerate(summaries["ids"]):
                    meta = summaries["metadatas"][i] if summaries["metadatas"] else {}
                    created[sid] = meta.get("created_at", "")
            except Exception:
                pass

            self._chunks_store = store
            self._doc_registry = [{
                "id": doc_id, "name": info["name"], "type": info["type"],
                "chunk_count": info["count"], "created_at": created.get(doc_id, ""),
            } for doc_id, info in by_doc.items()]
            self._rebuild_bm25()
        except Exception as e:
            log.warning("Restore warning: %s", e)
