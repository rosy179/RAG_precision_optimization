"""
Pure, stateless helpers shared across the RAG service modules
(user_rag.py, multihop.py). No I/O, no service state — kept apart so
multihop.py can reuse them without importing back into user_rag.py.
"""

import math
import re


def tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def merge_lexical(session_hits: list[dict], global_hits: list[dict], n: int = 20) -> list[dict]:
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


def reattach_chunk_fields(fused: list[dict], *source_lists: list[dict]):
    """RRF (hybrid_rag.reciprocal_rank_fusion) rebuilds chunk dicts and drops
    score/doc_id/scope — re-attach them so the HyDE low-score fallback and
    the per-source scope badge keep working."""
    by_id: dict[str, dict] = {}
    for chunks in source_lists:  # later lists win (semantic cosine score preferred)
        for c in chunks:
            by_id[c["id"]] = c
    for c in fused:
        orig = by_id.get(c["id"], {})
        for key in ("score", "doc_id", "scope", "page"):
            if key in orig:
                c.setdefault(key, orig[key])


def add_usage(usage: dict | None, res) -> None:
    """Accumulate real token usage from an OpenAI response (or streaming
    chunk carrying `.usage`) into the caller's accumulator dict — replaces
    the dashboard's rough cost estimates whenever available."""
    u = getattr(res, "usage", None)
    if usage is None or u is None:
        return
    usage["in"] = usage.get("in", 0) + (u.prompt_tokens or 0)
    usage["out"] = usage.get("out", 0) + (u.completion_tokens or 0)
    usage["real"] = True


def display_score(chunk: dict) -> float:
    """Normalize a chunk's relevance score to 0-1 for the UI badge.

    Depending on the model config, sentence-transformers ≥4 returns rerank
    scores either sigmoid-activated in [0, 1] (bge-reranker-v2-m3) or as
    raw unbounded logits (mmarco-mMiniLMv2) — squash only the latter.
    Semantic scores (1 - distance) can dip below 0 depending on the
    distance metric, so clamp instead.
    """
    if "rerank_score" in chunk:
        s = chunk["rerank_score"]
        if not 0.0 <= s <= 1.0:
            s = 1.0 / (1.0 + math.exp(-s))
        return round(s, 4)
    return round(max(0.0, min(1.0, chunk.get("score", 0.0))), 4)


def chunks_to_sources(chunks: list[dict]) -> list[dict]:
    """UI-ready source cards for a list of retrieved chunks."""
    return [{
        "rank":     c.get("rank", i + 1),
        "title":    c.get("title", "Unknown"),
        "snippet":  c.get("text", "")[:300],
        "score":    display_score(c),
        "scope":    c.get("scope", "session"),  # "global" = shared KB
        "doc_id":   c.get("doc_id", ""),
        "chunk_id": c.get("id", ""),
        **({"page": c["page"]} if c.get("page") else {}),
    } for i, c in enumerate(chunks)]
