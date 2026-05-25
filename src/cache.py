#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/cache.py
------------
Lightweight caching layer for RAG pipelines.

Three independent caches (all disk-backed via JSON):
  - EmbeddingCache  : query text → embedding vector
  - ResponseCache   : question → generated answer
  - RerankerCache   : (query, doc_id) → reranker score

All caches are write-through: reads check memory first, then disk.
Cache keys are sha256 hashes of normalized inputs.
"""

import hashlib
import json
import os
import time
from pathlib import Path


def _sha(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]


class _DiskCache:
    """Generic disk-backed dictionary cache."""

    def __init__(self, cache_file: Path, max_entries: int = 2000):
        self._file = cache_file
        self._max = max_entries
        self._data: dict = {}
        self._hits = 0
        self._misses = 0
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self):
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)

    def get(self, key: str):
        value = self._data.get(key)
        if value is not None:
            self._hits += 1
        else:
            self._misses += 1
        return value

    def set(self, key: str, value):
        if len(self._data) >= self._max:
            # Evict oldest 10%
            evict_n = max(1, self._max // 10)
            for k in list(self._data.keys())[:evict_n]:
                del self._data[k]
        self._data[key] = value
        self._save()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def size(self) -> int:
        return len(self._data)

    def stats(self) -> dict:
        return {
            "size":      self.size,
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(self.hit_rate, 3),
        }

    def clear(self):
        self._data = {}
        if self._file.exists():
            self._file.unlink()


class EmbeddingCache(_DiskCache):
    """Cache for text → embedding vector lookups."""

    def __init__(self, cache_dir: str = "cache"):
        super().__init__(Path(cache_dir) / "embeddings.json")

    def get_embedding(self, text: str):
        return self.get(_sha(text))

    def set_embedding(self, text: str, vector: list):
        self.set(_sha(text), vector)


class ResponseCache(_DiskCache):
    """Cache for question → answer lookups."""

    def __init__(self, cache_dir: str = "cache"):
        super().__init__(Path(cache_dir) / "responses.json")

    def get_response(self, question: str, pipeline: str = ""):
        key = _sha(question + "|" + pipeline)
        return self.get(key)

    def set_response(self, question: str, answer: str, pipeline: str = "",
                     context: list = None):
        key = _sha(question + "|" + pipeline)
        self.set(key, {
            "answer":    answer,
            "context":   context or [],
            "cached_at": time.time(),
        })


class RerankerCache(_DiskCache):
    """Cache for (query, doc_id) → reranker score."""

    def __init__(self, cache_dir: str = "cache"):
        super().__init__(Path(cache_dir) / "reranker_scores.json")

    def get_score(self, query: str, doc_id: str) -> float | None:
        key = _sha(query + "|" + doc_id)
        return self.get(key)

    def set_score(self, query: str, doc_id: str, score: float):
        key = _sha(query + "|" + doc_id)
        self.set(key, score)


class RAGCache:
    """
    Combined cache facade — wraps all three sub-caches.

    Usage:
        cache = RAGCache()
        cache.embeddings.get_embedding("my query")
        cache.responses.get_response("question", pipeline="cot")
        cache.reranker.get_score("query", "doc_id_123")
    """

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.embeddings = EmbeddingCache(cache_dir)
        self.responses  = ResponseCache(cache_dir)
        self.reranker   = RerankerCache(cache_dir)

    def stats(self) -> dict:
        return {
            "embeddings": self.embeddings.stats(),
            "responses":  self.responses.stats(),
            "reranker":   self.reranker.stats(),
        }

    def clear_all(self):
        self.embeddings.clear()
        self.responses.clear()
        self.reranker.clear()

    def print_stats(self):
        s = self.stats()
        print("\n  Cache Statistics")
        print(f"  {'Store':<14} {'Size':>6} {'Hits':>6} {'Misses':>7} {'Hit Rate':>9}")
        print("  " + "-" * 48)
        for name, info in s.items():
            print(f"  {name:<14} {info['size']:>6} {info['hits']:>6} "
                  f"{info['misses']:>7} {info['hit_rate']:>9.1%}")


# ── Cached wrappers ───────────────────────────────────────

def cached_embed(text: str, embed_fn, cache: EmbeddingCache) -> list:
    """Return cached embedding or call embed_fn and cache the result."""
    vec = cache.get_embedding(text)
    if vec is not None:
        return vec
    vec = embed_fn(text)
    cache.set_embedding(text, vec)
    return vec


def cached_rerank(query: str, candidates: list, rerank_fn,
                  cache: RerankerCache) -> list:
    """
    Rerank candidates, using cache to skip already-scored (query, doc) pairs.
    candidates: list of dicts with at least 'id' and 'text' keys
    rerank_fn:  fn(query, texts) -> list[float]
    """
    uncached_indices = []
    scores = [None] * len(candidates)

    for i, cand in enumerate(candidates):
        cached_score = cache.get_score(query, cand["id"])
        if cached_score is not None:
            scores[i] = cached_score
        else:
            uncached_indices.append(i)

    if uncached_indices:
        texts_to_score = [candidates[i]["text"] for i in uncached_indices]
        new_scores = rerank_fn(query, texts_to_score)
        for idx, score in zip(uncached_indices, new_scores):
            scores[idx] = score
            cache.set_score(query, candidates[idx]["id"], score)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )
    result = []
    for rank, (score, cand) in enumerate(ranked, 1):
        result.append({**cand, "rank": rank, "reranker_score": score})
    return result
