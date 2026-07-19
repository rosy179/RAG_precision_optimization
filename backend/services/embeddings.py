"""
Shared embedding-function factory.

The per-user collections (user_rag.py) and the global KB (global_kb.py) must
embed into the SAME vector space — retrieval merges chunks from both pools by
raw cosine score. Keeping the model choice in one module guarantees the two
can never drift apart.

EMBED_MODEL accepts:
  BAAI/bge-m3 (default)  — local SentenceTransformer, multilingual EN/VI/JA,
                           free, no API dependency
  text-embedding-*       — any OpenAI embedding model (needs OPENAI_API_KEY;
                           falls back to a small local model in mock mode)
  other HF model id      — loaded with SentenceTransformer

Switching models invalidates every stored vector (and usually the vector
dimension) — run scripts/reembed_collections.py afterwards.
"""

import os
from functools import lru_cache

from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
_MOCK_FALLBACK = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_function():
    """One process-wide embedding function for EMBED_MODEL (cached so every
    collection and user instance shares a single loaded model)."""
    if EMBED_MODEL.startswith("text-embedding-"):
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and not api_key.startswith("sk-your"):
            return OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBED_MODEL)
        return SentenceTransformerEmbeddingFunction(model_name=_MOCK_FALLBACK)
    # BGE models are trained for cosine similarity on normalized vectors
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL, normalize_embeddings=True)
