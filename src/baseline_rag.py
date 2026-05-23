#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/baseline_rag.py
-------------------
Day 2: Baseline RAG Pipeline
- Load documents from data/rag_dataset.json
- Chunk + embed into ChromaDB (local vector store)
- Retrieve top-k + generate answer with OpenAI
- No tricks yet — pure vanilla RAG
"""

import sys
import os
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K         = int(os.getenv("TOP_K", 3))
EMBED_MODEL   = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
LLM_MODEL     = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
DATA_PATH     = Path("data/rag_dataset.json")
DB_PATH       = Path("data/chroma_db")
COLLECTION    = "rag_baseline"


# ── Document Loading ──────────────────────────────────────
def load_documents():
    """Load documents from rag_dataset.json."""
    print("[1/4] Loading documents...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    docs = dataset.get("documents", [])
    print(f"      Loaded {len(docs)} documents")
    return docs


# ── Chunking ──────────────────────────────────────────────
def chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split documents into fixed-size chunks with overlap."""
    print(f"[2/4] Chunking (size={chunk_size}, overlap={overlap})...")
    chunks = []

    for doc in docs:
        content = doc.get("content", "")
        title   = doc.get("title", doc.get("id", "unknown"))
        source  = doc.get("source", "unknown")

        # Sliding window chunking
        words = content.split()
        i = 0
        chunk_idx = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunk_text  = " ".join(chunk_words)

            chunks.append({
                "id":      f"{doc['id']}_chunk{chunk_idx}",
                "text":    chunk_text,
                "title":   title,
                "source":  source,
                "doc_id":  doc["id"],
            })
            i += chunk_size - overlap
            chunk_idx += 1

    print(f"      Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks


# ── Vector Store (ChromaDB) ───────────────────────────────
def build_vector_store(chunks):
    """Embed chunks and store in local ChromaDB."""
    print(f"[3/4] Building vector store (embedding model: {EMBED_MODEL})...")

    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        print("      [WARN] OPENAI_API_KEY not set — using mock embeddings for testing")
        embed_fn = None
    else:
        embed_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=EMBED_MODEL,
        )

    client = chromadb.PersistentClient(path=str(DB_PATH))

    # Drop and recreate collection
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch upsert (ChromaDB max 5461 per batch)
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids       = [c["id"] for c in batch],
            documents = [c["text"] for c in batch],
            metadatas = [{"title": c["title"], "source": c["source"], "doc_id": c["doc_id"]} for c in batch],
        )
        print(f"      Indexed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks...", end="\r")

    print(f"\n      Vector store ready: {collection.count()} vectors in '{COLLECTION}'")
    return collection


# ── Retrieval ─────────────────────────────────────────────
def retrieve(collection, query: str, top_k=TOP_K):
    """Retrieve top-k relevant chunks for a query."""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )
    chunks = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        chunks.append({
            "rank":     i + 1,
            "text":     doc,
            "title":    meta.get("title", ""),
            "source":   meta.get("source", ""),
            "distance": round(dist, 4),
        })
    return chunks


# ── Generation ────────────────────────────────────────────
def generate_answer(query: str, context_chunks: list, model=LLM_MODEL) -> str:
    """Generate answer using retrieved context."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        # Mock response for testing without API key
        context_preview = context_chunks[0]["text"][:200] if context_chunks else "No context"
        return f"[MOCK ANSWER] Based on context: '{context_preview}...'\nAnswer to: {query}"

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Build context string
    context = "\n\n---\n\n".join(
        f"[Source {c['rank']}: {c['title']}]\n{c['text']}"
        for c in context_chunks
    )

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


# ── Full RAG Pipeline ─────────────────────────────────────
class BaselineRAG:
    """End-to-end baseline RAG pipeline."""

    def __init__(self):
        self.collection = None

    def build(self):
        """Load data, chunk, embed, index."""
        docs   = load_documents()
        chunks = chunk_documents(docs)
        self.collection = build_vector_store(chunks)
        print("\n[4/4] Pipeline ready!")
        return self

    def query(self, question: str, top_k=TOP_K, verbose=True) -> dict:
        """Run full RAG: retrieve + generate."""
        if not self.collection:
            raise RuntimeError("Call .build() first")

        context = retrieve(self.collection, question, top_k)
        answer  = generate_answer(question, context)

        result = {
            "question": question,
            "answer":   answer,
            "context":  context,
            "top_k":    top_k,
        }

        if verbose:
            print(f"\n{'='*55}")
            print(f"Q: {question}")
            print(f"{'='*55}")
            print(f"A: {answer}")
            print(f"\n--- Retrieved {len(context)} chunks ---")
            for c in context:
                print(f"  [{c['rank']}] {c['title']} (dist={c['distance']})")
                print(f"      {c['text'][:120]}...")

        return result


# ── Quick Test ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Baseline RAG Pipeline -- Day 2")
    print("=" * 55)

    rag = BaselineRAG().build()

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
    out = Path("results/baseline_test_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] Test results -> {out}")
    print("\nBaseline RAG pipeline is working!")
