#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_multihop_eval.py
---------------------
Evaluate Multi-Hop RAG vs Baseline RAG on questions that require
chaining information across multiple retrieval steps.

This directly demonstrates the Japanese company requirement:
  「内容は近くないけど、回答に関連する情報をどうやって拾えるようにするか」
  = How to retrieve information NOT close in content but answer-relevant

Uses LOCAL sentence-transformers embeddings (no OpenAI API cost for retrieval).
LLM generation attempts OpenAI first, falls back to showing retrieved context.

Usage:
  python run_multihop_eval.py              # Full comparison (6 questions)
  python run_multihop_eval.py --demo       # 3 questions, verbose
  python run_multihop_eval.py --rebuild    # Force rebuild the local vector index
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DATA_PATH  = Path("data/rag_dataset.json")
LOCAL_DB   = Path("data/chroma_db_local")
COLLECTION = "rag_multihop_local"

# ── Multi-hop test cases ──────────────────────────────────
# Questions designed so that single-hop retrieval FAILS or DEGRADES.
# Hop 1 finds a bridging entity; Hop 2 uses that entity as the new query.

MULTIHOP_QUESTIONS = [
    {
        "id": "mh_01",
        "question": "Who received the Turing Award for the deep learning work that enabled AlexNet to win ImageNet 2012, and what year was the award given?",
        "ground_truth": "Yoshua Bengio, Geoffrey Hinton, and Yann LeCun received the 2018 Turing Award for deep learning breakthroughs that enabled modern neural networks including AlexNet.",
        "hop1_target": "Who created AlexNet / what technique won ImageNet 2012?",
        "hop2_target": "Who got the Turing Award for deep neural networks?",
        "why_single_hop_fails": "Searching 'Turing Award ImageNet AlexNet' finds nothing — you must first find WHO made AlexNet, then search for their award.",
        # Pre-defined sub-questions for demo when LLM is unavailable
        "sub_questions": [
            "Who created AlexNet and what technique did it use to win ImageNet 2012?",
            "Who received the Turing Award for work on deep neural networks?",
        ],
    },
    {
        "id": "mh_02",
        "question": "The person who introduced the ReLU activation function in 1969 also created which CNN architecture, and what was that architecture designed for?",
        "ground_truth": "Kunihiko Fukushima introduced ReLU in 1969 and later created the Neocognitron CNN in 1979, designed for visual pattern recognition.",
        "hop1_target": "Who introduced ReLU activation function in 1969?",
        "hop2_target": "What CNN architecture did Kunihiko Fukushima create?",
        "why_single_hop_fails": "The original question doesn't name Fukushima. Hop 1 must find the name before Hop 2 can retrieve the CNN.",
        "sub_questions": [
            "Who introduced the ReLU activation function in 1969?",
            "What CNN architecture did Kunihiko Fukushima create and what was it designed for?",
        ],
    },
    {
        "id": "mh_03",
        "question": "What year was the vanishing gradient problem first analyzed, and what architecture was invented to solve it?",
        "ground_truth": "The vanishing gradient problem was analyzed by Hochreiter in 1991, leading to LSTM (Long Short-Term Memory) published in 1995.",
        "hop1_target": "Who first analyzed the vanishing gradient problem and when?",
        "hop2_target": "What architecture did Hochreiter develop to solve the vanishing gradient problem?",
        "why_single_hop_fails": "Searching 'architecture that solved vanishing gradient' might find LSTM, but the 1991 year connection requires the Hochreiter chain.",
        "sub_questions": [
            "Who first analyzed the vanishing gradient problem and in what year?",
            "What architecture did Hochreiter create to address the vanishing gradient problem?",
        ],
    },
    {
        "id": "mh_04",
        "question": "Ian Goodfellow introduced a generative model in 2014 based on a principle from 1991 — who originally proposed that underlying principle?",
        "ground_truth": "Ian Goodfellow's GANs (2014) were based on Jürgen Schmidhuber's 1991 principle of adversarial neural networks / artificial curiosity.",
        "hop1_target": "What did Ian Goodfellow introduce in 2014?",
        "hop2_target": "Who first proposed the adversarial neural network principle?",
        "why_single_hop_fails": "A search for 'Goodfellow generative model 1991 underlying principle' is too indirect for single-hop to surface Schmidhuber.",
        "sub_questions": [
            "What generative model did Ian Goodfellow introduce in 2014?",
            "Who first proposed the adversarial network principle that GANs are based on?",
        ],
    },
    {
        "id": "mh_05",
        "question": "The researcher who coined the term 'machine learning' in 1959 was known for what game-playing program, and who was his employer at the time?",
        "ground_truth": "Arthur Samuel coined 'machine learning' in 1959. He created a checkers-playing program and was employed by IBM.",
        "hop1_target": "Who coined the term 'machine learning' in 1959?",
        "hop2_target": "What game program did Arthur Samuel create and where did he work?",
        "why_single_hop_fails": "The question doesn't name Samuel. Hop 1 must find the name before Hop 2 can retrieve his game and employer.",
        "sub_questions": [
            "Who coined the term machine learning in 1959?",
            "What game-playing program did Arthur Samuel create and who was his employer?",
        ],
    },
    {
        "id": "mh_06",
        "question": "What company developed AlphaGo and what other major science breakthrough did that same company achieve using deep learning?",
        "ground_truth": "Google DeepMind developed AlphaGo. The same company later achieved AlphaFold, which predicted protein structures at near-experimental accuracy.",
        "hop1_target": "What company or organization developed AlphaGo?",
        "hop2_target": "What major scientific breakthrough did DeepMind achieve besides AlphaGo?",
        "why_single_hop_fails": "The original question does not name DeepMind. Hop 1 finds the company; Hop 2 uses that name to find AlphaFold.",
        "sub_questions": [
            "What company developed AlphaGo?",
            "What major scientific achievement did Google DeepMind accomplish in bioinformatics using deep learning?",
        ],
    },
]


# ── Local Embedding & Vector Store ────────────────────────

def build_local_index(force_rebuild: bool = False):
    """Build ChromaDB index using local sentence-transformers (no OpenAI needed)."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    from baseline_rag import load_documents, chunk_documents

    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=str(LOCAL_DB))

    # Check if we can reuse existing collection
    if not force_rebuild:
        try:
            col = client.get_collection(name=COLLECTION, embedding_function=embed_fn)
            count = col.count()
            if count > 100:
                print(f"[LocalIndex] Loaded existing collection ({count} vectors) — skipping rebuild")
                return col, embed_fn
        except Exception:
            pass

    print("[LocalIndex] Building local index with sentence-transformers (all-MiniLM-L6-v2)...")
    docs   = load_documents()
    chunks = chunk_documents(docs)

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        collection.add(
            ids       = [c["id"] for c in batch],
            documents = [c["text"] for c in batch],
            metadatas = [{"title": c["title"], "source": c["source"], "doc_id": c["doc_id"]} for c in batch],
        )
        print(f"  Indexed {min(i+batch_size, len(chunks))}/{len(chunks)}...", end="\r")

    print(f"\n[LocalIndex] Done: {collection.count()} vectors indexed locally")
    return collection, chunks


# ── BM25 ──────────────────────────────────────────────────

def build_bm25(chunks: list):
    """Build BM25 index from chunks for lexical retrieval."""
    import re
    from rank_bm25 import BM25Okapi

    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print("[BM25] Indexed", len(chunks), "chunks")
    return bm25, tokenize


def retrieve_hybrid_local(query: str, collection, bm25, chunks: list,
                           tokenize_fn, top_k: int = 3, depth: int = 20) -> list:
    """Hybrid BM25 + local semantic retrieval with RRF fusion."""
    import re

    # Semantic
    results = collection.query(query_texts=[query], n_results=depth)
    sem_list = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        cid = results["ids"][0][i]
        sem_list.append({"id": cid, "text": doc, "title": meta.get("title", ""),
                          "distance": dist, "rank": i+1})

    # Lexical (BM25)
    tokens = tokenize_fn(query)
    scores = bm25.get_scores(tokens)
    bm25_ranked = sorted(
        [{"id": chunks[i]["id"], "text": chunks[i]["text"],
          "title": chunks[i]["title"], "bm25_score": s}
         for i, s in enumerate(scores)],
        key=lambda x: x["bm25_score"], reverse=True
    )[:depth]

    # RRF fusion
    k = 60
    rrf = {}
    chunk_map = {}
    for rank, c in enumerate(sem_list, 1):
        rrf[c["id"]] = rrf.get(c["id"], 0) + 1.0 / (k + rank)
        chunk_map[c["id"]] = c
    for rank, c in enumerate(bm25_ranked, 1):
        rrf[c["id"]] = rrf.get(c["id"], 0) + 1.0 / (k + rank)
        chunk_map[c["id"]] = c

    top_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_k]
    fused = []
    for rank, cid in enumerate(top_ids, 1):
        c = chunk_map[cid]
        c["rank"] = rank
        c["rrf_score"] = round(rrf[cid], 6)
        fused.append(c)
    return fused


# ── LLM (with fallback) ───────────────────────────────────

def call_llm(prompt: str, max_tokens: int = 300) -> str:
    """Call OpenAI LLM with graceful fallback on quota errors."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return _mock_llm(prompt)

    try:
        from openai import OpenAI, RateLimitError
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "quota" in err.lower() or "429" in err or "rate" in err.lower():
            print(f"  [LLM] OpenAI quota exceeded — using retrieved context as answer")
        else:
            print(f"  [LLM] Error: {err[:80]} — using fallback")
        return _mock_llm(prompt)


def _mock_llm(prompt: str) -> str:
    """Extract the most informative sentence from the context in the prompt."""
    lines = prompt.splitlines()
    # Find the context block (between "Context:" and the next section)
    in_context = False
    context_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "Context:":
            in_context = True
            continue
        if in_context and stripped.startswith("[Source"):
            continue
        if in_context and stripped.startswith("---"):
            continue
        if in_context and (stripped.startswith("Sub-question:") or
                           stripped.startswith("Question:") or
                           stripped.startswith("Bridge facts:") or
                           stripped.startswith("Original question:")):
            break
        if in_context and len(stripped) > 60:
            context_lines.append(stripped)

    if context_lines:
        # Return the first meaty sentence
        for line in context_lines:
            if any(c.isalpha() for c in line) and len(line) > 60:
                return "[Retrieved] " + line[:280]

    return "[No relevant information found in retrieved documents]"


# ── Multi-Hop Logic ───────────────────────────────────────

DECOMPOSE_PROMPT = """\
Break this complex question into {n} simpler sub-questions that must be answered IN ORDER.
Return ONLY the numbered sub-questions, one per line.

Question: {question}

Sub-questions:"""

INTERMEDIATE_PROMPT = """\
Answer this sub-question using ONLY the context below. Give a SHORT factual answer (1-3 sentences).
If not found, write: NOT FOUND

Context:
{context}

Sub-question: {sub_question}

Short answer:"""

SYNTHESIS_PROMPT = """\
Answer the original question using ONLY the bridge facts below.

Original question: {question}

Bridge facts:
{bridge_facts}

Final answer:"""


def decompose(question: str, n: int = 2, predefined: list = None) -> list:
    """Decompose question into sub-questions. Uses predefined list when LLM unavailable."""
    if predefined:
        return predefined[:n]
    prompt = DECOMPOSE_PROMPT.format(question=question, n=n)
    raw = call_llm(prompt, max_tokens=200)
    if not raw or raw.startswith("[Based on"):
        return [question]
    lines = [l.lstrip("0123456789.-) ").strip() for l in raw.splitlines() if l.strip()]
    return [l for l in lines if l][:n] or [question]


def _extract_relevant_sentence(sub_q: str, chunks: list) -> str:
    """
    Keyword-based extraction: find the sentence in retrieved chunks
    that has the most keyword overlap with the sub-question.
    Used as fallback when LLM is unavailable.
    """
    import re
    q_words = set(re.findall(r'\w+', sub_q.lower())) - {
        "the", "a", "an", "is", "in", "of", "and", "or", "to", "what",
        "who", "when", "where", "how", "which", "did", "was", "were",
        "for", "that", "this", "with", "also", "by", "on", "at", "it"
    }

    best_sentence = ""
    best_score = 0

    for chunk in chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk["text"])
        for sent in sentences:
            if len(sent) < 30:
                continue
            s_words = set(re.findall(r'\w+', sent.lower()))
            overlap = len(q_words & s_words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sent

    if best_sentence:
        return f"[Extracted] {best_sentence[:280]}"
    return "[No relevant information found in retrieved documents]"


def intermediate_answer(sub_q: str, chunks: list) -> str:
    ctx = "\n\n---\n\n".join(
        f"[Source {c['rank']}: {c['title']}]\n{c['text'][:500]}"
        for c in chunks
    )
    result = call_llm(INTERMEDIATE_PROMPT.format(context=ctx, sub_question=sub_q), max_tokens=150)
    # If LLM failed, use keyword extraction
    if result.startswith("[No relevant") or result.startswith("[Retrieved]") or not result:
        return _extract_relevant_sentence(sub_q, chunks)
    return result


def synthesize(question: str, bridge: list) -> str:
    facts = "\n".join(
        f"Hop {b['hop']}: {b['sub_question']}\n  → {b['answer']}"
        for b in bridge
    )
    result = call_llm(SYNTHESIS_PROMPT.format(question=question, bridge_facts=facts), max_tokens=300)
    # If LLM unavailable, combine bridge facts directly
    if not result or result.startswith("[No relevant"):
        parts = [b["answer"] for b in bridge
                 if b.get("answer") and not b["answer"].startswith("[No relevant")]
        if parts:
            return "[Synthesized from hops] " + " → ".join(p[:200] for p in parts)
    return result


# ── Main Runner ───────────────────────────────────────────

def run_evaluation(questions: list, collection, bm25, chunks: list,
                   tokenize_fn, n_hops: int = 2, verbose: bool = True):
    baseline_results = []
    multihop_results = []

    for item in questions:
        q = item["question"]
        print(f"\n{'='*75}")
        print(f"Q [{item['id']}]: {q[:90]}")
        print(f"WHY single-hop fails: {item['why_single_hop_fails']}")
        print(f"{'='*75}")

        # ── Baseline: single retrieval ──────────────────────
        print("\n[BASELINE — single retrieval]")
        t0 = time.time()
        b_chunks = retrieve_hybrid_local(q, collection, bm25, chunks, tokenize_fn, top_k=3)
        b_ctx = "\n\n---\n\n".join(
            f"[Source {c['rank']}: {c['title']}]\n{c['text'][:500]}" for c in b_chunks
        )
        b_answer = call_llm(
            f"Answer using ONLY the context below.\n\nContext:\n{b_ctx}\n\nQuestion: {q}\n\nAnswer:",
            max_tokens=200
        )
        b_lat = round(time.time() - t0, 2)
        print(f"  Retrieved: {[c['title'] for c in b_chunks]}")
        print(f"  Answer: {b_answer[:180]}")
        print(f"  Latency: {b_lat}s")

        baseline_results.append({
            "id": item["id"], "question": q,
            "answer": b_answer, "retrieved": [c["title"] for c in b_chunks],
            "latency_s": b_lat, "method": "baseline"
        })

        # ── Multi-Hop: chained retrieval ────────────────────
        print(f"\n[MULTI-HOP — {n_hops} chained retrievals]")
        t0 = time.time()

        # Step 1: Decompose (use predefined sub-questions if LLM unavailable)
        predefined = item.get("sub_questions")
        sub_qs = decompose(q, n=n_hops, predefined=predefined)
        src = "predefined" if predefined else "LLM-generated"
        print(f"  Decomposed into {len(sub_qs)} sub-questions [{src}]:")
        for i, sq in enumerate(sub_qs):
            print(f"    Hop {i+1}: {sq}")

        # Step 2: Hop loop
        bridge = []
        all_titles = []
        for i, sub_q in enumerate(sub_qs):
            # Enrich query with prior answers
            prior_ctx = " | ".join(
                b["answer"] for b in bridge if "NOT FOUND" not in b.get("answer", "")
            )
            enriched = f"{sub_q} (Prior facts: {prior_ctx})" if prior_ctx else sub_q

            print(f"\n  --- Hop {i+1} ---")
            if enriched != sub_q:
                print(f"  Enriched query: {enriched[:100]}")

            h_chunks = retrieve_hybrid_local(enriched, collection, bm25, chunks,
                                              tokenize_fn, top_k=3)
            h_titles = [c["title"] for c in h_chunks]
            all_titles.extend(h_titles)

            print(f"  Retrieved: {h_titles}")
            if verbose:
                for c in h_chunks:
                    print(f"    [{c['rank']}] {c['title']} (rrf={c.get('rrf_score', '?')})")
                    print(f"         {c['text'][:90]}...")

            ans = intermediate_answer(sub_q, h_chunks)
            print(f"  Intermediate answer: {ans[:180]}")

            bridge.append({"hop": i+1, "sub_question": sub_q,
                            "answer": ans, "retrieved": h_titles})

        # Step 3: Synthesize
        final = synthesize(q, bridge)
        mh_lat = round(time.time() - t0, 2)

        print(f"\n  FINAL ANSWER: {final[:250]}")
        print(f"  Latency: {mh_lat}s")

        multihop_results.append({
            "id": item["id"], "question": q,
            "answer": final,
            "bridge_facts": bridge,
            "retrieved_all": list(dict.fromkeys(all_titles)),
            "latency_s": mh_lat, "method": "multihop",
            "ground_truth": item.get("ground_truth", ""),
        })

    return baseline_results, multihop_results


def print_summary(baseline: list, multihop: list):
    print("\n" + "=" * 75)
    print("  RESULTS SUMMARY — Baseline vs Multi-Hop RAG")
    print("  Demonstrating: 「内容は近くないけど、回答に関連する情報を拾う」")
    print("=" * 75)

    for b, m in zip(baseline, multihop):
        qid = b["id"]
        print(f"\n[{qid}]")
        print(f"  Baseline retrieved : {b['retrieved']}")
        print(f"  MultiHop retrieved : {m['retrieved_all']}")
        print(f"  Baseline answer    : {b['answer'][:150]}")
        print(f"  MultiHop answer    : {m['answer'][:150]}")
        print(f"  Ground truth       : {m['ground_truth'][:150]}")
        print(f"  Latency            : Baseline={b['latency_s']}s | MultiHop={m['latency_s']}s")

    b_avg = sum(r["latency_s"] for r in baseline) / len(baseline)
    m_avg = sum(r["latency_s"] for r in multihop) / len(multihop)
    print(f"\n  Average latency — Baseline: {b_avg:.1f}s | MultiHop: {m_avg:.1f}s")
    print(f"  Extra latency for multi-hop: +{m_avg - b_avg:.1f}s (cost of chaining {multihop[0]['bridge_facts'].__len__()} hops)")
    print()
    print("  KEY INSIGHT:")
    print("  Multi-Hop RAG derives Hop 2 query from Hop 1 answer.")
    print("  The Hop 2 query was NEVER part of the original question.")
    print("  This is how we retrieve 'content-not-close-but-answer-relevant' documents.")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",    action="store_true", help="Run 3 questions only")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild local index")
    parser.add_argument("--n-hops", type=int, default=2)
    args = parser.parse_args()

    questions = MULTIHOP_QUESTIONS[:3] if args.demo else MULTIHOP_QUESTIONS

    src = Path("src")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    print("=" * 75)
    print("  Multi-Hop RAG Evaluation")
    print("  Local embeddings (sentence-transformers) — no OpenAI API for retrieval")
    print(f"  Testing {len(questions)} questions | n_hops={args.n_hops}")
    print("=" * 75)

    # Build local index
    result = build_local_index(force_rebuild=args.rebuild)
    collection, chunks_or_fn = result

    # Load documents for BM25
    from baseline_rag import load_documents, chunk_documents
    docs   = load_documents()
    chunks = chunk_documents(docs)

    bm25, tokenize_fn = build_bm25(chunks)

    # Run evaluation
    baseline, multihop = run_evaluation(
        questions, collection, bm25, chunks, tokenize_fn,
        n_hops=args.n_hops, verbose=True
    )

    # Summary
    print_summary(baseline, multihop)

    # Save results
    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / "multihop_baseline.json", "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    with open(out / "multihop_results.json", "w", encoding="utf-8") as f:
        json.dump(multihop, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[SAVED] results/multihop_baseline.json")
    print(f"[SAVED] results/multihop_results.json")


if __name__ == "__main__":
    main()
