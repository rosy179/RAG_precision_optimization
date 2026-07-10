#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/multihop_rag.py
--------------------
Multi-Hop RAG — chained retrieval for questions requiring bridging facts

Core problem this solves (Japanese company requirement):
  「内容は近くないけど、回答に関連する情報をどうやって拾えるようにするか」
  = How to retrieve information that is NOT semantically close to the question,
    but IS necessary to answer it.

Root cause of standard RAG failure:
  Standard RAG embeds the QUESTION and finds semantically similar documents.
  But for complex questions, the answer requires a CHAIN of facts:

  Example:
    Q: "Who received the Turing Award for the breakthrough that enabled AlphaGo?"
    - Standard RAG searches: "Turing Award AlphaGo" → may find nothing
    - Multi-Hop RAG:
        Hop 1: "What technique enabled AlphaGo?" → "deep reinforcement learning / deep neural nets"
        Hop 2: "Who got the Turing Award for deep neural networks?" → "Hinton, LeCun, Bengio (2018)"
    The Hop 2 query was NEVER in the original question — it was derived from Hop 1.

How Multi-Hop RAG works:
  1. Decompose: LLM splits question into ordered sub-questions (hops)
  2. Loop (for each hop):
     a. Build enriched query = sub-question + prior intermediate answers
     b. Retrieve docs for the enriched query
     c. Generate intermediate answer
     d. Append to "bridge context"
  3. Synthesize: LLM produces final answer from all bridge context

Built on top of: ChainOfThoughtRAG (best full stack: Hybrid + CrossEncoder + CoT)
"""

import sys
import os
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K          = int(os.getenv("TOP_K", 3))
RETRIEVE_DEPTH = int(os.getenv("RETRIEVE_DEPTH", 20))
MAX_HOPS       = 3

# ── Prompts ───────────────────────────────────────────────

DECOMPOSE_PROMPT = """\
You are an expert at breaking complex questions into simpler sub-questions.

Given the complex question below, generate a numbered list of {n_hops} sub-questions \
that must be answered IN ORDER to answer the main question.

Rules:
- Each sub-question should be answerable on its own from a knowledge base
- Later sub-questions CAN reference answers from earlier ones (write them as templates)
- Return ONLY the numbered sub-questions, one per line, no extra text

Complex question: {question}

Sub-questions:"""

INTERMEDIATE_ANSWER_PROMPT = """\
You are a precise assistant. Answer the sub-question using ONLY the provided context.
If the answer is not in the context, say "NOT FOUND".
Give a SHORT, factual answer (1-3 sentences max).

Context:
{context}

Sub-question: {sub_question}

Short answer:"""

SYNTHESIS_PROMPT = """\
You are a precise assistant. Answer the original question using the bridge facts collected \
across multiple retrieval steps. Use ONLY the information provided below.

Original question: {question}

Bridge facts collected step by step:
{bridge_facts}

Final answer (based strictly on the bridge facts above):"""


# ── LLM Helper ────────────────────────────────────────────

def _call_llm(prompt: str, model: str = LLM_MODEL, max_tokens: int = 400) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return ""

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def decompose_question(question: str, n_hops: int = 2) -> list:
    """Use LLM to break question into ordered sub-questions."""
    prompt = DECOMPOSE_PROMPT.format(question=question, n_hops=n_hops)
    raw = _call_llm(prompt, max_tokens=300)

    if not raw:
        return [question]

    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    sub_questions = []
    for line in lines:
        # Strip leading "1." "2." etc.
        clean = line.lstrip("0123456789.-) ").strip()
        if clean:
            sub_questions.append(clean)

    return sub_questions[:n_hops] if sub_questions else [question]


def generate_intermediate_answer(sub_question: str, context_chunks: list,
                                  model: str = LLM_MODEL) -> str:
    """Generate a short bridging answer for a single hop."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return context_chunks[0]["text"][:200] if context_chunks else "NOT FOUND"

    context = "\n\n---\n\n".join(
        f"[Source {c['rank']}: {c['title']}]\n{c['text']}"
        for c in context_chunks
    )
    prompt = INTERMEDIATE_ANSWER_PROMPT.format(
        context=context, sub_question=sub_question
    )
    return _call_llm(prompt, model=model, max_tokens=150)


def synthesize_final_answer(question: str, bridge_facts: list,
                             model: str = LLM_MODEL) -> str:
    """Combine all hop results into final answer."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return "\n".join(f"Hop {i+1}: {f}" for i, f in enumerate(bridge_facts))

    formatted = "\n".join(
        f"Hop {i+1} — {item['sub_question']}\n  Answer: {item['answer']}"
        for i, item in enumerate(bridge_facts)
    )
    prompt = SYNTHESIS_PROMPT.format(question=question, bridge_facts=formatted)
    return _call_llm(prompt, model=model, max_tokens=300)


# ── Main Class ────────────────────────────────────────────

class MultiHopRAG:
    """
    Multi-Hop RAG pipeline.

    Architecture:
      Built on ChainOfThoughtRAG retrieval stack (Hybrid BM25+Semantic → CrossEncoder).
      Adds a decompose → hop-loop → synthesize layer on top.

    n_hops:
      2 (default) — most multi-step questions need exactly 2 hops
      3 — for deeply nested questions
    """

    def __init__(self, n_hops: int = 2, use_cot_stack: bool = True):
        self.n_hops = n_hops
        self.use_cot_stack = use_cot_stack
        self._retriever = None

    def build(self):
        src_dir = Path(__file__).parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        if self.use_cot_stack:
            from reranker_rag import RerankerRAG
            print("[MultiHop] Building RerankerRAG stack (Hybrid + CrossEncoder)...")
            self._retriever = RerankerRAG()
            self._retriever.build()
            self._retrieve_fn = self._retrieve_with_reranker
        else:
            from baseline_rag import BaselineRAG
            print("[MultiHop] Building BaselineRAG stack (semantic only)...")
            self._retriever = BaselineRAG()
            self._retriever.build()
            self._retrieve_fn = self._retrieve_baseline

        print(f"\n[MultiHop] Pipeline ready! n_hops={self.n_hops}")
        return self

    def _retrieve_with_reranker(self, query: str, top_k: int) -> list:
        return self._retriever.retrieve_and_rerank(
            query, top_k=top_k, retrieve_depth=RETRIEVE_DEPTH
        )

    def _retrieve_baseline(self, query: str, top_k: int) -> list:
        from baseline_rag import retrieve
        return retrieve(self._retriever.collection, query, top_k)

    def _enrich_query(self, sub_question: str, prior_answers: list) -> str:
        """Add prior intermediate answers as context to focus the next retrieval."""
        if not prior_answers:
            return sub_question
        context = " | ".join(
            f"[Established: {a['answer']}]" for a in prior_answers
            if a.get("answer") and "NOT FOUND" not in a.get("answer", "")
        )
        return f"{sub_question} (Context: {context})" if context else sub_question

    def query(self, question: str, top_k: int = TOP_K, verbose: bool = True) -> dict:
        if self._retriever is None:
            raise RuntimeError("Call .build() first")

        if verbose:
            print(f"\n{'='*70}")
            print(f"[MultiHop] Q: {question}")
            print(f"{'='*70}")

        # Step 1 — Decompose
        sub_questions = decompose_question(question, n_hops=self.n_hops)

        if verbose:
            print(f"\n[MultiHop] Decomposed into {len(sub_questions)} hops:")
            for i, sq in enumerate(sub_questions):
                print(f"  Hop {i+1}: {sq}")

        # Step 2 — Hop-loop
        bridge_facts = []
        all_retrieved_chunks = []

        for i, sub_q in enumerate(sub_questions):
            if verbose:
                print(f"\n--- Hop {i+1}/{len(sub_questions)} ---")

            # Enrich query with prior answers
            enriched_query = self._enrich_query(sub_q, bridge_facts)

            if verbose and enriched_query != sub_q:
                print(f"  [Enriched query] {enriched_query[:120]}")

            # Retrieve
            chunks = self._retrieve_fn(enriched_query, top_k)
            all_retrieved_chunks.extend(chunks)

            if verbose:
                print(f"  Retrieved {len(chunks)} chunks:")
                for c in chunks:
                    score = c.get("reranker_score", c.get("distance", "N/A"))
                    score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
                    print(f"    [{c['rank']}] {c['title']} (score={score_str})")
                    print(f"         {c['text'][:90]}...")

            # Generate intermediate answer
            intermediate = generate_intermediate_answer(sub_q, chunks)

            if verbose:
                print(f"  [Hop {i+1} answer] {intermediate}")

            bridge_facts.append({
                "hop":          i + 1,
                "sub_question": sub_q,
                "answer":       intermediate,
                "chunks":       chunks,
            })

        # Step 3 — Final synthesis
        if verbose:
            print(f"\n--- Final Synthesis ---")

        final_answer = synthesize_final_answer(question, bridge_facts)

        if verbose:
            print(f"[MultiHop] Final Answer: {final_answer}")

        # Deduplicate retrieved chunks for reporting
        seen_ids = set()
        unique_chunks = []
        for c in all_retrieved_chunks:
            cid = c.get("id", c.get("text", "")[:50])
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(c)

        return {
            "question":     question,
            "answer":       final_answer,
            "bridge_facts": bridge_facts,
            "context":      unique_chunks,
            "n_hops":       len(bridge_facts),
            "sub_questions": sub_questions,
        }


# ── Quick Test ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  Multi-Hop RAG Pipeline")
    print("  Solving: content-not-close but answer-relevant retrieval")
    print("=" * 70)

    rag = MultiHopRAG(n_hops=2).build()

    test_questions = [
        "Who received the Turing Award for the work that enabled AlexNet to win the ImageNet competition in 2012?",
        "What technique was introduced by the researcher who also proposed the first working deep learning algorithm, and what year was that technique introduced?",
        "The ReLU activation function was introduced by the same person who created what CNN architecture, and when was that architecture introduced?",
    ]

    results = []
    for q in test_questions:
        r = rag.query(q)
        results.append(r)

    out = Path("results/multihop_test_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {out}")
    print("\nMulti-Hop RAG pipeline working!")
