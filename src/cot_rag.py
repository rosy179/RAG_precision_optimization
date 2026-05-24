#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/cot_rag.py
--------------
Day 22-24: Chain-of-Thought (CoT) Retrieval
Target metric: Faithfulness (currently 0.8389) — best technique so far is Reranker

Problem this solves:
  Faithfulness measures whether the answer is grounded in retrieved context.
  Even with great retrieval, the LLM can hallucinate details not in the context.
  CoT forces the model to explicitly trace facts before concluding → fewer hallucinations.

How it works:
  Standard prompt: context → "Answer the question" → Answer
  CoT prompt:      context → "Extract facts → Reason step by step → Final Answer" → Answer

  The intermediate reasoning step anchors the LLM to the retrieved passages.

Three modes:
  "structured" (default) — 3-step prompt: facts → reasoning → "Final Answer:" (extracted)
  "simple"               — adds "Think step by step" before the answer
  "citation"             — forces [Source N] citations in the reasoning chain

Architecture:
  Built on RerankerRAG (best retrieval: Hybrid + CrossEncoder)
  Only the generation layer changes — retrieval is identical to Reranker baseline.
"""

import sys
import os
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────
TOP_K          = int(os.getenv("TOP_K", 3))
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-4o-mini")
RETRIEVE_DEPTH = int(os.getenv("RETRIEVE_DEPTH", 20))

# ── CoT Prompt Templates ──────────────────────────────────

STRUCTURED_COT_PROMPT = """\
You are a precise QA assistant. Use ONLY the provided context to answer the question.

Context:
{context}

Question: {question}

Work through these steps:

Step 1 - Relevant Facts:
List the specific facts from the context that are relevant to the question.

Step 2 - Reasoning:
Based only on the facts above, reason through what the answer must be.

Step 3 - Final Answer:
Provide a concise, direct answer grounded in the context.
If the context lacks sufficient information, write exactly: I don't have enough information.

Step 1 - Relevant Facts:"""

SIMPLE_COT_PROMPT = """\
You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Let's think step by step before giving the final answer.

Reasoning and Answer:"""

CITATION_COT_PROMPT = """\
You are a precise QA assistant. Use ONLY the provided context to answer.
For every claim in your answer, add a citation like [Source N] referencing the context passage.

Context:
{context}

Question: {question}

Step 1 - Evidence (cite sources as [Source N]):
Step 2 - Reasoning:
Step 3 - Answer: """


def _build_context_string(context_chunks: list) -> str:
    """Format context chunks into a numbered string for the prompt."""
    return "\n\n---\n\n".join(
        f"[Source {c['rank']}: {c['title']}]\n{c['text']}"
        for c in context_chunks
    )


def _extract_final_answer(response: str, mode: str) -> str:
    """
    Extract the final answer from a CoT response.
    For 'structured' mode, parses out everything after 'Final Answer:'.
    """
    if mode == "structured":
        marker = "Final Answer:"
        if marker in response:
            return response.split(marker, 1)[-1].strip()
        # Fallback: last non-empty paragraph
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        return paragraphs[-1] if paragraphs else response.strip()

    elif mode == "citation":
        # Look for "Step 3 - Answer:" or just "Answer:"
        for marker in ("Step 3 - Answer:", "Answer:"):
            if marker in response:
                return response.split(marker, 1)[-1].strip()
        return response.strip()

    else:  # simple
        return response.strip()


def generate_cot_answer(query: str, context_chunks: list,
                        model: str = LLM_MODEL,
                        mode: str = "structured") -> tuple:
    """
    Generate answer using Chain-of-Thought prompting.
    Returns (final_answer, full_reasoning) — reasoning is logged but not used in eval.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        context_preview = context_chunks[0]["text"][:200] if context_chunks else "No context"
        mock = f"[MOCK CoT ANSWER] Based on context: '{context_preview}...'"
        return mock, mock

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    context_str = _build_context_string(context_chunks)

    if mode == "structured":
        prompt = STRUCTURED_COT_PROMPT.format(context=context_str, question=query)
    elif mode == "citation":
        prompt = CITATION_COT_PROMPT.format(context=context_str, question=query)
    else:  # simple
        prompt = SIMPLE_COT_PROMPT.format(context=context_str, question=query)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=600,   # more tokens needed for reasoning chain
    )
    full_reasoning = response.choices[0].message.content.strip()
    final_answer = _extract_final_answer(full_reasoning, mode)

    return final_answer, full_reasoning


# ── Main Class ────────────────────────────────────────────
class ChainOfThoughtRAG:
    """
    Chain-of-Thought RAG: RerankerRAG retrieval + CoT generation.

    Retrieval stack: Hybrid (BM25 + Semantic + RRF) → CrossEncoder Rerank
    Generation:      Structured CoT prompt → extract Final Answer

    mode options:
      "structured" (default) — 3-step facts/reasoning/answer, parses "Final Answer:"
      "simple"               — "think step by step" prefix, minimal change
      "citation"             — forces [Source N] citations before concluding
    """

    def __init__(self, mode: str = "structured",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.mode = mode
        self.reranker_model = reranker_model
        self._reranker_rag = None

    def build(self):
        """Build underlying RerankerRAG pipeline."""
        src_dir = Path(__file__).parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        from reranker_rag import RerankerRAG

        self._reranker_rag = RerankerRAG(reranker_model=self.reranker_model)
        self._reranker_rag.build()

        print(f"\n[CoT] Chain-of-Thought mode: '{self.mode}'")
        print("[CoT] Chain-of-Thought Pipeline ready!")
        return self

    def query(self, question: str,
              top_k: int = TOP_K,
              retrieve_depth: int = RETRIEVE_DEPTH,
              verbose: bool = True) -> dict:
        """Full pipeline: Hybrid Retrieve → Rerank → CoT Generate."""
        if self._reranker_rag is None:
            raise RuntimeError("Call .build() first")

        # Retrieval: identical to RerankerRAG
        context = self._reranker_rag.retrieve_and_rerank(
            question, top_k=top_k, retrieve_depth=retrieve_depth
        )

        # Generation: CoT prompt
        final_answer, full_reasoning = generate_cot_answer(
            question, context, model=LLM_MODEL, mode=self.mode
        )

        result = {
            "question":      question,
            "answer":        final_answer,       # used by Ragas evaluation
            "reasoning":     full_reasoning,     # full CoT trace (logged only)
            "context":       context,
            "top_k":         top_k,
            "cot_mode":      self.mode,
        }

        if verbose:
            print(f"\n{'='*65}")
            print(f"Q: {question} [CoT mode={self.mode}]")
            print(f"{'='*65}")
            print(f"\n--- Reasoning ---\n{full_reasoning[:400]}...")
            print(f"\n--- Final Answer ---\n{final_answer}")
            print(f"\n--- Top {len(context)} passages ---")
            for c in context:
                rer = c.get("reranker_score", "N/A")
                score_str = f"{rer:.4f}" if isinstance(rer, float) else str(rer)
                print(f"  [{c['rank']}] {c['title']} (reranker={score_str})")
                print(f"      {c['text'][:120]}...")

        return result


# ── Quick Test ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Chain-of-Thought RAG Pipeline -- Day 22-24")
    print("=" * 65)

    rag = ChainOfThoughtRAG(mode="structured").build()

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

    out = Path("results/cot_test_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] Test results -> {out}")
    print("\nChain-of-Thought RAG pipeline is working!")
