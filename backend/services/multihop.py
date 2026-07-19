"""
Multi-Hop RAG (Layer 6): chained retrieval for bridging questions.

A question needs multi-hop when it references an entity that must first be
discovered by a separate search ("Who won X for the work behind Y?" → find
"the work behind Y" first). Provided as a mixin so UserRAGService stays a
single class even though its code lives across files; the methods here call
`self.retrieve` / `self.generate_stream` from the host class.
"""

from backend.services import llm
from backend.services.prompts import MULTIHOP_HOP_PROMPT, MULTIHOP_ROUTE_PROMPT
from backend.services.rag_helpers import add_usage, chunks_to_sources


class MultihopMixin:
    """Chained-retrieval methods mixed into UserRAGService."""

    def route_multihop(self, question: str,
                       usage: dict | None = None) -> list[str]:
        """Decide whether the question needs chained retrieval.

        Returns ordered sub-queries (2-3) for multi-hop, or [] for the
        normal single-retrieval path. Callers should only invoke this for
        heuristically complex questions to keep simple lookups fast.
        """
        if llm.is_mock():
            return []
        try:
            client = llm.get_client()
            res = client.chat.completions.create(
                model=llm.LLM_MODEL,
                messages=[{"role": "user",
                           "content": MULTIHOP_ROUTE_PROMPT.format(question=question)}],
                temperature=0.0,
                max_tokens=200,
            )
            add_usage(usage, res)
            raw = (res.choices[0].message.content or "").strip()
        except Exception:
            return []
        if not raw or raw.upper().startswith("SINGLE"):
            return []
        subs = [line.strip().lstrip("0123456789.-) ").strip()
                for line in raw.splitlines() if line.strip()]
        subs = [s for s in subs if len(s) > 5]
        # A single sub-question is just a rephrase — not worth the extra hops
        return subs[:3] if len(subs) >= 2 else []

    def _hop_answer(self, sub_question: str, chunks: list[dict],
                    usage: dict | None = None) -> str:
        """Short bridging answer for one hop (feeds the next hop's query)."""
        if not chunks:
            return "NOT FOUND"
        if llm.is_mock():
            return chunks[0]["text"][:200]
        context = "\n\n---\n\n".join(
            f"[Nguồn {c.get('rank', i + 1)}: {c.get('title', '')}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )
        try:
            client = llm.get_client()
            res = client.chat.completions.create(
                model=llm.LLM_MODEL,
                messages=[{"role": "user", "content": MULTIHOP_HOP_PROMPT.format(
                    context=context, sub_question=sub_question)}],
                temperature=0.0,
                max_tokens=150,
            )
            add_usage(usage, res)
            return (res.choices[0].message.content or "").strip() or "NOT FOUND"
        except Exception:
            return "NOT FOUND"

    def multihop_events(self, question: str, sub_questions: list[str],
                        history: list[dict] | None = None,
                        session_id: str | None = None,
                        include_doc_ids: list[str] | None = None,
                        use_global_kb: bool = True,
                        attached_doc_ids: list[str] | None = None,
                        usage: dict | None = None):
        """Chained-retrieval pipeline as an event generator.

        Yields ("step", dict) for each reasoning step, then
        ("sources", {"sources": [...]}) with the deduplicated union of all
        hops' chunks, then ("delta", str) tokens of the final answer.
        """
        history = history or []
        bridge: list[dict] = []
        all_chunks: list[dict] = []

        for i, sub_q in enumerate(sub_questions, 1):
            # Facts found so far steer the next retrieval — this is the
            # whole point of multi-hop: the bridging entity was never in
            # the original question.
            established = " | ".join(
                f"[Đã biết: {b['answer']}]" for b in bridge
                if b["answer"] and "NOT FOUND" not in b["answer"]
            )
            enriched = f"{sub_q} ({established})" if established else sub_q
            yield ("step", {"stage": "hop_start", "hop": i,
                            "total": len(sub_questions), "query": enriched})

            ret = self.retrieve(enriched, [], session_id,
                                include_doc_ids=include_doc_ids,
                                use_global_kb=use_global_kb,
                                attached_doc_ids=attached_doc_ids,
                                digest_ok=False, usage=usage)
            chunks = ret["top_chunks"] if not ret["empty"] else []
            answer = self._hop_answer(sub_q, chunks, usage=usage)
            all_chunks.extend(chunks)
            bridge.append({"hop": i, "sub_question": sub_q, "answer": answer})
            yield ("step", {"stage": "hop_done", "hop": i, "answer": answer,
                            "titles": [c.get("title", "") for c in chunks[:3]]})

        # Union of every hop's evidence, deduplicated, capped for context
        seen: set = set()
        unique: list[dict] = []
        for c in all_chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                unique.append(c)
        unique = unique[:8]
        for idx, c in enumerate(unique, 1):
            c["rank"] = idx
        yield ("sources", {"sources": chunks_to_sources(unique)})

        facts = "\n".join(f"- Bước {b['hop']}: {b['sub_question']} → {b['answer']}"
                          for b in bridge)
        synth_question = (
            f"{question}\n\n"
            f"(Các dữ kiện trung gian đã xác minh qua tìm kiếm nhiều bước — "
            f"dùng chúng để trả lời, vẫn trích dẫn [n] theo Nguồn:\n{facts})"
        )
        for delta in self.generate_stream(synth_question, unique, history,
                                          usage=usage):
            yield ("delta", delta)
