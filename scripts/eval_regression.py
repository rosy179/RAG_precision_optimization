#!/usr/bin/env python3
"""
Golden retrieval-regression test (TASKLIST D3).

Runs the FULL retrieval stack (heading-aware chunking → semantic + BM25 →
RRF → cross-encoder rerank → KB merge) over a small self-contained golden set
and fails (exit 1) if the answer-bearing document stops surviving into the
top-k for enough questions. Meant to run in CI on every PR that touches the
pipeline, catching silent retrieval regressions before they ship.

Why retrieval-only: it needs NO OpenAI key (no history → no query condensing;
routing is heuristic; contextualization is disabled here for determinism), so
it runs in CI with just the local embedder + reranker. It guards exactly the
layers most PRs change.

Usage:
  python scripts/eval_regression.py                 # top_k=3, min hit-rate 0.85
  python scripts/eval_regression.py --top-k 5 --min-hit-rate 0.90
  python scripts/eval_regression.py --verbose
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

GOLDEN_PATH = ROOT / "data" / "golden_regression.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--top-k", type=int, default=3,
                    help="expected source must appear within this many results")
    ap.add_argument("--min-hit-rate", type=float, default=0.85,
                    help="fail if the fraction of questions that hit falls below this")
    ap.add_argument("--verbose", action="store_true", help="print every question")
    args = ap.parse_args()

    if not GOLDEN_PATH.exists():
        print(f"[FAIL] golden set not found: {GOLDEN_PATH}")
        return 2
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    docs, questions = golden["documents"], golden["questions"]

    # Deterministic + key-free: no LLM-generated contextual prefixes.
    os.environ["CONTEXTUAL_RETRIEVAL"] = "0"

    # Point the vector store at a throwaway dir so a real KB is never touched.
    tmp = Path(tempfile.mkdtemp(prefix="eval_reg_"))
    try:
        import backend.services.global_kb as gk
        import backend.services.user_rag as ur
        gk.DB_PATH = tmp
        ur.DB_PATH = tmp
        from backend.services import document_processor as dp

        print(f"Ingesting {len(docs)} golden documents…")
        kb = gk.GlobalKBService()
        for d in docs:
            doc_id = dp._make_doc_id(d["title"])
            chunks, summary = dp._finalize_text_doc(
                dp._clean_text(d["content"]), doc_id, d["title"], "golden")
            kb.add_document(chunks, summary, {"id": doc_id, "name": d["title"], "type": "golden"})

        ur.warm_up()  # cross-encoder reranker — same one the webapp serves
        rag = ur.get_service("eval-regression")

        hits = 0
        by_lang: dict[str, list[int]] = {}  # lang → [hits, total]
        for q in questions:
            ret = rag.retrieve(q["question"], [], session_id=None)
            titles = [c.get("title", "") for c in ret["top_chunks"][:args.top_k]]
            ok = q["expected_source"] in titles
            hits += ok
            tally = by_lang.setdefault(q["lang"], [0, 0])
            tally[0] += ok
            tally[1] += 1
            if args.verbose or not ok:
                print(f"  [{'ok ' if ok else 'MISS'}] ({q['lang']}) {q['question'][:55]:57} "
                      f"→ {titles[0] if titles else '-'}")

        n = len(questions)
        rate = hits / n if n else 0.0
        print(f"\nGolden retrieval hit-rate @{args.top_k}: {hits}/{n} = {rate:.3f} "
              f"(threshold {args.min_hit_rate:.2f})")
        print("  by language: " + ", ".join(
            f"{lang} {h}/{t}" for lang, (h, t) in sorted(by_lang.items())))

        if rate < args.min_hit_rate:
            print(f"\n[FAIL] hit-rate {rate:.3f} below threshold {args.min_hit_rate:.2f} "
                  f"— retrieval regression detected.")
            return 1
        print("\n[PASS] retrieval regression check passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
