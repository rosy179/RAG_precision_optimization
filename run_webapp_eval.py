#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_webapp_eval.py
------------------
Evaluate the WEBAPP RAG pipeline (UserRAGService over the shared global KB)
with Ragas — not the research pipeline in src/. This answers "how good is the
thing users actually talk to", including Vietnamese/Japanese questions.

Usage:
  python run_webapp_eval.py --generate 100     # build data/webapp_eval_questions.json
                                               # (EN from KB corpus + VI/JA slices)
  python run_webapp_eval.py --limit 10         # smoke run: answer + Ragas on 10 samples
  python run_webapp_eval.py                    # full run over the whole question file
  python run_webapp_eval.py --no-ragas         # answers only (no metric cost)
  python run_webapp_eval.py --rescore          # re-run Ragas on the last run's answers

Requires the KB corpus ingested (scripts/ingest_knowledge.py) and OPENAI_API_KEY.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

import os
import openai

QUESTIONS_PATH = ROOT / "data" / "webapp_eval_questions.json"
RESULTS_PATH = ROOT / "results" / "webapp_eval_results.json"
# Full rows of the last run INCLUDING contexts — lets --rescore re-run the
# Ragas scoring (e.g. after a judge-config fix) without re-answering.
ROWS_PATH = ROOT / "results" / "webapp_eval_rows_last.json"
CORPUS_PATH = ROOT / "data" / "rag_dataset.json"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

GEN_PROMPT = {
    "en": ("Generate {n} factual question-answer pairs that can be answered ONLY "
           "from the text below. Questions must be self-contained (mention the "
           "subject by name). Answers: 1-2 sentences, factual.\n"
           'Reply as JSON array: [{{"question": "...", "answer": "..."}}]\n\n'
           "Title: {title}\n\nText:\n{text}"),
    "vi": ("Tạo {n} cặp câu hỏi-trả lời tiếng Việt dựa HOÀN TOÀN vào văn bản dưới "
           "đây (văn bản tiếng Anh, nhưng câu hỏi và câu trả lời phải bằng tiếng "
           "Việt). Câu hỏi phải tự chứa ngữ cảnh (nêu rõ tên chủ thể). Trả lời "
           "ngắn gọn 1-2 câu.\n"
           'Trả về JSON array: [{{"question": "...", "answer": "..."}}]\n\n'
           "Tiêu đề: {title}\n\nVăn bản:\n{text}"),
    "ja": ("以下の英語テキストの内容だけに基づいて、日本語の質問と回答のペアを{n}組"
           "作成してください。質問は主題名を明記した自己完結型にしてください。"
           "回答は1〜2文の事実ベースで。\n"
           'JSON配列で返答: [{{"question": "...", "answer": "..."}}]\n\n'
           "タイトル: {title}\n\n本文:\n{text}"),
}


def load_corpus() -> list[dict]:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    docs = data.get("documents", data) if isinstance(data, dict) else data
    return [d for d in docs if len(d.get("content", "")) > 400]


def gen_pairs(client, doc: dict, lang: str, n: int) -> list[dict]:
    prompt = GEN_PROMPT[lang].format(n=n, title=doc.get("title", ""),
                                     text=doc["content"][:3500])
    try:
        res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        raw = (res.choices[0].message.content or "").strip()
        # Tolerate ```json fences
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        pairs = json.loads(raw)
        if isinstance(pairs, dict):
            pairs = pairs.get("pairs", pairs.get("questions", []))
        return [p for p in pairs if p.get("question") and p.get("answer")][:n]
    except Exception as e:
        print(f"    [skip] {doc.get('title', '')[:40]}: {e}")
        return []


def generate_questions(target_en: int):
    """Build the eval set: EN questions across the KB corpus + VI/JA slices."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        print("OPENAI_API_KEY required for --generate")
        sys.exit(1)
    client = openai.OpenAI(api_key=api_key)

    docs = load_corpus()
    random.Random(42).shuffle(docs)
    print(f"Corpus: {len(docs)} usable docs. Target: {target_en} EN + 10 VI + 10 JA")

    questions: list[dict] = []
    per_doc = max(2, round(target_en / max(1, len(docs)) + 0.5))
    for doc in docs:
        if sum(1 for q in questions if q["lang"] == "en") >= target_en:
            break
        pairs = gen_pairs(client, doc, "en", per_doc)
        for p in pairs:
            questions.append({
                "id": f"en_{len(questions):03d}", "lang": "en",
                "question": p["question"], "ground_truth": p["answer"],
                "source_title": doc.get("title", ""),
            })
        print(f"  EN {sum(1 for q in questions if q['lang'] == 'en'):3d}/{target_en}"
              f"  ← {doc.get('title', '')[:50]}")

    for lang, n_total in (("vi", 10), ("ja", 10)):
        made = 0
        for doc in docs:
            if made >= n_total:
                break
            pairs = gen_pairs(client, doc, lang, 2)
            for p in pairs[: n_total - made]:
                questions.append({
                    "id": f"{lang}_{made:03d}", "lang": lang,
                    "question": p["question"], "ground_truth": p["answer"],
                    "source_title": doc.get("title", ""),
                })
                made += 1
            print(f"  {lang.upper()} {made:2d}/{n_total}  ← {doc.get('title', '')[:50]}")

    QUESTIONS_PATH.write_text(json.dumps(questions, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    by_lang = {}
    for q in questions:
        by_lang[q["lang"]] = by_lang.get(q["lang"], 0) + 1
    print(f"\n[SAVED] {QUESTIONS_PATH} — {len(questions)} questions {by_lang}")


def answer_questions(limit: int | None) -> list[dict]:
    """Answer the eval questions through the live webapp pipeline."""
    from backend.services.user_rag import get_service, warm_up

    if not QUESTIONS_PATH.exists():
        print(f"{QUESTIONS_PATH} not found — run with --generate first.")
        sys.exit(1)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if limit:
        # Even per-language slice so VI/JA are always represented
        by_lang: dict[str, list] = {}
        for q in questions:
            by_lang.setdefault(q["lang"], []).append(q)
        take = []
        langs = list(by_lang)
        i = 0
        while len(take) < limit and any(by_lang.values()):
            lang = langs[i % len(langs)]
            if by_lang[lang]:
                take.append(by_lang[lang].pop(0))
            i += 1
        questions = take
    print(f"Evaluating webapp pipeline on {len(questions)} questions "
          f"({', '.join(sorted(set(q['lang'] for q in questions)))})")

    warm_up()  # CrossEncoder reranker — same stack the webapp serves
    rag = get_service("webapp-eval")

    rows = []
    for i, q in enumerate(questions, 1):
        t0 = time.time()
        try:
            ret = rag.retrieve(q["question"], [], session_id=None)
            if ret["empty"]:
                print("KB is empty — ingest it first (scripts/ingest_knowledge.py)")
                sys.exit(1)
            answer = rag._generate_chat_answer(q["question"], ret["top_chunks"], [])
            contexts = [c["text"] for c in ret["top_chunks"]]
            status = "ok"
        except Exception as e:
            answer, contexts, status = f"[ERROR] {e}", [], "error"
        ms = int((time.time() - t0) * 1000)
        rows.append({**q, "answer": answer, "contexts": contexts,
                     "latency_ms": ms, "status": status})
        print(f"  [{i}/{len(questions)}] ({q['lang']}) {ms:5d}ms  {q['question'][:60]}")
    return rows


def run_eval(limit: int | None, use_ragas: bool, rescore: bool = False):
    from evaluation import run_ragas_evaluation

    if rescore:
        if not ROWS_PATH.exists():
            print(f"{ROWS_PATH} not found — run a full eval first.")
            sys.exit(1)
        rows = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
        print(f"Re-scoring {len(rows)} saved answers (no re-answering).")
    else:
        rows = answer_questions(limit)
        ROWS_PATH.parent.mkdir(exist_ok=True)
        ROWS_PATH.write_text(json.dumps(rows, ensure_ascii=False),
                             encoding="utf-8")

    ok = [r for r in rows if r["status"] == "ok"]
    report: dict = {
        "run": "webapp_pipeline",
        "n_total": len(rows),
        "n_ok": len(ok),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in ok) / max(1, len(ok))),
    }

    if use_ragas and ok:
        overall = run_ragas_evaluation({
            "question":     [r["question"] for r in ok],
            "answer":       [r["answer"] for r in ok],
            "contexts":     [r["contexts"] for r in ok],
            "ground_truth": [r["ground_truth"] for r in ok],
        }, run_name="webapp_pipeline")
        report["ragas"] = overall

        # Per-language breakdown (each language scored separately)
        report["by_lang"] = {}
        for lang in sorted(set(r["lang"] for r in ok)):
            subset = [r for r in ok if r["lang"] == lang]
            if len(subset) < 2:
                continue
            report["by_lang"][lang] = run_ragas_evaluation({
                "question":     [r["question"] for r in subset],
                "answer":       [r["answer"] for r in subset],
                "contexts":     [r["contexts"] for r in subset],
                "ground_truth": [r["ground_truth"] for r in subset],
            }, run_name=f"webapp_{lang}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    report["samples"] = [{k: v for k, v in r.items() if k != "contexts"} for r in rows]
    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print(f"\n[SAVED] {RESULTS_PATH}")
    if "ragas" in report:
        m = report["ragas"]
        print(f"\n=== Webapp pipeline Ragas ({m['n_samples']} samples) ===")
        print(f"  Faithfulness      {m['faithfulness']:.4f}")
        print(f"  Answer relevancy  {m['answer_relevancy']:.4f}")
        print(f"  Context precision {m['context_precision']:.4f}")
        print(f"  Context recall    {m['context_recall']:.4f}")
        print(f"  AVG               {m['avg_score']:.4f}")
        for lang, lm in report.get("by_lang", {}).items():
            print(f"  [{lang}] avg {lm['avg_score']:.4f} "
                  f"(faith {lm['faithfulness']:.2f}, relev {lm['answer_relevancy']:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", type=int, metavar="N",
                    help="generate N English questions from the KB corpus (+10 VI, +10 JA)")
    ap.add_argument("--limit", type=int, help="evaluate only this many questions")
    ap.add_argument("--no-ragas", action="store_true", help="answers only, skip Ragas")
    ap.add_argument("--rescore", action="store_true",
                    help="re-run Ragas on the last run's saved answers (no re-answering)")
    args = ap.parse_args()

    if args.generate:
        generate_questions(args.generate)
    else:
        run_eval(args.limit, use_ragas=not args.no_ragas, rescore=args.rescore)
