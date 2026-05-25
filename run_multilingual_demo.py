#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_multilingual_demo.py
-------------------------
Demo: Multilingual RAG with Vietnamese and Japanese inputs.

Demonstrates both strategies side-by-side:
  Strategy 1 "translate"     — translate query → English → CoT pipeline → translate back
  Strategy 2 "multilingual"  — multilingual embedding + CrossEncoder + CoT in source lang

Usage:
    python run_multilingual_demo.py                      # both strategies, VI + JA
    python run_multilingual_demo.py --strategy translate  # translation only
    python run_multilingual_demo.py --strategy multilingual
    python run_multilingual_demo.py --lang vi            # Vietnamese only
    python run_multilingual_demo.py --lang ja            # Japanese only
"""

import sys
import json
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from multilingual_rag import MultilingualRAG

# ── Test Questions ─────────────────────────────────────────
QUESTIONS = {
    "vi": [
        "RAG là gì và hoạt động như thế nào?",
        "Sự khác biệt giữa BM25 và tìm kiếm ngữ nghĩa là gì?",
        "CrossEncoder reranking hoạt động như thế nào?",
        "Mô hình Transformer có những thành phần chính nào?",
        "Vector database lưu trữ embeddings như thế nào?",
    ],
    "ja": [
        "RAGとはどのような技術ですか？",
        "BM25とセマンティック検索の違いは何ですか？",
        "密な文章検索はどのように機能しますか？",
        "Transformerモデルの主要なコンポーネントは何ですか？",
        "ベクトルデータベースはエンベッディングをどのように保存しますか？",
    ],
    "en": [
        "What is Retrieval Augmented Generation?",
        "What is the difference between BM25 and semantic search?",
    ],
}

QUESTION_MAP = {
    # Each VI question paired with its EN ground truth (for manual comparison)
    "RAG là gì và hoạt động như thế nào?":
        "What is Retrieval Augmented Generation?",
    "Sự khác biệt giữa BM25 và tìm kiếm ngữ nghĩa là gì?":
        "What is the difference between BM25 and semantic search?",
    "RAGとはどのような技術ですか？":
        "What is Retrieval Augmented Generation?",
    "BM25とセマンティック検索の違いは何ですか？":
        "What is the difference between BM25 and semantic search?",
}


def run_demo(strategy: str, lang_filter: str = "all", n_questions: int = 3):
    print("\n" + "=" * 70)
    print(f"  MULTILINGUAL RAG DEMO — strategy='{strategy}'")
    print("=" * 70)

    # Build pipeline
    rag = MultilingualRAG(strategy=strategy)
    rag.build()

    results = []

    # Select target languages
    if lang_filter == "all":
        langs = ["vi", "ja"]
    else:
        langs = [lang_filter]

    for lang in langs:
        questions = QUESTIONS.get(lang, [])[:n_questions]
        print(f"\n{'─'*70}")
        print(f"  Language: {lang.upper()}")
        print(f"{'─'*70}")

        for q in questions:
            try:
                result = rag.query(q, verbose=True)
                results.append(result)
            except Exception as e:
                print(f"  [ERROR] {q[:50]}... → {e}")

    return results


def compare_strategies(lang: str = "vi", n_questions: int = 2):
    """Run the same questions through both strategies and show comparison."""
    questions = QUESTIONS.get(lang, [])[:n_questions]
    lang_label = {"vi": "Vietnamese", "ja": "Japanese"}.get(lang, lang)

    print("\n" + "=" * 70)
    print(f"  STRATEGY COMPARISON — {lang_label}")
    print("=" * 70)

    # Build both pipelines
    print("\n[Building TRANSLATE strategy...]")
    rag_t = MultilingualRAG(strategy="translate").build()

    print("\n[Building MULTILINGUAL strategy...]")
    rag_m = MultilingualRAG(strategy="multilingual").build()

    comparison = []

    for q in questions:
        print(f"\n{'─'*70}")
        print(f"Q: {q}")
        print(f"{'─'*70}")

        r_translate = rag_t.query(q, verbose=False)
        r_multilingual = rag_m.query(q, verbose=False)

        print(f"\n[Translate strategy]")
        print(f"  Query (EN): {r_translate.get('query_en', 'N/A')}")
        print(f"  Answer ({lang_label}): {r_translate['answer']}")

        print(f"\n[Multilingual strategy]")
        print(f"  Answer ({lang_label}): {r_multilingual['answer']}")

        en_ref = QUESTION_MAP.get(q, "")
        if en_ref:
            print(f"\n[Reference EN question: {en_ref}]")

        comparison.append({
            "question": q,
            "lang": lang,
            "translate": {
                "query_en": r_translate.get("query_en", ""),
                "answer": r_translate["answer"],
                "answer_en": r_translate.get("answer_en", ""),
            },
            "multilingual": {
                "answer": r_multilingual["answer"],
            },
        })

    return comparison


def print_analysis():
    print("\n" + "=" * 70)
    print("  MULTILINGUAL IMPROVEMENT ANALYSIS")
    print("=" * 70)
    print("""
PROBLEM BREAKDOWN — What fails with non-English input:

┌─────────────────────────┬──────────────────┬──────────────────────┐
│ Component               │ Vietnamese       │ Japanese             │
├─────────────────────────┼──────────────────┼──────────────────────┤
│ BM25 tokenizer          │ ~OK (has spaces) │ FAIL (no spaces)     │
│ text-embedding-ada-002  │ ~65% quality     │ ~55% quality         │
│ ms-marco CrossEncoder   │ ~30% accuracy    │ ~20% accuracy        │
│ Response language       │ Returns English  │ Returns English      │
├─────────────────────────┼──────────────────┼──────────────────────┤
│ Est. quality (no fix)   │ ~0.65-0.72       │ ~0.40-0.55           │
└─────────────────────────┴──────────────────┴──────────────────────┘

SOLUTION COMPARISON:

Strategy "translate" (this demo):
  + Quick: no reindexing (uses existing English index)
  + Works for all languages (GPT-4o-mini translates everything)
  - +1-2s latency per query (2 extra LLM calls)
  - Translation errors can propagate (esp. proper nouns, technical terms)
  - Est. quality: Vietnamese ~0.85 / Japanese ~0.83

Strategy "multilingual" (this demo):
  + No translation latency
  + multilingual-e5-small: cross-lingual semantic search
  + mmarco CrossEncoder: trained on 13 languages (VI, JA, ZH, KO, ...)
  + BM25 bigrams: handles CJK no-space tokenization
  + CoT responds in source language directly
  - Requires reindexing (~2 min first run, cached after)
  - Est. quality: Vietnamese ~0.90 / Japanese ~0.87

RECOMMENDED MODELS:
  Embedding:   intfloat/multilingual-e5-small  (117MB, 100+ langs)
  CrossEncoder: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (120MB)
  Supported languages (mmarco): EN DE FR ES IT PT NL VI ID AR ZH JA RU
""")


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual RAG demo — Vietnamese and Japanese"
    )
    parser.add_argument(
        "--strategy",
        choices=["translate", "multilingual", "compare", "all"],
        default="all",
        help="Which strategy to demo (default: all = translate + multilingual)"
    )
    parser.add_argument(
        "--lang",
        choices=["vi", "ja", "all"],
        default="all",
        help="Language to test (default: all = Vietnamese + Japanese)"
    )
    parser.add_argument(
        "--n", type=int, default=3,
        help="Number of questions per language (default: 3)"
    )
    parser.add_argument(
        "--analysis", action="store_true",
        help="Print analysis table only"
    )
    args = parser.parse_args()

    if args.analysis:
        print_analysis()
        return

    all_results = []

    if args.strategy in ("all", "translate"):
        lang_filter = args.lang if args.lang != "all" else "all"
        results = run_demo("translate", lang_filter, args.n)
        all_results.extend(results)

    if args.strategy in ("all", "multilingual"):
        lang_filter = args.lang if args.lang != "all" else "all"
        results = run_demo("multilingual", lang_filter, args.n)
        all_results.extend(results)

    if args.strategy == "compare":
        lang = args.lang if args.lang != "all" else "vi"
        comparison = compare_strategies(lang, args.n)
        all_results = comparison

    # Print analysis summary
    print_analysis()

    # Save results
    if all_results:
        out = Path("results/multilingual_demo_results.json")
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[SAVED] Results → {out}")


if __name__ == "__main__":
    main()
