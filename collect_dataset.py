#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================
 RAG Precision Optimization -- Dataset Collection Script
 Domain: AI / ML / NLP (Technical Documents)
 Output: data/squad_qa.json | wiki_documents.json | arxiv_papers.json | rag_dataset.json
=============================================================
"""

import sys
import json
import os
import time
from pathlib import Path
from datetime import datetime

# Fix Windows terminal encoding (cp1252 -> utf-8)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ──────────────────────────────────────────────
#  CONFIG  (Chỉnh ở đây nếu muốn thay đổi)
# ──────────────────────────────────────────────
SQUAD_SAMPLES = 150          # Số QA pairs từ SQUAD (tối đa 88k)
ARXIV_MAX     = 30           # Số ArXiv papers (abstract)
DATA_DIR      = Path("data") # Thư mục output

WIKI_TOPICS = [
    # Core AI/ML
    "Machine Learning",
    "Deep Learning",
    "Artificial Intelligence",
    "Natural Language Processing",
    "Transformer (machine learning)",
    "Large language model",
    # RAG / Search
    "Retrieval-augmented generation",
    "Semantic search",
    "Vector database",
    "Information retrieval",
    # Embeddings / Models
    "Word2vec",
    "BERT (language model)",
    "Attention mechanism",
    # Evaluation / Infrastructure
    "Question answering",
    "Named-entity recognition",
]

ARXIV_QUERIES = [
    "retrieval augmented generation RAG",
    "dense passage retrieval question answering",
    "semantic search embedding neural",
    "large language model evaluation benchmark",
]


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────
def banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size_kb = path.stat().st_size / 1024
    print(f"  [SAVED] --> {path}  ({size_kb:.1f} KB)")


# ──────────────────────────────────────────────
#  1. SQUAD QA pairs
# ──────────────────────────────────────────────
def collect_squad(n=SQUAD_SAMPLES):
    banner("STEP 1 / 3 — SQUAD QA Pairs")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] Run: pip install datasets")
        return []

    print(f"  Downloading SQUAD (first {n} samples)...")
    try:
        squad = load_dataset("squad", split=f"train[:{n}]")
        qa_pairs = []
        for idx, item in enumerate(squad):
            qa_pairs.append({
                "id":       f"squad_{idx:04d}",
                "query":    item["question"],
                "document": item["context"],
                "answer":   item["answers"]["text"][0] if item["answers"]["text"] else "",
                "title":    item.get("title", ""),
                "source":   "SQUAD_v1.1",
            })
        out = DATA_DIR / "squad_qa.json"
        save_json(qa_pairs, out)
        print(f"  [OK] Downloaded {len(qa_pairs)} QA pairs")
        return qa_pairs
    except Exception as e:
        print(f"  [ERROR] downloading SQUAD: {e}")
        return []


# ──────────────────────────────────────────────
#  2. Wikipedia Technical Documents
# ──────────────────────────────────────────────
def collect_wikipedia(topics=WIKI_TOPICS):
    banner("STEP 2 / 3 — Wikipedia Documents")
    print(f"\n  Downloading Wikipedia documents ({len(topics)} topics)...")
    try:
        import wikipediaapi
    except ImportError:
        print("  [ERROR] Run: pip install wikipedia-api")
        return []

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="RAG-Research-Bot/1.0 (research@example.com)"
    )
    documents = []

    for i, topic in enumerate(topics, 1):
        print(f"  [{i:02d}/{len(topics)}] {topic} ...", end=" ", flush=True)
        try:
            page = wiki.page(topic)
            if page.exists():
                documents.append({
                    "id":      topic.lower().replace(" ", "_"),
                    "title":   page.title,
                    "content": page.text,
                    "url":     page.fullurl,
                    "chars":   len(page.text),
                    "source":  "Wikipedia",
                })
                print(f"OK  ({len(page.text):,} chars)")
            else:
                print("SKIP (not found)")
            time.sleep(0.3)   # Be polite to Wikipedia API
        except Exception as e:
            print(f"FAIL ({e})")

    out = DATA_DIR / "wiki_documents.json"
    save_json(documents, out)
    print(f"\n  [OK] {len(documents)} Wikipedia documents collected")
    return documents


# ──────────────────────────────────────────────
#  3. ArXiv Papers (abstract + metadata)
# ──────────────────────────────────────────────
def collect_arxiv(queries=ARXIV_QUERIES, max_per_query=ARXIV_MAX // len(ARXIV_QUERIES)):
    banner("STEP 3 / 3 — ArXiv Papers")
    try:
        import arxiv
    except ImportError:
        print("  [ERROR] Run: pip install arxiv")
        return []

    client = arxiv.Client()
    seen_ids = set()
    papers = []

    for query in queries:
        print(f"\n  [QUERY] \"{query}\"")
        try:
            results = client.results(
                arxiv.Search(
                    query=query,
                    max_results=max_per_query,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
            )
            count = 0
            for r in results:
                pid = r.entry_id.split("/abs/")[-1]
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                papers.append({
                    "id":         pid,
                    "title":      r.title,
                    "authors":    [a.name for a in r.authors[:5]],
                    "abstract":   r.summary,
                    "published":  str(r.published.date()),
                    "url":        r.entry_id,
                    "pdf_url":    r.pdf_url,
                    "categories": r.categories,
                    "source":     "ArXiv",
                    # Use abstract as the document content for RAG indexing
                    "content":    f"Title: {r.title}\n\nAbstract:\n{r.summary}",
                })
                print(f"    + {r.title[:70]}...")
                count += 1
            print(f"    -> {count} papers added")
        except Exception as e:
            print(f"    [ERROR]: {e}")

    out = DATA_DIR / "arxiv_papers.json"
    save_json(papers, out)
    print(f"\n  [OK] {len(papers)} ArXiv papers collected")
    return papers


# ──────────────────────────────────────────────
#  4. Combine into final RAG dataset
# ──────────────────────────────────────────────
def combine_all(qa_pairs, wiki_docs, arxiv_papers):
    banner("FINAL -- Combining into rag_dataset.json")

    all_docs = wiki_docs + [
        {**p, "type": "arxiv_abstract"}
        for p in arxiv_papers
    ]

    dataset = {
        "metadata": {
            "created_at":        datetime.now().isoformat(),
            "domain":            "AI / ML / NLP",
            "qa_pairs_count":    len(qa_pairs),
            "documents_count":   len(all_docs),
            "wiki_count":        len(wiki_docs),
            "arxiv_count":       len(arxiv_papers),
            "total_chars":       sum(len(d.get("content","")) for d in all_docs),
            "split": {
                "train": "70%  (~105 QA pairs)",
                "test":  "30%  (~45 QA pairs)",
            },
        },
        "qa_pairs":  qa_pairs,
        "documents": all_docs,
    }

    out = DATA_DIR / "rag_dataset.json"
    save_json(dataset, out)

    print("\n  DATASET SUMMARY:")
    print(f"   QA pairs  : {len(qa_pairs)}")
    print(f"   Documents : {len(all_docs)}  (Wikipedia: {len(wiki_docs)} | ArXiv: {len(arxiv_papers)})")
    print(f"   Total text: {dataset['metadata']['total_chars']:,} characters")


# ──────────────────────────────────────────────
#  5. Print sample for verification
# ──────────────────────────────────────────────
def print_sample(qa_pairs, docs):
    banner("SAMPLE CHECK")
    if qa_pairs:
        q = qa_pairs[0]
        print("  [QA pair #0]")
        print(f"    Query  : {q['query']}")
        print(f"    Answer : {q['answer'][:100]}")
        print(f"    Source : {q['source']}")
    if docs:
        d = docs[0]
        print("\n  [Document #0]")
        print(f"    Title  : {d['title']}")
        print(f"    Source : {d['source']}")
        print(f"    Chars  : {len(d.get('content', ''))}")
        print(f"    Preview: {d.get('content', '')[:200]}...")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RAG DATASET COLLECTION -- AI/ML Domain")
    print("=" * 60)
    print(f"  Output directory : ./{DATA_DIR}/")
    print(f"  SQUAD samples    : {SQUAD_SAMPLES}")
    print(f"  Wikipedia topics : {len(WIKI_TOPICS)}")
    print(f"  ArXiv papers     : up to {ARXIV_MAX}")

    DATA_DIR.mkdir(exist_ok=True)

    # --- Collect ---
    qa_pairs     = collect_squad(SQUAD_SAMPLES)
    wiki_docs    = collect_wikipedia(WIKI_TOPICS)
    arxiv_papers = collect_arxiv()

    # --- Combine ---
    if qa_pairs or wiki_docs or arxiv_papers:
        combine_all(qa_pairs, wiki_docs, arxiv_papers)

    # --- Sample check ---
    print_sample(qa_pairs, wiki_docs + arxiv_papers)

    print("\n" + "=" * 60)
    print("  [DONE] COLLECTION COMPLETE!")
    print("=" * 60)
    print("\nFiles in ./data/:")
    for f in sorted(DATA_DIR.glob("*.json")):
        print(f"   {f.name:30s}  ({f.stat().st_size/1024:.1f} KB)")

    print("\nNext steps:")
    print("  1. Review ./data/rag_dataset.json")
    print("  2. Setup vector DB (Chroma or Pinecone)")
    print("  3. Build baseline RAG pipeline (Day 2 of 45-day plan)")
    print("\nHappy RAG building!\n")


if __name__ == "__main__":
    main()
