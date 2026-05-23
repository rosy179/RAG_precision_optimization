#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArXiv Paper Collector with Retry + Backoff
Fetches RAG/NLP/ML paper abstracts for the RAG dataset.
Saves to: data/arxiv_papers.json
"""

import sys
import json
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ARXIV_QUERIES = [
    ("retrieval augmented generation", 10),
    ("dense passage retrieval question answering", 8),
    ("semantic search neural embedding", 8),
    ("large language model evaluation", 8),
]

MAX_RETRIES = 5
BASE_WAIT   = 10   # seconds before first retry


def fetch_with_retry(client, search, label):
    """Fetch arxiv results with exponential backoff on 429."""
    import arxiv

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            results = list(client.results(search))
            return results
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                wait = BASE_WAIT * (2 ** (attempt - 1))   # 10, 20, 40, 80, 160s
                print(f"    [429] Rate limited. Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                print(f"    [ERROR] {msg}")
                return []
    print(f"    [FAIL] Gave up after {MAX_RETRIES} retries for: {label}")
    return []


def collect_arxiv():
    try:
        import arxiv
    except ImportError:
        print("[ERROR] Run: pip install arxiv")
        return []

    print("=" * 60)
    print("  ArXiv Paper Collector (with retry + backoff)")
    print("=" * 60)

    client   = arxiv.Client(num_retries=1, delay_seconds=5.0)
    seen_ids = set()
    papers   = []

    for query, max_results in ARXIV_QUERIES:
        print(f"\n  [QUERY] \"{query}\" (max {max_results})")
        time.sleep(5)   # polite delay between queries

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = fetch_with_retry(client, search, query)

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
                "content":    f"Title: {r.title}\n\nAbstract:\n{r.summary}",
            })
            print(f"    + [{r.published.year}] {r.title[:65]}...")
            count += 1

        print(f"    -> {count} new papers added (total so far: {len(papers)})")

    # Save
    out = DATA_DIR / "arxiv_papers.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"\n  [SAVED] --> {out}  ({out.stat().st_size/1024:.1f} KB)")
    print(f"  [OK] {len(papers)} ArXiv papers collected")
    return papers


def merge_into_main_dataset(arxiv_papers):
    """Merge new ArXiv papers into existing rag_dataset.json."""
    main_path = DATA_DIR / "rag_dataset.json"
    if not main_path.exists():
        print("\n  [WARN] rag_dataset.json not found — skipping merge")
        return

    with open(main_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    existing_docs = dataset.get("documents", [])
    existing_ids  = {d["id"] for d in existing_docs}

    new_docs = [
        {**p, "type": "arxiv_abstract"}
        for p in arxiv_papers
        if p["id"] not in existing_ids
    ]

    dataset["documents"].extend(new_docs)
    dataset["metadata"]["arxiv_count"]    = len(arxiv_papers)
    dataset["metadata"]["documents_count"] = len(dataset["documents"])
    dataset["metadata"]["total_chars"]    = sum(
        len(d.get("content", "")) for d in dataset["documents"]
    )

    with open(main_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n  [MERGED] rag_dataset.json updated:")
    print(f"    QA pairs  : {len(dataset['qa_pairs'])}")
    print(f"    Documents : {len(dataset['documents'])} (added {len(new_docs)} ArXiv)")
    print(f"    Total text: {dataset['metadata']['total_chars']:,} chars")


if __name__ == "__main__":
    papers = collect_arxiv()
    if papers:
        merge_into_main_dataset(papers)
    else:
        print("\n  [INFO] No papers collected. Check network / try again later.")
    print("\nDone!\n")
