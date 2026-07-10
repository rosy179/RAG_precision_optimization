#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_dataset.py
----------------
Extracts the unique context paragraphs from the SQuAD QA pairs
and adds them to the documents list in rag_dataset.json.
This ensures the retriever has access to the actual content
needed to answer SQuAD questions.
"""

import sys
import json
from pathlib import Path

# Fix Windows terminal encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_PATH = Path("data/rag_dataset.json")

def main():
    print("=" * 60)
    print("  Aligning RAG Dataset: Merging SQuAD contexts into Documents")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"[ERROR] {DATA_PATH} does not exist.")
        return

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    qa_pairs = dataset.get("qa_pairs", [])
    documents = dataset.get("documents", [])

    print(f"Current QA pairs : {len(qa_pairs)}")
    print(f"Current Documents: {len(documents)}")

    # Extract unique context documents from QA pairs
    unique_contexts = {}
    for qa in qa_pairs:
        ctx_text = qa.get("document", "").strip()
        if ctx_text:
            unique_contexts[ctx_text] = qa.get("title", "University_of_Notre_Dame")

    print(f"Found {len(unique_contexts)} unique context paragraphs in QA pairs")

    # Check which ones are already in documents list (to avoid duplicates)
    existing_contents = {d.get("content", "").strip() for d in documents}
    
    added_count = 0
    for idx, (ctx_text, title) in enumerate(unique_contexts.items()):
        if ctx_text not in existing_contents:
            doc_id = f"squad_ctx_{idx:03d}"
            documents.append({
                "id": doc_id,
                "title": title,
                "content": ctx_text,
                "url": f"https://en.wikipedia.org/wiki/{title}",
                "source": "SQUAD_v1.1",
                "type": "wikipedia_squad",
                "chars": len(ctx_text)
            })
            added_count += 1

    print(f"Added {added_count} new SQuAD contexts to Documents list")

    # Update metadata
    meta = dataset.setdefault("metadata", {})
    meta["documents_count"] = len(documents)
    meta["total_chars"] = sum(len(d.get("content", "")) for d in documents)
    meta["squad_contexts_count"] = len(unique_contexts)

    # Save
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Saved updated dataset to {DATA_PATH}")
    print(f"Total Documents now: {len(documents)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
