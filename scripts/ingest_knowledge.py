"""
Bulk-load documents into the GLOBAL knowledge base (shared by all users).

Usage:
  python scripts/ingest_knowledge.py path/to/folder another.pdf notes.txt
  python scripts/ingest_knowledge.py --urls urls.txt
  python scripts/ingest_knowledge.py --corpus data/rag_dataset.json
  python scripts/ingest_knowledge.py docs/ --urls urls.txt --force

Supported files: .pdf .txt .png .jpg .jpeg .webp .mp3 .wav .m4a .ogg .webm
(images/audio call OpenAI vision/Whisper, so they need OPENAI_API_KEY).
The --urls file contains one URL per line; blank lines and lines starting
with '#' are skipped.
The --corpus file is this project's JSON corpus format: either a list of
{title, content, ...} objects or a dict with a "documents" list
(data/rag_dataset.json, data/wiki_documents.json, data/arxiv_papers.json).

Documents whose name already exists in the KB are skipped unless --force,
which removes the old copy first and re-ingests.

The script writes to the same ChromaDB directory the server uses. If the
server is running while you ingest, restart it afterwards so its in-memory
BM25 index and registry pick up the new documents.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services import document_processor as dp
from backend.services.global_kb import get_kb


def _existing_ids(kb, name: str) -> list[str]:
    return [d["id"] for d in kb.get_documents() if d["name"] == name]


def ingest_file(kb, path: Path, force: bool) -> tuple[str, int]:
    """Returns (status, chunk_count) where status is ok/skipped/empty."""
    old = _existing_ids(kb, path.name)
    if old and not force:
        return "skipped", 0
    chunks, summary, doc_type = dp.process_file(path.read_bytes(), path.name)
    if not chunks:
        return "empty", 0
    for doc_id in old:  # --force: replace, don't duplicate
        kb.remove_document(doc_id)
    kb.add_document(chunks, summary, {
        "id": chunks[0]["doc_id"], "name": path.name, "type": doc_type,
    })
    return "ok", len(chunks)


def ingest_url(kb, url: str, force: bool) -> tuple[str, int, str]:
    """Returns (status, chunk_count, title)."""
    chunks, summary = dp.process_url(url)
    if not chunks:
        return "empty", 0, url
    title = chunks[0].get("title", url)
    old = _existing_ids(kb, title)
    if old and not force:
        return "skipped", 0, title
    for doc_id in old:
        kb.remove_document(doc_id)
    kb.add_document(chunks, summary, {
        "id": chunks[0]["doc_id"], "name": title, "type": "url",
    })
    return "ok", len(chunks), title


def load_corpus(path: Path) -> list[dict]:
    """Read the project's JSON corpus format: a list of {title, content, ...}
    objects, or a dict holding that list under "documents".

    Several items can share a title (e.g. the SQUAD contexts are different
    paragraphs of one article), and the KB dedupes by name — so colliding
    titles get the item id appended to stay distinct."""
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = data.get("documents", []) if isinstance(data, dict) else data
    docs = [d for d in docs if isinstance(d, dict) and d.get("title") and d.get("content")]
    seen: dict[str, int] = {}
    for d in docs:
        seen[d["title"].strip()] = seen.get(d["title"].strip(), 0) + 1
    for i, d in enumerate(docs):
        title = d["title"].strip()
        d["_kb_name"] = f"{title} ({d.get('id', i)})" if seen[title] > 1 else title
    return docs


def ingest_corpus_item(kb, item: dict, force: bool) -> tuple[str, int]:
    """Returns (status, chunk_count) where status is ok/skipped/empty."""
    title = item.get("_kb_name") or item["title"].strip()
    old = _existing_ids(kb, title)
    if old and not force:
        return "skipped", 0
    doc_type = item.get("source", "corpus")
    doc_id = dp._make_doc_id(title)
    chunks = dp._sliding_window_chunks(dp._clean_text(item["content"]), doc_id, title, doc_type)
    if not chunks:
        return "empty", 0
    summary = dp._generate_summary(item["content"][:4000], title)
    for oid in old:
        kb.remove_document(oid)
    kb.add_document(chunks, summary, {"id": doc_id, "name": title, "type": doc_type})
    return "ok", len(chunks)


def collect_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(f for f in p.rglob("*")
                           if f.is_file() and f.suffix.lower() in dp.ALLOWED_EXTENSIONS)
            if not found:
                print(f"[WARN] No supported files under {p}")
            files.extend(found)
        elif p.is_file():
            if p.suffix.lower() not in dp.ALLOWED_EXTENSIONS:
                print(f"[WARN] Skipping unsupported file: {p}")
            else:
                files.append(p)
        else:
            print(f"[WARN] Path not found: {p}")
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Bulk-load documents into the global knowledge base.")
    parser.add_argument("paths", nargs="*", help="Files or folders to ingest")
    parser.add_argument("--urls", help="Text file with one URL per line")
    parser.add_argument("--corpus", action="append", default=[],
                        help="JSON corpus file (list of {title, content} or dict with 'documents'); repeatable")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest documents that already exist (replaces the old copy)")
    args = parser.parse_args()

    urls: list[str] = []
    if args.urls:
        urls_file = Path(args.urls)
        if not urls_file.is_file():
            parser.error(f"--urls file not found: {urls_file}")
        urls = [line.strip() for line in urls_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")]

    corpus_docs: list[dict] = []
    for raw in args.corpus:
        corpus_file = Path(raw)
        if not corpus_file.is_file():
            parser.error(f"--corpus file not found: {corpus_file}")
        docs = load_corpus(corpus_file)
        if not docs:
            print(f"[WARN] No usable documents in {corpus_file}")
        corpus_docs.extend(docs)

    files = collect_files(args.paths)
    if not files and not urls and not corpus_docs:
        parser.error("Nothing to ingest — pass files/folders, --urls and/or --corpus.")

    kb = get_kb()
    total = len(files) + len(urls) + len(corpus_docs)
    ok = skipped = failed = 0
    added_chunks = 0

    for i, path in enumerate(files, 1):
        label = f"[{i}/{total}] {path.name}"
        try:
            status, n = ingest_file(kb, path, args.force)
        except Exception as e:
            failed += 1
            print(f"{label} — FAILED: {e}")
            traceback.print_exc(limit=1)
            continue
        if status == "ok":
            ok += 1
            added_chunks += n
            print(f"{label} — {n} chunks")
        elif status == "skipped":
            skipped += 1
            print(f"{label} — already in KB, skipped (use --force to replace)")
        else:
            failed += 1
            print(f"{label} — no text extracted")

    for j, url in enumerate(urls, len(files) + 1):
        label = f"[{j}/{total}] {url}"
        try:
            status, n, title = ingest_url(kb, url, args.force)
        except Exception as e:
            failed += 1
            print(f"{label} — FAILED: {e}")
            continue
        if status == "ok":
            ok += 1
            added_chunks += n
            print(f"{label} — '{title}', {n} chunks")
        elif status == "skipped":
            skipped += 1
            print(f"{label} — '{title}' already in KB, skipped (use --force to replace)")
        else:
            failed += 1
            print(f"{label} — no content extracted")

    for k, item in enumerate(corpus_docs, len(files) + len(urls) + 1):
        label = f"[{k}/{total}] {item['title'][:60]}"
        try:
            status, n = ingest_corpus_item(kb, item, args.force)
        except Exception as e:
            failed += 1
            print(f"{label} — FAILED: {e}")
            traceback.print_exc(limit=1)
            continue
        if status == "ok":
            ok += 1
            added_chunks += n
            print(f"{label} — {n} chunks")
        elif status == "skipped":
            skipped += 1
            print(f"{label} — already in KB, skipped (use --force to replace)")
        else:
            failed += 1
            print(f"{label} — no text extracted")

    print(f"\nDone: {ok} ingested (+{added_chunks} chunks), "
          f"{skipped} skipped, {failed} failed.")
    print(f"Knowledge base now holds {len(kb.get_documents())} documents, "
          f"{kb.chunk_count()} chunks.")
    if ok:
        print("If the API server is running, restart it so the in-memory "
              "index picks up the new documents.")


if __name__ == "__main__":
    main()
