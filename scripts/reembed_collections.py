"""
Re-embed every Chroma collection after changing EMBED_MODEL.

Chroma stores one vector per chunk, so switching embedding models (e.g.
text-embedding-ada-002 → BAAI/bge-m3) invalidates every stored vector and
usually changes the vector DIMENSION — collections must be rebuilt, not
updated in place.

For safety the whole data/chroma_db_users directory is copied to a
timestamped backup first. Then each collection is read out (ids, documents,
metadatas), dropped, recreated with the embedding function currently
configured in backend/services/embeddings.py, and refilled in batches so the
new model embeds everything.

Run this while the API server is STOPPED, then start the server again.

Usage:
  python scripts/reembed_collections.py            # backup + re-embed all
  python scripts/reembed_collections.py --no-backup
"""

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb

from backend.services.embeddings import EMBED_MODEL, get_embedding_function

DB_PATH = ROOT / "data" / "chroma_db_users"
BATCH = 64


def reembed_collection(client, name: str, embed_fn) -> tuple[int, float]:
    """Rebuild one collection with embed_fn; returns (row_count, seconds)."""
    rows = client.get_collection(name).get(include=["documents", "metadatas"])
    n = len(rows["ids"])

    t0 = time.time()
    client.delete_collection(name)
    col = client.get_or_create_collection(
        name, embedding_function=embed_fn, metadata={"hnsw:space": "cosine"})
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        col.add(
            ids=rows["ids"][start:end],
            documents=rows["documents"][start:end],
            metadatas=rows["metadatas"][start:end],
        )
        print(f"    {end}/{n} rows", end="\r", flush=True)

    if col.count() != n:
        raise RuntimeError(f"{name}: expected {n} rows, got {col.count()}")
    return n, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the directory backup (not recommended)")
    args = ap.parse_args()

    if not DB_PATH.exists():
        sys.exit(f"{DB_PATH} not found — nothing to re-embed.")

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = DB_PATH.with_name(f"{DB_PATH.name}_backup_{stamp}")
        print(f"Backing up {DB_PATH.name} → {backup.name} ...")
        shutil.copytree(DB_PATH, backup)

    print(f"Embedding model: {EMBED_MODEL} (loading...)")
    embed_fn = get_embedding_function()

    client = chromadb.PersistentClient(path=str(DB_PATH))
    names = sorted(c.name for c in client.list_collections())
    print(f"{len(names)} collections to rebuild.\n")

    total_rows = 0
    for i, name in enumerate(names, 1):
        n, secs = reembed_collection(client, name, embed_fn)
        total_rows += n
        print(f"[{i}/{len(names)}] {name:<28} {n:4d} rows  {secs:5.1f}s")

    print(f"\nDone — {total_rows} rows re-embedded with {EMBED_MODEL}.")
    print("Restart the API server so it reopens the rebuilt collections.")


if __name__ == "__main__":
    main()
