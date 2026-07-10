"""
Multi-format document ingestion:
  PDF     → pdfplumber
  URL     → requests + BeautifulSoup
  Image   → OpenAI GPT-4o Vision
  Audio   → OpenAI Whisper transcription
"""

import os
import re
import sys
import uuid
import base64
from pathlib import Path
from typing import Optional

_SRC = str(Path(__file__).parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dotenv import load_dotenv
load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


def _make_doc_id(name: str) -> str:
    # Must be unique per upload (not just per name) — two users, or the same
    # user re-uploading, can share a filename/URL, and Document.id is a
    # single global primary key.
    return uuid.uuid4().hex[:12]


def _sliding_window_chunks(text: str, doc_id: str, title: str, source_type: str) -> list[dict]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk_text = " ".join(words[i: i + CHUNK_SIZE])
        chunks.append({
            "id":    f"{doc_id}_c{idx}",
            "text":  chunk_text,
            "title": title,
            "source": source_type,
            "doc_id": doc_id,
        })
        i += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks


def _clean_text(text: str) -> str:
    """Remove excessive whitespace/newlines."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# ── Format Handlers ───────────────────────────────────────

def process_pdf(file_bytes: bytes, filename: str) -> tuple[list[dict], str]:
    """
    Returns (chunks, doc_summary).
    """
    import pdfplumber, io
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    full_text = _clean_text("\n\n".join(text_parts))
    doc_id = _make_doc_id(filename)
    chunks = _sliding_window_chunks(full_text, doc_id, filename, "pdf")
    summary = _generate_summary(full_text[:4000], filename)
    return chunks, summary


def process_url(url: str) -> tuple[list[dict], str]:
    import requests
    from bs4 import BeautifulSoup
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-bot/1.0)"}
    resp = requests.get(url, timeout=20, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    # Try to get main content first
    main = soup.find("main") or soup.find("article") or soup.body
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)
    text = _clean_text(text)
    title = soup.title.string.strip() if soup.title else url
    doc_id = _make_doc_id(url)
    chunks = _sliding_window_chunks(text, doc_id, title, "url")
    summary = _generate_summary(text[:4000], title)
    return chunks, summary


def process_image(file_bytes: bytes, filename: str) -> tuple[list[dict], str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        description = f"[MOCK] Image description for {filename}"
    else:
        from openai import OpenAI
        b64 = base64.b64encode(file_bytes).decode()
        ext = Path(filename).suffix.lower().lstrip(".")
        mime = "image/png" if ext == "png" else "image/jpeg"
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": (
                    "You are a technical knowledge base assistant. Describe this image/diagram "
                    "in detail for indexing purposes. Include: all text labels, values, trends, "
                    "components, relationships, and key technical insights. Be thorough."
                )},
            ]}],
            max_tokens=1000,
        )
        description = resp.choices[0].message.content.strip()

    doc_id = _make_doc_id(filename)
    chunks = _sliding_window_chunks(description, doc_id, filename, "image")
    summary = description[:500]
    return chunks, summary


def process_audio(file_bytes: bytes, filename: str) -> tuple[list[dict], str]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        transcript = f"[MOCK] Audio transcription for {filename}"
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, file_bytes),
        )
        transcript = resp.text.strip()

    transcript = _clean_text(transcript)
    doc_id = _make_doc_id(filename)
    chunks = _sliding_window_chunks(transcript, doc_id, filename, "audio")
    summary = _generate_summary(transcript[:4000], filename)
    return chunks, summary


# ── Document Summary ──────────────────────────────────────

def _generate_summary(text: str, title: str) -> str:
    """Generate a short summary for hierarchical indexing."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return text[:300]
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": (
                f"Summarize this document in 3-5 sentences for a technical search index. "
                f"Document title: {title}\n\n{text}"
            )}],
            max_tokens=200,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return text[:300]
