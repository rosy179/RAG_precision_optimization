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

AUDIO_EXTENSIONS   = {".mp3", ".wav", ".m4a", ".ogg", ".webm"}
IMAGE_EXTENSIONS   = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_EXTENSIONS = {".pdf", ".txt"} | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS


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
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    }
    resp = requests.get(url, timeout=20, headers=headers)
    resp.raise_for_status()
    html = resp.text

    text, title = "", ""

    # Primary: trafilatura locates the actual article body. Naive
    # first-<main>/<article> selection grabs cookie banners and promo
    # widgets on real-world pages (that's how we indexed 2 KB of cookie
    # consent text instead of a 36 KB article).
    try:
        import trafilatura
        text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
        meta = trafilatura.extract_metadata(html)
        if meta and meta.title:
            title = meta.title
    except Exception:
        pass

    # Fallback: BeautifulSoup with junk stripping, picking the candidate
    # with the MOST text instead of the first one in document order.
    if len(text) < 200:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                         "noscript", "form", "iframe", "svg", "button"]):
            tag.decompose()
        junk = re.compile(r"cookie|consent|banner|popup|modal|menu|sidebar|share|"
                          r"social|subscribe|newsletter|advert|promo|breadcrumb", re.I)
        for attr in ("class", "id"):
            for tag in soup.find_all(attrs={attr: junk}):
                tag.decompose()
        candidates = soup.find_all(["main", "article"]) or ([soup.body] if soup.body else [])
        best = max(candidates, key=lambda t: len(t.get_text()), default=None)
        text = (best or soup).get_text(separator="\n", strip=True)

    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        title = _clean_text(m.group(1)) if m else url

    text = _clean_text(text)
    if len(text) < 100:
        raise ValueError("Trang không có nội dung văn bản đọc được (có thể nội dung được render bằng JavaScript)")

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


def process_txt(file_bytes: bytes, filename: str) -> tuple[list[dict], str]:
    text = _clean_text(file_bytes.decode("utf-8", errors="replace"))
    doc_id = _make_doc_id(filename)
    chunks = _sliding_window_chunks(text, doc_id, filename, "txt")
    summary = _generate_summary(text[:4000], filename)
    return chunks, summary


def process_file(file_bytes: bytes, filename: str,
                 content_type: str = "") -> tuple[list[dict], str, str]:
    """Dispatch to the right handler by extension/MIME.

    Returns (chunks, summary, doc_type). Raises ValueError for
    unsupported formats.
    """
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == ".pdf" or "pdf" in content_type:
        chunks, summary = process_pdf(file_bytes, filename)
        return chunks, summary, "pdf"
    if ext in AUDIO_EXTENSIONS or content_type.startswith("audio/"):
        chunks, summary = process_audio(file_bytes, filename)
        return chunks, summary, "audio"
    if ext in IMAGE_EXTENSIONS or content_type.startswith("image/"):
        chunks, summary = process_image(file_bytes, filename)
        return chunks, summary, "image"
    if ext == ".txt" or content_type.startswith("text/"):
        chunks, summary = process_txt(file_bytes, filename)
        return chunks, summary, "txt"
    raise ValueError(f"Unsupported file type: {ext or content_type or 'unknown'}")


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
