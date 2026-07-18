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


def _sliding_window_chunks(text: str, doc_id: str, title: str, source_type: str,
                           page_word_starts: Optional[list[int]] = None) -> list[dict]:
    """Split text into overlapping word-window chunks.

    page_word_starts: word index where each page begins (PDFs) — used to tag
    every chunk with the 1-based page its first word falls on, so the UI can
    jump straight to that page in the viewer.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk_text = " ".join(words[i: i + CHUNK_SIZE])
        chunk = {
            "id":    f"{doc_id}_c{idx}",
            "text":  chunk_text,
            "title": title,
            "source": source_type,
            "doc_id": doc_id,
        }
        if page_word_starts:
            page = 1
            for p, start in enumerate(page_word_starts, 1):
                if i >= start:
                    page = p
                else:
                    break
            chunk["page"] = page
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks


def reconstruct_from_chunks(chunk_texts: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Rebuild a document's canonical text from its ordered sliding-window
    chunks, returning (text, [(char_start, char_end) per chunk]).

    The overlap is found by suffix/prefix word matching rather than trusting
    CHUNK_OVERLAP, so documents ingested under older chunking settings still
    reconstruct correctly.
    """
    words: list[str] = []
    ranges: list[tuple[int, int]] = []  # word ranges per chunk
    for text in chunk_texts:
        cw = text.split()
        overlap = 0
        max_k = min(len(words), len(cw))
        for k in range(max_k, 0, -1):
            if words[len(words) - k:] == cw[:k]:
                overlap = k
                break
        start_word = len(words) - overlap
        words.extend(cw[overlap:])
        ranges.append((start_word, start_word + len(cw)))

    # word index → char offset in " ".join(words)
    char_at: list[int] = []
    pos = 0
    for w in words:
        char_at.append(pos)
        pos += len(w) + 1
    text = " ".join(words)
    offsets = [
        (char_at[s] if s < len(char_at) else len(text),
         (char_at[e - 1] + len(words[e - 1])) if 0 < e <= len(words) else len(text))
        for s, e in ranges
    ]
    return text, offsets


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
    # Track how many words each page contributes so chunks can be tagged
    # with the page they start on (viewer jumps straight to it).
    page_word_starts: list[int] = []
    words: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_word_starts.append(len(words))
            t = page.extract_text()
            if t:
                words.extend(_clean_text(t).split())
    full_text = " ".join(words)
    doc_id = _make_doc_id(filename)
    chunks = _sliding_window_chunks(full_text, doc_id, filename, "pdf",
                                    page_word_starts=page_word_starts)
    summary = _generate_summary(full_text[:4000], filename)
    return chunks, summary


# URL ingestion download cap — bounds both memory use and what an SSRF
# attempt could exfiltrate in one response.
MAX_FETCH_BYTES = 5 * 1024 * 1024


def _require_public_url(url: str) -> None:
    """Reject URLs that would make the server request itself or the internal
    network (SSRF): non-http(s) schemes, and hosts resolving to loopback,
    private, link-local or any other non-global address (this also covers
    cloud metadata endpoints like 169.254.169.254)."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Chỉ hỗ trợ URL http/https")
    if not parsed.hostname:
        raise ValueError("URL không hợp lệ")
    try:
        infos = socket.getaddrinfo(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
            proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"Không phân giải được tên miền: {parsed.hostname}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ValueError(
                "URL trỏ tới địa chỉ mạng nội bộ — đã chặn để tránh SSRF")


def _fetch_public_url(url: str) -> tuple[bytes, str]:
    """GET with SSRF validation re-run on every redirect hop (max 5) and a
    MAX_FETCH_BYTES download cap. Returns (body, encoding guess).

    Known limit: validation resolves DNS separately from the request itself,
    so a hostile nameserver flipping records between the two lookups (DNS
    rebinding) is not covered — acceptable for this deployment.
    """
    import requests
    from urllib.parse import urljoin

    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
        "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
    }
    for _ in range(5):
        _require_public_url(url)
        resp = requests.get(url, timeout=20, headers=headers,
                            allow_redirects=False, stream=True)
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            if not loc:
                raise ValueError("Chuyển hướng không có địa chỉ đích")
            url = urljoin(url, loc)
            continue
        resp.raise_for_status()
        raw = bytearray()
        for chunk in resp.iter_content(65536):
            raw.extend(chunk)
            if len(raw) > MAX_FETCH_BYTES:
                raise ValueError("Trang quá lớn (giới hạn 5 MB)")
        # requests defaults to ISO-8859-1 when the header has no charset —
        # sniff the meta tag instead, falling back to UTF-8.
        enc = resp.encoding
        if not enc or enc.lower() == "iso-8859-1":
            m = re.search(rb'charset=["\']?([\w-]{2,20})', bytes(raw[:4096]), re.I)
            enc = m.group(1).decode("ascii", "ignore") if m else "utf-8"
        return bytes(raw), enc
    raise ValueError("Quá nhiều lần chuyển hướng (redirect)")


def process_url(url: str) -> tuple[list[dict], str]:
    raw, enc = _fetch_public_url(url)
    try:
        html = raw.decode(enc, errors="replace")
    except LookupError:
        html = raw.decode("utf-8", errors="replace")

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


def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    """Whisper speech-to-text; mock text when no API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return f"[MOCK] Audio transcription for {filename}"
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, file_bytes),
    )
    return resp.text.strip()


def process_audio(file_bytes: bytes, filename: str) -> tuple[list[dict], str]:
    transcript = _clean_text(transcribe_audio(file_bytes, filename))
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
