"""
Per-user Modular RAG pipeline (5 layers).

Imports helper functions from src/ directly — does NOT call .build()
on any existing RAG class (those hard-code JSON loading).
"""

import os
import re
import sys
import json
import math
import time
import hashlib
from pathlib import Path
from typing import Optional

# Add src/ to path so we can import helpers
_SRC = str(Path(__file__).parent.parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from dotenv import load_dotenv
load_dotenv()

from backend.services.global_kb import get_kb

DB_PATH     = Path(__file__).parent.parent.parent / "data" / "chroma_db_users"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")
LLM_MODEL   = os.getenv("LLM_MODEL", "gpt-4o-mini")
RERANKER    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# System prompt for the chat UI. Unlike the eval pipeline (cot_rag.py, tuned
# for short Ragas-scored answers), chat answers must be structured Markdown.
CHAT_SYSTEM_PROMPT = """\
Bạn là trợ lý tri thức IT, trả lời câu hỏi dựa trên tài liệu người dùng đã cung cấp.

NGUYÊN TẮC NỘI DUNG:
1. Chỉ dùng thông tin từ phần "Ngữ cảnh" và lịch sử hội thoại — tuyệt đối không bịa thêm ngoài hai nguồn đó.
2. Với yêu cầu trình bày lại nội dung đã trả lời trước đó (dịch sang ngôn ngữ khác, tóm tắt, rút gọn, đổi định dạng), hãy THỰC HIỆN dựa trên câu trả lời trước trong lịch sử hội thoại và ngữ cảnh — không được từ chối vì lý do "chỉ dùng ngữ cảnh".
3. Nếu ngữ cảnh không đủ để trả lời trọn vẹn, trả lời phần có thể và nói rõ phần nào còn thiếu thông tin.
4. Khách quan và chính xác: giữ nguyên số liệu, tên riêng, thuật ngữ trong tài liệu; giải thích ngắn gọn thuật ngữ khó.
4b. Khi NHIỀU đoạn ngữ cảnh chứa dữ kiện có vẻ cùng trả lời câu hỏi (ví dụ nhiều mốc năm, nhiều tên chương trình), hãy đối chiếu từng dữ kiện với ĐÚNG chủ thể và phạm vi của câu hỏi (toàn tổ chức ≠ một đơn vị con; "đầu tiên/bắt đầu" = mốc sớm nhất) rồi chọn dữ kiện khớp trực tiếp nhất; nếu vẫn mơ hồ, nêu cả hai và giải thích khác biệt.
5. Ngôn ngữ trả lời: nếu người dùng yêu cầu ngôn ngữ cụ thể (ví dụ "bằng tiếng Nhật") thì dùng đúng ngôn ngữ đó; nếu không, trả lời bằng ngôn ngữ của câu hỏi.

ĐỊNH DẠNG BẮT BUỘC (Markdown):
- Mở đầu bằng 1-2 câu trả lời thẳng vào ý chính của câu hỏi.
- Triển khai chi tiết bằng gạch đầu dòng "- ", mỗi ý một dòng, in đậm **từ khóa** ở đầu mỗi ý.
- Nếu câu trả lời có nhiều khía cạnh, chia mục bằng tiêu đề "### ".
- Dùng danh sách đánh số (1. 2. 3.) cho các bước hoặc quy trình.
- Dùng bảng Markdown khi cần so sánh từ 2 đối tượng trở lên.
- Không bao giờ dồn toàn bộ câu trả lời vào một dòng hay một đoạn văn duy nhất.

TRÍCH DẪN NGUỒN (bắt buộc):
- Mỗi đoạn "Ngữ cảnh" được đánh số dạng [Nguồn 1: tên], [Nguồn 2: tên]...
- Khi một ý lấy từ nguồn nào, chèn chỉ số [1], [2]... ngay sau ý đó (trước dấu chấm câu hoặc cuối gạch đầu dòng). Ví dụ: "- **RAG** kết hợp truy xuất và sinh văn bản [1]."
- Một ý tổng hợp từ nhiều nguồn thì ghi liền các chỉ số: [1][3].
- Chỉ dùng số nguồn có thật trong Ngữ cảnh; không chèn chỉ số cho ý lấy từ lịch sử hội thoại hay kiến thức chung."""

# Router + decomposer for Multi-Hop RAG: only invoked for heuristically
# complex questions, so simple lookups pay zero extra latency.
MULTIHOP_ROUTE_PROMPT = """\
Bạn là bộ định tuyến truy vấn cho hệ thống RAG. Câu hỏi CẦN multi-hop khi phải \
tìm một thực thể/dữ kiện trung gian trước, rồi mới dùng nó để tìm tiếp câu trả lời \
(ví dụ: "Ai nhận giải X cho công trình đứng sau hệ thống Y?" — phải tìm "công trình \
đứng sau Y" trước).

Nếu câu hỏi trả lời được bằng MỘT lần tìm kiếm (kể cả câu so sánh/giải thích thông \
thường), trả về đúng một từ: SINGLE

Nếu cần multi-hop, trả về 2-3 truy vấn con theo thứ tự, mỗi truy vấn một dòng, \
không đánh số, không giải thích. Truy vấn sau được phép tham chiếu kết quả của \
truy vấn trước. Dùng cùng ngôn ngữ với câu hỏi.

Câu hỏi: {question}"""

MULTIHOP_HOP_PROMPT = """\
Bạn là trợ lý chính xác. Trả lời truy vấn con CHỈ dựa trên ngữ cảnh dưới đây.
Nếu ngữ cảnh không chứa câu trả lời, trả về đúng: NOT FOUND
Trả lời NGẮN GỌN (tối đa 2 câu), giữ nguyên tên riêng/số liệu, cùng ngôn ngữ với truy vấn.

Ngữ cảnh:
{context}

Truy vấn con: {sub_question}

Trả lời ngắn:"""

# Cross-lingual retrieval fix: the corpus is predominantly English, and every
# stage degrades on a non-English query — ada-002 cross-lingual similarity is
# weak, BM25 has zero term overlap, and the ms-marco CrossEncoder only
# understands English. Retrieval therefore also runs an English translation
# of the query; the ANSWER language still follows the user's question.
TRANSLATE_QUERY_PROMPT = """\
Dịch truy vấn tìm kiếm sau sang tiếng Anh để tìm trong kho tài liệu tiếng Anh.
- Giữ nguyên tên riêng, thuật ngữ kỹ thuật, số liệu.
- Nếu truy vấn đã là tiếng Anh, trả lại nguyên văn.
Chỉ trả về bản dịch, không giải thích."""

# Rewrites follow-up questions ("dịch sang tiếng Nhật", "nói rõ hơn ý 2")
# into standalone retrieval queries using the conversation history.
CONDENSE_PROMPT = """\
Bạn nhận lịch sử hội thoại và một câu hỏi mới. Viết lại câu hỏi mới thành MỘT câu truy vấn tìm kiếm độc lập, nêu rõ chủ đề đang được nói tới, để tìm các đoạn tài liệu liên quan.
- Giữ nguyên ngôn ngữ chính của chủ đề trong hội thoại.
- Bỏ các yêu cầu về cách trình bày (ví dụ: "dịch sang tiếng Nhật", "tóm tắt lại", "viết ngắn hơn") — chỉ giữ chủ đề nội dung.
- Nếu câu hỏi mới đã độc lập và rõ ràng, trả lại nguyên văn.
Chỉ trả về câu truy vấn, không giải thích gì thêm."""

# "Summarize this link/document"-type questions carry no topical signal —
# similarity search returns noise for them (often unrelated global-KB chunks).
# They are detected here and served by reading the target document directly.
_SUMMARY_INTENT_RE = re.compile(
    r"tóm\s*tắt|tóm\s*lược|nội\s*dung\s*(chính|của|có|trong|ở)|ý\s*chính|"
    r"điểm\s*chính|nói\s*(về\s*)?(gì|cái\s*gì|điều\s*gì)|viết\s*về\s*gì|"
    r"đề\s*cập\s*(đến|tới)?\s*gì|giới\s*thiệu\s*gì|"
    r"summar(y|ize|ise)|tl;?dr|main\s+(points?|ideas?|content)|"
    r"key\s+(points?|takeaways?)|what\s+is\s+(it|this|the\s+\w+)\s+about|"
    r"\boverview\b|\bgist\b", re.IGNORECASE)

# Mentions of an attached-document object ("đường link", "bài viết", "file"…)
# used to pick a digest target when the message has no attachment of its own.
_DOC_REF_RE = re.compile(
    r"https?://|đường\s*link|\blink\b|\burl\b|bài\s*(viết|báo|blog)|"
    r"trang\s*web|tài\s*liệu|văn\s*bản|tệp|\bfile\b|\bpdf\b|hình\s*ảnh|"
    r"bản\s*ghi\s*âm|article|\bpage\b|document|\bblog\b|\bpost\b|website|"
    r"recording|image|picture", re.IGNORECASE)


def is_summary_question(question: str) -> bool:
    """True for topically-empty summarize/what-is-it-about questions."""
    return bool(_SUMMARY_INTENT_RE.search(question))


# "Dịch bản tóm tắt đó sang tiếng Nhật"-type requests transform the PREVIOUS
# answer — retrieval has nothing to add and its strict grounded-only prompt
# makes the model refuse. They are served by a dedicated transform path.
_LANG_PHRASE_RE = re.compile(
    r"(sang|ra|qua|bằng|thành)\s+tiếng\s+\w+|\btranslate\b|"
    r"\b(in|into|to)\s+(japanese|english|vietnamese|chinese|korean|french|german|spanish)\b|"
    r"日本語|英語|ベトナム語", re.IGNORECASE)
# Starts as an imperative transform command…
_TRANSFORM_CMD_RE = re.compile(
    r"^\s*(hãy|vui\s*lòng|làm\s*ơn|please)?\s*(dịch|chuyển|translate|viết\s*lại|đổi)\b",
    re.IGNORECASE)
# …or explicitly points back at the previous output
_PREV_OUTPUT_RE = re.compile(
    r"(bản|câu|phần|nội\s*dung|đoạn)\s*(tóm\s*tắt|trả\s*lời|tổng\s*hợp|dịch)?\s*"
    r"(đó|này|trên|vừa\s*rồi|ở\s*trên)|câu\s*trả\s*lời|"
    r"the\s+(answer|summary|response)|\babove\b", re.IGNORECASE)


def wants_prev_transform(question: str, history: list[dict] | None) -> bool:
    """True when the question asks to re-render the previous answer in
    another language/format rather than asking something new."""
    prev = next((m for m in reversed(history or [])
                 if m.get("role") == "assistant" and m.get("content")), None)
    if prev is None or not _LANG_PHRASE_RE.search(question):
        return False
    return bool(_TRANSFORM_CMD_RE.match(question) or _PREV_OUTPUT_RE.search(question))


TRANSFORM_SYSTEM_PROMPT = """\
Bạn nhận "Câu trả lời trước" của trợ lý và một yêu cầu trình bày lại nó \
(dịch sang ngôn ngữ khác, rút gọn, đổi định dạng...).
- Thực hiện đúng yêu cầu trên TOÀN BỘ câu trả lời trước, không bỏ sót ý.
- Giữ nguyên cấu trúc Markdown (tiêu đề, gạch đầu dòng, bảng), số liệu và các chỉ số trích dẫn [1], [2]...
- Khi dịch: giữ nguyên tên riêng, tên sản phẩm và thuật ngữ kỹ thuật thông dụng; phần còn lại dịch tự nhiên, trôi chảy.
- Chỉ trả về kết quả, không thêm lời giải thích hay lời dẫn."""


# Shared reranker (loaded once at startup)
_reranker: Optional[CrossEncoder] = None
# Per-user RAG service instances
_instances: dict[str, "UserRAGService"] = {}


def warm_up():
    global _reranker
    print("[RAG] Loading CrossEncoder reranker...")
    try:
        # Local cache first — HF Hub outages must not block startup.
        _reranker = CrossEncoder(RERANKER, local_files_only=True)
        print("[RAG] Reranker ready (local cache).")
    except Exception:
        try:
            _reranker = CrossEncoder(RERANKER)
            print("[RAG] Reranker ready.")
        except Exception as e:
            print(f"[RAG] Reranker load failed: {e}")


def get_service(user_id: str) -> "UserRAGService":
    if user_id not in _instances:
        _instances[user_id] = UserRAGService(user_id)
    return _instances[user_id]


def _tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


def _merge_lexical(session_hits: list[dict], global_hits: list[dict], n: int = 20) -> list[dict]:
    """Merge BM25 results from the session corpus and the global KB.

    Raw BM25 scores are not comparable across corpora (different IDF
    statistics), so normalize each list by its own max before merging.
    RRF downstream only consumes ranks, so this just needs to produce a
    fair combined ordering.
    """
    merged: list[dict] = []
    for hits in (session_hits, global_hits):
        if hits:
            mx = max(h["score"] for h in hits) or 1.0
            merged.extend({**h, "score": round(h["score"] / mx, 4)} for h in hits)
    merged.sort(key=lambda h: h["score"], reverse=True)
    return merged[:n]


def _reattach_chunk_fields(fused: list[dict], *source_lists: list[dict]):
    """RRF (hybrid_rag.reciprocal_rank_fusion) rebuilds chunk dicts and drops
    score/doc_id/scope — re-attach them so the HyDE low-score fallback and
    the per-source scope badge keep working."""
    by_id: dict[str, dict] = {}
    for chunks in source_lists:  # later lists win (semantic cosine score preferred)
        for c in chunks:
            by_id[c["id"]] = c
    for c in fused:
        orig = by_id.get(c["id"], {})
        for key in ("score", "doc_id", "scope", "page"):
            if key in orig:
                c.setdefault(key, orig[key])


# The Vietnamese system prompt biases gpt-4o-mini toward Vietnamese answers
# regardless of the question's language — pin the answer language explicitly.
_JA_RE = re.compile(r'[぀-ヿㇰ-ㇿｦ-ﾟ一-鿿]')
_VI_RE = re.compile(
    r'[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ'
    r'ùúủũụưừứửữựỳýỷỹỵ]', re.IGNORECASE)


def _lang_directive(question: str) -> str:
    """Explicit answer-language instruction based on the question's language.
    Uses the first non-empty line so multi-hop scaffolding (Vietnamese framing
    text appended below the question) can't skew detection."""
    head = next((l for l in question.splitlines() if l.strip()), question)
    if _JA_RE.search(head):
        return ("QUAN TRỌNG: Câu hỏi bằng tiếng Nhật — toàn bộ câu trả lời phải bằng "
                "tiếng Nhật (回答はすべて日本語で書いてください), trừ khi người dùng "
                "yêu cầu rõ một ngôn ngữ khác.")
    if _VI_RE.search(head):
        return ("QUAN TRỌNG: Trả lời hoàn toàn bằng tiếng Việt, trừ khi người dùng "
                "yêu cầu rõ một ngôn ngữ khác.")
    return ("IMPORTANT: The question is in English — write the ENTIRE answer in "
            "English, unless the user explicitly asked for another language.")


def _chunks_to_sources(chunks: list[dict]) -> list[dict]:
    """UI-ready source cards for a list of retrieved chunks."""
    return [{
        "rank":     c.get("rank", i + 1),
        "title":    c.get("title", "Unknown"),
        "snippet":  c.get("text", "")[:300],
        "score":    _display_score(c),
        "scope":    c.get("scope", "session"),  # "global" = shared KB
        "doc_id":   c.get("doc_id", ""),
        "chunk_id": c.get("id", ""),
        **({"page": c["page"]} if c.get("page") else {}),
    } for i, c in enumerate(chunks)]


def _display_score(chunk: dict) -> float:
    """Normalize a chunk's relevance score to 0-1 for the UI badge.

    Cross-encoder rerank scores are raw, unbounded logits — squash them
    through a sigmoid. Semantic scores (1 - distance) can dip below 0
    depending on the distance metric, so clamp instead.
    """
    if "rerank_score" in chunk:
        return round(1.0 / (1.0 + math.exp(-chunk["rerank_score"])), 4)
    return round(max(0.0, min(1.0, chunk.get("score", 0.0))), 4)


def generate_session_title(question: str, answer: str = "") -> str:
    """Short LLM-generated conversation title; falls back to a truncated
    question in mock mode or on any API failure."""
    fallback = question[:60] + ("…" if len(question) > 60 else "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return fallback
    try:
        client = openai.OpenAI(api_key=api_key)
        res = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": (
                "Đặt tiêu đề thật ngắn gọn (tối đa 8 từ, không dấu ngoặc kép, "
                "không dấu chấm cuối) tóm tắt chủ đề đoạn hội thoại sau. "
                "Dùng đúng ngôn ngữ của câu hỏi.\n\n"
                f"Câu hỏi: {question[:300]}\n\nTrả lời: {answer[:300]}"
            )}],
            temperature=0.2,
            max_tokens=30,
        )
        title = (res.choices[0].message.content or "").strip().strip('"').strip()
        return title[:60] if title else fallback
    except Exception:
        return fallback


def openai_error_detail(e: openai.APIError) -> str:
    """Translate an OpenAI SDK error into a user-facing Vietnamese message."""
    if isinstance(e, openai.RateLimitError):
        return "Tài khoản OpenAI đã hết quota. Vui lòng kiểm tra billing tại platform.openai.com."
    if isinstance(e, openai.AuthenticationError):
        return "OPENAI_API_KEY không hợp lệ. Vui lòng kiểm tra lại key trong file .env."
    return f"Lỗi kết nối OpenAI: {e.message if hasattr(e, 'message') else str(e)}"


class UserRAGService:
    def __init__(self, user_id: str):
        self._uid   = user_id
        self._api_key = os.getenv("OPENAI_API_KEY", "")

        DB_PATH.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(DB_PATH))

        if self._api_key and not self._api_key.startswith("sk-your"):
            embed_fn = OpenAIEmbeddingFunction(api_key=self._api_key, model_name=EMBED_MODEL)
        else:
            embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        self._chunk_col   = self._chroma.get_or_create_collection(
            f"u_{user_id[:8]}_chunks",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._summary_col = self._chroma.get_or_create_collection(
            f"u_{user_id[:8]}_summaries",
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self._chunks_store: list[dict] = []  # in-memory for BM25
        self._doc_registry: list[dict] = []  # {id, name, type, session_id, chunk_count}

        # Restore from ChromaDB on init (handles server restarts)
        self._restore_from_chroma()

    # ── Ingestion ──────────────────────────────────────────

    def add_document(self, chunks: list[dict], doc_summary: str, doc_meta: dict) -> int:
        """Upsert chunks + summary into ChromaDB and rebuild BM25."""
        if not chunks:
            return 0

        doc_id = doc_meta["id"]
        session_id = doc_meta.get("session_id", "")

        # Upsert chunks into chunk collection (page only exists for PDFs —
        # Chroma metadata rejects None values, so include it conditionally)
        self._chunk_col.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{
                "doc_id": doc_id,
                "session_id": session_id,
                "title":  doc_meta["name"],
                "source": doc_meta.get("type", "unknown"),
                "chunk_index": i,
                **({"page": c["page"]} if "page" in c else {}),
            } for i, c in enumerate(chunks)],
        )

        # Upsert doc summary into summary collection
        self._summary_col.upsert(
            ids=[doc_id],
            documents=[doc_summary],
            metadatas=[{"name": doc_meta["name"], "type": doc_meta.get("type", "unknown"),
                        "session_id": session_id}],
        )

        # Update in-memory BM25 store
        for c in chunks:
            self._chunks_store.append({"id": c["id"], "text": c["text"],
                                       "title": doc_meta["name"], "doc_id": doc_id,
                                       "session_id": session_id,
                                       **({"page": c["page"]} if "page" in c else {})})

        # Update registry
        self._doc_registry.append({
            "id": doc_id,
            "name": doc_meta["name"],
            "type": doc_meta.get("type", "unknown"),
            "session_id": session_id,
            "chunk_count": len(chunks),
        })

        return len(chunks)

    def remove_document(self, doc_id: str):
        """Remove all chunks for a document."""
        existing = self._chunk_col.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self._chunk_col.delete(ids=existing["ids"])
        try:
            self._summary_col.delete(ids=[doc_id])
        except Exception:
            pass
        self._chunks_store = [c for c in self._chunks_store if c.get("doc_id") != doc_id]
        self._doc_registry = [d for d in self._doc_registry if d["id"] != doc_id]

    def get_documents(self) -> list[dict]:
        return self._doc_registry

    def chunk_count(self) -> int:
        return self._chunk_col.count()

    def get_document_content(self, doc_id: str) -> dict | None:
        """Reconstruct a document's full text from its stored chunks for the
        viewer, with per-chunk char offsets so a cited chunk can be
        highlighted. Returns None if the doc has no chunks."""
        res = self._chunk_col.get(where={"doc_id": doc_id},
                                  include=["documents", "metadatas"])
        if not res["ids"]:
            return None
        rows = sorted(
            zip(res["ids"], res["documents"], res["metadatas"]),
            key=lambda r: r[2].get("chunk_index", 0),
        )
        from backend.services.document_processor import reconstruct_from_chunks
        text, offsets = reconstruct_from_chunks([r[1] for r in rows])
        return {
            "text": text,
            "chunks": [{
                "id": rid, "start": start, "end": end,
                **({"page": meta["page"]} if "page" in meta else {}),
            } for (rid, _, meta), (start, end) in zip(rows, offsets)],
        }

    # ── Query ──────────────────────────────────────────────

    def retrieve(self, question: str, history: list[dict] | None = None,
                 session_id: str | None = None,
                 include_doc_ids: list[str] | None = None,
                 use_global_kb: bool = True,
                 attached_doc_ids: list[str] | None = None,
                 digest_ok: bool = True) -> dict:
        """Run the full retrieval stack (Layers 1-4) without generation.

        Returns top_chunks + UI-ready sources so callers can either generate
        in one shot (query) or stream tokens (generate_stream).

        include_doc_ids: session doc ids the user checked in the source
        picker (None = all session docs). use_global_kb=False excludes the
        shared KB for this question.

        attached_doc_ids: docs attached to THIS message — they are the
        question's most likely subject, so they lead the doc shortlist and
        anchor the summarize digest path. digest_ok=False disables that path
        (multi-hop sub-queries must always run real retrieval).
        """
        # Retrieval draws from two pools: chunks uploaded in THIS session
        # (per-conversation scope, as before) plus the shared global
        # knowledge base available to every user and session.
        kb = get_kb()
        session_chunks = [c for c in self._chunks_store
                          if not session_id or c.get("session_id") == session_id]
        if include_doc_ids is not None:
            allowed = set(include_doc_ids)
            session_chunks = [c for c in session_chunks if c.get("doc_id") in allowed]
        kb_enabled = use_global_kb and not kb.is_empty()
        if not session_chunks and not kb_enabled:
            filtered_out = include_doc_ids is not None or not use_global_kb
            return {
                "empty": True,
                "notice": (
                    "Tất cả nguồn tham khảo đang bị tắt. Hãy bật lại ít nhất một tài liệu "
                    "hoặc kho kiến thức chung trong bộ chọn nguồn rồi hỏi lại."
                    if filtered_out else
                    "Chưa có tài liệu nào — cả trong cuộc trò chuyện này lẫn kho "
                    "kiến thức chung. Hãy đính kèm PDF, dán URL hoặc tải ảnh/ghi âm "
                    "lên trước khi hỏi."
                ),
                "top_chunks": [],
                "sources": [],
                "complexity": "simple",
            }

        t0 = time.time()
        history = history or []
        session_filter = {"session_id": session_id} if session_id else None

        allowed_ids = set(include_doc_ids) if include_doc_ids is not None else None
        session_docs = [d for d in self._doc_registry
                        if (not session_id or d.get("session_id") == session_id)
                        and (allowed_ids is None or d["id"] in allowed_ids)]

        # Summarize-type questions targeting an attached/referenced document
        # skip search entirely and read that document's chunks in order.
        if digest_ok:
            digest = self._digest_retrieve(question, session_docs, attached_doc_ids)
            if digest is not None:
                digest["retrieval_ms"] = int((time.time() - t0) * 1000)
                return digest

        # Follow-ups ("dịch sang tiếng Nhật", "nói rõ hơn") carry no retrieval
        # signal on their own — condense with history into a standalone query.
        # Generation still sees the original question + history.
        search_query = self._condense_question(question, history)

        # Non-English queries also search via an English translation (the
        # ASCII check is a cheap language proxy: VI diacritics / CJK fail it).
        queries = [search_query]
        if not search_query.isascii():
            translated = self._translate_to_english(search_query)
            if translated and translated.lower() != search_query.lower():
                queries.append(translated)
        # English-side query: the CrossEncoder, HyDE and the keyword-based
        # complexity heuristic all only work well in English.
        q_en = queries[-1]

        # Layer 1 — Adaptive Router
        from adaptive_rag import classify_heuristic
        complexity = classify_heuristic(q_en)

        # Layer 2 — Hierarchical: find relevant docs by summary, separately
        # per pool (session docs and global KB docs live in different
        # collections, so each needs its own doc-id shortlist).
        top_k_docs = min(5, len(session_docs))
        relevant_doc_ids: list[str] = []
        if top_k_docs > 0:
            kwargs = {"query_texts": queries, "n_results": top_k_docs}
            if session_filter:
                kwargs["where"] = session_filter
            summary_res = self._summary_col.query(**kwargs)
            # Union of both language variants' shortlists, order-preserving
            seen_docs: set = set()
            for id_list in (summary_res["ids"] or []):
                for did in id_list:
                    if did not in seen_docs:
                        seen_docs.add(did)
                        relevant_doc_ids.append(did)
        if allowed_ids is not None:
            # The summary shortlist isn't checkbox-aware — constrain it, and
            # never fall through to an unfiltered chunk search.
            relevant_doc_ids = [d for d in relevant_doc_ids if d in allowed_ids] \
                or [d["id"] for d in session_docs]
        if attached_doc_ids:
            # Docs attached to this message are its most likely subject —
            # guarantee them a spot in the chunk-search shortlist.
            valid = {d["id"] for d in session_docs}
            boost = [i for i in attached_doc_ids if i in valid]
            relevant_doc_ids = list(dict.fromkeys(boost + relevant_doc_ids))
        kb_doc_ids = kb.search_summaries(queries, top_k=5) if kb_enabled else []

        # Layer 3 — Advanced: Semantic + BM25 + RRF, over both pools
        semantic_chunks = []
        if session_chunks:
            filters = []
            if session_filter:
                filters.append(session_filter)
            if relevant_doc_ids:
                filters.append({"doc_id": {"$in": relevant_doc_ids}})
            where_filter = filters[0] if len(filters) == 1 else ({"$and": filters} if filters else None)

            kwargs = {"query_texts": queries, "n_results": min(20, len(session_chunks))}
            if where_filter:
                kwargs["where"] = where_filter
            semantic_res = self._chunk_col.query(**kwargs)

            # Both language variants search in one Chroma call; a chunk hit
            # by both keeps its best score.
            best: dict[str, dict] = {}
            for qi in range(len(semantic_res["ids"] or [])):
                for i, chunk_id in enumerate(semantic_res["ids"][qi]):
                    meta = semantic_res["metadatas"][qi][i] if semantic_res["metadatas"] else {}
                    dist = semantic_res["distances"][qi][i] if semantic_res["distances"] else 1.0
                    score = round(1.0 - float(dist), 4)
                    if chunk_id in best and best[chunk_id]["score"] >= score:
                        continue
                    best[chunk_id] = {
                        "id":     chunk_id,
                        "text":   semantic_res["documents"][qi][i],
                        "title":  meta.get("title", ""),
                        "source": meta.get("source", ""),
                        "doc_id": meta.get("doc_id", ""),
                        "scope":  "session",
                        "score":  score,
                        **({"page": meta["page"]} if "page" in meta else {}),
                    }
            semantic_chunks = list(best.values())

        # Both collections use the same embedding model and cosine space,
        # so their (1 - distance) scores are directly comparable.
        kb_semantic = kb.search_semantic(queries, n_results=20,
                                         doc_ids=kb_doc_ids or None) if kb_enabled else []
        semantic_chunks = sorted(semantic_chunks + kb_semantic,
                                 key=lambda c: c["score"], reverse=True)[:20]

        bm25_chunks = _merge_lexical(
            self._bm25_search(queries, n=20, session_id=session_id,
                              allowed_doc_ids=allowed_ids),
            kb.search_bm25(queries, n=20) if kb_enabled else [],
        )

        from hybrid_rag import reciprocal_rank_fusion
        # simple gets 4 (not 3): with competing near-answer chunks in the
        # corpus, the correct one must survive into the generation context.
        top_k = 5 if complexity == "complex" else 4
        fused = reciprocal_rank_fusion(semantic_chunks, bm25_chunks, top_k=top_k * 4)
        _reattach_chunk_fields(fused, bm25_chunks, semantic_chunks)

        # Rerank ALWAYS (against the English-side query — the ms-marco
        # CrossEncoder scores non-English pairs as noise). Skipping rerank
        # for "simple" questions saved ~0.5s but let near-duplicate facts
        # outrank the exact answer chunk — accuracy wins here.
        # The CE ordering is BLENDED with the fusion ordering (RRF over both
        # rank lists): when the corpus holds two competing near-answers, one
        # CE miss must not evict the chunk BM25+semantic both agreed on.
        if _reranker and len(fused) > 1:
            pairs = [(q_en, c["text"]) for c in fused]
            scores = _reranker.predict(pairs)
            ce_rank = {i: r for r, i in enumerate(
                sorted(range(len(fused)), key=lambda i: -float(scores[i])))}
            for i, chunk in enumerate(fused):
                chunk["rerank_score"] = float(scores[i])
                chunk["blend"] = 1.0 / (10 + i) + 1.0 / (10 + ce_rank[i])
            fused.sort(key=lambda c: c["blend"], reverse=True)

        top_chunks = fused[:top_k]

        # Layer 4 — Iterative fallback: if best score < 0.5, HyDE re-query
        best_score = top_chunks[0].get("score", 1.0) if top_chunks else 0.0
        if best_score < 0.5:
            from query_expansion import generate_hypothetical_doc
            hyde = generate_hypothetical_doc(q_en)
            if hyde and hyde != q_en:
                hyde_chunks = []
                if session_chunks:
                    hyde_filters = []
                    if session_filter:
                        hyde_filters.append(session_filter)
                    if allowed_ids is not None:
                        hyde_filters.append({"doc_id": {"$in": list(allowed_ids)}})
                    hyde_kwargs = {"query_texts": [hyde],
                                   "n_results": min(10, len(session_chunks))}
                    if hyde_filters:
                        hyde_kwargs["where"] = (hyde_filters[0] if len(hyde_filters) == 1
                                                else {"$and": hyde_filters})
                    hyde_res = self._chunk_col.query(**hyde_kwargs)
                    if hyde_res["ids"] and hyde_res["ids"][0]:
                        for i, cid in enumerate(hyde_res["ids"][0]):
                            meta = hyde_res["metadatas"][0][i] if hyde_res["metadatas"] else {}
                            dist = hyde_res["distances"][0][i] if hyde_res["distances"] else 1.0
                            hyde_chunks.append({
                                "id": cid, "text": hyde_res["documents"][0][i],
                                "title": meta.get("title", ""), "source": meta.get("source", ""),
                                "doc_id": meta.get("doc_id", ""),
                                "scope": "session",
                                "score": round(1.0 - float(dist), 4),
                                **({"page": meta["page"]} if "page" in meta else {}),
                            })
                if kb_enabled:
                    hyde_chunks += kb.search_semantic(hyde, n_results=10)
                fused2 = reciprocal_rank_fusion(top_chunks, hyde_chunks, top_k=top_k)
                _reattach_chunk_fields(fused2, top_chunks, hyde_chunks)
                top_chunks = fused2[:top_k]

        # Add rank field for CoT context builder
        for i, c in enumerate(top_chunks, 1):
            c["rank"] = i

        return {
            "empty":        False,
            "notice":       None,
            "top_chunks":   top_chunks,
            "sources":      _chunks_to_sources(top_chunks),
            "complexity":   complexity,
            "retrieval_ms": int((time.time() - t0) * 1000),
        }

    def query(self, question: str, history: list[dict] | None = None,
              session_id: str | None = None,
              include_doc_ids: list[str] | None = None,
              use_global_kb: bool = True,
              attached_doc_ids: list[str] | None = None) -> dict:
        t0 = time.time()
        history = history or []

        if wants_prev_transform(question, history):
            prev = next(str(m["content"]) for m in reversed(history)
                        if m.get("role") == "assistant" and m.get("content"))
            answer = "".join(self.transform_stream(question, prev))
            return {"answer": answer, "sources": [], "reasoning": "",
                    "latency_ms": int((time.time() - t0) * 1000),
                    "from_cache": False, "complexity": "transform"}

        ret = self.retrieve(question, history, session_id,
                            include_doc_ids=include_doc_ids,
                            use_global_kb=use_global_kb,
                            attached_doc_ids=attached_doc_ids)
        if ret["empty"]:
            return {"answer": ret["notice"], "sources": [], "reasoning": "",
                    "latency_ms": 0, "from_cache": False, "complexity": "simple"}

        # Layer 5 — Structured chat generation (grounded in top_chunks + history)
        answer = self._generate_chat_answer(question, ret["top_chunks"], history)

        return {
            "answer":     answer,
            "reasoning":  "",
            "sources":    ret["sources"],
            "latency_ms": int((time.time() - t0) * 1000),
            "from_cache": False,
            "complexity": ret["complexity"],
        }

    def generate_stream(self, question: str, chunks: list[dict],
                        history: list[dict] | None = None):
        """Yield answer text deltas (Layer 5, streaming variant).

        openai.APIError propagates to the caller so the endpoint can emit
        an SSE error event.
        """
        if not self._api_key or self._api_key.startswith("sk-your"):
            preview = chunks[0]["text"][:200] if chunks else "No context"
            # Chunked mock so the UI streaming path is exercised without a key
            mock = f"[MOCK ANSWER] Based on context: '{preview}...'"
            for i in range(0, len(mock), 24):
                yield mock[i:i + 24]
            return

        client = openai.OpenAI(api_key=self._api_key)
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=self._chat_messages(question, chunks, history or []),
            temperature=0.2,
            max_tokens=1500,
            stream=True,
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content

    def transform_stream(self, question: str, prev_answer: str):
        """Yield the previous answer re-rendered per the user's instruction
        (translate, shorten, reformat) — no retrieval involved."""
        if not self._api_key or self._api_key.startswith("sk-your"):
            mock = f"[MOCK TRANSFORM] {question}: {prev_answer[:200]}"
            for i in range(0, len(mock), 24):
                yield mock[i:i + 24]
            return

        client = openai.OpenAI(api_key=self._api_key)
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": TRANSFORM_SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"Câu trả lời trước:\n{prev_answer}\n\nYêu cầu: {question}"},
            ],
            temperature=0.2,
            max_tokens=2000,
            stream=True,
        )
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content

    # ── Multi-Hop (Layer 6) ────────────────────────────────

    def has_any_source(self, session_id: str | None = None,
                       include_doc_ids: list[str] | None = None,
                       use_global_kb: bool = True) -> bool:
        """Cheap check whether retrieval has anything to search."""
        chunks = [c for c in self._chunks_store
                  if not session_id or c.get("session_id") == session_id]
        if include_doc_ids is not None:
            allowed = set(include_doc_ids)
            chunks = [c for c in chunks if c.get("doc_id") in allowed]
        return bool(chunks) or (use_global_kb and not get_kb().is_empty())

    def route_multihop(self, question: str) -> list[str]:
        """Decide whether the question needs chained retrieval.

        Returns ordered sub-queries (2-3) for multi-hop, or [] for the
        normal single-retrieval path. Callers should only invoke this for
        heuristically complex questions to keep simple lookups fast.
        """
        if not self._api_key or self._api_key.startswith("sk-your"):
            return []
        try:
            client = openai.OpenAI(api_key=self._api_key)
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user",
                           "content": MULTIHOP_ROUTE_PROMPT.format(question=question)}],
                temperature=0.0,
                max_tokens=200,
            )
            raw = (res.choices[0].message.content or "").strip()
        except Exception:
            return []
        if not raw or raw.upper().startswith("SINGLE"):
            return []
        subs = [line.strip().lstrip("0123456789.-) ").strip()
                for line in raw.splitlines() if line.strip()]
        subs = [s for s in subs if len(s) > 5]
        # A single sub-question is just a rephrase — not worth the extra hops
        return subs[:3] if len(subs) >= 2 else []

    def _hop_answer(self, sub_question: str, chunks: list[dict]) -> str:
        """Short bridging answer for one hop (feeds the next hop's query)."""
        if not chunks:
            return "NOT FOUND"
        if not self._api_key or self._api_key.startswith("sk-your"):
            return chunks[0]["text"][:200]
        context = "\n\n---\n\n".join(
            f"[Nguồn {c.get('rank', i + 1)}: {c.get('title', '')}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )
        try:
            client = openai.OpenAI(api_key=self._api_key)
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": MULTIHOP_HOP_PROMPT.format(
                    context=context, sub_question=sub_question)}],
                temperature=0.0,
                max_tokens=150,
            )
            return (res.choices[0].message.content or "").strip() or "NOT FOUND"
        except Exception:
            return "NOT FOUND"

    def multihop_events(self, question: str, sub_questions: list[str],
                        history: list[dict] | None = None,
                        session_id: str | None = None,
                        include_doc_ids: list[str] | None = None,
                        use_global_kb: bool = True,
                        attached_doc_ids: list[str] | None = None):
        """Chained-retrieval pipeline as an event generator.

        Yields ("step", dict) for each reasoning step, then
        ("sources", {"sources": [...]}) with the deduplicated union of all
        hops' chunks, then ("delta", str) tokens of the final answer.
        """
        history = history or []
        bridge: list[dict] = []
        all_chunks: list[dict] = []

        for i, sub_q in enumerate(sub_questions, 1):
            # Facts found so far steer the next retrieval — this is the
            # whole point of multi-hop: the bridging entity was never in
            # the original question.
            established = " | ".join(
                f"[Đã biết: {b['answer']}]" for b in bridge
                if b["answer"] and "NOT FOUND" not in b["answer"]
            )
            enriched = f"{sub_q} ({established})" if established else sub_q
            yield ("step", {"stage": "hop_start", "hop": i,
                            "total": len(sub_questions), "query": enriched})

            ret = self.retrieve(enriched, [], session_id,
                                include_doc_ids=include_doc_ids,
                                use_global_kb=use_global_kb,
                                attached_doc_ids=attached_doc_ids,
                                digest_ok=False)
            chunks = ret["top_chunks"] if not ret["empty"] else []
            answer = self._hop_answer(sub_q, chunks)
            all_chunks.extend(chunks)
            bridge.append({"hop": i, "sub_question": sub_q, "answer": answer})
            yield ("step", {"stage": "hop_done", "hop": i, "answer": answer,
                            "titles": [c.get("title", "") for c in chunks[:3]]})

        # Union of every hop's evidence, deduplicated, capped for context
        seen: set = set()
        unique: list[dict] = []
        for c in all_chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                unique.append(c)
        unique = unique[:8]
        for idx, c in enumerate(unique, 1):
            c["rank"] = idx
        yield ("sources", {"sources": _chunks_to_sources(unique)})

        facts = "\n".join(f"- Bước {b['hop']}: {b['sub_question']} → {b['answer']}"
                          for b in bridge)
        synth_question = (
            f"{question}\n\n"
            f"(Các dữ kiện trung gian đã xác minh qua tìm kiếm nhiều bước — "
            f"dùng chúng để trả lời, vẫn trích dẫn [n] theo Nguồn:\n{facts})"
        )
        for delta in self.generate_stream(synth_question, unique, history):
            yield ("delta", delta)

    # ── Internals ──────────────────────────────────────────

    def _digest_retrieve(self, question: str, session_docs: list[dict],
                         attached_doc_ids: list[str] | None) -> dict | None:
        """Direct-read path for "tóm tắt link này"-type questions.

        Such questions are topically empty, so similarity search returns
        noise. When the target document is identifiable — attached to this
        message, named in the question, or referenced deictically ("đường
        link đó") — answer from its chunks in document order instead.
        Returns None when this path does not apply.
        """
        if not session_docs or not _SUMMARY_INTENT_RE.search(question):
            return None

        targets: list[str] = []
        if attached_doc_ids:
            valid = {d["id"] for d in session_docs}
            targets = [i for i in attached_doc_ids if i in valid]
        if not targets:
            q = question.lower()
            # Doc named in the question ("tóm tắt file report.pdf")
            targets = [d["id"] for d in session_docs
                       if len(d["name"]) > 4
                       and d["name"].lower().rsplit(".", 1)[0] in q][:2]
        if not targets and _DOC_REF_RE.search(question):
            # "đường link đó" / "bài viết này" — most recently added doc
            targets = [session_docs[-1]["id"]]
        if not targets:
            return None

        chunks = self._ordered_doc_chunks(targets)
        if not chunks:
            return None
        for i, c in enumerate(chunks, 1):
            c["rank"] = i
        return {
            "empty":      False,
            "notice":     None,
            "top_chunks": chunks,
            "sources":    _chunks_to_sources(chunks),
            "complexity": "simple",
        }

    def _ordered_doc_chunks(self, doc_ids: list[str], cap: int = 8) -> list[dict]:
        """A document's chunks in reading order, evenly sampled down to
        `cap` (first and last always kept) so long docs still fit the
        generation context."""
        def chunk_index(c: dict) -> int:
            m = re.search(r"_c(\d+)$", c["id"])
            return int(m.group(1)) if m else 0

        out: list[dict] = []
        for did in doc_ids:
            rows = sorted((c for c in self._chunks_store if c.get("doc_id") == did),
                          key=chunk_index)
            out.extend(rows)
        if len(out) > cap:
            picks = sorted({round(j * (len(out) - 1) / (cap - 1)) for j in range(cap)})
            out = [out[i] for i in picks]
        # score 1.0: chunks are read from the requested doc, not ranked guesses
        return [{**c, "scope": "session", "score": 1.0} for c in out]

    def _translate_to_english(self, query: str) -> str:
        """English version of a retrieval query (see TRANSLATE_QUERY_PROMPT).
        Returns "" on any failure so callers fall back to the original."""
        if not self._api_key or self._api_key.startswith("sk-your"):
            return ""
        try:
            client = openai.OpenAI(api_key=self._api_key)
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": TRANSLATE_QUERY_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            return (res.choices[0].message.content or "").strip().strip('"')
        except Exception:
            return ""

    def _condense_question(self, question: str, history: list[dict]) -> str:
        """Rewrite a follow-up question into a standalone retrieval query.

        First messages and mock mode skip the extra LLM call. Any failure
        falls back to the raw question so retrieval still runs.
        """
        if not history or not self._api_key or self._api_key.startswith("sk-your"):
            return question
        convo = "\n".join(
            f"{'Người dùng' if m.get('role') == 'user' else 'Trợ lý'}: {str(m['content'])[:500]}"
            for m in history[-6:]
            if m.get("role") in ("user", "assistant") and m.get("content")
        )
        try:
            client = openai.OpenAI(api_key=self._api_key)
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": CONDENSE_PROMPT},
                    {"role": "user",
                     "content": f"Lịch sử hội thoại:\n{convo}\n\nCâu hỏi mới: {question}\n\nCâu truy vấn độc lập:"},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            rewritten = (res.choices[0].message.content or "").strip().strip('"')
            return rewritten or question
        except Exception:
            return question

    def _chat_messages(self, question: str, chunks: list[dict],
                       history: list[dict]) -> list[dict]:
        """Build the LLM message list for chat generation.

        Includes recent conversation history so follow-up questions
        ("nó là gì?", "còn cách nào khác?") resolve correctly.
        """
        context_str = "\n\n---\n\n".join(
            f"[Nguồn {c.get('rank', i + 1)}: {c.get('title', 'Unknown')}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )

        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        for m in history[-6:]:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": str(m["content"])})
        messages.append({
            "role": "user",
            "content": (f"Ngữ cảnh:\n{context_str}\n\nCâu hỏi: {question}\n\n"
                        f"{_lang_directive(question)}"),
        })
        return messages

    def _generate_chat_answer(self, question: str, chunks: list[dict],
                              history: list[dict]) -> str:
        """Generate a structured Markdown answer for the chat UI in one shot."""
        if not self._api_key or self._api_key.startswith("sk-your"):
            preview = chunks[0]["text"][:200] if chunks else "No context"
            return f"[MOCK ANSWER] Based on context: '{preview}...'"

        client = openai.OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=self._chat_messages(question, chunks, history),
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()

    def _bm25_search(self, queries: str | list[str], n: int = 20,
                     session_id: str | None = None,
                     allowed_doc_ids: set[str] | None = None) -> list[dict]:
        # Score only within the session's chunks so IDF weights are not
        # skewed by documents from other conversations.
        if isinstance(queries, str):
            queries = [queries]
        pool = [c for c in self._chunks_store
                if (not session_id or c.get("session_id") == session_id)
                and (allowed_doc_ids is None or c.get("doc_id") in allowed_doc_ids)]
        if not pool:
            return []
        bm25 = BM25Okapi([_tokenize(c["text"]) for c in pool])
        # Multi-language query variants: keep each chunk's best score
        # (an EN translation is what actually matches an EN corpus).
        merged = [0.0] * len(pool)
        for q in queries:
            for idx, score in enumerate(bm25.get_scores(_tokenize(q))):
                if score > merged[idx]:
                    merged[idx] = score
        ranked = sorted(enumerate(merged), key=lambda x: x[1], reverse=True)[:n]
        return [{**pool[idx], "scope": "session", "score": round(float(score), 4)}
                for idx, score in ranked if score > 0]

    def _rebuild_bm25(self):
        if not self._chunks_store:
            self._bm25 = None
            return
        tokenized = [_tokenize(c["text"]) for c in self._chunks_store]
        self._bm25 = BM25Okapi(tokenized)

    def _restore_from_chroma(self):
        """Rebuild in-memory state from persisted ChromaDB on startup."""
        try:
            all_chunks = self._chunk_col.get(include=["documents", "metadatas"])
            if not all_chunks["ids"]:
                return
            chunk_by_doc: dict[str, dict] = {}
            for i, cid in enumerate(all_chunks["ids"]):
                meta = all_chunks["metadatas"][i] if all_chunks["metadatas"] else {}
                text = all_chunks["documents"][i] if all_chunks["documents"] else ""
                doc_id = meta.get("doc_id", "unknown")
                session_id = meta.get("session_id", "")
                self._chunks_store.append({
                    "id": cid, "text": text,
                    "title": meta.get("title", ""), "doc_id": doc_id,
                    "session_id": session_id,
                    **({"page": meta["page"]} if "page" in meta else {}),
                })
                if doc_id not in chunk_by_doc:
                    chunk_by_doc[doc_id] = {"name": meta.get("title", ""), "type": meta.get("source", ""),
                                            "session_id": session_id, "count": 0}
                chunk_by_doc[doc_id]["count"] += 1

            for doc_id, info in chunk_by_doc.items():
                self._doc_registry.append({
                    "id": doc_id, "name": info["name"],
                    "type": info["type"], "session_id": info["session_id"],
                    "chunk_count": info["count"],
                })
        except Exception as e:
            print(f"[RAG] Restore warning: {e}")
