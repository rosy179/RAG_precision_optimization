"""
Query-intent heuristics for the RAG pipeline.

Regex-based detectors, split out of user_rag.py so they can be unit-tested in
isolation (tests/test_intent.py) — the patterns are brittle and language-mixed
(VN + EN + JA), exactly the kind of code that needs its own tests.
"""

import re

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


# The Vietnamese system prompt biases gpt-4o-mini toward Vietnamese answers
# regardless of the question's language — pin the answer language explicitly.
_JA_RE = re.compile(r'[぀-ヿㇰ-ㇿｦ-ﾟ一-鿿]')
_VI_RE = re.compile(
    r'[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ'
    r'ùúủũụưừứửữựỳýỷỹỵ]', re.IGNORECASE)


def lang_directive(question: str) -> str:
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
