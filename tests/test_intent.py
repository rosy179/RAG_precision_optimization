"""
tests/test_intent.py
--------------------
Unit tests for the query-intent regex heuristics (backend/services/intent.py),
extracted from user_rag.py in refactor E2. These patterns are brittle and
language-mixed (VN + EN + JA), so they get their own focused tests.

Run: pytest tests/test_intent.py -v
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.services.intent import (
    is_summary_question, wants_prev_transform, lang_directive,
)


# ── is_summary_question ──────────────────────────────────────────────────────

class TestIsSummaryQuestion:
    def test_vietnamese_summarize(self):
        assert is_summary_question("Tóm tắt tài liệu này")
        assert is_summary_question("tóm  tắt bài viết")
        assert is_summary_question("Nội dung chính của file là gì")
        assert is_summary_question("Bài này nói về gì")
        assert is_summary_question("Ý chính của đoạn văn")

    def test_english_summarize(self):
        assert is_summary_question("Summarize this document")
        assert is_summary_question("Give me a TL;DR")
        assert is_summary_question("What is this about?")
        assert is_summary_question("main points please")
        assert is_summary_question("Give an overview")

    def test_not_summary(self):
        # Topical questions that must NOT trigger the digest path.
        assert not is_summary_question("Khi nào BERT được phát hành?")
        assert not is_summary_question("What are the two parameters of BM25?")
        assert not is_summary_question("So sánh RAG và fine-tuning")
        assert not is_summary_question("Ai phát minh ra Transformer?")


# ── wants_prev_transform ─────────────────────────────────────────────────────

def _history(*, with_assistant=True):
    h = [{"role": "user", "content": "RAG là gì?"}]
    if with_assistant:
        h.append({"role": "assistant", "content": "RAG kết hợp truy xuất và sinh văn bản."})
    return h


class TestWantsPrevTransform:
    def test_translate_previous_answer(self):
        assert wants_prev_transform("Dịch câu trả lời đó sang tiếng Nhật", _history())
        assert wants_prev_transform("Hãy dịch sang tiếng Anh", _history())
        assert wants_prev_transform("translate the answer to Japanese", _history())
        assert wants_prev_transform("Viết lại bằng tiếng Việt", _history())

    def test_needs_language_phrase(self):
        # An imperative without a target language is NOT a transform request.
        assert not wants_prev_transform("Viết lại cho ngắn hơn", _history())
        assert not wants_prev_transform("Giải thích rõ hơn", _history())

    def test_needs_prior_assistant_answer(self):
        # No previous assistant answer → nothing to transform.
        assert not wants_prev_transform("Dịch sang tiếng Nhật", _history(with_assistant=False))
        assert not wants_prev_transform("Dịch sang tiếng Nhật", [])
        assert not wants_prev_transform("Dịch sang tiếng Nhật", None)

    def test_new_question_not_transform(self):
        # A fresh topical question, even mentioning a language, isn't a re-render
        # of the previous answer unless it commands/points back at it.
        assert not wants_prev_transform("BERT được phát hành năm nào?", _history())


# ── lang_directive ───────────────────────────────────────────────────────────

class TestLangDirective:
    def test_japanese(self):
        assert "日本語" in lang_directive("BERTはいつリリースされましたか？")

    def test_vietnamese(self):
        assert "tiếng Việt" in lang_directive("BERT được phát hành năm nào?")

    def test_english(self):
        assert "English" in lang_directive("When was BERT released?")

    def test_uses_first_nonempty_line(self):
        # Multi-hop scaffolding (Vietnamese) is appended BELOW the question —
        # detection must key off the first line so an English question stays EN.
        q = "When was BERT released?\n(Các dữ kiện trung gian đã xác minh...)"
        assert "English" in lang_directive(q)
