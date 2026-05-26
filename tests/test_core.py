"""
tests/test_core.py
------------------
Unit tests for pure functions that don't require API keys or network access.
Run with: pytest tests/test_core.py -v
"""

import sys
from pathlib import Path

# Add project root and src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest


# ── chunk_documents (baseline_rag) ───────────────────────────────────────────

from baseline_rag import chunk_documents


def _make_doc(content: str, doc_id: str = "doc1") -> dict:
    return {"id": doc_id, "content": content, "title": "Test", "source": "test"}


def test_chunk_empty_doc():
    chunks = chunk_documents([_make_doc("")])
    # A single empty-content chunk is produced (0 words → single empty chunk)
    assert isinstance(chunks, list)


def test_chunk_short_doc_single_chunk():
    doc = _make_doc("hello world foo bar")
    chunks = chunk_documents([doc], chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert "hello" in chunks[0]["text"]


def test_chunk_long_doc_multiple_chunks():
    words = " ".join(["word"] * 600)
    doc = _make_doc(words)
    chunks = chunk_documents([doc], chunk_size=100, overlap=10)
    assert len(chunks) > 1


def test_chunk_overlap_produces_shared_words():
    words = list(range(200))
    content = " ".join(str(w) for w in words)
    doc = _make_doc(content)
    chunks = chunk_documents([doc], chunk_size=100, overlap=20)
    # Second chunk should start 80 words after first chunk (100 - 20 = 80)
    first_end_words = set(chunks[0]["text"].split()[-20:])
    second_start_words = set(chunks[1]["text"].split()[:20])
    assert len(first_end_words & second_start_words) > 0


def test_chunk_preserves_metadata():
    doc = _make_doc("some text here", doc_id="my_doc")
    chunks = chunk_documents([doc])
    assert chunks[0]["doc_id"] == "my_doc"
    assert chunks[0]["title"] == "Test"
    assert chunks[0]["source"] == "test"


# ── reciprocal_rank_fusion (hybrid_rag) ──────────────────────────────────────

from hybrid_rag import reciprocal_rank_fusion


def _chunk(chunk_id: str) -> dict:
    return {"id": chunk_id, "text": f"text of {chunk_id}", "title": "", "source": ""}


def test_rrf_basic_merge():
    semantic = [_chunk("a"), _chunk("b"), _chunk("c")]
    lexical  = [_chunk("b"), _chunk("a"), _chunk("d")]
    result = reciprocal_rank_fusion(semantic, lexical, k=60, top_k=3)
    ids = [r["id"] for r in result]
    # "b" and "a" appear in both lists → should be ranked higher than "c" or "d"
    assert ids[0] in ("a", "b")
    assert ids[1] in ("a", "b")


def test_rrf_top_k_respected():
    semantic = [_chunk(f"s{i}") for i in range(10)]
    lexical  = [_chunk(f"l{i}") for i in range(10)]
    result = reciprocal_rank_fusion(semantic, lexical, k=60, top_k=5)
    assert len(result) == 5


def test_rrf_overlap_item_beats_non_overlap():
    semantic = [_chunk("shared"), _chunk("only_sem")]
    lexical  = [_chunk("shared"), _chunk("only_lex")]
    result = reciprocal_rank_fusion(semantic, lexical, k=60, top_k=3)
    assert result[0]["id"] == "shared"


def test_rrf_empty_inputs():
    result = reciprocal_rank_fusion([], [], k=60, top_k=3)
    assert result == []


# ── _extract_final_answer (cot_rag) ──────────────────────────────────────────

from cot_rag import _extract_final_answer


def test_extract_structured_with_marker():
    response = "Step 1: facts.\nStep 2: reasoning.\nFinal Answer: Paris"
    assert _extract_final_answer(response, "structured") == "Paris"


def test_extract_structured_no_marker_fallback():
    response = "Some reasoning.\n\nThe answer is London."
    result = _extract_final_answer(response, "structured")
    assert "London" in result


def test_extract_citation_mode():
    response = "Analysis here.\nStep 3 - Answer: Berlin"
    assert _extract_final_answer(response, "citation") == "Berlin"


def test_extract_simple_mode():
    response = "  Tokyo  "
    assert _extract_final_answer(response, "simple") == "Tokyo"


# ── _token_overlap (error_analyzer) ──────────────────────────────────────────

from error_analyzer import _token_overlap, classify_difficulty, classify_error


def test_token_overlap_identical():
    assert _token_overlap("hello world", "hello world") == pytest.approx(1.0)


def test_token_overlap_no_overlap():
    assert _token_overlap("foo bar", "baz qux") == pytest.approx(0.0)


def test_token_overlap_partial():
    score = _token_overlap("hello world", "hello there")
    assert 0.0 < score < 1.0


def test_token_overlap_empty():
    assert _token_overlap("", "anything") == pytest.approx(0.0)


# ── classify_difficulty ───────────────────────────────────────────────────────

def test_difficulty_simple_short_answer():
    assert classify_difficulty("Who founded Apple?", "Steve Jobs") == "simple"


def test_difficulty_hard_long_answer():
    long = " ".join(["word"] * 20)
    assert classify_difficulty("Describe the process.", long) == "hard"


def test_difficulty_medium_moderate():
    # 4-word answer, non-complex question → medium
    assert classify_difficulty("What year was it?", "nineteen eighty four hundred") == "medium"


# ── classify_error ───────────────────────────────────────────────────────────

def _make_ctx(text: str) -> list:
    return [{"text": text}]


def test_classify_error_correct():
    # Answer closely matches ground truth → f1 >= 0.6 → "correct"
    result = classify_error(
        question="Who is the president?",
        answer="Barack Obama",
        ground_truth="Barack Obama",
        context_chunks=_make_ctx("Barack Obama was the 44th president."),
    )
    assert result["category"] == "correct"
    assert result["f1_score"] == pytest.approx(1.0)


def test_classify_error_hallucination():
    result = classify_error(
        question="What color is the sky?",
        answer="The sky is green with purple polka dots and aliens live there",
        ground_truth="blue",
        context_chunks=_make_ctx("The sky appears blue due to Rayleigh scattering."),
    )
    assert result["category"] in ("hallucination", "incomplete", "low_relevance")


# ── _sha (cache) ─────────────────────────────────────────────────────────────

from cache import _sha


def test_sha_deterministic():
    assert _sha("hello") == _sha("hello")


def test_sha_case_insensitive():
    assert _sha("Hello") == _sha("hello")


def test_sha_different_inputs():
    assert _sha("hello") != _sha("world")


def test_sha_length():
    assert len(_sha("any text")) == 16
