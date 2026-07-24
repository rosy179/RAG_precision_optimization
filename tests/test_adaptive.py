"""
tests/test_adaptive.py
----------------------
Unit tests for the query-complexity heuristic (src/adaptive_rag.py).

Regression guard for the CJK fix: Japanese/Chinese questions are written
without spaces, so `.split()` used to collapse them to ~1 "word" and every
Japanese query fell through to the `word_count <= 8` "simple" tiebreak — which
meant the multi-hop router (gated on medium/complex in chat.py) never fired
for Japanese. classify_heuristic now derives an approximate word count from
CJK character runs so unspaced questions classify by length like spaced ones.

Run: pytest tests/test_adaptive.py -v
"""

import sys
from pathlib import Path

# Add project root and src to path (mirrors test_core.py)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from adaptive_rag import classify_heuristic


# ── The CJK regression: long Japanese questions must NOT be "simple" ──────────

class TestJapaneseComplexity:
    def test_long_bridge_question_not_simple(self):
        # The exact question from the demo that previously returned "simple"
        # (so multi-hop never fired). It must now open the router gate.
        q = ("DeepSeek-R1が推論能力を獲得するために用いた学習手法は何で、"
             "その手法はどのアルゴリズムを基にしていますか")
        assert classify_heuristic(q) in ("medium", "complex")

    def test_two_complex_signals_is_complex(self):
        # 「仕組み」+「どのように」 → two complex signals → complex outright.
        q = "DeepSeek-V3が基盤とするアーキテクチャは何で、その仕組みはどのように機能しますか"
        assert classify_heuristic(q) == "complex"

    def test_short_factual_stays_simple(self):
        # A genuinely single-hop factual lookup should still be "simple".
        assert classify_heuristic("誰がAlexNetを発表しましたか") == "simple"


# ── EN / VI behaviour must be unchanged (cjk_chars == 0 for them) ─────────────

class TestNonCjkUnchanged:
    def test_english_short_factual_simple(self):
        assert classify_heuristic("When was BERT released?") == "simple"

    def test_english_multi_signal_complex(self):
        q = "Why is RAG better than fine-tuning and how does reranking help?"
        assert classify_heuristic(q) == "complex"

    def test_vietnamese_multi_signal_complex(self):
        # Vietnamese uses Latin+diacritics (outside the CJK ranges), so the fix
        # leaves its word count untouched.
        q = "Tại sao RAG tốt hơn fine-tuning và nó ảnh hưởng thế nào đến độ chính xác?"
        assert classify_heuristic(q) == "complex"
