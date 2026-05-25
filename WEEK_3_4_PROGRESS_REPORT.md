# Week 3–4 Progress Report
# RAG Precision Optimization — Days 15–28

**Ngày báo cáo:** 24/05/2026  
**Giai đoạn:** Week 3 (Day 15-21) + Week 4 (Day 22-28)  
**Tác giả:** Nguyễn Trọng Huy

---

## Tóm tắt nhanh

| | Week 3 | Week 4 |
|--|--|--|
| **Kỹ thuật** | Query Expansion + Adaptive Retrieval | Chain-of-Thought Generation |
| **Avg score** | 0.9071 (QE) · 0.9228 (Adaptive) | **0.9434** ← best overall |
| **Metric nổi bật** | Precision 0.9639, Recall 1.0000 | Faithfulness 0.9000 ↑ |
| **Trạng thái** | ✅ Hoàn thành | ✅ Hoàn thành |

---

## 1. Kết quả tích lũy (tính đến hết Week 4)

```
===========================================================================
  TECHNIQUE COMPARISON (Ragas, 30 SQUAD test samples)
===========================================================================
  Run                       Faith   Relev    Prec  Recall     AVG
  ─────────────────────── ─────── ─────── ─────── ─────── ───────
  cot_structured           0.9000  0.9097  0.9639  1.0000  0.9434  ← BEST
  adaptive_heuristic       0.8333  0.8939  0.9639  1.0000  0.9228
  reranked                 0.8389  0.8755  0.9639  1.0000  0.9196
  query_expansion_combined 0.8222  0.8423  0.9639  1.0000  0.9071
  hybrid                   0.8833  0.8717  0.8778  0.9667  0.8999
  baseline                 0.8389  0.8405  0.9000  0.9333  0.8782
===========================================================================
```

**Tiến độ tổng thể:** 0.8782 → **0.9434** (+7.4% so với baseline)

---

## 2. Week 3 — Kỹ thuật 3 & 4 (Day 15-21)

### 2.1 Query Expansion — Multi-Query + HyDE (Day 15-17)

**Mục tiêu:** Cải thiện Faithfulness bằng cách mở rộng query thành nhiều variants trước retrieval.

**Triển khai:**
- `src/query_expansion.py` — `QueryExpansionRAG` class
- Modes: `multi_query`, `hyde`, `combined`
- Runner: `run_query_expansion_eval.py --mode [multi_query|hyde|combined]`

**Kết quả:**

| Metric | Reranked | QE Combined | Δ |
|--------|----------|-------------|---|
| Faithfulness | 0.8389 | 0.8222 | **−2.0% ↓** |
| Answer Relevancy | 0.8755 | 0.8423 | −3.8% ↓ |
| Context Precision | 0.9639 | **0.9639** | 0% |
| Context Recall | 1.0000 | **1.0000** | 0% |
| **AVG** | 0.9196 | 0.9071 | **−1.4%** |

**Phân tích thất bại của HyDE trên SQUAD:**

HyDE sinh ra "hypothetical answer" bằng LLM rồi dùng embedding của đó làm query. Vấn đề với factual QA:

```
Q: "In what year was the LaFortune Center renamed?"

HyDE sinh ra: "...The LaFortune Center was renamed in 1980..."
                                            ↑ hallucinate năm sai

Embedding của "1980" → kéo về document nói về năm 1980 khác → Faithfulness giảm
```

**Kết luận:** QE phát huy tốt với open-ended/conceptual queries. Không phù hợp SQUAD-style factual QA.

---

### 2.2 Adaptive Retrieval — Dynamic top_k (Day 18-20)

**Mục tiêu:** Tự động điều chỉnh retrieval config theo độ phức tạp câu hỏi.

**Triển khai:**
- `src/adaptive_rag.py` — `AdaptiveRAG` class với `classify_heuristic()` + `classify_llm()`
- Word-boundary regex cho signal matching (tránh `"cons"` match trong `"constructed"`)
- 3 tiers:

| Tier | Signals | top_k | depth | QE |
|------|---------|-------|-------|----|
| simple | who/when/where/which | 3 | 20 | off |
| medium | contextual, multi-clause | 4 | 20 | off |
| complex | why/compare/explain/how does | 5 | 30 | multi_query |

**Phân phối trên SQUAD test (30 câu):** simple=13 / medium=17 / complex=0

**Kết quả:**

| Metric | CoT | Adaptive | Δ |
|--------|-----|----------|---|
| Faithfulness | **0.9000** | 0.8333 | −7.4% ↓ |
| Answer Relevancy | 0.9097 | **0.8939** | −1.7% |
| Context Precision | **0.9639** | **0.9639** | 0% |
| Context Recall | **1.0000** | **1.0000** | 0% |
| **AVG** | **0.9434** | 0.9228 | **−2.2%** |

**Phân tích — tại sao Adaptive thấp hơn CoT trên SQUAD:**

SQUAD là 100% factual QA → toàn bộ câu hỏi nằm ở simple/medium tier. Medium dùng top_k=4 thay vì 3 — context rộng hơn → LLM đôi khi blend thông tin từ passage thứ 4 → Faithfulness giảm nhẹ.

**Khi nào Adaptive vượt trội:** Dataset có mix complexity thực sự (vừa "who/when" vừa "why/compare"). Trên SQUAD thuần factual → CoT fixed top_k=3 vẫn tốt hơn.

---

## 3. Week 4 — Chain-of-Thought Generation (Day 22-24)

**Mục tiêu:** Giải quyết Faithfulness thấp (0.8389) tồn tại từ Reranker.

**Insight chính:**
> Retrieval đã hoàn hảo (Recall=1.0000, Precision=0.9639) → bottleneck là GENERATION, không phải retrieval.

**Cơ chế CoT — tại sao giảm hallucination:**

```
Standard prompt:
  Context → "Answer the question" → LLM có thể "nhớ" từ training → Hallucinate

CoT structured prompt:
  Context → "Step 1: List facts from context" ← buộc explicit grounding
           → "Step 2: Reason from those facts"
           → "Final Answer: [parser extract this only]"

Khi LLM đã liệt kê facts (Step 1), nó không thể generate nội dung ngoài context
mà không tự mâu thuẫn với Step 1 → Faithfulness tăng tự nhiên
```

**Triển khai:**
- `src/cot_rag.py` — `ChainOfThoughtRAG`, `generate_cot_answer()`, `_extract_final_answer()`
- 3 modes: `structured`, `simple`, `citation`
- `max_tokens=600` (gấp đôi baseline để accommodate reasoning chain)
- Parser: extract chỉ phần sau `"Final Answer:"` → Ragas chỉ đánh giá final answer, không gồm reasoning

**Kết quả:**

| Metric | Reranked | CoT Structured | Δ |
|--------|----------|----------------|---|
| Faithfulness | 0.8389 | **0.9000** | **+7.3% ↑↑** |
| Answer Relevancy | 0.8755 | **0.9097** | **+3.9% ↑** |
| Context Precision | 0.9639 | **0.9639** | 0% (same retrieval) |
| Context Recall | 1.0000 | **1.0000** | 0% (same retrieval) |
| **AVG** | 0.9196 | **0.9434** | **+2.6% ↑** |

---

## 4. Phân tích tổng hợp Week 3-4

### 4.1 Contribution Breakdown (từ baseline)

```
Baseline RAG                      0.8782   (100%)
  + Hybrid Search      +0.0217    0.8999   retrieval breadth
  + Reranking          +0.0197    0.9196   retrieval precision
  + CoT Generation     +0.0238    0.9434   generation quality
  ─────────────────────────────────────────
  Total gain           +0.0652    +7.4%
```

**Quan sát quan trọng:** Mỗi kỹ thuật đóng góp xấp xỉ đồng đều — retrieval optimization và generation optimization có cùng tầm quan trọng.

### 4.2 Kỹ thuật không cải thiện trên SQUAD

| Kỹ thuật | Lý do không cải thiện | Khi nào hiệu quả |
|----------|----------------------|-----------------|
| Query Expansion (HyDE) | Hallucinate facts trong hypothetical answer | Open-ended, conceptual QA |
| Adaptive top_k (medium=4) | SQUAD toàn simple/factual → top_k=3 tối ưu | Mixed-complexity datasets |
| Multi-Query | Variants không thêm angle mới cho factual QA | Ambiguous, multi-topic queries |

### 4.3 Kiến trúc pipeline tối ưu

```
BEST PIPELINE (avg 0.9434):
  Query
    → Hybrid Retrieve (BM25 + Semantic → RRF, top-20)
    → CrossEncoder Rerank (ms-marco-MiniLM-L-6-v2, top-3)
    → CoT Generate (structured 3-step prompt, extract "Final Answer:")
    → Answer
```

---

## 5. Deliverables Week 3-4

| File | Mô tả |
|------|-------|
| `src/query_expansion.py` | Multi-Query + HyDE pipeline |
| `src/adaptive_rag.py` | Adaptive complexity classifier + retrieval routing |
| `src/cot_rag.py` | Chain-of-Thought generation (3 modes) |
| `run_query_expansion_eval.py` | QE evaluation runner |
| `run_adaptive_eval.py` | Adaptive evaluation + tier distribution |
| `run_cot_eval.py` | CoT evaluation runner |
| `results/query_expansion_combined_metrics.json` | QE: avg 0.9071 |
| `results/adaptive_heuristic_metrics.json` | Adaptive: avg 0.9228 |
| `results/cot_structured_metrics.json` | CoT: avg **0.9434** |
| `RESEARCH_REPORT.md` | Full research report (Vietnamese) |
| `PRESENTATION_SLIDES.md` | 15-slide Marp presentation |

---

## 6. Kế hoạch Week 5-6

| Ngày | Task |
|------|------|
| Day 29-31 | Unit tests (`tests/test_core.py`) — kiểm tra pipeline robustness |
| Day 32-33 | Latency & cost analysis — đo thời gian và số token mỗi technique |
| Day 34-35 | Multilingual eval với Ragas (tiếng Việt test set) |
| Day 36-38 | Code cleanup, docstrings, config YAML files |
| Day 39-41 | Final presentation polish |
| Day 42-44 | End-to-end testing + edge cases |
| Day 45 | Presentation day |

**Mục tiêu còn lại:** Đạt 95%+ avg trên mixed-complexity dataset (hiện tại 94.34% trên SQUAD factual).
