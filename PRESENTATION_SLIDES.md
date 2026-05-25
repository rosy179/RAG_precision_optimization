---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
color: #1a1a2e
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 22px;
  }
  h1 { color: #16213e; font-size: 2em; }
  h2 { color: #0f3460; font-size: 1.4em; border-bottom: 2px solid #e94560; padding-bottom: 6px; }
  table { font-size: 0.85em; }
  th { background: #0f3460; color: white; }
  tr:nth-child(even) { background: #f0f4ff; }
  code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
  pre { background: #1e1e2e; color: #cdd6f4; padding: 16px; border-radius: 8px; font-size: 0.8em; }
  .highlight { color: #e94560; font-weight: bold; }
---

# Tối Ưu Hóa Độ Chính Xác Hệ Thống RAG
## Bằng Kỹ Thuật Kết Hợp

**Nguyễn Trọng Huy** · 24/05/2026

Framework: **Ragas** · Dataset: **SQUAD v1.1** (30 test samples)

---

> ## Kết quả: `0.8782` → **`0.9434`** (**+7.4%** trong 24 ngày)

Hybrid Search + CrossEncoder Reranking + Chain-of-Thought Generation

---

# Vấn Đề: RAG Vanilla Có Gì Hạn Chế?

## 3 Điểm Yếu Cốt Lõi

```
Documents → Chunk → Embed(ada-002) → ChromaDB
Query     → Embed → cosine similarity → top-3 → LLM → Answer
```

| Hạn chế | Ví dụ thực tế |
|---------|--------------|
| Semantic search bỏ sót keyword-heavy queries | "LaFortune Center renamed in what year?" → bỏ sót |
| Bi-encoder retrieval thiếu chính xác | Lấy về sai passages khi nhiều document cùng topic |
| LLM hallucinate kể cả khi context đúng | Thêm "year 1980" dù context nói "1987" |

**Mục tiêu nghiên cứu:** Đo đóng góp từng kỹ thuật — mỗi layer cải thiện bao nhiêu %?

---

# Framework Đánh Giá — Ragas

## 4 Metrics, 30 SQUAD Test Samples

| Metric | Đo gì | Cách tính |
|--------|-------|-----------|
| **Faithfulness** | Answer grounded trong context? | LLM kiểm tra từng claim |
| **Answer Relevancy** | Answer relevant với question? | Embedding similarity |
| **Context Precision** | Chunks retrieved có hữu ích? | Proportion useful chunks |
| **Context Recall** | Context đủ info cần thiết? | Coverage of ground truth |

**Dataset:** SQUAD v1.1 · 150 QA pairs · 56 documents (Wikipedia + ArXiv) · ~750K chars

```
Split: 70% train (105 QA) / 30% test (45 QA) → 30 samples dùng cho eval
```

---

# Kiến Trúc Pipeline Tối Ưu

## Best Stack: Avg 0.9434

```
                   ┌─────────────────────────────────────┐
  Query ──────────►│  Hybrid Retrieve                    │
                   │  BM25 (lexical) + Semantic (dense)  │
                   │  → RRF fusion → top-20 candidates   │
                   └─────────────────┬───────────────────┘
                                     │
                   ┌─────────────────▼───────────────────┐
                   │  CrossEncoder Rerank                 │
                   │  ms-marco-MiniLM-L-6-v2             │
                   │  joint-encode (query, passage) → top-3│
                   └─────────────────┬───────────────────┘
                                     │
                   ┌─────────────────▼───────────────────┐
                   │  CoT Generation                     │
                   │  Step 1 Facts → Step 2 Reasoning    │
                   │  → "Final Answer:" [extracted only] │
                   └─────────────────────────────────────┘
```

---

# Kỹ Thuật 1: Hybrid Search + RRF

## BM25 (Sparse) + Semantic (Dense) → Reciprocal Rank Fusion

**Vấn đề:** Pure semantic search bỏ sót keyword-heavy queries (tên, số, năm)

```python
# Reciprocal Rank Fusion
RRF_score(d) = Σᵢ  1 / (60 + rankᵢ(d))
```

| Metric | Baseline | + Hybrid | Δ |
|--------|:--------:|:--------:|:---:|
| Faithfulness | 0.8389 | **0.8833** | +5.3% ↑ |
| Answer Relevancy | 0.8405 | 0.8717 | +3.7% ↑ |
| Context Precision | 0.9000 | 0.8778 | −2.5% ↓ |
| Context Recall | 0.9333 | **0.9667** | +3.6% ↑ |
| **AVG** | 0.8782 | **0.8999** | **+2.5%** |

**Trade-off:** Context Precision giảm nhẹ (mang về nhiều candidates hơn) — sẽ được CrossEncoder fix.

---

# Kỹ Thuật 2: CrossEncoder Reranking

## Bi-Encoder (fast) → CrossEncoder (precise)

| | Bi-Encoder | CrossEncoder |
|--|--|--|
| Encode | query & doc riêng | (query, doc) cùng nhau |
| Cross-attention | ❌ Không có | ✅ Đầy đủ |
| Speed | O(n) fast | O(n) slow |
| Accuracy | Thấp hơn | Cao hơn |

**Strategy:** Bi-Encoder lấy top-20, CrossEncoder rerank → top-3

| Metric | + Hybrid | + Reranking | Δ |
|--------|:--------:|:-----------:|:---:|
| Context Precision | 0.8778 | **0.9639** | **+8.6%** ↑↑ |
| Context Recall | 0.9667 | **1.0000** | **+3.4%** ↑↑ |
| **AVG** | 0.8999 | **0.9196** | **+2.2%** |

---

# Kỹ Thuật 3: Query Expansion — Thất Bại Có Giá Trị

## HyDE Hallucinate Facts trong Factual QA

```
Q: "In what year was the LaFortune Center renamed?"

HyDE sinh: "...The LaFortune Center was renamed in 1980..."
                                            ↑ hallucinate năm sai (thực tế: 1987)

Embedding "1980" → kéo về wrong documents → Faithfulness giảm
```

| Metric | + Reranking | + QE Combined | Δ |
|--------|:-----------:|:-------------:|:---:|
| Faithfulness | 0.8389 | 0.8222 | **−2.0% ↓** |
| **AVG** | 0.9196 | 0.9071 | **−1.4%** |

**Kết luận:** HyDE phù hợp open-ended/conceptual queries — **không phù hợp factual QA**

Query Expansion tốt khi: "Explain advantages of RAG over fine-tuning" ✅

---

# Kỹ Thuật 4: Adaptive Retrieval

## Dynamic top_k theo Độ Phức Tạp Query

| Tier | Signal words | top_k | depth | QE |
|------|-------------|-------|-------|----|
| **simple** | who/when/where/which | 3 | 20 | ❌ |
| **medium** | contextual, multi-clause | 4 | 20 | ❌ |
| **complex** | why/compare/explain/how does | 5 | 30 | multi_query |

**Phân phối SQUAD test (30 câu):** simple=13 / medium=17 / **complex=0**

| Metric | + Reranking | + Adaptive | Δ |
|--------|:-----------:|:----------:|:---:|
| Faithfulness | 0.8389 | 0.8333 | −0.7% |
| **AVG** | 0.9196 | 0.9228 | +0.3% |

**SQUAD toàn factual → top_k=3 là tối ưu.** Adaptive vượt trội trên mixed-complexity datasets.

---

# Kỹ Thuật 5: Chain-of-Thought Generation ⭐

## Giải Quyết Hallucination Ở Tầng Generation

**Insight:** Retrieval đã hoàn hảo (Recall=1.0000, Precision=0.9639) → bottleneck là GENERATION

```
Standard:  Context → "Answer the question" → LLM có thể nhớ từ training → Hallucinate

CoT:       Context → "Step 1: List ONLY facts from context"   ← explicit grounding
                   → "Step 2: Reason ONLY from those facts"
                   → "Final Answer: ..."  ← parser extract this only
```

| Metric | + Reranking | + **CoT Structured** | Δ |
|--------|:-----------:|:--------------------:|:---:|
| Faithfulness | 0.8389 | **0.9000** | **+7.3% ↑↑** |
| Answer Relevancy | 0.8755 | **0.9097** | **+3.9% ↑** |
| Context Precision | 0.9639 | 0.9639 | 0% |
| **AVG** | 0.9196 | **0.9434** | **+2.6% ↑** |

---

# Kết Quả Tổng Hợp

## Ragas · 30 SQUAD Test Samples

| Kỹ thuật | Faith | Relev | Prec | Recall | **AVG** |
|----------|:-----:|:-----:|:----:|:------:|:-------:|
| Baseline RAG | 0.8389 | 0.8405 | 0.9000 | 0.9333 | 0.8782 |
| + Hybrid Search | 0.8833 | 0.8717 | 0.8778 | 0.9667 | 0.8999 |
| + Reranking | 0.8389 | 0.8755 | **0.9639** | **1.0000** | 0.9196 |
| + Query Expansion | 0.8222 | 0.8423 | **0.9639** | **1.0000** | 0.9071 |
| + Adaptive Retrieval | 0.8333 | 0.8939 | **0.9639** | **1.0000** | 0.9228 |
| **+ CoT Structured** | **0.9000** | **0.9097** | **0.9639** | **1.0000** | **0.9434** |

---

> **Best pipeline:** Hybrid + CrossEncoder + CoT · Avg **0.9434** (+7.4%)

---

# Phân Tích Đóng Góp

## Mỗi Layer Đóng Góp Xấp Xỉ Đồng Đều

```
Baseline RAG                        0.8782  (100%)
  ├─ + Hybrid Search    +0.0217     0.8999  retrieval breadth  ↑2.5%
  ├─ + Reranking        +0.0197     0.9196  retrieval precision ↑2.2%
  └─ + CoT Generation   +0.0238     0.9434  generation quality ↑2.6%
     ─────────────────────────────────────
     Total gain          +0.0652            +7.4%
```

**Bài học quan trọng:**
- Retrieval optimization **và** generation optimization có cùng tầm quan trọng
- Không thể chỉ tối ưu retrieval rồi bỏ qua generation layer
- CoT hiệu quả vì **force explicit grounding** — không phải "magic prompting"

| Kỹ thuật | Không hiệu quả vì | Hiệu quả khi |
|----------|-------------------|--------------|
| HyDE | Hallucinate năm/tên trong hypothetical | Open-ended, conceptual QA |
| Adaptive | SQUAD toàn simple/factual | Mixed-complexity datasets |

---

# Thách Thức Đa Ngôn Ngữ

## Input Tiếng Việt / Tiếng Nhật

| Tầng | Tiếng Việt | Tiếng Nhật |
|------|-----------|------------|
| BM25 `text.split()` | ~OK (có space) | ❌ FAIL (không space) |
| `text-embedding-ada-002` | ~65% quality | ~55% quality |
| `ms-marco` CrossEncoder | ~30% accuracy | ~20% accuracy |
| CoT response language | Trả lời tiếng Anh | Trả lời tiếng Anh |
| **Ước tính quality** | **~0.65–0.72** | **~0.40–0.55** |

**Nguyên nhân sâu:**
- `ada-002` tối ưu English → cross-language semantic gap lớn
- `ms-marco-MiniLM` train 100% English MS MARCO → score cross-language ≈ random
- Japanese/Chinese: không có spaces → BM25 tokenize cả câu thành 1 token → zero overlap

---

# Giải Pháp Đa Ngôn Ngữ

## 2 Chiến Lược — Trade-offs Khác Nhau

**Strategy "translate" (quick win):**
```
Query (VI/JA) → GPT-4o-mini translate → English query
  → existing CoT pipeline (không thay đổi)
  → English answer → GPT-4o-mini translate ngược
  → Answer (VI/JA)
```
✅ Không cần reindex | ⚠️ +1-2s latency | ⚠️ Lỗi dịch proper nouns

**Strategy "multilingual" (best quality):**

| Component | Thay thế |
|-----------|---------|
| `text-embedding-ada-002` | `multilingual-e5-small` (100+ ngôn ngữ) |
| `text.split()` BM25 | Character bigrams cho CJK |
| `ms-marco-MiniLM` | `mmarco-mMiniLMv2-L12-H384-v1` (13 ngôn ngữ) |

```
Q (JA): "RAGとはどのような技術ですか？"
A (multilingual): "RAGは、大規模言語モデルが外部データソースから
                  情報を取得し、応答に組み込む技術です。" ✅
```

---

# Key Takeaways

## 5 Bài Học Từ 24 Ngày Thực Nghiệm

1. **Retrieval VÀ generation đều là bottleneck** — tối ưu retrieval không đủ để đạt 94%+
2. **CoT hiệu quả vì cơ chế, không phải "magic"** — Step 1 (explicit fact listing) prevents self-contradiction
3. **Query Expansion hại trên factual QA** — HyDE hallucinate facts trong hypothetical answer làm sai embedding direction
4. **Adaptive Retrieval cần mixed-complexity** — SQUAD thuần factual không có complex queries → không có routing benefit
5. **Multilingual cần stack riêng** — ada-002 + ms-marco không generalize; cần multilingual-e5 + mmarco + bigram BM25

## Lộ Trình Cải Thiện Tiếp Theo

```
Hiện tại: 0.9434 (SQUAD factual)
Mục tiêu: 0.95+  (mixed-complexity dataset)

Next: Unit tests · Latency analysis · Vietnamese eval set · End-to-end testing
```

---

# Tech Stack & Tài Liệu Tham Khảo

## Công Nghệ Sử Dụng

| Layer | Technology |
|-------|-----------|
| LLM | GPT-4o-mini (temp=0.0) |
| Embeddings | text-embedding-ada-002 / multilingual-e5-small |
| Vector Store | ChromaDB |
| BM25 | rank_bm25 (word split / character bigrams) |
| CrossEncoder | ms-marco-MiniLM-L-6-v2 / mmarco-mMiniLMv2-L12-H384-v1 |
| Evaluation | Ragas 0.1.x |
| Language Detection | langdetect |

## References

Lewis et al. (2020) · Robertson & Zaragoza (2009) · Nogueira & Cho (2019)
Gao et al. (2022) HyDE · Wei et al. (2022) CoT · Es et al. (2023) RAGAS
Rajpurkar et al. (2016) SQuAD · Wang et al. (2024) Multilingual E5 · Asai et al. (2023) Self-RAG

---

> **Q & A**
>
> Source code: `src/` · Results: `results/` · Report: `RESEARCH_REPORT.md`
>
> *Nguyễn Trọng Huy · RAG Precision Optimization · 24/05/2026*
