# HOW TO RUN — RAG Precision Optimization

Hướng dẫn chi tiết cách chạy từng kỹ thuật trong pipeline RAG, từ baseline đến reranking.

---

## Yêu cầu chung

```bash
pip install -r requirements.txt
cp .env.example .env
# Mở .env và điền OPENAI_API_KEY
```

Kiểm tra API key hoạt động:
```bash
python -c "import openai; openai.OpenAI().models.list(); print('API key OK')"
```

---

## Kỹ thuật 1: Baseline RAG (Day 2)

**File:** `src/baseline_rag.py`

**Giải thích:**
Đây là pipeline RAG thuần túy (vanilla). Luồng xử lý:
1. Load documents từ `data/rag_dataset.json`
2. Chunk mỗi document thành các đoạn 512 từ (overlap 50 từ)
3. Embed từng chunk bằng `text-embedding-ada-002` → lưu vào ChromaDB
4. Với mỗi câu hỏi: tìm top-3 chunk gần nhất (cosine similarity)
5. Ghép context vào prompt → sinh câu trả lời bằng `gpt-4o-mini`

**Chạy pipeline test (5 câu hỏi mẫu):**
```bash
python src/baseline_rag.py
# Output: results/baseline_test_results.json
```

**Chạy evaluation (30 câu hỏi SQUAD):**
```bash
python src/evaluation.py
# Output: results/baseline_metrics.json
```

**Kết quả đạt được:**
| Metric | Score |
|--------|-------|
| Faithfulness | 0.8389 |
| Answer Relevancy | 0.8405 |
| Context Precision | 0.9000 |
| Context Recall | 0.9333 |
| **Average** | **0.8782** |

**Điểm mạnh/yếu:**
- Đơn giản, dễ debug
- Chỉ dùng semantic similarity → miss các câu hỏi keyword-heavy (tên người, số liệu)

---

## Kỹ thuật 2: Hybrid Search — BM25 + Semantic (Day 8-10)

**File:** `src/hybrid_rag.py`

**Giải thích:**
Kết hợp hai phương pháp retrieval khác nhau để bù đắp nhược điểm của nhau:

| | BM25 (Lexical) | Semantic (Vector) |
|---|---|---|
| Cơ chế | Đếm tần suất từ xuất hiện (TF-IDF cải tiến) | Khoảng cách cosine giữa embedding |
| Giỏi với | Tên riêng, số liệu, keyword chính xác | Câu hỏi mơ hồ, paraphrase, ngữ nghĩa |
| Yếu với | Paraphrase, đồng nghĩa | Keyword chính xác, out-of-vocab |

Sau khi cả hai trả về top-20, kết quả được merge bằng **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(chunk) = Σ [ 1 / (60 + rank_i) ]   (tổng qua từng phương pháp i)
```

Chunk xuất hiện cao trong cả hai danh sách sẽ được ưu tiên.

**Chạy pipeline test:**
```bash
python src/hybrid_rag.py
# Output: results/hybrid_test_results.json
```

**Chạy evaluation:**
```bash
# Sửa evaluation.py để chỉ chạy hybrid, hoặc chạy file so sánh:
python src/evaluation.py
# Output: results/hybrid_metrics.json
```

**Kết quả đạt được:**
| Metric | Baseline | Hybrid | Delta |
|--------|----------|--------|-------|
| Faithfulness | 0.8389 | 0.8833 | +5.3% ↑ |
| Answer Relevancy | 0.8405 | 0.8717 | +3.7% ↑ |
| Context Precision | 0.9000 | 0.8778 | -2.5% ↓ |
| Context Recall | 0.9333 | 0.9667 | +3.6% ↑ |
| **Average** | **0.8782** | **0.8999** | **+2.5%** |

**Lưu ý:** Context Precision giảm nhẹ là trade-off bình thường — hybrid mang về nhiều context hơn nhưng một số không thực sự cần thiết. Reranking ở bước tiếp theo sẽ khắc phục điều này.

---

## Kỹ thuật 3: Reranking — CrossEncoder (Day 11-13)

**File:** `src/reranker_rag.py`  
**Runner:** `run_reranker_eval.py`

**Giải thích:**
Thêm một tầng "giám khảo" thứ hai sau khi hybrid search trả về 20 candidates.

**Tại sao cần Reranker?**

Bi-Encoder (dùng ở bước retrieval) encode query và document **riêng lẻ** → nhanh nhưng kém chính xác.
CrossEncoder encode cặp **(query, document) cùng nhau** → chậm hơn nhưng chính xác hơn nhiều vì model thấy cả hai phía khi ra quyết định.

```
Hybrid Retrieval (BM25 + Semantic)
       ↓  top 20 candidates
CrossEncoder: score(query, passage₁), score(query, passage₂), ..., score(query, passage₂₀)
       ↓  sort by score descending
     Top 3 passages  →  LLM  →  Answer
```

Model dùng: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Chạy **local**, không cần API key
- Download tự động lần đầu (~85 MB, cached tại `~/.cache/huggingface/`)
- Được train trên MS MARCO (1M+ passage retrieval pairs)

**Chạy pipeline test (5 câu hỏi):**
```bash
python src/reranker_rag.py
# Output: results/reranker_test_results.json
```

**Chạy evaluation đầy đủ + so sánh 3 kỹ thuật:**
```bash
python run_reranker_eval.py
# Output: results/reranked_metrics.json
#         In bảng so sánh baseline / hybrid / reranked
```

**Kết quả đạt được:**
| Metric | Hybrid | Reranked | Delta |
|--------|--------|----------|-------|
| Faithfulness | 0.8833 | 0.8389 | -5.0% |
| Answer Relevancy | 0.8717 | 0.8755 | +0.4% ↑ |
| Context Precision | 0.8778 | **0.9639** | **+8.6% ↑↑** |
| Context Recall | 0.9667 | **1.0000** | **+3.3% ↑** |
| **Average** | **0.8999** | **0.9196** | **+2.2%** |

**Phân tích:**
- **Context Precision +8.6%**: Mục tiêu chính đạt được — CrossEncoder loại bỏ noise hiệu quả.
- **Context Recall 1.0000**: Candidate pool 20 kết quả đảm bảo không bỏ sót thông tin.
- **Faithfulness giảm**: LLM đôi khi generate nội dung beyond context khi được cung cấp context quá tốt. Sẽ được xử lý ở technique 4 (Chain-of-Thought).

**Config tunable:**
```bash
# Trong .env
TOP_K=3              # Số passage cuối cùng truyền vào LLM
RETRIEVE_DEPTH=20    # Số candidates trước khi rerank (nên 5-10x TOP_K)
```

---

## So sánh toàn bộ (tính đến hiện tại)

```
python run_reranker_eval.py
```

Sẽ in ra bảng:

```
===========================================================================
  TECHNIQUE COMPARISON TABLE
===========================================================================
  Run                         Faith   Relev    Prec  Recall     AVG
  ───────────────────────── ─────── ─────── ─────── ─────── ───────
  reranked                   0.8389  0.8755  0.9639  1.0000  0.9196
  hybrid                     0.8833  0.8717  0.8778  0.9667  0.8999
  baseline                   0.8389  0.8405  0.9000  0.9333  0.8782
===========================================================================
```

---

## Kỹ thuật 4: Query Expansion — Multi-Query + HyDE (Day 15-17) [Upcoming]

**File:** `src/query_expansion.py` (chưa implement)

**Kế hoạch:**

**Multi-Query:** Dùng LLM sinh ra 3-5 biến thể của câu hỏi gốc, chạy retrieval cho từng biến thể, gộp kết quả. Giải quyết vấn đề câu hỏi mơ hồ hoặc có nhiều cách diễn đạt.

**HyDE (Hypothetical Document Embeddings):** Thay vì embed câu hỏi, dùng LLM sinh ra một đoạn văn *giả định* trả lời câu hỏi, rồi embed đoạn đó để retrieval. Embedding của "câu trả lời giả" gần với embedding của document thật hơn so với embedding của câu hỏi.

```bash
# Sẽ chạy bằng (chưa có):
python src/query_expansion.py
python run_query_expansion_eval.py
```

**Kỳ vọng:** avg_score ~0.93+

---

## Tóm tắt lệnh

| Mục tiêu | Lệnh |
|----------|------|
| Test baseline pipeline | `python src/baseline_rag.py` |
| Test hybrid pipeline | `python src/hybrid_rag.py` |
| Test reranker pipeline | `python src/reranker_rag.py` |
| Evaluate reranker + so sánh | `python run_reranker_eval.py` |
| Evaluate tất cả cùng lúc | `python src/evaluation.py` |

---

## Cấu trúc output

Mỗi lần chạy pipeline test tạo ra file `results/<technique>_test_results.json`:

```json
[
  {
    "question": "What is Retrieval Augmented Generation?",
    "answer": "RAG is a technique that...",
    "context": [
      {
        "rank": 1,
        "text": "Retrieval-Augmented Generation (RAG)...",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "source": "ArXiv",
        "rrf_score": 0.032258,
        "reranker_score": 8.432
      }
    ],
    "top_k": 3
  }
]
```

Mỗi lần chạy evaluation tạo ra `results/<technique>_metrics.json`:

```json
{
  "run_name": "reranked",
  "timestamp": "2026-05-24T10:46:24",
  "n_samples": 30,
  "faithfulness": 0.8389,
  "answer_relevancy": 0.8755,
  "context_precision": 0.9639,
  "context_recall": 1.0,
  "avg_score": 0.9196
}
```
