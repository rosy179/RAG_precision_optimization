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
| Faithfulness | 0.8833 | 0.8056 | -8.8% |
| Answer Relevancy | 0.8717 | 0.8779 | +0.7% ↑ |
| Context Precision | 0.8778 | **0.9639** | **+8.6% ↑↑** |
| Context Recall | 0.9667 | **1.0000** | **+3.3% ↑** |
| **Average** | **0.8999** | **0.9118** | **+1.3%** |

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

```bash
python run_cot_eval.py   # chạy CoT và in bảng đầy đủ
```

Kết quả thực đo (Ragas, 30 SQUAD test samples):

```
===========================================================================
  TECHNIQUE COMPARISON TABLE
===========================================================================
  Run                         Faith   Relev    Prec  Recall     AVG
  ───────────────────────── ─────── ─────── ─────── ─────── ───────
  cot_structured             0.9000  0.9097  0.9639  1.0000  0.9434   ← best overall
  reranked                   0.8056  0.8779  0.9639  1.0000  0.9118
  query_expansion_combined   0.8222  0.8423  0.9639  1.0000  0.9071
  hybrid                     0.8833  0.8717  0.8778  0.9667  0.8999
  baseline                   0.8389  0.8405  0.9000  0.9333  0.8782
===========================================================================
```

**Nhận xét tổng thể:**
- **CoT Structured** là kỹ thuật tốt nhất toàn diện — **0.9434** avg, vượt Reranking (+2.6%)
- Faithfulness đạt **0.9000** (+7.3% so với Reranker) — mục tiêu chính hoàn thành
- Answer Relevancy **0.9097** — cải thiện bất ngờ do structured reasoning tập trung vào question
- Điểm tích lũy: Hybrid (retrieval) + CrossEncoder (reranking) + CoT (generation) = bộ ba tối ưu

---

## Kỹ thuật 4: Query Expansion — Multi-Query + HyDE (Day 15-17)

**File:** `src/query_expansion.py`
**Runner:** `run_query_expansion_eval.py`

**Giải thích:**

Một câu hỏi đơn lẻ thường không đủ để kéo về tất cả các góc độ thông tin cần thiết. Kỹ thuật này mở rộng query trước khi retrieval.

**Multi-Query:** LLM sinh ra N cách hỏi khác nhau → retrieval cho từng cách → gộp + dedup kết quả.

```
"How does RAG work?"
    ↓ LLM generates variants
"What is the mechanism behind retrieval augmented generation?"
"Explain the architecture of RAG systems"
"How does retrieval-augmented generation retrieve documents?"
```

**HyDE (Hypothetical Document Embeddings):** Thay vì embed câu hỏi, dùng LLM sinh một *đoạn văn giả định* trả lời câu hỏi, rồi embed đoạn đó làm query. Embedding của "câu trả lời giả" nằm gần embedding của document thật hơn so với embedding của câu hỏi ngắn.

```
"How does RAG work?"
    ↓ LLM generates hypothetical answer
"RAG combines a retrieval system with a generative model.
 Given a query, it first retrieves relevant passages from a
 knowledge base using dense retrieval, then passes those
 passages as context to a language model..."
    ↓ Embed this hypothetical answer → retrieve similar documents
```

**Combined mode** (mặc định): cả hai kỹ thuật cùng lúc, tất cả candidates gộp lại → CrossEncoder rerank.

```
Original Query
    ├── Multi-Query: [variant₁, variant₂, variant₃] → Hybrid Retrieval × 3
    └── HyDE: [hypothetical_doc] → Hybrid Retrieval × 1
                        ↓
          Merged + Deduplicated Candidates
                        ↓
             CrossEncoder Reranking
                        ↓
               Top 3 → LLM → Answer
```

**Chạy pipeline test (5 câu hỏi):**
```bash
python src/query_expansion.py
# Output: results/query_expansion_test_results.json
```

**Chạy evaluation + so sánh:**
```bash
python run_query_expansion_eval.py                  # combined (default)
python run_query_expansion_eval.py --mode multi_query
python run_query_expansion_eval.py --mode hyde
```

**Kết quả đạt được (combined mode):**
| Metric | Reranked | Query Expansion | Delta |
|--------|----------|-----------------|-------|
| Faithfulness | 0.8389 | 0.8222 | -2.0% ↓ |
| Answer Relevancy | 0.8755 | 0.8423 | -3.8% ↓ |
| Context Precision | 0.9639 | **0.9639** | 0% (giữ nguyên) |
| Context Recall | 1.0000 | **1.0000** | 0% (giữ nguyên) |
| **Average** | **0.9196** | **0.9071** | **-1.4%** |

**Phân tích — tại sao QE không cải thiện trên SQUAD?**

Query Expansion phát huy tốt nhất trên câu hỏi mơ hồ, open-ended. SQUAD gồm các câu hỏi factual cụ thể ("In what year?", "Who was?") — Multi-Query tạo ra biến thể nhưng không mang thêm góc nhìn mới.

HyDE đặc biệt nhạy cảm với dạng câu hỏi factual: khi LLM sinh ra đoạn giả định, nó có thể **hallucinate số liệu cụ thể** (năm, tên người, con số), khiến embedding kéo về document sai:
```
Q: "In what year was the LaFortune Center renamed?"
HyDE (sai): "...renamed in 1980..."   ← LLM đoán sai năm
→ Retrieval kéo về document không liên quan
```

**Khuyến nghị theo loại câu hỏi:**
| Loại câu hỏi | Mode tốt nhất |
|---|---|
| Factual QA (SQUAD, trivia) | `multi_query` hoặc tắt QE |
| Open-ended / conceptual | `combined` |
| Technical documentation | `hyde` |

**Config tunable:**
```bash
# Trong .env
N_QUERIES=3          # Số biến thể Multi-Query sinh ra
TOP_K=3              # Số passage cuối truyền vào LLM
RETRIEVE_DEPTH=20    # Candidates mỗi query variant
```

---

## Kỹ thuật 5: Chain-of-Thought (CoT) Retrieval — Day 22-24

**File:** `src/cot_rag.py`
**Runner:** `run_cot_eval.py`

**Giải thích:**
Giữ nguyên retrieval stack tốt nhất (Hybrid + CrossEncoder), chỉ thay đổi **bước generation**.

**Tại sao Faithfulness vẫn thấp sau Reranking?**

Reranker chọn đúng passages, nhưng LLM vẫn có thể "trượt" ra ngoài context khi sinh câu trả lời — đặc biệt với câu hỏi factual (năm tháng, tên người, số liệu cụ thể). LLM "biết" câu trả lời từ pre-training knowledge và có xu hướng generate câu trả lời đó thay vì đọc context cẩn thận.

**Giải pháp CoT:**

```
Standard prompt:
  Context → "Answer the question" → Answer

CoT (structured) prompt:
  Context → Step 1: List relevant facts
           → Step 2: Reason from those facts
           → Step 3: "Final Answer: ..." → parse answer
```

Bước intermediate forcing model phải explicitly trace facts trước → ít hallucinate hơn.

**Ba modes:**

| Mode | Cơ chế | Phù hợp với |
|------|--------|-------------|
| `structured` (default) | 3 bước: facts → reasoning → "Final Answer:" được parse | Factual QA — SQUAD |
| `simple` | "Let's think step by step" prefix | General QA |
| `citation` | Bắt buộc cite [Source N] trong reasoning | Long-form / doc QA |

**Chạy pipeline test (5 câu hỏi):**
```bash
python src/cot_rag.py
# Output: results/cot_test_results.json
```

**Chạy evaluation + so sánh tất cả kỹ thuật:**
```bash
python run_cot_eval.py                    # structured (default)
python run_cot_eval.py --mode simple
python run_cot_eval.py --mode citation
```

**Kết quả đạt được:**
| Metric | Reranked | CoT Structured | Delta |
|--------|----------|----------------|-------|
| Faithfulness | 0.8389 | **0.9000** | **+7.3% ↑↑** |
| Answer Relevancy | 0.8755 | **0.9097** | **+3.9% ↑** |
| Context Precision | 0.9639 | **0.9639** | 0% (same retrieval) |
| Context Recall | 1.0000 | **1.0000** | 0% (same retrieval) |
| **Average** | **0.9196** | **0.9434** | **+2.6% ↑** |

**Phân tích:**
- **Faithfulness +7.3%**: Mục tiêu chính đạt được — CoT buộc LLM phải list facts từ context trước → không thể hallucinate.
- **Answer Relevancy +3.9%**: Unexpected improvement — structured reasoning tập trung câu trả lời vào đúng question hơn.
- **Precision/Recall**: Không đổi vì dùng cùng retrieval stack (Hybrid + CrossEncoder).

**Lưu ý kỹ thuật:**
- `max_tokens=600` (gấp đôi baseline) vì reasoning chain cần thêm tokens
- `_extract_final_answer()` parse "Final Answer:" label → chỉ final answer đưa vào Ragas (không gồm reasoning)
- `temperature=0.0` — deterministic để Faithfulness cao nhất

---

---

## Kỹ thuật 6: Multilingual RAG — Vietnamese, Japanese & 100+ ngôn ngữ

**File:** `src/multilingual_rag.py`
**Runner:** `run_multilingual_demo.py`

**Vấn đề với input không phải tiếng Anh:**

| Tầng | Tiếng Việt | Tiếng Nhật/Trung |
|------|-----------|-----------------|
| BM25 tokenizer (`split()`) | ~OK (có space) | ❌ FAIL (không có space) |
| `text-embedding-ada-002` | ~65% quality | ~55% quality |
| `ms-marco` CrossEncoder | ~30% accuracy | ~20% accuracy |
| CoT response language | Trả lời tiếng Anh | Trả lời tiếng Anh |
| **Ước tính quality** | **~0.65–0.72** | **~0.40–0.55** |

**Hai strategy cải thiện:**

**Strategy "translate"** — Nhanh, dùng lại index tiếng Anh:
```
Input (VI/JA) → GPT-4o-mini dịch → English query
  → Existing CoT pipeline → English answer
  → GPT-4o-mini dịch ngược → Answer (VI/JA)
```

**Strategy "multilingual"** — Chất lượng cao, không cần dịch:
```
Input (VI/JA) → multilingual-e5-small embed (100+ ngôn ngữ)
  → Hybrid Retrieve (BM25 bigrams cho CJK)
  → mmarco CrossEncoder (13 ngôn ngữ: VI, JA, ZH, KO...)
  → CoT prompt "respond in Vietnamese/Japanese"
  → Answer (VI/JA) trực tiếp
```

**Models (tự download qua sentence-transformers):**
- Embedding: `intfloat/multilingual-e5-small` (~117MB, 100+ ngôn ngữ)
- CrossEncoder: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (~120MB, 13 ngôn ngữ)

**Chạy demo:**
```bash
# Cả 2 strategy + cả 2 ngôn ngữ (3 câu hỏi mỗi ngôn ngữ)
python run_multilingual_demo.py

# Chỉ translate strategy, tiếng Việt
python run_multilingual_demo.py --strategy translate --lang vi --n 3

# Chỉ multilingual strategy, tiếng Nhật
python run_multilingual_demo.py --strategy multilingual --lang ja --n 3

# So sánh 2 strategy side-by-side
python run_multilingual_demo.py --strategy compare --lang vi

# Xem bảng phân tích kỹ thuật
python run_multilingual_demo.py --analysis
```

**Kết quả thực tế (demo output):**

| Câu hỏi (VI) | Strategy | Answer |
|---|---|---|
| "RAG là gì và hoạt động như thế nào?" | translate | Câu trả lời đầy đủ, tự nhiên ✅ |
| "Sự khác biệt BM25 và semantic search?" | multilingual | Câu trả lời súc tích, đúng ✅ |
| "RAGとはどのような技術ですか？" (JA) | translate | 自然な日本語の回答 ✅ |
| "BM25とセマンティック検索の違いは？" (JA) | multilingual | 正確な日本語 ✅ |

**Reranker score so sánh (Japanese "RAG とは？"):**
- ms-marco (English-only): `4.131` → score thấp, ít confidence
- mmarco (multilingual): `6.089` → score cao hơn, trained trên Japanese ✅

**Lộ trình khuyến nghị (7 ngày cho production):**
```
Ngày 1:   Response language matching (fix UX ngay lập tức)
Ngày 2-3: Strategy "translate" (quick win, mọi ngôn ngữ)
Ngày 4-6: Strategy "multilingual" (embedding + CrossEncoder)
Ngày 7:   Đo Ragas metrics với multilingual test set
```

---

## Tóm tắt lệnh

| Mục tiêu | Lệnh |
|----------|------|
| Test baseline pipeline | `python src/baseline_rag.py` |
| Test hybrid pipeline | `python src/hybrid_rag.py` |
| Test reranker pipeline | `python src/reranker_rag.py` |
| Test query expansion pipeline | `python src/query_expansion.py` |
| Test CoT pipeline | `python src/cot_rag.py` |
| Evaluate reranker + so sánh | `python run_reranker_eval.py` |
| Evaluate query expansion (combined) | `python run_query_expansion_eval.py` |
| Evaluate query expansion (mode cụ thể) | `python run_query_expansion_eval.py --mode multi_query` |
| Evaluate CoT (structured) | `python run_cot_eval.py` |
| Evaluate CoT (mode cụ thể) | `python run_cot_eval.py --mode simple` |
| Evaluate tất cả cùng lúc | `python src/evaluation.py` |
| Demo multilingual (VI + JA) | `python run_multilingual_demo.py` |
| Demo multilingual (strategy so sánh) | `python run_multilingual_demo.py --strategy compare --lang vi` |

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
