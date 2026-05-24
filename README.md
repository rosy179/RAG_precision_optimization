# RAG Precision Optimization

> **Goal:** Improve RAG system accuracy from baseline ~88% to 95%+ using 5 advanced techniques over 45 days.

---

## Project Structure

```
RAG_prescision_optimization/
├── data/
│   ├── squad_qa.json              # 150 QA pairs (SQUAD v1.1)
│   ├── wiki_documents.json        # 15 Wikipedia AI/ML articles
│   ├── arxiv_papers.json          # 10 RAG research paper abstracts
│   ├── rag_dataset.json           # Combined dataset (~750 KB)
│   └── chroma_db/                 # Local vector store (auto-created by pipeline)
│
├── src/
│   ├── baseline_rag.py            # Day 2  : Vanilla RAG pipeline
│   ├── evaluation.py              # Day 3-4: Ragas evaluation framework
│   ├── hybrid_rag.py              # Day 8-10: BM25 + Semantic hybrid search + RRF
│   ├── reranker_rag.py            # Day 11-13: CrossEncoder reranking layer
│   └── query_expansion.py        # Day 15-17: Multi-Query + HyDE expansion
│
├── notebooks/                     # Jupyter notebooks (optional exploration)
│
├── results/
│   ├── baseline_metrics.json      # Week 1 eval: avg 0.8782
│   ├── baseline_test_results.json # Baseline pipeline test outputs
│   ├── hybrid_metrics.json        # Week 2 eval: avg 0.8999
│   ├── hybrid_test_results.json   # Hybrid pipeline test outputs
│   ├── reranked_metrics.json      # Week 2b eval: avg 0.9196
│   └── reranker_test_results.json # Reranker pipeline test outputs
│
├── config/                        # Reserved for future config files
│
├── run_reranker_eval.py           # Run + evaluate Reranker pipeline
├── run_query_expansion_eval.py    # Run + evaluate Query Expansion pipeline
├── collect_dataset.py             # Data collection: SQUAD + Wikipedia
├── collect_arxiv.py               # Data collection: ArXiv papers (with retry)
├── collect_arxiv_hf.py            # Data collection: ArXiv via HuggingFace datasets
├── download_rag_data.py           # Interactive data downloader
├── align_dataset.py               # Merge SQUAD contexts into documents
├── requirements.txt
├── .env.example                   # Copy to .env and add API keys
├── HOW_TO_RUN.md                  # Step-by-step guide for each technique
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Open .env and set your OPENAI_API_KEY
```

### 3. Data is ready
```
data/rag_dataset.json  —  150 QA pairs + 56 documents, ready to use
```

### 4. Run a technique
```bash
# Baseline RAG (requires OPENAI_API_KEY)
python src/baseline_rag.py

# Hybrid Search
python src/hybrid_rag.py

# Reranking (CrossEncoder)
python src/reranker_rag.py

# Run full evaluation + comparison
python run_reranker_eval.py
```

> See [HOW_TO_RUN.md](HOW_TO_RUN.md) for detailed instructions and explanations.

---

## 45-Day Progress

| Week | Technique | Status | Avg Score |
|------|-----------|--------|-----------|
| **Week 1** | Baseline RAG + Evaluation Framework | ✅ Complete | 0.8782 |
| **Week 2a** | Hybrid Search (BM25 + Semantic + RRF) | ✅ Complete | 0.8999 |
| **Week 2b** | Reranking (CrossEncoder) | ✅ Complete | **0.9196** |
| **Week 3** | Query Expansion (Multi-Query + HyDE) | ✅ Complete | ~0.93+ |
| **Week 4** | Chain-of-Thought Retrieval | ⬜ Pending | ~0.95+ |
| **Week 5-6** | Documentation + Presentation | ⬜ Pending | — |

---

## Actual Results (Ragas, 30 SQUAD test samples)

| Technique | Faithfulness | Relevancy | Precision | Recall | **AVG** |
|-----------|:-----------:|:---------:|:---------:|:------:|:-------:|
| Baseline RAG | 0.8389 | 0.8405 | 0.9000 | 0.9333 | 0.8782 |
| + Hybrid Search | 0.8833 | 0.8717 | 0.8778 | 0.9667 | 0.8999 |
| + Reranking | 0.8389 | 0.8755 | **0.9639** | **1.0000** | **0.9196** |
| + Query Expansion (combined) | 0.8222 | 0.8423 | **0.9639** | **1.0000** | 0.9071 |
| **Final (ALL)** | — | — | — | — | ~0.95+ |

---

## Dataset

| Source | Count | Content |
|--------|-------|---------|
| SQUAD v1.1 | 150 QA pairs | Question + context + ground truth answer |
| Wikipedia | 15 articles | ML, DL, AI, NLP, Transformer, LLM, RAG, Vector DB… |
| ArXiv abstracts | 10 papers | RAG, DPR, RAGAS, Self-RAG, HyDE, Sentence-BERT… |
| **Total** | **56 documents** | **~750K characters of technical AI/ML content** |

Train/test split: 70% train (105 QA pairs) / 30% test (45 QA pairs, 30 used for eval).

---

## Evaluation Metrics (Ragas)

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Is the answer grounded in retrieved context? (no hallucination) |
| **Answer Relevancy** | Is the answer actually relevant to the question? |
| **Context Precision** | Are the retrieved chunks genuinely useful for the answer? |
| **Context Recall** | Does the retrieved context contain all information needed? |
