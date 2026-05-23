# RAG Precision Optimization

> **Goal:** Improve RAG system accuracy from ~68% to 85-90% faithfulness using 5 advanced techniques.

---

## Project Structure

```
RAG_prescision_optimization/
├── data/
│   ├── squad_qa.json          # 150 QA pairs (SQUAD v1.1)
│   ├── wiki_documents.json    # 15 Wikipedia AI/ML articles
│   ├── arxiv_papers.json      # 10 RAG research paper abstracts
│   ├── rag_dataset.json       # Combined dataset (734 KB)
│   └── chroma_db/             # Local vector store (auto-created)
├── src/
│   ├── baseline_rag.py        # Day 2: Vanilla RAG pipeline
│   ├── evaluation.py          # Day 3-4: Ragas evaluation framework
│   ├── hybrid_retriever.py    # Week 2: BM25 + Semantic hybrid search
│   ├── reranker.py            # Week 2: CrossEncoder reranking
│   ├── query_expansion.py     # Week 3: Multi-query + HyDE
│   └── utils.py               # Shared utilities
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_hybrid_search.ipynb
│   └── 03_evaluation.ipynb
├── results/
│   └── *_metrics.json         # Evaluation results per technique
├── config/
├── collect_dataset.py         # Step 1: SQUAD + Wikipedia collection
├── collect_arxiv_hf.py        # Step 1b: ArXiv papers
├── requirements.txt
├── .env.example               # Copy to .env and add API keys
└── README.md
```

---

## Quick Start

### 1. Setup environment
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Data is already collected
```
data/rag_dataset.json  -- 150 QA pairs + 25 documents ready
```

### 3. Run baseline RAG (mock mode, no API key needed)
```bash
cd src
python baseline_rag.py
```

### 4. Run with real API key
```bash
# Set OPENAI_API_KEY in .env, then:
cd src
python baseline_rag.py     # Test pipeline
python evaluation.py       # Ragas evaluation (30 samples)
```

---

## 45-Day Progress

| Week | Focus | Status |
|------|-------|--------|
| **Week 1** | Data + Baseline RAG + Evaluation framework | ✅ In Progress |
| **Week 2** | Hybrid Search (BM25 + Semantic) + Reranking | ⬜ Pending |
| **Week 3** | Query Expansion + Adaptive Context | ⬜ Pending |
| **Week 4** | Chain-of-Thought Retrieval + Evaluation | ⬜ Pending |
| **Week 5-6** | Documentation + Presentation | ⬜ Pending |

---

## Expected Results

| Technique | Faithfulness | Relevancy | vs Baseline |
|-----------|-------------|-----------|-------------|
| Baseline | 0.68 | 0.65 | — |
| + Hybrid Search | 0.73 | 0.70 | +5-10% |
| + Reranking | 0.76 | 0.75 | +8-12% |
| + Query Expansion | 0.78 | 0.77 | +6-10% |
| **Final (ALL)** | **0.83** | **0.83** | **+22%** |

---

## Dataset Summary

- **150 QA pairs** — SQUAD v1.1 (question + context + ground truth answer)
- **15 Wikipedia docs** — Machine Learning, Deep Learning, AI, NLP, Transformer, LLM, RAG, Vector DB...
- **10 ArXiv papers** — RAG, DPR, RAGAS, Self-RAG, HyDE, Sentence-BERT, Hybrid Search...
- **Total:** ~550K characters of technical AI/ML content

---

## Evaluation Metrics (Ragas)

| Metric | Meaning |
|--------|---------|
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer Relevancy** | Is the answer relevant to the question? |
| **Context Precision** | Are retrieved chunks actually useful? |
| **Context Recall** | Does context contain all info needed? |
