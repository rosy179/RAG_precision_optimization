# RAG Precision Optimization

> **Goal:** Improve RAG system accuracy from baseline ~88% to 95%+ using 5 advanced techniques over 45 days.
> **Result:** Achieved **0.9434** (94.34%) with Chain-of-Thought pipeline — a +7.4% gain over baseline.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  RETRIEVAL STAGE                    │
├─────────────────────────────────────────────────────┤
│  BM25 (keyword)  +  Semantic (dense)  +  HyDE/MQ   │
│             ↓               ↓              ↓        │
│         Reciprocal Rank Fusion (RRF)                │
│                  ↓ (20 candidates)                  │
├─────────────────────────────────────────────────────┤
│                  RANKING STAGE                      │
├─────────────────────────────────────────────────────┤
│       CrossEncoder Reranking (local model)          │
│                  ↓ (top 3 passages)                 │
├─────────────────────────────────────────────────────┤
│                 GENERATION STAGE                    │
├─────────────────────────────────────────────────────┤
│   LLM + Chain-of-Thought Structured Prompt          │
│   Step 1: Extract facts → Step 2: Reason → Answer  │
│                  ↓                                  │
│              Final Answer                           │
└─────────────────────────────────────────────────────┘
```

---

## Performance at a Glance

```
Accuracy │
0.9434  ─────────────────────────────● CoT Structured  ← BEST
0.9228  ──────────────────────────● Adaptive
0.9196  ────────────────────────● Reranked
0.9071  ──────────────────────● Query Expansion
0.8999  ────────────────────● Hybrid Search
0.8782  ●─────────────────────────── Baseline
        └──────────────────────────────────────────
               Each step builds on the previous
```

### Cost vs Accuracy Tradeoff

| Technique | Accuracy | Latency (P50) | Cost/Query | Best For |
|-----------|:--------:|:-------------:|:----------:|----------|
| Baseline | 0.8782 | ~520ms | ~$0.000057 | Demo, budget-constrained |
| Hybrid | 0.8999 | ~680ms | ~$0.000065 | Web API, balanced |
| Reranked | 0.9196 | ~950ms | ~$0.000065 | **Production default** |
| Query Expansion | 0.9071 | ~1800ms | ~$0.000195 | Broad search |
| CoT | **0.9434** | ~2100ms | ~$0.000181 | High-stakes, hallucination-critical |

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
│   ├── query_expansion.py        # Day 15-17: Multi-Query + HyDE expansion
│   ├── cot_rag.py                # Day 22-24: Chain-of-Thought generation
│   ├── adaptive_rag.py           # Day 18-20: Adaptive top_k by query complexity
│   ├── multilingual_rag.py       # Bonus: Multilingual support (VI, JA, 100+ langs)
│   ├── cost_analyzer.py          # Latency, token counting, API cost estimation
│   ├── error_analyzer.py         # Failure mode classification (hallucination, etc.)
│   ├── cache.py                  # Disk-backed cache (embeddings, responses, reranker)
│   ├── resilience.py             # Retry + timeout + fallback wrapper
│   └── monitoring.py             # Query logging and latency/cost monitoring
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
├── config/                        # YAML configuration profiles
│   ├── baseline.yaml              # Fast, minimal cost
│   ├── balanced.yaml              # Hybrid retrieval (recommended API default)
│   ├── production.yaml            # Full stack: Hybrid + Reranker + CoT
│   └── demo.yaml                  # Lightweight demo (no API key needed)
│
├── run_reranker_eval.py           # Run + evaluate Reranker pipeline
├── run_query_expansion_eval.py    # Run + evaluate Query Expansion pipeline
├── run_cot_eval.py                # Run + evaluate Chain-of-Thought pipeline
├── run_adaptive_eval.py           # Run + evaluate Adaptive Retrieval pipeline
├── run_multilingual_demo.py       # Demo multilingual RAG (Vietnamese + Japanese)
├── run_cost_analysis.py           # Latency + token + API cost analysis per technique
├── run_error_analysis.py          # Error classification and failure mode analysis
├── run_ablation_study.py          # Technique combination ablation study
├── collect_dataset.py             # Data collection: SQUAD + Wikipedia
├── collect_arxiv.py               # Data collection: ArXiv papers (with retry)
├── collect_arxiv_hf.py            # Data collection: ArXiv via HuggingFace datasets
├── download_rag_data.py           # Interactive data downloader
├── align_dataset.py               # Merge SQUAD contexts into documents
├── requirements.txt
├── .env.example                   # Copy to .env and add API keys
├── HOW_TO_RUN.md                  # Step-by-step guide for each technique
├── RESEARCH_REPORT.md             # Full research report (Vietnamese)
├── PRESENTATION_SLIDES.md         # Marp presentation slides (15 slides)
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
| **Week 3** | Query Expansion (Multi-Query + HyDE) | ✅ Complete | 0.9071 |
| **Week 4** | Chain-of-Thought (CoT) Generation | ✅ Complete | **0.9434** |
| **Week 5-6** | Documentation + Presentation | ✅ Complete | — |

---

## Actual Results (Ragas, 30 SQUAD test samples)

| Technique | Faithfulness | Relevancy | Precision | Recall | **AVG** |
|-----------|:-----------:|:---------:|:---------:|:------:|:-------:|
| Baseline RAG | 0.8389 | 0.8405 | 0.9000 | 0.9333 | 0.8782 |
| + Hybrid Search | 0.8833 | 0.8717 | 0.8778 | 0.9667 | 0.8999 |
| + Reranking | 0.8389 | 0.8755 | **0.9639** | **1.0000** | **0.9196** |
| + Query Expansion (combined) | 0.8222 | 0.8423 | **0.9639** | **1.0000** | 0.9071 |
| + CoT Structured | **0.9000** | **0.9097** | **0.9639** | **1.0000** | **0.9434** |
| + Adaptive Retrieval | 0.8333 | 0.8939 | **0.9639** | **1.0000** | 0.9228 |
| **Best Overall** | **0.9000** | **0.9097** | **0.9639** | **1.0000** | **0.9434** |

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

---

## Analysis Tools

```bash
# Cost & latency analysis (estimate mode — no API calls needed)
python run_cost_analysis.py

# Live measurement (API key required)
python run_cost_analysis.py --live --n 10

# Error classification: hallucination, low relevance, incomplete
python run_error_analysis.py

# Ablation study: find optimal technique combination
python run_ablation_study.py
```

---

## Production Modules

| Module | Purpose |
|--------|---------|
| `src/cache.py` | Disk-backed cache for embeddings, responses, reranker scores |
| `src/resilience.py` | Retry + exponential backoff + timeout + fallback |
| `src/monitoring.py` | Per-query latency/cost logging with P50/P95/P99 stats |

```python
from src.cot_rag import ChainOfThoughtRAG
from src.cache import RAGCache
from src.resilience import ResilientRAG
from src.monitoring import RAGMonitor

rag     = ChainOfThoughtRAG().build()
cache   = RAGCache(cache_dir="cache")
monitor = RAGMonitor(log_dir="logs", pipeline="cot")
robust  = ResilientRAG(rag, max_retries=3, timeout=15.0, cache=cache)

result = robust.query("What is RAG?")
monitor.log_query("What is RAG?", result["answer"], latency_ms=450)
monitor.print_summary()
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `OPENAI_API_KEY` not set | Copy `.env.example` → `.env` and set your key |
| ChromaDB path error | Delete `data/chroma_db/` and re-run any pipeline to rebuild |
| `cross-encoder` model download fails | Run `pip install sentence-transformers` and check internet |
| `ragas` import error | Run `pip install ragas==0.1.21 datasets langchain-openai` |
| Slow first run | CrossEncoder model (~85MB) downloads once automatically |
| Out of memory | Reduce `CHUNK_SIZE` in `.env` (default 512, try 256) |

---

## References

- [RAGAS Paper](https://arxiv.org/abs/2309.15217) — Evaluation framework
- [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906)
- [HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496)
- [Self-RAG](https://arxiv.org/abs/2310.11511)
- [ms-marco CrossEncoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
