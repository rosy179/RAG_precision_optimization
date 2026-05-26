# Decision Framework: Which RAG Technique to Use?

Use this guide to choose the right technique (or combination) for your use case.

---

## Quick Decision Tree

```
START: I want to deploy a RAG system
│
├── Q1: Do I have an OpenAI API key?
│   └── No  → Use demo profile (mock mode). Limited to structure demo only.
│
├── Q2: What is the primary constraint?
│   ├── Latency (<1s response)     → Use Baseline or Balanced profile
│   ├── Cost (minimize API spend)  → Use Baseline profile
│   ├── Accuracy (maximize score)  → Use Production profile (CoT)
│   └── Balance                    → Use Balanced profile (Hybrid)
│
├── Q3: What type of queries?
│   ├── Simple factual ("Who is X?", "What year?")
│   │   → Hybrid is sufficient (0.90 accuracy)
│   ├── Multi-hop reasoning ("Why did X cause Y?")
│   │   → Reranker helps (0.91 accuracy)
│   └── Complex / hallucination-risk ("Explain the mechanism of...")
│       → CoT is best (0.94 accuracy)
│
└── Q4: Expected query volume?
    ├── <1 QPS    → Use CoT with caching (best quality)
    ├── 1-10 QPS  → Use Reranker + caching (balanced)
    └── >10 QPS   → Use Baseline/Hybrid + caching + batch processing
```

---

## Use Case Matrix

| Scenario | Recommended Profile | Accuracy | P50 Latency | Cost/Query | Config File |
|----------|:------------------:|:--------:|:-----------:|:----------:|:-----------:|
| **Internal demo** | demo | ~0.88 | <100ms | ~$0 | `config/demo.yaml` |
| **Prototyping** | baseline | ~0.88 | ~520ms | $0.000057 | `config/baseline.yaml` |
| **Web API** | balanced | ~0.90 | ~680ms | $0.000065 | `config/balanced.yaml` |
| **Production** | production | ~0.94 | ~2100ms | $0.000181 | `config/production.yaml` |
| **Batch processing** | balanced + cache | ~0.90 | ~200ms* | $0.000020* | Custom |

*with warm cache

---

## Technique Comparison

### By accuracy contribution

```
Technique         │ Adds to Baseline  │ Cumulative  │ Cost multiplier
──────────────────┼───────────────────┼─────────────┼────────────────
Baseline          │ —                 │ 0.8782      │ 1.0x
+ Hybrid Search   │ +0.0217 (+2.5%)   │ 0.8999      │ 1.1x
+ Reranker        │ +0.0119 (+1.3%)   │ 0.9118      │ 1.1x
+ CoT             │ +0.0238 (+2.6%)   │ 0.9434      │ 2.8x
+ Query Expansion │ -0.0125 (hurts)   │ 0.9071      │ 3.0x (avoid solo)
```

> **Note:** Query Expansion alone reduces accuracy vs reranked. It is more useful for
> broad/exploratory search where recall matters more than precision.

### By technique's primary benefit

| Technique | Improves Most | Why |
|-----------|:------------:|-----|
| Hybrid Search | Context Recall (+3.3%) | BM25 catches exact-match terms semantic search misses |
| Reranker | Context Precision (+8.6%) | CrossEncoder re-scores (query, passage) pairs more accurately |
| Query Expansion | Context Recall (maintains 100%) | Multi-Query / HyDE explores more of the document space |
| CoT | Faithfulness (+6.1%) | Step-by-step reasoning anchors LLM to retrieved facts |

---

## When to Enable Each Component

### Hybrid Search — enable when:
- Queries contain domain-specific terminology or acronyms
- Users mix keyword and semantic intent (most real-world cases)
- You need recall improvement without adding latency of a reranker

### Reranker — enable when:
- Precision matters more than speed (production APIs, enterprise)
- You have a wide candidate pool (RETRIEVE_DEPTH ≥ 15)
- Your query load is < 5 QPS without caching

### Query Expansion — enable when:
- Queries are short, ambiguous, or under-specified (1-3 words)
- You're building a search engine where recall is critical
- You have budget for extra LLM calls (3x cost overhead)

### Chain-of-Thought — enable when:
- Hallucination is unacceptable (medical, legal, financial)
- Faithfulness score is a primary KPI
- Questions require multi-step reasoning
- You can accept 2-3x latency overhead

---

## Configuration-Switching Example

```python
import yaml
import os
from pathlib import Path

def load_config(profile: str = "production") -> dict:
    config_path = Path(f"config/{profile}.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def build_rag_from_config(config: dict):
    retrieval_type = config["retrieval"]["type"]
    use_reranker   = config["reranking"]["enabled"]
    use_cot        = config["generation"]["use_cot"]

    if use_cot and use_reranker:
        from src.cot_rag import ChainOfThoughtRAG
        return ChainOfThoughtRAG(mode=config["generation"]["cot_mode"])
    elif use_reranker:
        from src.reranker_rag import RerankerRAG
        return RerankerRAG()
    elif retrieval_type == "hybrid":
        from src.hybrid_rag import HybridRAG
        return HybridRAG()
    else:
        from src.baseline_rag import BaselineRAG
        return BaselineRAG()

# Usage
profile = os.getenv("CONFIG_PROFILE", "production")
config  = load_config(profile)
rag     = build_rag_from_config(config).build()
```

---

## Cost-Accuracy Tradeoff Chart

```
Accuracy │
  0.95 ──┤
         │                                    ● CoT
  0.94 ──┤
         │
  0.93 ──┤
         │     ● Reranked
  0.92 ──┤
         │
  0.91 ──┤
         │                  ● Query Expansion
  0.90 ──┤
         │   ● Hybrid
  0.89 ──┤
         │
  0.88 ──● Baseline
         │
         └──────────────────────────────────────
            $0.00006  $0.00007  $0.00019  $0.00020
                       Cost per query (USD)
                             ↑
                  QE at ~$0.000195 (3x overhead)

Pareto front: Baseline → Hybrid → Reranked → CoT
(Query Expansion is off the Pareto front — high cost ~$0.000195, lower accuracy 0.9071)
```

---

## Frequently Asked Questions

**Q: Can I skip Hybrid and go directly from Baseline to CoT?**
A: Yes. `Baseline → Reranker → CoT` gives 0.93+ accuracy at lower latency than full Hybrid+Reranker+CoT.
   Run `python run_ablation_study.py` to see estimates for all combinations.

**Q: Does enabling caching affect accuracy?**
A: No. Cache stores the exact answer — accuracy is identical on cache hit. Only new queries hit the pipeline.

**Q: Query Expansion shows lower accuracy than Reranker — should I skip it?**
A: For precision-focused use cases, yes. Query Expansion excels when you need high recall
   (finding all relevant docs). Combine with Reranker to recover precision.

**Q: What's the minimum setup for a production deployment?**
A: `Hybrid + Reranker` (`config/production.yaml` with `use_cot: false`) gives 0.91 accuracy
   at ~950ms latency and is the most cost-effective production configuration.

**Q: How many queries can I serve per dollar?**

| Technique | Cost/query | Queries per $1 |
|-----------|:----------:|:--------------:|
| Baseline | ~$0.000057 | ~17,500 |
| Hybrid | ~$0.000065 | ~15,400 |
| Reranked | ~$0.000065 | ~15,400 |
| Query Expansion | ~$0.000195 | ~5,100 |
| CoT | ~$0.000181 | ~5,500 |
