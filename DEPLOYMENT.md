# Deployment Guide

This guide covers deploying the RAG Precision Optimization system to production.

---

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Docker Deployment](#2-docker-deployment)
3. [FastAPI Server](#3-fastapi-server)
4. [Configuration Profiles](#4-configuration-profiles)
5. [Load Testing](#5-load-testing)
6. [Monitoring in Production](#6-monitoring-in-production)
7. [Cost Estimation](#7-cost-estimation)

---

## 1. Environment Setup

### Requirements

```bash
# Python 3.10+
pip install -r requirements.txt
```

### Environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-...          # Required for embeddings + generation
LLM_MODEL=gpt-4o-mini          # Recommended: cost-effective, fast
EMBED_MODEL=text-embedding-ada-002
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=3
RETRIEVE_DEPTH=20
```

### Build vector index (one-time)

```bash
# Build ChromaDB index — required before serving queries
python src/baseline_rag.py
# Index saved to data/chroma_db/ — reused by all pipelines
```

---

## 2. Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build the index at container startup
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and run

```bash
docker build -t rag-precision .

docker run \
  -e OPENAI_API_KEY="sk-..." \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/logs:/app/logs \
  rag-precision
```

### Docker Compose (with volume persistence)

```yaml
version: "3.9"
services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_MODEL=gpt-4o-mini
    volumes:
      - ./data:/app/data
      - ./cache:/app/cache
      - ./logs:/app/logs
    restart: unless-stopped
```

```bash
docker compose up -d
```

---

## 3. FastAPI Server

Create `api/server.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cot_rag import ChainOfThoughtRAG
from cache import RAGCache
from resilience import ResilientRAG
from monitoring import RAGMonitor

app = FastAPI(title="RAG Precision API", version="1.0")

# Initialize once at startup
_rag     = ChainOfThoughtRAG(mode="structured")
_cache   = RAGCache(cache_dir="cache")
_monitor = RAGMonitor(log_dir="logs", pipeline="cot")
_robust  = ResilientRAG(_rag, max_retries=3, timeout=15.0, cache=_cache)


@app.on_event("startup")
async def startup():
    _rag.build()


class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    from_cache: bool = False


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    import time
    t0 = time.perf_counter()
    result = _robust.query(req.question)
    latency_ms = (time.perf_counter() - t0) * 1000

    _monitor.log_query(
        req.question, result["answer"],
        latency_ms=latency_ms,
        status="success" if not result.get("fallback") else "fallback",
    )

    return QueryResponse(
        answer=result["answer"],
        from_cache=result.get("from_cache", False),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return _monitor.summary()
```

### Run server

```bash
pip install fastapi uvicorn

# Development
uvicorn api.server:app --reload --port 8000

# Production
uvicorn api.server:app --workers 4 --port 8000
```

### Example API call

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Retrieval Augmented Generation?"}'
```

---

## 4. Configuration Profiles

Switch between profiles by setting `CONFIG_PROFILE`:

```bash
# Fastest / cheapest (demo)
export CONFIG_PROFILE=demo

# Balanced (web API)
export CONFIG_PROFILE=balanced

# Full accuracy (production)
export CONFIG_PROFILE=production
```

See [config/](config/) for YAML definitions of each profile.

| Profile | Accuracy | P50 Latency | Cost/Query | Best For |
|---------|:--------:|:-----------:|:----------:|----------|
| `demo` | ~0.88 | ~100ms | ~$0 (cached) | Presentations, CI |
| `baseline` | ~0.88 | ~520ms | ~$0.000057 | Budget API |
| `balanced` | ~0.90 | ~680ms | ~$0.000065 | Web API default |
| `production` | ~0.94 | ~2100ms | ~$0.000181 | Enterprise Q&A |

---

## 5. Load Testing

### Using locust

```bash
pip install locust
```

Create `load_test.py`:

```python
from locust import HttpUser, task, between

class RAGUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def query(self):
        self.client.post("/query", json={
            "question": "What is Retrieval Augmented Generation?"
        })
```

```bash
# Run: 50 users, ramp up 5/sec, target localhost:8000
locust -f load_test.py -u 50 -r 5 --host=http://localhost:8000
```

### Using Apache Bench (quick test)

```bash
ab -n 100 -c 10 \
   -H "Content-Type: application/json" \
   -p query.json \
   http://localhost:8000/query
```

### Expected throughput (single worker, production profile)

| Concurrency | Avg Latency | RPS |
|-------------|:-----------:|:---:|
| 1 | ~2.1s | ~0.5 |
| 5 | ~3.5s | ~1.4 |
| 10 | ~6s | ~1.6 |

For higher throughput: enable caching (`cache_enabled: true`) and use `balanced` profile.

---

## 6. Monitoring in Production

### Query logs

All queries are logged to `logs/<pipeline>_<timestamp>.jsonl`:

```json
{"ts": "2026-05-24T10:00:01", "pipeline": "cot", "question": "...",
 "answer": "...", "latency_ms": 1850.2, "tokens": 520, "cost_usd": 0.000181,
 "status": "success", "error": ""}
```

### View real-time metrics

```bash
# GET /metrics endpoint
curl http://localhost:8000/metrics
```

Response:
```json
{
  "pipeline": "cot",
  "n_queries": 150,
  "error_rate": 0.013,
  "latency_ms": {"avg": 2050, "p50": 1900, "p95": 3800, "p99": 5100},
  "total_cost_usd": 0.027,
  "qps": 0.48
}
```

### Alerts

The `RAGMonitor` logs warnings automatically when:
- Any query exceeds `alert_latency_ms` (default 5000ms)
- Rolling error rate exceeds `alert_error_rate` (default 10%)

---

## 7. Cost Estimation

Run the cost analysis tool:

```bash
python run_cost_analysis.py
```

### Quick estimates (per 1,000 queries)

| Profile | Cost | Accuracy |
|---------|:----:|:--------:|
| Baseline | ~$0.057 | 0.88 |
| Balanced | ~$0.065 | 0.90 |
| Production (CoT) | ~$0.181 | 0.94 |

### Monthly cost at different scales

| QPS | Daily Queries | Monthly Cost (production) |
|:---:|:------------:|:------------------------:|
| 0.1 | ~8,640 | ~$47 |
| 1 | ~86,400 | ~$470 |
| 10 | ~864,000 | ~$4,700 |

Enabling caching reduces repeated-query costs by 30-70%.
