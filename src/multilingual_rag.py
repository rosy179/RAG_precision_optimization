#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/multilingual_rag.py
------------------------
Multilingual RAG: Handle queries in Vietnamese, Japanese, and 100+ other languages.

Problem: Existing pipeline is English-only:
  - BM25 whitespace tokenizer fails for CJK (no spaces)
  - text-embedding-ada-002 has cross-language semantic gap
  - CrossEncoder trained on English MS MARCO only
  - CoT prompt responds in English regardless of input

Two strategies:
  "translate"     — Detect lang → translate query to EN (GPT-4o-mini)
                    → existing CoT pipeline → translate answer back
                    Fast, uses existing index, ~1-2s extra latency

  "multilingual"  — Rebuild index with multilingual-e5-small (100+ langs)
                    → language-aware BM25 tokenizer (bigrams for CJK)
                    → multilingual CrossEncoder (mmarco, 13 langs)
                    → CoT prompt instructs LLM to respond in source language
                    Best quality, no translation overhead, requires reindex

Models (auto-download via sentence-transformers):
  Embedding:   intfloat/multilingual-e5-small  (~117MB, 100+ langs)
  CrossEncoder: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (~120MB, 13 langs)
"""

import sys
import os
import re
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────
TOP_K          = int(os.getenv("TOP_K", 3))
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-4o-mini")
RETRIEVE_DEPTH = int(os.getenv("RETRIEVE_DEPTH", 20))
DB_PATH        = Path("data/chroma_db")

ML_EMBED_MODEL   = "intfloat/multilingual-e5-small"
ML_RERANKER      = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
ML_COLLECTION    = "rag_multilingual"

LANG_NAMES = {
    "vi": "Vietnamese", "ja": "Japanese",
    "zh-cn": "Chinese",  "zh-tw": "Chinese",  "ko": "Korean",
    "fr": "French",      "de": "German",       "es": "Spanish",
    "it": "Italian",     "ar": "Arabic",       "ru": "Russian",
    "pt": "Portuguese",  "id": "Indonesian",   "en": "English",
}


# ── Language Utilities ────────────────────────────────────

def detect_language(text: str) -> str:
    """Detect language code (e.g. 'vi', 'ja', 'en'). Falls back to 'en'."""
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return "en"


def tokenize_multilingual(text: str, lang: str) -> list:
    """
    Language-aware tokenizer for BM25.
    CJK languages (Japanese/Chinese/Korean) have no spaces → use character bigrams.
    Other languages → word tokenization.
    """
    if lang in ("ja", "zh", "zh-cn", "zh-tw", "ko"):
        clean = re.sub(r'\s+', '', text.lower())
        unigrams = list(clean)
        bigrams  = [clean[i:i+2] for i in range(len(clean) - 1)]
        return unigrams + bigrams
    # Vietnamese and most European languages use spaces
    return re.findall(r'\w+', text.lower())


# ── Translation Helpers ───────────────────────────────────

def _call_llm_translate(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return ""
    from openai import OpenAI
    resp = OpenAI(api_key=api_key).chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def translate_to_english(text: str, source_lang_name: str) -> str:
    prompt = f"""Translate the following {source_lang_name} text to English.
Return ONLY the translated text — no explanation, no quotes.

{source_lang_name} text: {text}

English translation:"""
    result = _call_llm_translate(prompt)
    return result if result else text


def translate_from_english(text: str, target_lang_name: str) -> str:
    prompt = f"""Translate the following English text to {target_lang_name}.
Return ONLY the translated text — no explanation, no quotes.

English text: {text}

{target_lang_name} translation:"""
    result = _call_llm_translate(prompt)
    return result if result else text


# ── CoT Generation (Multilingual-aware) ──────────────────

MULTILINGUAL_COT_PROMPT = """\
You are a precise QA assistant. Use ONLY the provided context to answer the question.
The question is in {lang_name}. Write all content in {lang_name} EXCEPT the structural labels below.

Context:
{context}

Question: {question}

Follow this EXACT structure (keep labels in English, write content in {lang_name}):

Step 1 - Relevant Facts:
[list relevant facts from context here, in {lang_name}]

Step 2 - Reasoning:
[reason from the facts, in {lang_name}]

Final Answer:
[your concise answer in {lang_name} — this label MUST stay "Final Answer:" in English]

If the context lacks sufficient information write after "Final Answer:": I don't have enough information.

Step 1 - Relevant Facts:"""


def _build_context_string(context_chunks: list) -> str:
    return "\n\n---\n\n".join(
        f"[Source {c['rank']}: {c['title']}]\n{c['text']}"
        for c in context_chunks
    )


def _extract_final_answer(response: str) -> str:
    marker = "Final Answer:"
    if marker in response:
        return response.split(marker, 1)[-1].strip()
    # Fallback: last non-empty paragraph
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else response.strip()


def generate_multilingual_cot(question: str, context_chunks: list,
                               lang_name: str) -> tuple:
    """Generate CoT answer explicitly in the detected language."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your"):
        return "[MOCK] No API key set.", "[MOCK]"

    from openai import OpenAI
    context_str = _build_context_string(context_chunks)
    prompt = MULTILINGUAL_COT_PROMPT.format(
        lang_name=lang_name,
        context=context_str,
        question=question,
    )
    resp = OpenAI(api_key=api_key).chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=700,
    )
    reasoning = resp.choices[0].message.content.strip()
    final_answer = _extract_final_answer(reasoning)
    return final_answer, reasoning


# ── Main Class ────────────────────────────────────────────

class MultilingualRAG:
    """
    Multilingual RAG pipeline supporting Vietnamese, Japanese, and 100+ languages.

    strategy="translate":
      Uses existing English CoT pipeline (Hybrid + Rerank + CoT).
      Adds a translation layer before and after retrieval.
      Fastest path; no reindexing required.

    strategy="multilingual":
      Rebuilds index with multilingual-e5-small embeddings.
      Uses language-aware BM25 tokenizer (bigrams for CJK).
      Uses multilingual CrossEncoder (mmarco, 13 languages).
      CoT prompt explicitly requests response in source language.
      Best quality for cross-language retrieval.
    """

    def __init__(self, strategy: str = "translate"):
        if strategy not in ("translate", "multilingual"):
            raise ValueError("strategy must be 'translate' or 'multilingual'")
        self.strategy    = strategy
        self._cot_rag    = None    # translate strategy
        self._embed_model = None   # multilingual strategy
        self._collection  = None
        self._chunks      = []
        self._bm25        = None
        self._reranker    = None

    # ── Build ──────────────────────────────────────────────

    def build(self):
        src_dir = Path(__file__).parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        if self.strategy == "translate":
            self._build_translate()
        else:
            self._build_multilingual()
        return self

    def _build_translate(self):
        from cot_rag import ChainOfThoughtRAG
        print("[ML] Strategy: translate — using existing CoT pipeline")
        self._cot_rag = ChainOfThoughtRAG(mode="structured")
        self._cot_rag.build()
        print("[ML] Translate strategy ready. (Query: detect → translate → CoT → translate)")

    def _build_multilingual(self):
        from baseline_rag import load_documents, chunk_documents
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from rank_bm25 import BM25Okapi
        import chromadb

        print("[ML] Strategy: multilingual — building new index with multilingual embeddings")

        # 1. Load and chunk documents
        docs = load_documents()
        self._chunks = chunk_documents(docs)

        # 2. Load multilingual embedding model
        print(f"\n[ML1] Loading embedding model: {ML_EMBED_MODEL}")
        self._embed_model = SentenceTransformer(ML_EMBED_MODEL)

        # 3. Embed all chunks with "passage: " prefix (required by E5 models)
        print(f"[ML2] Embedding {len(self._chunks)} chunks...")
        passage_texts = [f"passage: {c['text']}" for c in self._chunks]
        embeddings = self._embed_model.encode(
            passage_texts, normalize_embeddings=True,
            show_progress_bar=True, batch_size=64,
        )

        # 4. Build ChromaDB collection (manual embeddings — no embedding_function)
        print(f"[ML3] Building vector store: collection='{ML_COLLECTION}'")
        client = chromadb.PersistentClient(path=str(DB_PATH))
        try:
            client.delete_collection(ML_COLLECTION)
        except Exception:
            pass
        self._collection = client.create_collection(
            name=ML_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        batch = 200
        for i in range(0, len(self._chunks), batch):
            bc = self._chunks[i:i+batch]
            be = embeddings[i:i+batch]
            self._collection.add(
                ids=[c["id"] for c in bc],
                documents=[c["text"] for c in bc],
                metadatas=[{
                    "title": c["title"], "source": c["source"], "doc_id": c["doc_id"]
                } for c in bc],
                embeddings=be.tolist(),
            )
        print(f"       {self._collection.count()} vectors indexed")

        # 5. Build BM25 index with English tokenizer (documents are English)
        print("[ML4] Building BM25 index...")
        tokenized = [tokenize_multilingual(c["text"], "en") for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

        # 6. Load multilingual CrossEncoder
        print(f"[ML5] Loading CrossEncoder: {ML_RERANKER}")
        self._reranker = CrossEncoder(ML_RERANKER)

        print("\n[ML] Multilingual full-stack pipeline ready!")
        print(f"     Embedding: {ML_EMBED_MODEL}")
        print(f"     CrossEncoder: {ML_RERANKER}")

    # ── Retrieval (multilingual strategy) ─────────────────

    def _retrieve_ml(self, query: str, lang: str,
                     retrieve_depth: int = RETRIEVE_DEPTH) -> list:
        from hybrid_rag import reciprocal_rank_fusion

        # Semantic: embed with "query: " prefix, query manually
        query_vec = self._embed_model.encode(
            f"query: {query}", normalize_embeddings=True
        )
        results = self._collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=retrieve_depth,
        )
        semantic_list = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            semantic_list.append({
                "id":       results["ids"][0][i],
                "text":     doc,
                "title":    meta.get("title", ""),
                "source":   meta.get("source", ""),
                "distance": dist,
            })

        # BM25: language-aware tokenizer for the query
        tok_q = tokenize_multilingual(query, lang)
        bm25_scores = self._bm25.get_scores(tok_q)
        scored = sorted(zip(bm25_scores, self._chunks),
                        key=lambda x: x[0], reverse=True)
        lexical_list = [c for score, c in scored[:retrieve_depth] if score > 0.0]

        # RRF fusion
        return reciprocal_rank_fusion(semantic_list, lexical_list,
                                      k=60, top_k=retrieve_depth)

    def _rerank_ml(self, query: str, candidates: list, top_k: int = TOP_K) -> list:
        if not candidates:
            return candidates
        pairs  = [(query, c["text"]) for c in candidates]
        scores = self._reranker.predict(pairs)
        for i, c in enumerate(candidates):
            c["reranker_score"] = round(float(scores[i]), 6)
        reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)
        for i, c in enumerate(reranked[:top_k], 1):
            c["rank"] = i
        return reranked[:top_k]

    # ── Query ─────────────────────────────────────────────

    def query(self, question: str,
              top_k: int = TOP_K,
              retrieve_depth: int = RETRIEVE_DEPTH,
              verbose: bool = True) -> dict:

        lang      = detect_language(question)
        lang_name = LANG_NAMES.get(lang, lang.upper())

        if verbose:
            print(f"\n{'='*65}")
            print(f"Q: {question}")
            print(f"{'='*65}")
            print(f"[ML] Language detected: {lang_name} ({lang}) | strategy={self.strategy}")

        # English: no translation overhead — route directly
        if lang == "en":
            if self.strategy == "translate":
                result = self._cot_rag.query(
                    question, top_k=top_k, retrieve_depth=retrieve_depth, verbose=False
                )
            else:
                ctx = self._retrieve_ml(question, "en", retrieve_depth)
                ctx = self._rerank_ml(question, ctx, top_k)
                answer, reasoning = generate_multilingual_cot(question, ctx, "English")
                result = {"question": question, "answer": answer,
                          "reasoning": reasoning, "context": ctx, "top_k": top_k}
            result.update({"detected_lang": "en", "strategy": self.strategy})
            if verbose:
                print(f"A: {result['answer']}")
            return result

        # Non-English
        if self.strategy == "translate":
            return self._query_translate(question, lang, lang_name, top_k, retrieve_depth, verbose)
        else:
            return self._query_multilingual(question, lang, lang_name, top_k, retrieve_depth, verbose)

    def _query_translate(self, question, lang, lang_name,
                         top_k, retrieve_depth, verbose):
        # Step 1 — translate query → English
        print(f"[T1] Translating query to English...")
        en_query = translate_to_english(question, lang_name)
        print(f"     → {en_query}")

        # Step 2 — English CoT pipeline
        result = self._cot_rag.query(
            en_query, top_k=top_k, retrieve_depth=retrieve_depth, verbose=False
        )
        en_answer = result["answer"]

        # Step 3 — translate answer → source language
        print(f"[T2] Translating answer back to {lang_name}...")
        native_answer = translate_from_english(en_answer, lang_name)

        if verbose:
            print(f"\nA ({lang_name}):\n{native_answer}")
            print(f"\n[EN source answer: {en_answer[:200]}...]")
            print(f"\n--- Top {len(result['context'])} passages retrieved ---")
            for c in result["context"]:
                rer = c.get("reranker_score", "N/A")
                rer_str = f"{rer:.3f}" if isinstance(rer, float) else str(rer)
                print(f"  [{c['rank']}] {c['title']} (reranker={rer_str})")
                print(f"       {c['text'][:100]}...")

        result["answer"]        = native_answer
        result["answer_en"]     = en_answer
        result["query_en"]      = en_query
        result["detected_lang"] = lang
        result["lang_name"]     = lang_name
        result["strategy"]      = "translate"
        return result

    def _query_multilingual(self, question, lang, lang_name,
                             top_k, retrieve_depth, verbose):
        # Step 1 — multilingual retrieval
        candidates = self._retrieve_ml(question, lang, retrieve_depth)
        context    = self._rerank_ml(question, candidates, top_k)

        # Step 2 — CoT generate in source language
        answer, reasoning = generate_multilingual_cot(question, context, lang_name)

        if verbose:
            print(f"\nA ({lang_name}):\n{answer}")
            print(f"\n--- Top {len(context)} passages (multilingual stack) ---")
            for c in context:
                rer = c.get("reranker_score", 0)
                print(f"  [{c['rank']}] {c['title']} (reranker={rer:.3f})")
                print(f"       {c['text'][:100]}...")

        return {
            "question":      question,
            "answer":        answer,
            "reasoning":     reasoning,
            "context":       context,
            "detected_lang": lang,
            "lang_name":     lang_name,
            "strategy":      "multilingual",
            "top_k":         top_k,
        }


# ── Quick Test ────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Multilingual RAG -- Quick Test")
    print("=" * 65)

    # Test translate strategy (faster to build)
    rag = MultilingualRAG(strategy="translate").build()

    test_questions = [
        ("vi", "RAG là gì và hoạt động như thế nào?"),
        ("vi", "Sự khác biệt giữa BM25 và tìm kiếm ngữ nghĩa là gì?"),
        ("ja", "RAGとはどのような技術ですか？"),
        ("ja", "密な文章検索はどのように機能しますか？"),
    ]

    results = []
    for lang, q in test_questions:
        r = rag.query(q)
        results.append(r)

    out = Path("results/multilingual_test_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {out}")
    print("Multilingual RAG pipeline working!")
