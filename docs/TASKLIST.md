# 📋 Task List — Lộ trình sau đợt rà soát 18/07/2026

> Thứ tự sắp theo nguyên tắc đã thống nhất: **bảo mật/đúng đắn → thay đổi buộc re-index (làm khi corpus còn nhỏ) → tính năng → kỹ thuật RAG nâng cao → mở rộng**.
> Mỗi task ghi: độ ưu tiên (🔴 cao / 🟡 vừa / 🟢 thấp), công sức ước tính (S ≤ nửa ngày, M = 1–2 ngày, L ≥ 3 ngày), và phụ thuộc.

---

## Giai đoạn A — Dọn nốt nền tảng (không phụ thuộc gì, làm song song được)

- [x] **A1. Vá SSRF + giới hạn upload** — 🔴 S — *xong 18/07* (README §12.1)
- [x] **A2. Cache BM25 + xóa dead code** — 🔴 S — *xong 18/07* (README §12.2)
- [x] **A3. LRU + lock cho instance RAG** — 🔴 S — *xong 18/07* (README §12.3)
- [x] **A4. Gộp `condense` + `translate` thành 1 LLM call** — 🔴 S — *xong 18/07* (README §12.4)
  - `_prepare_queries()` trả JSON `{query, query_en}` trong 1 call (skip hẳn khi câu hỏi tiếng Anh + không có history). Sau B2 (reranker đa ngôn ngữ) có thể bỏ luôn phần translate.
- [x] **A5. Nối semantic cache vào webapp** — 🟡 M — *xong 18/07* (README §12.4)
  - Answer cache theo scope (session, bộ lọc nguồn, KB): exact-match trước, embedding cosine ≥ 0.95 sau; invalidate theo version tài liệu phiên + version KB chung; TTL 1h, LRU 50.
- [x] **A6. Đồng nhất endpoint `/chat` non-stream với `/chat/stream`** — 🟡 S — *xong 18/07*
  - Đã chọn phương án **xóa** endpoint non-stream (frontend không còn gọi — chỉ dùng `chatStream`); xóa cả `UserRAGService.query()` và hàm client tương ứng.
- [x] **A7. Đếm token thật thay vì ước lượng** — 🟢 S — *xong 18/07*
  - `stream_options={"include_usage": True}` + gom usage của mọi call LLM trong request (router, hop, query-prep, generation) vào QueryLog; công thức ước lượng chỉ còn là fallback.
- [x] **A8. Thay `print()` bằng module `logging`** — 🟢 S — *xong 18/07*
  - Logger `rag.user` / `rag.kb`, `basicConfig` có timestamp trong `main.py`.

## Giai đoạn B — Nâng cấp đa ngôn ngữ (LÀM TRƯỚC KHI INGEST THÊM TÀI LIỆU — vì phải re-embed toàn bộ)

- [x] **B1. Đổi embedding model** — 🔴 M — *xong 19/07*
  - Chọn **BGE-M3** (local, miễn phí, đa ngôn ngữ; query embed ~0.5s trên CPU). Tạo module dùng chung `backend/services/embeddings.py` (một factory cho cả user collections + global KB, hết trùng lặp logic chọn model); `EMBED_MODEL` đổi qua `.env`. Script `scripts/reembed_collections.py` tự backup thư mục Chroma rồi rebuild toàn bộ — đã chạy: 392 rows / 42 collections, backup tại `data/chroma_db_users_backup_20260719_085449`.
- [x] **B2. Đổi reranker đa ngôn ngữ + bỏ dịch query** — 🔴 S — *xong 19/07*
  - **Phát hiện khi đo thực tế:** `bge-reranker-v2-m3` (568M) mất **57s/query trên CPU 4 threads** → không dùng được trên máy này. Default chuyển sang **`mmarco-mMiniLMv2-L12-H384-v1`** (~120MB, 14 ngôn ngữ gồm VI+JA, ~4s/16 cặp, phân biệt rel/irr tốt cả 3 thứ tiếng); `bge-reranker-v2-m3` vẫn chọn được qua `RERANKER_MODEL` khi deploy máy GPU. Đã bỏ cơ chế dịch query EN: `_prepare_queries` (JSON 2 khóa) → `_condense_query` (chỉ condense follow-up) — bớt 1 LLM call, `retrieve()` gọn 1 query xuyên suốt. Heuristic phân loại độ phức tạp thêm tín hiệu VI/JA (fix `\b` không match tiếng Nhật). Ngưỡng HyDE 0.5 → 0.4 theo thang cosine BGE-M3. Latency retrieval: 40–52s → **1.3–5.2s/câu**.
- [x] **B3. Chạy lại eval 120 mẫu EN/VI/JA, so trước/sau** — 🔴 S — *xong 19/07*
  - Cùng judge (ada-002): tổng thể 0.9445 → **0.9498**; EN 0.9536 → **0.9624**; JA 0.8795 → **0.8929**; VI 0.8942 → 0.8700 (faithfulness VI 0.875 → **0.926** ↑, nhưng precision 0.97 → 0.82 ↓ vì BM25 tiếng Việt không còn bản dịch EN để khớp corpus). Latency −26% (6952 → 5142ms). Kết quả 2 phía: `webapp_eval_results_ada002.json` / `webapp_eval_results_bgem3_mmarco.json`.
  - Bài học phương pháp đo: judge embeddings của Ragas từng ăn theo `EMBED_MODEL` → vỡ khi trỏ model local; đổi judge sang `3-small` làm lệch thang cosine ~0.1 → đã **ghim `ada-002`** trong `evaluation.py` (override qua `RAGAS_EMBED_MODEL`). Thêm `--rescore` + sidecar `webapp_eval_rows_last.json` để chấm lại không cần trả lời lại.
  - **Nợ kỹ thuật VI-precision** (0.97 → 0.82): thử `bge-reranker-v2-m3` khi có GPU, hoặc D1 (contextual retrieval) — cả hai đều đánh trực diện vào precision.
- [x] **B4. Cập nhật README + memory với kết quả B3** — 🟡 S — *xong 19/07*
  - README: tiêu đề 95.0%, KPI, timeline tuần 7, §7 (chart + bảng trước/sau), §10, §12.5 (nhật ký kỹ thuật B1–B3), Tham khảo. Kèm `.env.example`, `docs/DEPLOYMENT.md`, memory.

## Giai đoạn C — Tính năng người dùng (sau B để tài liệu mới chỉ embed 1 lần)

- [x] **C1. Suggested follow-up questions** — 🔴 S — *xong 19/07*
  - `suggest_questions()` (user_rag.py): 1 call gpt-4o-mini (`response_format=json_object`) sinh 3 câu hỏi từ câu hỏi + câu trả lời + snippet nguồn, **cùng ngôn ngữ câu gốc** và chỉ hỏi điều trả lời được từ nguồn. Chạy **sau** event `done` nên không làm chậm câu trả lời (event SSE mới `suggestions`); lưu vào cột `messages.suggestions_json` (migration) + answer cache để reload/cache-hit vẫn hiện. Frontend: chip `Sparkles` bấm-để-hỏi dưới MessageBubble (chỉ trên câu trả lời cuối), `askSuggestion()` ở ChatPage. Đã E2E qua endpoint SSE (thứ tự meta→done→suggestions, persist OK) + tsc sạch + 26/26 unit test.
- [x] **C2. Hỗ trợ DOCX / PPTX / XLSX / Markdown** — 🔴 M — *xong 19/07*
  - Handler `process_docx` (đoạn văn + bảng), `process_pptx` (text frame + bảng + ghi chú, đánh dấu `### Slide n`), `process_xlsx` (mỗi sheet → hàng `col | col`), `process_txt` mở rộng cho `.md/.markdown`. `ALLOWED_EXTENSIONS` + dispatch trong `process_file` cập nhật. Viewer dùng text mode dựng lại từ chunk (không cần lưu file gốc). Frontend: mở rộng `DOC_ACCEPT`/`FILE_ACCEPT`, icon `FileSpreadsheet`/`Presentation`, nhãn loại. Đã test file thật cả 4 định dạng (giữ nguyên tiếng Việt + bảng).
- [x] **C3. OCR fallback cho PDF scan** — 🔴 S–M — *xong 19/07*
  - `process_pdf` giờ phát hiện trang có <40 ký tự text → coi là scan, render bằng PyMuPDF (zoom 2x) rồi OCR qua **đường GPT-4o Vision sẵn có** (tách helper `_vision_call`/`_vision_extract_text`, dùng chung với `process_image`). Giữ đúng thứ tự & căn trang (`page_word_starts`); cap 30 trang/tài liệu để chặn chi phí. Đã test PDF scan thuần (0 text → OCR ra text) và PDF hỗn hợp text+scan (căn trang đúng).
- [x] **C4. Trang xem & sửa chunk (admin)** — 🟡 M — *xong 19/07*
  - `GlobalKBService.list_chunks/update_chunk/delete_chunk`: sửa chunk → upsert Chroma (re-embed) + cập nhật in-memory store + rebuild BM25 + bump version; xóa → gỡ khỏi Chroma/store/BM25 + giảm `chunk_count`. Endpoints admin `GET/PATCH/DELETE /api/knowledge/{doc}/chunks[/{chunk}]`. Frontend `ChunkManagerPanel` (drawer như DocViewer, sửa inline + xóa có xác nhận), nút `Layers` mỗi tài liệu ở KnowledgePage (chỉ admin). Test endpoint: 200/404(sai doc)/400(rỗng)/403(non-admin), edit re-embed + BM25 tìm được text mới.
- [x] **C5. Export hội thoại ra Markdown kèm citation** — 🟢 S — *xong 19/07*
  - Nút `Xuất` nổi góc phải khung chat (hiện khi có tin nhắn), sinh file `.md` client-side: giữ nguyên chỉ số trích dẫn `[n]` trong câu trả lời + liệt kê nguồn (tên, trang, nhãn kho chung) sau mỗi câu. Không thêm phụ thuộc; PDF để người dùng in từ Markdown nếu cần.
- [x] **C6. Trang "câu trả lời bị 👎" trong Dashboard** — 🟡 S — *xong 19/07*
  - `GET /api/monitoring/downvoted` (join câu hỏi người dùng liền trước), `POST /api/monitoring/eval-queue` (idempotent theo message_id, ghi `data/feedback_eval_queue.json` kèm câu hỏi + đáp án tệ + ground_truth để reviewer điền + tên nguồn). `DownvotedPanel` trong Dashboard: list có thể mở rộng + nút "Vào eval". Khép vòng lặp online→offline, dẫn thẳng vào D4. E2E: downvote → hiện trong list → thêm queue → cờ `in_eval_queue` bật.

## Giai đoạn D — Kỹ thuật RAG nâng cao (mỗi task phải có số Ragas trước/sau)

- [x] **D1. Contextual Retrieval (kiểu Anthropic) + heading-aware chunking** — 🔴 M — *xong 19/07*
  - **Cài đặt:** chunker mới bám heading (Markdown/`### Slide`/`### Sheet`/ALL-CAPS), chunk không bắc qua ranh giới mục, gắn `heading` path. Lúc ingest, 1 LLM call/tài liệu (batch, JSON, gpt-4o-mini) sinh câu ngữ cảnh mỗi chunk **cùng ngôn ngữ tài liệu**; fallback theo heading khi offline/lệch; bỏ qua tài liệu 1-chunk (không có gì để định vị). Context ghép trước text **chỉ để embed + BM25** — text gốc vẫn lưu/hiển thị (embed thủ công qua `embeddings=`, Chroma giữ `documents` gốc; metadata `context`/`heading`; restore dựng lại token BM25 từ context). Env `CONTEXTUAL_RETRIEVAL` (mặc định bật).
  - **Số Ragas 120 mẫu (cùng judge ada-002), trước → sau:** tổng thể 0.9498 → **0.9459**; EN 0.9624 → 0.9593; VI 0.8700 → 0.8709; **JA 0.8929 → 0.9038** (faithfulness 0.822 → 0.909). Latency query **không đổi** (median ~5.0s — D1 chỉ tốn lúc ingest). Kết quả lưu tại `webapp_eval_results_D1_contextual.json`.
  - **Kết luận (trung thực):** trên corpus eval này (**42/57 tài liệu là đoạn SQUAD 1-chunk → bị bỏ qua**, chỉ 15 tài liệu wiki/arxiv nhiều-chunk được contextualize → 164 chunk), kỹ thuật **net-neutral** (−0.4đ, trong biên nhiễu của LLM-judge). JA (slice yếu nhất) tăng thật +1.1đ. Kỹ thuật nhắm vào **tài liệu lớn nhiều mục** — đúng loại mà C2 (DOCX/PPTX/XLSX) + dùng thật đem lại — nên giữ **bật** mặc định, tắt được qua env cho corpus dạng SQUAD. Đây chính là giá trị của quy tắc "mỗi task phải có số trước/sau": đo được rằng technique nào *không* giúp trên corpus nào.
- [x] **D2. Grounding check sau generation** — 🟡 M — *xong 19/07*
  - `verify_grounding(answer, chunks)` (user_rag.py): 1 call gpt-4o-mini (`json_object`) **sau** khi answer stream xong, kiểm mỗi câu có `[n]` có được nguồn tương ứng hỗ trợ trực tiếp không → `{total, verified, unsupported[]}`. Gate rẻ: không có `[n]` thì bỏ qua. Event SSE mới `grounding` (phát sau `done`, không chặn answer), lưu cột `messages.grounding_json` + answer cache. Frontend: badge `ShieldCheck` "Đã kiểm chứng" (xanh) / `ShieldAlert` "x/y có căn cứ" (hổ phách, tooltip liệt kê câu chưa được đỡ). Đây là bản online của Faithfulness offline. E2E: 3 câu trích dẫn → verified 3/3, persist đúng.
- [x] **D3. Eval regression trong CI** — 🟡 S–M — *xong 19/07*
  - `scripts/eval_regression.py`: chạy **toàn bộ stack retrieval** (heading chunk → semantic+BM25 → RRF → rerank) trên golden set tự chứa `data/golden_regression.json` (8 tài liệu AI + 22 câu EN/VI/JA, mỗi câu có `expected_source`), assert nguồn đúng nằm trong top-k; fail (exit 1) nếu hit-rate < ngưỡng. **Không cần OpenAI** (retrieval-only, contextual tắt để tất định) → chạy CI chỉ với model local. `.github/workflows/eval-regression.yml` chạy trên PR đổi `backend/services|src|golden`, cache HF models. Hiện tại: **22/22 hit-rate @3** (en 15/15, vi 4/4, ja 3/3).
- [x] **D4. Vòng lặp feedback → eval set** — 🟢 S — *xong 19/07*
  - `scripts/promote_feedback_to_eval.py`: đọc `feedback_eval_queue.json` (C6 ghi), lấy entry đã có `ground_truth` reviewer điền → append vào `webapp_eval_questions.json` (id `fb_XXX`, tự nhận diện ngôn ngữ EN/VI/JA, `origin: feedback`), dedup + đánh dấu `promoted` (idempotent). `--list` xem trạng thái. Khép trọn vòng: 👎 online → reviewer gán đáp án → câu hỏi regression offline. Test: promote 1 entry đã review, bỏ entry chưa có gt, re-run không thêm gì.
- [ ] **D5. (Tùy chọn nghiên cứu) RAPTOR** — 🟢 L
  - Cây tóm tắt đa cấp — mở rộng tự nhiên của tầng doc-summary hiện có; giúp câu hỏi "tổng quan toàn corpus".
- [ ] **D6. (Tùy chọn nghiên cứu) LazyGraphRAG / ColPali** — 🟢 L
  - Chỉ làm nếu muốn thêm 1 chương nghiên cứu; không cần cho sản phẩm.

## Giai đoạn E — Mở rộng & tái cấu trúc (làm cuốn chiếu, không làm dồn)

- [x] **E1. Provider abstraction (1 client layer duy nhất)** — 🟡 M — *xong 19/07*
  - `backend/services/llm.py`: `get_client()` / `is_mock()` / `has_api_key()` + `LLM_BASE_URL`. Gom **13 chỗ** tạo `openai.OpenAI(...)` (9 ở user_rag.py, 4 ở document_processor.py) về 1 nơi duy nhất; mock-check rải rác cũng gom về `llm.is_mock()`. Đổi sang Ollama/vLLM chỉ cần đặt `LLM_BASE_URL=http://localhost:11434/v1` (OpenAI-compatible) — không đổi code. E2E chat/suggestions/grounding/title chạy qua layer mới.
- [x] **E2. Tách `user_rag.py` (1.378 → 1.008 dòng) thành module** — 🟢 M — *xong 19/07*
  - `prompts.py` (7 prompt), `intent.py` (regex + `is_summary_question`/`wants_prev_transform`/`lang_directive`), `rag_helpers.py` (7 helper thuần), `multihop.py` (`MultihopMixin`: `route_multihop`/`_hop_answer`/`multihop_events` — mixin để UserRAGService vẫn là 1 class). Tránh vòng lặp import bằng `rag_helpers`. **`tests/test_intent.py` 11 test** cho các regex intent (task yêu cầu). Verify: import OK + mixin resolve + E2E chat/multihop chạy + **eval_regression vẫn 22/22** (retrieval nguyên vẹn) + 37 unit test. *(retrieval.py chưa tách — `retrieve()` là lõi coupling cao, để lại giảm rủi ro; user_rag vẫn giảm 370 dòng.)*
- [x] **E3. Tách `ChatPage.tsx` (1.067 → 964 dòng)** — 🟢 M — *xong 19/07*
  - Hook `hooks/useChatStream.ts` (message list + SSE streaming plumbing: `runStream`/`patchMessage`/`stopStreaming`/`sending`) + component `components/SourcePicker.tsx` (popover chọn nguồn). `tsc --noEmit` sạch + **full vite build pass** (1867 modules). *(useSessions không tách — danh sách session ở component cha, không nằm trong ChatPage.)*
- [ ] **E4. Connector Google Drive / crawl URL định kỳ** — 🟢 L — *chỉ khi định hướng team/enterprise*

---


> Quy tắc khi thực hiện: mỗi task chạm vào pipeline retrieval/generation phải chạy lại eval (tối thiểu bộ smoke 9–30 mẫu) và ghi số trước/sau vào README §12.
