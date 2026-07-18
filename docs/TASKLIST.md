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

- [ ] **B1. Đổi embedding model** — 🔴 M — *chặn C2/C3 (mọi tài liệu ingest sau này)*
  - `text-embedding-ada-002` (2022) → **BGE-M3** (local, miễn phí, đa ngôn ngữ) hoặc `text-embedding-3-large`. Sửa `EMBED_MODEL` ở `user_rag.py` + `global_kb.py`, viết script re-embed 2 cặp collection Chroma.
- [ ] **B2. Đổi reranker đa ngôn ngữ** — 🔴 S — *sau B1 hoặc song song*
  - `ms-marco-MiniLM` (chỉ tiếng Anh) → **bge-reranker-v2-m3**. Sau đó bỏ được cơ chế dịch query sang tiếng Anh (đơn giản hóa `retrieve()`, bớt 1 LLM call — thay thế luôn A4).
- [ ] **B3. Chạy lại eval 120 mẫu EN/VI/JA, so trước/sau** — 🔴 S — *sau B1+B2*
  - `python run_webapp_eval.py`. Mục tiêu: kéo VI (0.8942) và JA (0.8795) về gần EN (0.9536). Backup kết quả cũ trước khi ghi đè.
- [ ] **B4. Cập nhật README + memory với kết quả B3** — 🟡 S

## Giai đoạn C — Tính năng người dùng (sau B để tài liệu mới chỉ embed 1 lần)

- [ ] **C1. Suggested follow-up questions** — 🔴 S — *không phụ thuộc B, có thể làm xen kẽ*
  - Sau mỗi câu trả lời, 1 call gpt-4o-mini sinh 3 câu hỏi gợi ý từ context + answer; render chip bấm-để-hỏi dưới MessageBubble. Chuẩn UX ChatPDF/NotebookLM, chi phí thấp, hiệu quả demo cao.
- [ ] **C2. Hỗ trợ DOCX / PPTX / XLSX / Markdown** — 🔴 M — *sau B1*
  - `python-docx`, `python-pptx`, `openpyxl`; thêm handler trong `document_processor.py` (pattern `process_pdf` có sẵn). Đây là rào cản dùng thật lớn nhất hiện nay.
- [ ] **C3. OCR fallback cho PDF scan** — 🔴 S–M — *sau B1*
  - Trang PDF không có text layer → hiện index ra 0 chunk **im lặng**. Tái dùng đường GPT-4o Vision sẵn có của `process_image` cho từng trang (render bằng pdfium/pymupdf), hoặc tesseract local.
- [ ] **C4. Trang xem & sửa chunk (admin)** — 🟡 M
  - Như RAGFlow: liệt kê chunk đã parse của mỗi tài liệu, cho xóa/sửa chunk rác. Rất hợp định vị "precision" của dự án; backend đã có `get_document_content()` làm nền.
- [ ] **C5. Export hội thoại ra Markdown/PDF kèm citation** — 🟢 S
- [ ] **C6. Trang "câu trả lời bị 👎" trong Dashboard** — 🟡 S
  - Dữ liệu feedback đã thu nhưng chưa dùng. List message bị downvote + nút "đưa vào bộ eval" → khép vòng lặp cải thiện, dẫn thẳng vào D4.

## Giai đoạn D — Kỹ thuật RAG nâng cao (mỗi task phải có số Ragas trước/sau)

- [ ] **D1. Contextual Retrieval (kiểu Anthropic)** — 🔴 M — *sau B3 để tách bạch nguồn cải thiện*
  - Lúc ingest: LLM sinh 1–2 câu ngữ cảnh ("chunk thuộc phần X của tài liệu Y") gắn đầu chunk trước khi embed. Corpus nhỏ nên re-index rẻ — làm pass riêng sau B để biết chính xác kỹ thuật nào đem lại bao nhiêu điểm. Kèm nâng chunking word-window → heading-aware.
- [ ] **D2. Grounding check sau generation** — 🟡 M
  - 1 call rẻ kiểm từng câu có `[n]` có thực sự được context đỡ không → badge "✓ verified" trên UI. Nối dài câu chuyện Faithfulness 0.84→0.90 của dự án.
- [ ] **D3. Eval regression trong CI** — 🟡 S–M
  - GitHub Actions chạy bộ golden 20–30 câu mỗi PR đổi pipeline, fail nếu tụt điểm. Điểm khác biệt lớn so với mọi webapp open-source cùng loại.
- [ ] **D4. Vòng lặp feedback → eval set** — 🟢 S — *sau C6*
- [ ] **D5. (Tùy chọn nghiên cứu) RAPTOR** — 🟢 L
  - Cây tóm tắt đa cấp — mở rộng tự nhiên của tầng doc-summary hiện có; giúp câu hỏi "tổng quan toàn corpus".
- [ ] **D6. (Tùy chọn nghiên cứu) LazyGraphRAG / ColPali** — 🟢 L
  - Chỉ làm nếu muốn thêm 1 chương nghiên cứu; không cần cho sản phẩm.

## Giai đoạn E — Mở rộng & tái cấu trúc (làm cuốn chiếu, không làm dồn)

- [ ] **E1. Provider abstraction (1 client layer duy nhất)** — 🟡 M
  - Gom ~8 chỗ `openai.OpenAI(...)` về 1 module → mở đường Ollama/local model và đổi provider. Làm khi đụng vào `user_rag.py` vì việc khác.
- [ ] **E2. Tách `user_rag.py` (1.190 dòng) thành module** — 🟢 M
  - `prompts.py` / `intent.py` / `retrieval.py` / `multihop.py`. Kèm unit test cho các regex intent (hiện brittle, chưa có test riêng).
- [ ] **E3. Tách `ChatPage.tsx` (986 dòng)** — 🟢 M
  - Hook `useChatStream`, `useSessions`; component hóa phần input bar + source picker.
- [ ] **E4. Connector Google Drive / crawl URL định kỳ** — 🟢 L — *chỉ khi định hướng team/enterprise*

---

## Gợi ý "tuần làm việc" tiếp theo

| Ngày | Task |
|---|---|
| 1 | A4 + A6 + A7 (ba việc nhỏ, gọn trong 1 ngày) |
| 2–3 | B1 + B2 (embedder + reranker đa ngôn ngữ, re-index) |
| 3 | B3 + B4 (eval 120 mẫu + cập nhật README) |
| 4 | C1 (suggested questions) + C3 (OCR fallback) |
| 5 | C2 (DOCX/PPTX) — bắt đầu, kéo dài sang tuần sau nếu cần |

> Quy tắc khi thực hiện: mỗi task chạm vào pipeline retrieval/generation phải chạy lại eval (tối thiểu bộ smoke 9–30 mẫu) và ghi số trước/sau vào README §12.
