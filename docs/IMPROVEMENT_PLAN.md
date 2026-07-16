# Kế hoạch hoàn thiện webapp RAG (2026-07-14)

Kết quả nghiên cứu các sản phẩm tương tự (NotebookLM, AnythingLLM, Open WebUI,
RAGFlow, Onyx, ChatPDF/Humata) và kế hoạch nâng cấp webapp theo 4 giai đoạn.

## Bối cảnh

Pipeline nghiên cứu (Hybrid BM25+Semantic → RRF → CrossEncoder rerank → CoT,
Ragas 0.9434, multi-hop RAG) đã đạt chuẩn "best quality-to-cost" mà các guide
production 2026 khuyến nghị. Khoảng cách với thị trường nằm ở **webapp**:
trải nghiệm chat, citation, vòng phản hồi, và quản lý tri thức.

## Nhận xét chính (so với thị trường)

1. **Không có streaming** — `/chat` trả nguyên câu trả lời sau 1–2 giây im lặng.
   Token-by-token là chuẩn tối thiểu 2026.
2. **Citation yếu** — nguồn là card gấp gọn, không đánh số inline `[1][2]`,
   không click mở tài liệu, không highlight đoạn trích (NotebookLM/ChatPDF coi
   đây là tính năng cốt lõi).
3. **Không có vòng phản hồi** — không thumbs up/down, regenerate, stop →
   không thu được tín hiệu chất lượng online.
4. **Quản lý tri thức thô** — ingest KB qua script + restart server (BM25
   in-memory); chưa có UI admin; chưa chọn/loại nguồn theo câu hỏi.
5. **Kỹ thuật nghiên cứu chưa vào sản phẩm** — multi-hop, CoT, monitoring
   nằm ở `src/`, chưa nối vào webapp.
6. **Eval mỏng** — 30 mẫu SQUAD, chưa có tiếng Việt/Nhật, chưa eval trên chính
   pipeline webapp.

## Giai đoạn 1 — Trải nghiệm chat đạt chuẩn 2026 ✅ HOÀN THÀNH (2026-07-14)

- [x] SSE streaming cho `/chat` — endpoint mới `POST /api/sessions/{id}/chat/stream`
      (meta → delta* → done|error); frontend dùng fetch + parse SSE, render
      markdown mid-stream, nút **Stop** (abort giữ lại phần đã sinh, backend
      cũng lưu partial khi client ngắt).
- [x] Inline citation `[1]`, `[2]` — quy tắc trích dẫn trong CHAT_SYSTEM_PROMPT;
      frontend render chip click được (nhảy + highlight nguồn), hover hiện
      snippet; bỏ qua `[n]` trong code block.
- [x] Thumbs up/down + **Regenerate** — cột `messages.feedback`,
      `POST /api/messages/{id}/feedback`; regenerate thay nội dung message cũ
      (`regenerate_message_id`), không tạo bản ghi trùng.
- Kiểm thử: 24/24 E2E pass (streaming, citation, persist, feedback, regen,
  ownership 404, endpoint cũ tương thích).

## Giai đoạn 2 — Citation & xem tài liệu như ChatPDF/NotebookLM ✅ HOÀN THÀNH (2026-07-14)

- [x] Panel xem tài liệu (`DocViewerPanel`): click citation/nguồn mở tài liệu
      bên phải — PDF render bằng react-pdf (nhảy đúng trang, highlight cụm từ
      trích dẫn trong text layer), tài liệu khác hiển thị văn bản tái tạo từ
      chunks với đoạn trích được highlight + auto-scroll.
- [x] Ingest PDF gắn số trang cho từng chunk; file PDF gốc lưu ở
      `backend/uploads/{doc_id}.pdf`; sources trong chat mang
      `doc_id`/`chunk_id`/`page`. Endpoint mới:
      `GET /api/documents/{id}/content|/file`, `GET /api/knowledge/{id}/content|/file`.
      Văn bản tái tạo từ chunks bằng khớp overlap (không cần re-ingest tài liệu cũ).
- [x] Checkbox chọn nguồn (nút ⚙ cạnh ô nhập): bật/tắt từng tài liệu của phiên
      + kho kiến thức chung; backend lọc theo `include_doc_ids`/`use_global_kb`
      xuyên suốt semantic + BM25 + HyDE.
- [ ] (dời sang backlog) YouTube transcript làm loại nguồn mới.
- Kiểm thử: 20/20 E2E pass (upload PDF → content/file/page, chat sources có
  doc_id/page, 3 kịch bản lọc nguồn, KB content, ownership 404).

## Giai đoạn 3 — Quản lý tri thức trưởng thành ✅ HOÀN THÀNH (2026-07-14)

- [x] Trang "Kho kiến thức" (`KnowledgePage.tsx`, nút ở sidebar): danh sách +
      tìm kiếm, thống kê docs/chunks, upload tệp (kéo-thả) & URL, xóa có xác
      nhận, click xem nội dung bằng DocViewerPanel. Quyền quản lý theo
      `ADMIN_EMAILS` (`can_manage` trong GET /api/knowledge — non-admin chỉ xem).
- [x] Ingest không cần restart: upload qua API rebuild BM25/registry in-process
      (đã có sẵn trong GlobalKBService); thêm `POST /api/knowledge/reload` +
      nút "Đồng bộ từ đĩa" để nhận dữ liệu do `scripts/ingest_knowledge.py`
      ghi từ process khác.
- [x] Auto-title session bằng LLM (`generate_session_title`, ≤8 từ, đúng ngôn
      ngữ câu hỏi; fallback cắt 60 ký tự khi lỗi/mock); sidebar tự refresh khi
      nhận `session_title` trong event `done`.
- Kiểm thử: 17/17 E2E pass (upload KB → truy vấn được ngay không restart,
  title LLM lưu đúng, reload giữ nguyên 57 docs/203 chunks, delete sạch).

## Giai đoạn 4 — Nghiên cứu vào sản phẩm + eval thuyết phục ✅ HOÀN THÀNH (2026-07-15)

- [x] **Multi-hop RAG trong webapp** (demo trực tiếp cho yêu cầu 「内容は近くない
      けど、回答に関連する情報をどうやって拾えるようにするか」): router LLM
      (`route_multihop`) chạy cho câu medium/complex — quyết định SINGLE hay
      2-3 truy vấn con; vòng lặp hop dùng đầy đủ stack retrieval của webapp
      (hybrid + rerank, 2 pool), dữ kiện tìm được ở hop trước được chèn vào
      truy vấn hop sau; tổng hợp cuối streaming với citation [n] trên union
      nguồn. UI hiển thị **các bước suy luận trực tiếp** (khối "Suy luận qua N
      bước", lưu vào `messages.steps_json` để xem lại). Câu simple đi đường
      nhanh, không thêm latency.
- [x] **Monitoring dashboard**: bảng `query_logs` (độ trễ, complexity, số hop,
      token/chi phí ước tính, trạng thái) ghi từ endpoint stream;
      `GET /api/monitoring/stats` (admin) trả P50/P95/P99, chi phí, phân bố
      complexity, feedback 👍/👎, chuỗi theo ngày; trang "Thống kê" (sidebar)
      với KPI tiles + biểu đồ cột truy vấn/ngày + breakdown độ phức tạp
      (single-hue đã validate bằng dataviz palette checker).
- [x] **Eval mở rộng**: `run_webapp_eval.py` — sinh bộ câu hỏi từ chính corpus
      KB (`--generate`, đã tạo `data/webapp_eval_questions.json`: 100 EN +
      10 VI + 10 JA) và chạy Ragas trên **đúng pipeline webapp**
      (UserRAGService + KB chung, có rerank). Smoke run đã pass; full run:
      `python run_webapp_eval.py` (120 mẫu, tốn API + ~30-60 phút).
- Kiểm thử: 18/18 E2E pass — chuỗi bắc cầu AlexNet→Turing Award chạy đúng
  (hop 2 chứa dữ kiện hop 1 tìm được), câu thường không bị router bắt nhầm,
  steps lưu DB, stats API trả đủ số liệu.

### Gói fix độ chính xác (2026-07-15, sau phát hiện của smoke eval)

Smoke eval 9 mẫu ban đầu lộ ra tiếng Việt yếu rõ rệt (faithfulness 0.33): câu
VI trên corpus EN tìm đúng tài liệu nhưng sai chunk. Soi từng mẫu lộ thêm 2
lỗi nữa. Đã fix cả 4 (tất cả trong `user_rag.py` + `global_kb.py`):

1. **Dịch truy vấn sang tiếng Anh ở tầng retrieval** (câu hỏi non-ASCII):
   tìm kiếm bằng CẢ truy vấn gốc lẫn bản dịch EN — semantic gộp 1 lần gọi
   Chroma đa truy vấn (lấy max điểm mỗi chunk), BM25 chấm cả hai (max),
   summary shortlist union. Ngôn ngữ TRẢ LỜI vẫn theo câu hỏi.
2. **Rerank cho mọi câu hỏi** (trước đây câu "simple" bỏ qua rerank) và
   CrossEncoder chấm theo truy vấn EN (ms-marco chỉ hiểu tiếng Anh); thứ hạng
   CE được **blend RRF với thứ hạng fusion** — một cú chấm sai của CE (thực tế
   xảy ra: chunk đúng bị chấm -0.9) không thể đá chunk mà BM25+semantic đồng
   thuận ra khỏi ngữ cảnh. top_k simple 3→4.
3. **Ghim ngôn ngữ trả lời** (`_lang_directive`): system prompt tiếng Việt
   từng kéo câu trả lời JA/EN trôi sang tiếng Việt — giờ chèn chỉ thị ngôn ngữ
   tường minh theo ngôn ngữ câu hỏi (phát hiện bằng regex ký tự JA/VI).
4. **Quy tắc xử lý dữ kiện cạnh tranh** trong prompt (nhiều mốc năm/chương
   trình cùng khớp → đối chiếu đúng chủ thể & phạm vi, "đầu tiên" = mốc sớm nhất).

Kết quả trên CÙNG 9 mẫu (n nhỏ, nhưng khớp với xác minh trực tiếp từng case):

| | Trước fix | Sau fix |
|---|---|---|
| AVG tổng | 0.629 | **0.771** |
| Faithfulness | 0.647 | **0.862** |
| VI avg / faith | 0.45 / 0.33 | **0.78 / 0.75** |
| JA avg | 0.84 | 0.80 |
| EN avg | 0.62 | 0.73 |

Xác minh trực tiếp: câu VI Notre Dame trả đúng "1854–1855" (3 lần chạy trước
đều sai), câu JA trả lời bằng tiếng Nhật, câu multi-hop EN trả lời bằng tiếng
Anh. Regression Phase 4: 18/18 pass.

### Kết quả FULL EVAL 120 mẫu (2026-07-15, sau gói fix) — số liệu chốt

`python run_webapp_eval.py` trên đúng pipeline webapp, 120 câu (100 EN +
10 VI + 10 JA) sinh từ corpus KB, kết quả tại `results/webapp_eval_results.json`:

| | Faith | Relev | Prec | Recall | **AVG** |
|---|---|---|---|---|---|
| **Tổng (120)** | 0.9154 | 0.9502 | 0.9290 | 0.9833 | **0.9445** |
| EN (100) | 0.93 | 0.97 | | | 0.9536 |
| VI (10) | 0.88 | 0.83 | | | 0.8942 |
| JA (10) | 0.85 | 0.89 | 0.78 | 1.00 | 0.8795 |

- **0.9445 trên 120 mẫu đa ngôn ngữ** — ngang mốc 0.9434 của pipeline nghiên
  cứu (vốn chỉ đo 30 mẫu SQUAD tiếng Anh), đạt mục tiêu ~95% của dự án trên
  chính sản phẩm thật.
- Latency trung bình 6.95s/câu (đo trong eval, gồm cả dịch truy vấn VI/JA).
- Lưu ý kỹ thuật: Ragas trả NaN cho dòng lỗi — `safe_mean` trong
  `src/evaluation.py` đã được sửa để lọc NaN (trước đó làm breakdown JA
  thành nan; đã chấm lại riêng 10 câu JA).

## Backlog

- Embedding đa ngôn ngữ (multilingual-e5 / text-embedding-3-large) nếu muốn
  kéo VI/JA (~0.88) sát EN (~0.95) — yêu cầu re-embed toàn bộ KB.
- YouTube transcript làm loại nguồn mới.
- Giảm time-to-first-token (~10s, nghẽn ở retrieval: nhiều lần embed query +
  rerank): cache query embedding, gộp truy vấn 2 pool, xem xét rerank bất đồng bộ.
- Code-split frontend (bundle > 500 kB sau khi thêm react-pdf).

## Nguồn tham khảo

- NotebookLM chat & citations: https://support.google.com/notebooklm/answer/16179559
- AnythingLLM vs Open WebUI: https://localaimaster.com/blog/anythingllm-vs-open-webui
- Open WebUI vs LibreChat vs AnythingLLM: https://www.local-llm.net/compare/open-webui-vs-librechat-vs-anythingllm/
- Onyx – alternatives for teams: https://onyx.app/insights/openwebui-alternatives
- AI Chat UI best practices 2026: https://thefrontkit.com/blogs/ai-chat-ui-best-practices
- Agentic RAG production patterns: https://www.brightter.com/articles/agentic-rag-five-retrieval-patterns-that-survive-production
- RAG techniques compared 2026: https://blog.starmorph.com/blog/rag-techniques-compared-best-practices-guide
