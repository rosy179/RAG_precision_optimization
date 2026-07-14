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

## Giai đoạn 3 — Quản lý tri thức trưởng thành (~1 tuần)

- [ ] Trang admin KB: upload/xóa/danh sách/thống kê (thay script + restart).
- [ ] BM25 rebuild in-process sau ingest (bỏ yêu cầu restart server).
- [ ] Auto-title session bằng LLM (thay cắt 60 ký tự).

## Giai đoạn 4 — Nghiên cứu vào sản phẩm + eval thuyết phục (~1,5 tuần)

- [ ] Adaptive routing đầy đủ trong webapp: câu đơn giản → hybrid; phức tạp →
      CoT/multi-hop; hiển thị reasoning steps khi multi-hop chạy (demo cho yêu
      cầu 「内容は近くないけど、回答に関連する情報をどうやって拾えるようにするか」).
- [ ] Nối `src/monitoring.py` vào backend + dashboard latency/cost đơn giản.
- [ ] Mở rộng eval: bộ câu hỏi từ corpus KB (56 docs), ≥100 mẫu, thêm bộ nhỏ
      tiếng Việt/Nhật; chạy Ragas trên chính pipeline webapp.

## Nguồn tham khảo

- NotebookLM chat & citations: https://support.google.com/notebooklm/answer/16179559
- AnythingLLM vs Open WebUI: https://localaimaster.com/blog/anythingllm-vs-open-webui
- Open WebUI vs LibreChat vs AnythingLLM: https://www.local-llm.net/compare/open-webui-vs-librechat-vs-anythingllm/
- Onyx – alternatives for teams: https://onyx.app/insights/openwebui-alternatives
- AI Chat UI best practices 2026: https://thefrontkit.com/blogs/ai-chat-ui-best-practices
- Agentic RAG production patterns: https://www.brightter.com/articles/agentic-rag-five-retrieval-patterns-that-survive-production
- RAG techniques compared 2026: https://blog.starmorph.com/blog/rag-techniques-compared-best-practices-guide
