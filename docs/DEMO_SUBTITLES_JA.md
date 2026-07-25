# デモ動画 字幕（日本語）— 完成版
# Phụ đề tiếng Nhật cho video demo (bản hoàn chỉnh)

> **Dùng cho:** lồng phụ đề tiếng Nhật lên video demo webapp RAG (khách HBA・RikkeiSoft).
> **Đã đối chiếu với code** (chat.py / user_rag.py / document_processor.py / multihop.py / monitoring.py / knowledge.py) — các câu sai ý ("retrain model", cơ chế từ chối) đã sửa cho khớp thực tế.
> **Thứ tự cảnh** theo đúng kịch bản bạn đã quay. Mỗi dòng số = 1 thẻ phụ đề (câu dài có thể tách 2 dòng khi hiển thị).

---

## 場面1 — ログイン・管理者画面とナレッジベース
*(Đăng nhập · giao diện admin · ナレッジベース)*

1. ログインすると、管理者向けの、Webアプリのメイン画面が表示されます。
2. こちらが、ナレッジベースです。
3. 管理者は、アップロードした資料を、追加・削除できます。
4. 資料内のチャンク（分割された文章）の、編集も可能です。

---

## 場面2 — 基本の質問「What is RAG?」
*(Câu hỏi đơn giản · trích dẫn · verified · gợi ý)*

5. まず、簡単な質問「What is RAG?」から始めます。
6. 回答には、引用番号が付きます。
7. これは、ナレッジベースのどの箇所を参照したかを示します。
8. 番号をクリックすると、根拠の箇所がハイライトされ、情報の誤りを防ぎます。
9. 回答は、コピー・高評価／低評価・再生成ができます。
10. 低評価がついた回答は、評価用のキューに集められ、オフライン評価と改善に活用されます。
11. 「検証済み（Verified）」バッジは、回答がナレッジベースの事実に基づき、創作でないことを示します。
12. さらに回答の下に、会話に関連する、おすすめの質問が表示されます。

> 💡 Dòng 10: bản đúng với code (KHÔNG phải "retrain model" — câu dislike chỉ được đưa vào bộ đánh giá offline).
> Nếu muốn nhấn tính trung thực, thêm 1 thẻ: 「モデルの再学習は行いません。」

---

## 場面3 — データが無い質問 → URLで追加
*(Hỏi cái không có → từ chối → dán URL → trả lời được)*

13. 次の会話では、参照できる資料が無い状態で質問します。
14. 根拠が無い場合、チャットボットは推測で答えず、資料の提供を求めます。
15. 根拠がある時だけ回答し、創作（ハルシネーション）を防いで、精度を高めます。
16. 次に、その話題を含むURLを添付して、もう一度質問します。
17. チャットボットは、リンク内の文章をチャンクに分割して、回答します。
18. 引用番号が付き、リンク由来の正しい事実のため、「検証済み」バッジも付きます。

> ⚠️ **QUAN TRỌNG khi quay cảnh này:** app chỉ hiện thông báo "không có tài liệu, hãy cung cấp"
> khi **scope rỗng**. Vì vậy trước khi hỏi câu (ở dòng 13–14), hãy **TẮT công tắc "Kho kiến thức chung"**
> trong bộ chọn nguồn → app mới hiện đúng thông báo từ chối. Sau đó dán URL rồi hỏi lại.
> (Nếu để KB bật, chatbot sẽ vẫn cố trả lời → không khớp phụ đề.)

---

## 場面4 — 画像の添付（GPT-4o Vision）
*(Đính ảnh sơ đồ RAG · hỏi input/output)*

19. 次の会話では、RAGモデルの画像を添付します。
20. 「この図の入力と出力は何ですか」と質問します。
21. 画像やPDFから文字や図表を読み取るため、視覚モデルのGPT-4o Visionを使用しています。

---

## 場面5 — Multi-hop 推論
*(Bắc cầu: tìm o1 → o1 sinh ra gì)*

22. 続いて、Multi-hop（マルチホップ）推論を使う会話です。
23. スライドで示した課題——「内容は近くないが、回答に関連する情報を、どう拾うか」——に対応します。
24. チャットボットは、橋渡しの推論過程を可視化します。
25. まず、中間モデル "o1" を見つけ出します。
26. 次に、その "o1" が最終的な回答の前に何を生成するか（思考の連鎖）を、たどって回答します。

---

## 場面6 — 統計・分析ダッシュボード（管理者のみ）
*(Thống kê: số câu hỏi · latency · cost · phân loại · review dislike)*

27. 最後に、統計・分析の画面です。管理者のみ、アクセスできます。
28. 質問数、レイテンシ（平均・P50・P95・P99）、トークンコスト、エラー率を確認できます。
29. 日ごとの質問数を、グラフで表示します。
30. 質問を、複雑なものから簡単なものへ、分類する機能もあります。
31. 低評価がついた回答を見直し、オフラインの評価セットに追加して、品質改善に活用できます。

> 💡 Dòng 28: dashboard hiện **percentiles (平均/P50/P95/P99)** và **tỷ lệ lỗi**, không phải độ trễ/danh sách từng câu.
> Dòng 31: bản đúng — review câu dislike để bổ sung **eval set offline**, không phải retrain.

---

## 読み方メモ — phát âm thuật ngữ khó (nếu lồng tiếng)
引用番号 (いんようばんごう) · 根拠 (こんきょ) · 検証済み (けんしょうずみ) · 創作 (そうさく) ·
橋渡し (はしわたし) · 中間モデル (ちゅうかんモデル) · 統計 (とうけい) · 分割 (ぶんかつ)

**Katakana (đọc chậm):** ナレッジベース · チャンク · ハイライト · ハルシネーション · マルチホップ · レイテンシ

---

## 補足 — sự thật kỹ thuật đằng sau (để trả lời nếu bị hỏi)
- **検証済み（Verified）:** hệ thống kiểm mỗi câu có trích dẫn `[n]` xem nguồn `n` có thực sự hỗ trợ không (`GROUNDING_PROMPT`). Xanh = mọi câu có căn cứ; nếu có câu không căn cứ, badge chỉ rõ câu đó.
- **GPT-4o Vision:** lúc đính ảnh, Vision (`gpt-4o`) chuyển ảnh thành mô tả text rồi index; câu trả lời sinh ra bằng RAG trên mô tả đó (sinh câu trả lời dùng `gpt-4o-mini`).
- **URL trong chat:** link được tải, bóc nội dung chính (trafilatura/BeautifulSoup), cắt chunk, thành **tài liệu của phiên chat** (không phải KB chung).
- **Multi-hop:** chỉ kích hoạt cho câu hỏi router phân loại là cần bắc cầu → **test trước** câu DeepSeek để chắc chắn nó chạy multi-hop khi quay.
- **Dislike → chất lượng:** lưu `feedback="down"` → admin xem ở `/downvoted` → thêm vào `feedback_eval_queue.json` (eval set offline, TASKLIST D4). **Không** re-train/fine-tune model — nhất quán với slide 5/6/14.
