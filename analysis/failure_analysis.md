# Báo cáo Phân tích Thất bại (Failure Analysis Report)
**Nhóm:** Hàn Quang Hiếu (2A202600056) · Nguyễn Bình Thành (2A202600138) · Phan Anh Khôi (2A202600276)
**Ngày chạy benchmark:** 2026-04-21

---

## 1. Tổng quan Benchmark

| Chỉ số | V1-Rewrite | V3-Clarify |
|--------|------------|------------|
| Tổng số cases | 50 | 50 |
| Điểm LLM-Judge trung bình | 4.301 / 5.0 | **4.458 / 5.0** |
| Hit Rate (Retrieval) | 94% | 94% |
| Agreement Rate (Multi-Judge) | 94.7% | 96.0% |
| Delta V3 vs V1 | — | **+0.157** |
| Release Gate | — | ✅ **APPROVE** |

**V3-Clarify cao hơn V1-Rewrite** về avg_score (+0.157) và agreement_rate (+1.3%). Delta dương, vượt ngưỡng `RELEASE_DELTA_TOLERANCE = -0.10` → hệ thống tự động quyết định **APPROVE**.

**Tỉ lệ Pass/Fail (V1-Rewrite, 50 cases):** ~47 Pass / ~3 Fail (threshold score < 3.0)

**Điểm RAGAS:**
- Faithfulness: 0.00 *(RAGAS chạy ở chế độ retrieval-only; faithfulness/relevancy không được tính do không có ground-truth context riêng biệt)*
- Relevancy: 0.00 *(tương tự)*
- Hit Rate: **0.94** — 47/50 test cases retrieve đúng tài liệu liên quan
- MRR: **0.94** — tài liệu đúng hầu hết xuất hiện ở vị trí đầu tiên

---

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Hallucination / Thông tin sai | 1 | Agent tổng hợp thông tin từ nhiều chunk, dẫn đến trả lời sai mức phí cụ thể |
| Incomplete / Thiếu chi tiết | 3 | Agent trả lời đúng ý chính nhưng bỏ sót điều kiện phụ (khu vực, loại tài xế) |
| Tone Mismatch / Thiếu chuyên nghiệp | 2 | Agent dùng ngôn ngữ cộc lốc, thiếu lịch sự với khách hàng |
| Ambiguity Handling | 2 | Agent không hỏi lại khi câu hỏi mơ hồ (thiếu thông tin loại tài xế/dịch vụ) |
| Judge Disagreement (accuracy gap ≥ 3) | 1 | Hai judge GPT-4o-mini và GPT-4o bất đồng lớn về accuracy (4 vs 1) |

---

## 3. Phân tích 5 Whys (3 case tệ nhất)

### Case #1: Mức phí bảo hiểm tai nạn Xanh SM Care (Score: 2.8 — FAIL)

**Câu hỏi:** "Xanh SM Care có mức phí bảo hiểm tai nạn là bao nhiêu?"
**Agent trả lời:** "Mức phí bảo hiểm tai nạn cho Xanh SM Care là 1.000 VNĐ cho chuyến xe Bike và 2.000 VNĐ cho chuyến xe Taxi."
**Ground Truth:** Chỉ đề cập mức phí cho một loại xe cụ thể, không phải cả hai.

1. **Symptom:** Agent trả lời sai mức phí — cung cấp thông tin cho cả Bike lẫn Taxi trong khi Ground Truth chỉ đề cập một loại.
2. **Why 1:** LLM tổng hợp thông tin từ nhiều chunk khác nhau trong context mà không phân biệt rõ ràng từng loại dịch vụ.
3. **Why 2:** Vector DB trả về nhiều chunk liên quan đến "bảo hiểm" và "phí" từ các tài liệu khác nhau (Bike và Taxi), tất cả đều có hit_rate = 1.0.
4. **Why 3:** Chunking strategy hiện tại (fixed-size theo Q&A pair) không tách biệt rõ ràng thông tin theo loại dịch vụ — một chunk có thể chứa thông tin của cả Bike và Taxi.
5. **Why 4:** System prompt không yêu cầu agent xác nhận loại dịch vụ trước khi trả lời câu hỏi về phí.
6. **Root Cause:** Thiếu bước clarification bắt buộc khi câu hỏi liên quan đến thông tin phân biệt theo loại dịch vụ (Bike vs Taxi). V3-Clarify được thiết kế để giải quyết vấn đề này nhưng chưa đủ mạnh với câu hỏi dạng "bao nhiêu" (không mơ hồ về mặt ngữ nghĩa).

---

### Case #2: Thu nhập ngoài giờ theo khu vực (Score: 4.15)

**Câu hỏi:** "Nếu tôi là tài xế ở khu vực A và có thu nhập ngoài giờ, tôi nên xem thu nhập như thế nào?"
**Agent trả lời:** Hướng dẫn vào mục "Thống kê thu nhập" nhưng không đề cập đến khu vực cụ thể và loại thu nhập.

1. **Symptom:** Agent trả lời đúng quy trình chung nhưng bỏ qua điều kiện cụ thể (khu vực A, thu nhập ngoài giờ).
2. **Why 1:** Query rewriter không giữ lại đủ context về "khu vực A" và "ngoài giờ" khi viết lại query.
3. **Why 2:** Chunk được retrieve chứa thông tin chung về "xem thu nhập" nhưng không có thông tin chi tiết về phân loại theo khu vực.
4. **Why 3:** Dữ liệu trong ChromaDB (qa.json) không có tài liệu riêng về thu nhập theo khu vực — thông tin này có thể nằm trong tài liệu chính sách chưa được ingest.
5. **Root Cause:** Khoảng trống trong knowledge base — một số chính sách chi tiết (phân loại theo khu vực, loại thu nhập) chưa được crawl và ingest vào ChromaDB.

---

### Case #3: Judge Disagreement — Phản hồi đánh giá khách hàng (Score: 3.75, Agreement: 0.67)

**Câu hỏi:** "Tôi muốn phản hồi lại đánh giá của khách hàng, nhưng tôi không chắc mình có thể làm vậy hay không?"
**Discrepancy:** GPT-4o-mini cho accuracy=4, GPT-4o cho accuracy=1 (chênh lệch 3 điểm).

1. **Symptom:** Hai judge model bất đồng lớn về accuracy — một judge cho rằng câu trả lời đúng, judge kia cho rằng không liên quan đến Ground Truth.
2. **Why 1:** Ground Truth yêu cầu agent hỏi lại loại tài xế/dịch vụ trước khi trả lời, nhưng agent lại trả lời trực tiếp với hướng dẫn cụ thể.
3. **Why 2:** GPT-4o-mini đánh giá dựa trên "thông tin có đúng không" (đúng), còn GPT-4o đánh giá dựa trên "có khớp với Ground Truth không" (không khớp vì Ground Truth yêu cầu clarification).
4. **Why 3:** Rubric đánh giá accuracy chưa phân biệt rõ giữa "câu trả lời đúng về mặt thực tế" và "câu trả lời đúng theo Ground Truth".
5. **Root Cause:** Ambiguity trong rubric chấm điểm — cần bổ sung hướng dẫn rõ ràng hơn cho judge về cách xử lý trường hợp agent trả lời đúng nhưng không theo đúng flow mà Ground Truth kỳ vọng.

---

## 4. Phân tích Regression (V1 vs V3)

| Chỉ số | V1-Rewrite | V3-Clarify | Delta |
|--------|------------|------------|-------|
| avg_score | 4.301 | **4.458** | **+0.157** |
| hit_rate | 0.940 | 0.940 | 0.000 |
| agreement_rate | 0.947 | **0.960** | +0.013 |

**Nhận xét:**
- **V3-Clarify thắng V1-Rewrite** — delta score = +0.157, vượt ngưỡng `RELEASE_DELTA_TOLERANCE = -0.10` → ✅ **APPROVE**
- Hit Rate giữ nguyên 94% — clarification prompt không làm ảnh hưởng đến retrieval quality
- V3 có agreement_rate cao hơn (+1.3%) — prompt clarification giúp agent trả lời rõ ràng hơn, hai judge đồng thuận hơn
- Cơ chế clarification (hỏi lại khi câu hỏi mơ hồ) giúp agent V3 tránh được các lỗi ambiguity mà V1 mắc phải

---

## 5. Kế hoạch cải tiến (Action Plan)

- [ ] **Chunking Strategy:** Chuyển từ fixed Q&A pair sang semantic chunking, tách biệt thông tin theo loại dịch vụ (Bike/Taxi/Car) để tránh cross-contamination.
- [ ] **Mandatory Clarification:** Bổ sung logic phát hiện câu hỏi về phí/chính sách phân biệt theo loại dịch vụ → bắt buộc hỏi lại loại tài xế trước khi trả lời.
- [ ] **Knowledge Base Expansion:** Crawl thêm tài liệu chính sách chi tiết (thu nhập theo khu vực, phụ cấp đặc biệt) và ingest vào ChromaDB.
- [ ] **Rubric Refinement:** Cập nhật rubric accuracy để phân biệt rõ "factually correct" vs "matches expected flow" — giảm judge disagreement.
- [ ] **Reranking Pipeline:** Thêm bước reranking (cross-encoder) sau retrieval để ưu tiên chunk chính xác nhất theo loại dịch vụ.
- [ ] **Cost Optimization:** Chạy judge bằng GPT-4o-mini cho lần đầu, chỉ escalate lên GPT-4o khi có discrepancy ≥ 2 — ước tính giảm ~35% chi phí eval.
