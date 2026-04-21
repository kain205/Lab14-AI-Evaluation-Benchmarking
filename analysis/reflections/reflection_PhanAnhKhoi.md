# Individual Reflection — Phan Anh Khôi

**Student ID**: 2A202600276
**Class**: E403
**Ngày:** 2026-04-21

---

## 1. Phần việc đảm nhận

Tôi phụ trách **Giai đoạn 2 — Evaluation Engine**, bao gồm toàn bộ logic chấm điểm cho pipeline benchmark:

| Module | File | Mô tả |
|--------|------|--------|
| LLM Judge | `engine/llm_judge.py` | Multi-model judge dùng GPT-4o-mini + GPT-4o, rubrics chi tiết, weighted scoring, position bias detection |
| Retrieval Evaluator | `engine/retrieval_eval.py` | Hit Rate@K, MRR, batch evaluation trên toàn bộ golden_set.jsonl |
| Retriever (fix) | `rag/retriever.py` | Thêm `id` field vào kết quả trả về để tính retrieval metrics |

---

## 2. Đóng góp kỹ thuật cụ thể

### 2.1 LLM Judge với Rubrics chi tiết

Thiết kế 3 rubrics cho use case FAQ của XanhSM:

- **Accuracy (weight 0.5)**: So sánh câu trả lời với ground truth — tiêu chí quan trọng nhất vì thông tin sai có thể khiến khách hàng hành động nhầm
- **Professionalism (weight 0.2)**: Văn phong dịch vụ khách hàng — lịch sự, rõ ràng, không lỗi ngữ pháp
- **Safety (weight 0.3)**: Ngăn rủi ro tiết lộ dữ liệu nhạy cảm hoặc lời khuyên sai gây thiệt hại

Mỗi rubric có mô tả cụ thể cho từng mức 1-5 thay vì chỉ cho điểm số, giúp giảm ambiguity khi judge chấm.

### 2.2 Multi-Judge với 2 model song song

`evaluate_multi_judge` gọi song song cả 3 rubric cho cả 2 model bằng `asyncio.gather` lồng nhau. Logic xử lý bất đồng:

- Khi 2 judge lệch nhau ≤ 1 điểm: lấy điểm của model_a
- Khi lệch > 1 điểm: lấy trung bình để tránh một model áp đặt kết quả

`agreement_rate` = tỷ lệ rubric mà 2 judge đồng thuận (lệch ≤ 1), dùng để theo dõi độ ổn định của judge theo thời gian.

### 2.3 Calibration cho câu trả lời ngắn gọn

Implement `_calibrate_scores()` để tránh phạt oan câu trả lời ngắn nhưng đúng: nếu accuracy ≥ 4.0 và safety ≥ 4.0 nhưng câu trả lời ≤ 90 ký tự, professionalism được cộng thêm 0.5. Lý do: FAQ thường có câu trả lời ngắn gọn là đúng phong cách, không phải thiếu chuyên nghiệp.

### 2.4 Retrieval Evaluator — evaluate_batch

`evaluate_batch` load toàn bộ `data/golden_set.jsonl`, gọi `retrieve()` bằng `asyncio.to_thread` (vì retriever là synchronous), tính Hit Rate và MRR cho từng case.

Phải fix `rag/retriever.py` để trả về `id` field từ `results["ids"][0]` của ChromaDB — trước đó retriever chỉ trả về text chunks, không có ID nên không thể so sánh với `ground_truth_ids`.

### 2.5 Position Bias Detection

`check_position_bias` score từng response bằng cả 3 rubric, sau đó hoán đổi vị trí và score lại. Bias được phát hiện nếu điểm của cùng một response thay đổi > 0.5 khi đổi vị trí — tức judge đang bị ảnh hưởng bởi thứ tự trình bày thay vì chất lượng thực sự.

---

## 3. Kiến thức kỹ thuật học được

### Hit Rate vs MRR
- **Hit Rate@K**: Binary — 1 nếu có ít nhất 1 expected doc trong top-K. Đơn giản nhưng không phân biệt "tìm thấy ở vị trí 1" vs "tìm thấy ở vị trí K".
- **MRR**: `1/rank` của expected doc đầu tiên. MRR = 1.0 nghĩa là document đúng luôn xuất hiện ở vị trí đầu tiên — đây là kết quả lý tưởng.

Hai metric này đánh giá **retrieval** độc lập với **generation** — pipeline có 2 tầng lỗi tách biệt nhau.

### Position Bias trong LLM-as-Judge
LLM judge có xu hướng ưu tiên response xuất hiện ở vị trí đầu tiên (primacy bias) hoặc cuối cùng (recency bias) bất kể chất lượng thực sự. Cách phát hiện: chạy judge 2 lần với thứ tự hoán đổi và đo drift trong điểm số. Cách giảm thiểu: thêm instruction rõ ràng trong prompt, hoặc dùng nhiều model judge và lấy trung bình.

### Async trong Evaluation Pipeline
Gọi LLM API là I/O-bound — dùng `asyncio.gather` để chạy song song giảm latency đáng kể. Với 50 test cases × 3 rubrics × 2 models = 300 API calls, song song hóa giảm thời gian từ ~10 phút xuống ~1-2 phút.

---

## 4. Vấn đề gặp phải và cách giải quyết

### Vấn đề 1: Retriever không trả về document ID
`rag/retriever.py` chỉ trả về text (question, answer, score) — không có ID nên không thể so sánh với `ground_truth_ids`. Giải pháp: thêm `"id": doc_id` vào chunk dict bằng cách zip thêm `results["ids"][0]` trong vòng lặp.

### Vấn đề 2: LLM judge trả về non-JSON
GPT-4o-mini đôi khi trả về text thay vì JSON dù đã set `response_format={"type": "json_object"}`. Giải pháp: implement `_parse_judge_payload()` với fallback regex `re.search(r"\b([1-5])\b", content)` để vẫn extract được điểm số.

### Vấn đề 3: Câu trả lời ngắn bị chấm thấp professionalism
Judge có xu hướng chấm thấp professionalism cho câu trả lời ngắn dù nội dung đúng — vì "ngắn" bị hiểu nhầm là "thiếu lịch sự". Giải pháp: `_calibrate_scores()` tự động điều chỉnh khi accuracy và safety đã đủ cao.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm | Lý do |
|----------|---------|-------|
| Engineering Contribution | 13/15 | Hoàn thành đầy đủ LLM Judge + Retrieval Evaluator. Thiếu điểm vì chưa wire `evaluate_batch` vào pipeline chính của `main.py`. |
| Technical Depth | 13/15 | Hiểu rõ Hit Rate, MRR, position bias, async optimization. Chưa đi sâu vào Cohen's Kappa để đo inter-rater agreement chính xác hơn. |
| Problem Solving | 9/10 | Giải quyết được cả 3 vấn đề phát sinh trong quá trình implement. |

**Tổng tự đánh giá: 35/40**

---

## 6. Điều rút ra

Phần thú vị nhất là nhận ra rằng **evaluation cũng cần được evaluate**. Judge bị position bias, rubric mơ hồ, hay calibration sai đều dẫn đến kết quả benchmark không đáng tin — và khi benchmark không đáng tin, mọi quyết định "approve/block release" dựa trên nó đều có thể sai.

Nếu làm lại, tôi sẽ chạy `check_position_bias` ngay từ đầu trên nhiều cặp response trước khi dùng judge để chấm toàn bộ dataset, thay vì implement xong rồi để đó.
