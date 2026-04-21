# Individual Reflection — Nguyen Binh Thanh
**Môn học:** AI Engineering — Lab Day 14: AI Evaluation Factory
**Ngày:** 2026-04-21

---

## 1. Phần việc đảm nhận

Em phụ trách vai trò lead team (điều phối công việc, theo sát tiến độ) và trực tiếp triển khai các phần kỹ thuật quan trọng của pipeline benchmark.

| Hạng mục | File | Đóng góp chính |
|---|---|---|
| Agent orchestration | `main.py`, `agent/main_agent.py` | Thiết kế lại prompt, wire nhiều agent version vào pipeline benchmark |
| LLM Judge | `engine/llm_judge.py` | Nâng cấp judge, chuẩn hóa output để dễ phân tích và so sánh |
| Benchmark runner | `engine/runner.py` | Tối ưu luồng chạy, bổ sung async/parallel để giảm thời gian chấm |
| Synthetic dataset | `data/synthetic_gen.py` | Cải thiện chất lượng dữ liệu test để benchmark ổn định hơn |

Các phần việc trên bám theo commit history em cung cấp trên GitHub (author `kain205`), đặc biệt các commit:
- `feat: improve synthetic data quality and switch judge to dual OpenAI models`
- `feat: enhance llm_judge with JSON reasoning output and update main configs`
- `feat: redesign agent prompts and run 3 agents in parallel`
- `feat: support multi-turn cases and wire retrieval eval`
- `feat: tag agent_version per result, save all 3 agents in benchmark_results.json`
- `feat: wire real agents V1/V2/V3 and LLMJudge into benchmark pipeline`

---

## 2. Đóng góp kỹ thuật cụ thể

### 2.1 Chuyển judge từ Gemini sang 2 model OpenAI
Khi Gemini API hết key, em chuyển hướng sang 2 model OpenAI để giữ hệ thống hoạt động liên tục. Đây là quyết định ưu tiên tính sẵn sàng của pipeline, tránh gián đoạn demo và benchmark.

### 2.2 Tối ưu thời gian chấm bằng async/parallel
Em triển khai async/parallel để chấm bộ 50 câu nhanh hơn, giảm latency nhưng vẫn giữ được chất lượng đánh giá. Song song đó, em giữ cấu trúc output nhất quán để tiện tổng hợp kết quả ở bước sau.

### 2.3 Nâng cấp LLM Judge cho khả năng phân tích
Phần judge được cải thiện theo hướng trả về output có cấu trúc (JSON reasoning) để dễ debug, đối chiếu và theo dõi sai khác giữa các phiên bản agent.

### 2.4 Cải thiện synthetic data
Em quay lại chất lượng synthetic dataset sau khi thấy kết quả thực tế không như kỳ vọng. Mục tiêu là làm bộ test phản ánh tốt hơn hành vi của hệ thống, từ đó benchmark có ý nghĩa hơn thay vì chỉ đẹp số.

---

## 3. Kiến thức kỹ thuật học được

### MRR trong Retrieval Evaluation
MRR cho biết tài liệu đúng xuất hiện sớm đến đâu trong danh sách retrieve. Em hiểu đây là metric quan trọng để đánh giá chất lượng truy xuất, không chỉ là “có tìm thấy hay không” mà còn là “tìm thấy ở vị trí nào”.

### Cohen's Kappa trong đánh giá đồng thuận
Cohen's Kappa đo mức đồng thuận giữa 2 judge sau khi loại trừ phần đồng thuận ngẫu nhiên. Em dùng góc nhìn này để đánh giá mức ổn định của judge thay vì chỉ nhìn accuracy thô.

### Position Bias trong LLM-as-a-Judge
Position Bias là hiện tượng judge bị ảnh hưởng bởi thứ tự A/B. Vì vậy cần hoán đổi vị trí để kiểm tra và giảm thiên lệch khi so sánh câu trả lời giữa các agent.

### Trade-off kỹ thuật
Trong bối cảnh giới hạn API, em ưu tiên tính vận hành (switch sang OpenAI) và hiệu năng (parallel hóa) để hệ thống chạy ổn định, sau đó mới tiếp tục tối ưu chiều sâu đánh giá.

---

## 4. Vấn đề gặp phải và cách giải quyết

### Vấn đề 1: Mất khả dụng Gemini API
- Hiện trạng: key Gemini hết, pipeline có nguy cơ dừng.
- Xử lý: chuyển sang 2 model OpenAI trong judge.
- Kết quả: benchmark tiếp tục chạy được, không gián đoạn tiến độ.

### Vấn đề 2: Kết quả benchmark ngược kỳ vọng (v1 > v2 > v3)
- Hiện trạng: thứ hạng version cho thấy chất lượng dữ liệu/judge chưa phản ánh đúng mục tiêu.
- Xử lý: quay lại cải thiện synthetic dataset và LLM judge.
- Kết quả: hệ thống đánh giá ổn định hơn, dễ giải thích hơn.

### Vấn đề 3: Scope demo chưa hợp lý
- Hiện trạng: nếu demo đủ cả phiên bản sẽ khó làm rõ insight khi kết quả chưa ổn định.
- Xử lý: thu hẹp scope còn v2 và v3.
- Kết quả: demo tập trung hơn, nhấn mạnh được khác biệt chính và bài học kỹ thuật.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm | Lý do |
|---|---:|---|
| Engineering Contribution | 14/15 | Vừa lead team vừa trực tiếp triển khai các phần lõi (agent, synthetic gen, judge, runner), có minh chứng commit rõ ràng. |
| Technical Depth | 13/15 | Nắm được MRR, Cohen's Kappa, Position Bias và trade-off vận hành; phần thực nghiệm bias có thể mở rộng sâu thêm. |
| Problem Solving | 9/10 | Xử lý tốt tình huống mất API, kết quả benchmark lệch kỳ vọng và tái cấu trúc scope demo đúng lúc. |

**Tổng tự đánh giá: 36/40**

---

## 6. Điều rút ra

Điểm em rút ra lớn nhất là cần làm kỹ theo từng bước đánh giá nhỏ thay vì tạo toàn bộ golden set rồi mới chạy batch lớn. Cách tốt hơn là làm incremental: kiểm tra từng nhóm synthetic data ngay sau khi sinh (ví dụ fact-check trước, multi-turn sau), chạy evaluate bằng LLM judge cho từng nhóm để xác nhận chất lượng rồi mới gộp lại. Vì mỗi loại synthetic data có lỗi khác nhau, nếu test sớm theo từng mục thì có thể phát hiện và chặn lỗi trước khi ảnh hưởng cả pipeline.

Ngoài ra, khi chia việc cho các bạn làm song song, lúc gộp kết quả thì khối lượng output rất lớn và khó kiểm soát nếu thiếu checkpoint trung gian. Bài học của em là cần chia nhỏ milestone, đặt format output thống nhất và review từng đợt nhỏ để điều hướng sớm. Làm được vậy thì chất lượng tổng thể và tốc độ ra quyết định sẽ tốt hơn nhiều.
