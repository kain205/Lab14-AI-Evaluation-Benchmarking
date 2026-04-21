# Reflection - Nguyen Binh Thanh

## Tiêu chí
- [x] Engineering Contribution
- [x] Technical Depth
- [x] Problem Solving

## Engineering Contribution
Em phụ trách lead team, phân công việc cho các bạn và theo sát tiến độ chung. Về kỹ thuật, em trực tiếp làm phần agent, cải thiện synthetic gen và LLM judge để bộ benchmark ổn định hơn. Khi Gemini API hết key, em chuyển sang dùng 2 model của OpenAI và triển khai async/parallel để chấm 50 câu nhanh hơn. Các phần này có thể đối chiếu qua những thay đổi trong main.py, engine/llm_judge.py, engine/runner.py và data/synthetic_gen.py.

## Technical Depth
Em hiểu MRR là thước đo xem tài liệu đúng xuất hiện sớm đến mức nào trong danh sách retrieve, Cohen's Kappa là mức đồng thuận giữa 2 judge sau khi loại trừ phần đồng thuận ngẫu nhiên, còn Position Bias là việc judge bị ảnh hưởng bởi thứ tự A/B nên cần hoán đổi vị trí để kiểm tra. Về trade-off, em ưu tiên dùng 2 model OpenAI thay vì Gemini để hệ thống vẫn chạy được, đồng thời parallel hóa để giảm latency mà vẫn giữ chất lượng đánh giá.

## Problem Solving
Ban đầu team đã có golden dataset khá sớm, nhưng khi test thực tế thì kết quả cho thấy v1 > v2 > v3. Vì vậy em phải quay lại cải thiện synthetic dataset và LLM judge, rồi thu hẹp scope demo xuống v2 và v3 để làm rõ khác biệt và tránh trình diễn một kết quả chưa ổn định.
