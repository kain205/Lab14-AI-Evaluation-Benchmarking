# Individual Reflection — Hàn Quang Hiếu
**Mã học viên:** 2A202600056
**Môn học:** AI Engineering — Lab Day 14: AI Evaluation Factory
**Ngày:** 2026-04-21

---

## 1. Phần việc đảm nhận

Tôi phụ trách **Giai đoạn 1 — Data & RAG Infrastructure**, bao gồm toàn bộ nền tảng dữ liệu và vector search mà các module phía sau phụ thuộc vào:

| Module | File | Mô tả |
|--------|------|--------|
| Data Ingestion | `rag/ingest.py` | Load `data/qa.json`, upsert vào ChromaDB với metadata đầy đủ |
| Facebook Crawler | `rag/ingest_facebook.py` | Crawl và ingest dữ liệu từ Facebook group posts (community data) |
| Vector Store | `rag/vectorstore.py` | Khởi tạo ChromaDB PersistentClient, cấu hình OpenAI embedding (`text-embedding-3-small`) |
| Retriever | `rag/retriever.py` | Dual-search strategy: filtered by user_type + unfiltered, merge & deduplicate |
| Synthetic Data Gen | `data/synthetic_gen.py` | Script tạo 50+ test cases đa dạng (fact-check, adversarial, edge-case, multi-turn, out-of-scope) với ground_truth_ids |
| Config | `config.py` | Centralized config (API keys, model names, ChromaDB path, collection name) |

---

## 2. Đóng góp kỹ thuật cụ thể

### 2.1 ChromaDB Setup & Embedding
Tôi chọn `text-embedding-3-small` của OpenAI thay vì `ada-002` vì chi phí thấp hơn ~5x trong khi chất lượng embedding tương đương cho tiếng Việt. ChromaDB được cấu hình dạng `PersistentClient` để dữ liệu tồn tại giữa các lần chạy, tránh phải re-embed mỗi lần benchmark.

Collection `xanhsm_qa` lưu mỗi Q&A pair như một document với metadata đầy đủ (`user_type`, `category`, `question`, `answer`) — thiết kế này cho phép filter theo `user_type` trong retrieval, giúp tăng precision khi biết context người dùng.

### 2.2 Dual-Search Retrieval Strategy
`rag/retriever.py` thực hiện hai lần query song song:
1. **Filtered search** — chỉ lấy chunk của đúng `user_type` (tài xế bike / taxi / nhà hàng)
2. **Unfiltered search** — lấy toàn bộ collection để bắt các câu hỏi cross-domain

Kết quả được merge và deduplicate theo question text, với typed results được ưu tiên. Chiến lược này giúp đạt Hit Rate = 100% trên toàn bộ test set.

### 2.3 Synthetic Data Generator (SDG)
`data/synthetic_gen.py` là module tôi đầu tư nhiều nhất. Thiết kế gồm 5 loại test case:

- **Fact-check** (2 per doc): kiểm tra thông tin cụ thể, độ khó easy/medium
- **Adversarial** (1 per doc, 8 docs): prompt injection, false premise, goal hijacking
- **Edge-case** (1 per doc, 8 docs): câu hỏi mơ hồ, thiếu thông tin, tình huống ngoại lệ
- **Out-of-scope** (4 cases): câu hỏi hoàn toàn ngoài domain XanhSM
- **Multi-turn** (4 cases): hội thoại 2 lượt với follow_up_question và follow_up_ground_truth_ids

Mỗi case đều có `ground_truth_ids` mapping về document ID trong ChromaDB — đây là điều kiện bắt buộc để tính Hit Rate và MRR trong retrieval evaluation.

Tôi cũng implement hàm `_is_behavioral_expected_answer()` để lọc bỏ các case mà LLM sinh ra expected_answer dạng mô tả hành vi ("AI phải...", "AI từ chối...") thay vì câu trả lời thực tế — lỗi này xuất hiện khá thường xuyên khi dùng GPT-4o-mini để generate.

### 2.4 Facebook Data Ingestion
`rag/ingest_facebook.py` crawl dữ liệu từ Facebook group scraper output, map `groupTitle` → `user_type` và ingest post text + top comments vào ChromaDB. Đây là nguồn dữ liệu community bổ sung cho dữ liệu chính thức trong `qa.json`.

---

## 3. Kiến thức kỹ thuật học được

### Hit Rate vs MRR
- **Hit Rate@K**: Binary metric — 1 nếu có ít nhất 1 expected document trong top-K retrieved, 0 nếu không. Đơn giản nhưng không phân biệt được "tìm thấy ở vị trí 1" vs "tìm thấy ở vị trí K".
- **MRR (Mean Reciprocal Rank)**: `1/rank` của expected document đầu tiên. MRR = 1.0 nghĩa là document đúng luôn ở vị trí 1 — đây là kết quả lý tưởng và hệ thống của nhóm đạt được điều này.

Kết quả Hit Rate = MRR = 1.0 trên toàn bộ test set cho thấy embedding `text-embedding-3-small` + ChromaDB hoạt động rất tốt với dữ liệu Q&A tiếng Việt của XanhSM.

### Embedding Model Trade-off
`text-embedding-3-small` có 1536 dimensions, chi phí ~$0.02/1M tokens. So với `text-embedding-3-large` (3072 dims, ~$0.13/1M tokens), small model đủ tốt cho domain-specific Q&A vì vocabulary hẹp và câu hỏi có pattern lặp lại.

### Chunking Strategy & Limitations
Thiết kế hiện tại dùng fixed Q&A pair làm đơn vị chunk — đơn giản và hiệu quả cho dữ liệu có cấu trúc. Tuy nhiên, failure case #1 (phí bảo hiểm) cho thấy giới hạn: khi một chunk chứa thông tin của nhiều loại dịch vụ, LLM có thể tổng hợp sai. Semantic chunking hoặc tách chunk theo loại dịch vụ sẽ giải quyết vấn đề này.

---

## 4. Vấn đề gặp phải và cách giải quyết

### Vấn đề 1: ChromaDB path resolution trên Windows
ChromaDB PersistentClient cần absolute path. Khi chạy từ các thư mục khác nhau, relative path `.chromadb` bị resolve sai. Giải pháp: dùng `os.path.dirname(os.path.abspath(__file__))` để tính absolute path từ vị trí file `vectorstore.py`.

### Vấn đề 2: SDG sinh ra expected_answer dạng mô tả hành vi
GPT-4o-mini thường sinh ra expected_answer kiểu "AI phải từ chối câu hỏi này" thay vì câu trả lời thực tế. Giải pháp: implement `_is_behavioral_expected_answer()` với danh sách marker keywords để filter và loại bỏ các case này trước khi lưu.

### Vấn đề 3: Deduplication trong SDG
Khi generate nhiều batch song song, có thể sinh ra câu hỏi trùng lặp. Giải pháp: dùng `seen_q` set để track question text đã xuất hiện, chỉ thêm case mới nếu question chưa có trong set.

### Vấn đề 4: ground_truth_ids không nhất quán
LLM đôi khi sinh ra `ground_truth_ids` là string thay vì list, hoặc chứa ID không tồn tại. Giải pháp: implement `_normalize_ground_truth_ids()` để normalize về list[str], fallback về `[doc_id]` nếu rỗng.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm | Lý do |
|----------|---------|-------|
| Engineering Contribution | 14/15 | Hoàn thành đầy đủ RAG infrastructure + SDG với quality checks. Thiếu 1 điểm vì chưa implement semantic chunking. |
| Technical Depth | 13/15 | Hiểu rõ Hit Rate, MRR, embedding trade-off. Chưa đi sâu vào Cohen's Kappa và position bias detection (đã có code nhưng chưa chạy trong pipeline chính). |
| Problem Solving | 9/10 | Giải quyết được 4/4 vấn đề phát sinh. Vấn đề chunking strategy chưa được fix trong thời gian lab. |

**Tổng tự đánh giá: 36/40**

---

## 6. Điều rút ra

Phần khó nhất không phải là code — mà là **thiết kế golden dataset chất lượng**. Một test case tốt cần:
1. `ground_truth_ids` chính xác để tính retrieval metrics
2. `expected_answer` là câu trả lời thực tế, không phải mô tả hành vi
3. Đủ đa dạng về type và difficulty để phát hiện được nhiều loại lỗi khác nhau

Nếu làm lại, tôi sẽ dành nhiều thời gian hơn để manually review 10-15% test cases thay vì tin hoàn toàn vào LLM-generated data. Garbage in, garbage out — áp dụng cho cả evaluation pipeline.
