from typing import List, Dict, Optional

class RetrievalEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def _normalize_ids(ids: List[str]) -> List[str]:
        normalized = []
        seen = set()
        for item in ids or []:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: Optional[int] = None,
    ) -> float:
        """
        Hit Rate = 1 nếu có ít nhất một expected_id xuất hiện trong retrieved_ids.
        Nếu top_k được truyền vào thì chỉ xét top_k kết quả đầu tiên.
        Với out-of-scope (không có expected_ids), trả về 1.0 để không phạt retrieval.
        """
        expected = self._normalize_ids(expected_ids)
        retrieved = self._normalize_ids(retrieved_ids)

        if not expected:
            return 1.0

        top_retrieved = retrieved if top_k is None else retrieved[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Mean Reciprocal Rank cho một query:
        Tìm vị trí đầu tiên của expected_id trong retrieved_ids.
        MRR = 1 / rank (rank bắt đầu từ 1). Nếu không thấy thì là 0.
        Với out-of-scope (không có expected_ids), trả về 1.0.
        """
        expected = set(self._normalize_ids(expected_ids))
        retrieved = self._normalize_ids(retrieved_ids)

        if not expected:
            return 1.0

        for i, doc_id in enumerate(retrieved):
            if doc_id in expected:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Chạy retrieval eval cho toàn bộ bộ dữ liệu.
        Mỗi item có thể chứa:
        - expected_retrieval_ids hoặc ground_truth_ids
        - retrieved_ids (trực tiếp) hoặc response.retrieved_ids
        """
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0, "total": 0}

        hit_rates = []
        mrr_scores = []

        for item in dataset:
            expected_ids = item.get("expected_retrieval_ids") or item.get("ground_truth_ids") or []
            retrieved_ids = item.get("retrieved_ids")

            if retrieved_ids is None and isinstance(item.get("response"), dict):
                retrieved_ids = item["response"].get("retrieved_ids", [])

            retrieved_ids = retrieved_ids or []

            hit_rates.append(self.calculate_hit_rate(expected_ids, retrieved_ids))
            mrr_scores.append(self.calculate_mrr(expected_ids, retrieved_ids))

        total = len(dataset)
        return {
            "avg_hit_rate": sum(hit_rates) / total,
            "avg_mrr": sum(mrr_scores) / total,
            "total": total,
        }
