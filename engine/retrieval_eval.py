import json
import os
import asyncio
from typing import List, Dict

from rag.retriever import retrieve

GOLDEN_SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "golden_set.jsonl")


class RetrievalEvaluator:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        TODO: Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        TODO: Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict] | None = None) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu từ data/golden_set.jsonl.
        Mỗi case cần có 'question', 'ground_truth_ids', và tùy chọn 'metadata.user_type'.
        """
        if dataset is None:
            with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f if line.strip()]

        hit_rates, mrrs = [], []
        results = []

        for case in dataset:
            question = case.get("question", "")
            expected_ids = case.get("ground_truth_ids", [])
            user_type = case.get("metadata", {}).get("user_type", "")

            chunks = await asyncio.to_thread(retrieve, question, user_type, self.top_k)
            retrieved_ids = [c["id"] for c in chunks]

            hit = self.calculate_hit_rate(expected_ids, retrieved_ids, self.top_k)
            mrr = self.calculate_mrr(expected_ids, retrieved_ids)
            hit_rates.append(hit)
            mrrs.append(mrr)
            results.append({
                "question": question,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
                "hit": hit,
                "mrr": mrr,
            })

        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0

        return {
            "avg_hit_rate": round(avg_hit_rate, 4),
            "avg_mrr": round(avg_mrr, 4),
            "total": len(dataset),
            "details": results,
        }
