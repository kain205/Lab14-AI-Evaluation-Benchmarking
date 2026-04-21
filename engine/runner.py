import asyncio
import time
from typing import List, Dict


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        # Turn 1
        response = await self.agent.query(test_case["question"])

        eval_question = test_case["question"]
        eval_expected = test_case.get("expected_answer", "")
        is_multi_turn = bool(test_case.get("follow_up_question"))

        # Turn 2 — nếu case có follow_up_question
        if is_multi_turn:
            history = [
                {"role": "user",      "content": test_case["question"]},
                {"role": "assistant", "content": response["answer"]},
            ]
            response = await self.agent.query(
                test_case["follow_up_question"],
                history=history,
            )
            eval_question = test_case["follow_up_question"]
            eval_expected = test_case.get("follow_up_expected", eval_expected)

        latency = time.perf_counter() - start_time

        # Eval
        ragas_scores = await self.evaluator.score(test_case, response)
        judge_result = await self.judge.evaluate_multi_judge(
            eval_question,
            response["answer"],
            eval_expected,
        )

        return {
            "test_case": test_case["question"],
            "is_multi_turn": is_multi_turn,
            "eval_question": eval_question,
            "agent_response": response["answer"],
            "latency": latency,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 2) -> List[Dict]:
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
