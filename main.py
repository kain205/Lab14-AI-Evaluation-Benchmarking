import asyncio
import json
import os
import time

from dotenv import load_dotenv
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from agent.main_agent import AgentV1, AgentV2, AgentV3

load_dotenv()

# ── Cấu hình ──────────────────────────────────────────────────────────────────
MAX_CASES = 10  # Giới hạn số test cases mỗi lần chạy (None = chạy hết)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ── Giữ nguyên tên class từ template, fill thật vào ──────────────────────────

class ExpertEvaluator:
    """Retrieval evaluator — stub cho đến khi retrieval_eval được wire đầy đủ."""
    async def score(self, case, resp):
        return {
            "faithfulness": 0.0,
            "relevancy": 0.0,
            "retrieval": {"hit_rate": 0.0, "mrr": 0.0},
        }

class MultiModelJudge:
    """Wrapper giữ tên gốc từ template, delegate sang LLMJudge thật."""
    def __init__(self):
        self._judge = LLMJudge(
            openai_api_key=OPENAI_API_KEY,
            gemini_api_key=GEMINI_API_KEY,
        )

    async def evaluate_multi_judge(self, q, a, gt):
        return await self._judge.evaluate_multi_judge(q, a, gt)


# ── Giữ nguyên signature gốc, thêm agent param ───────────────────────────────

async def run_benchmark_with_results(agent_version: str, agent=None):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    if MAX_CASES is not None:
        dataset = dataset[:MAX_CASES]
        print(f"  ⚡ Giới hạn {MAX_CASES} cases (đổi MAX_CASES trong main.py để chạy hết)")

    if agent is None:
        agent = AgentV1()

    runner = BenchmarkRunner(agent, ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)

    for r in results:
        r["agent_version"] = agent_version

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
        }
    }
    return results, summary

async def run_benchmark(version, agent=None):
    _, summary = await run_benchmark_with_results(version, agent)
    return summary


async def main():
    v1_summary = await run_benchmark("Agent_V1_Base", AgentV1())
    v2_summary = await run_benchmark("Agent_V2_Rewrite", AgentV2())
    v3_results, v3_summary = await run_benchmark_with_results("Agent_V3_Clarify", AgentV3())

    if not v1_summary or not v3_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    for label, s in [("V1-Base", v1_summary), ("V2-Rewrite", v2_summary), ("V3-Clarify", v3_summary)]:
        if s:
            print(f"  {label}: score={s['metrics']['avg_score']:.2f}  agreement={s['metrics']['agreement_rate']:.2f}")

    delta = v3_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"\nDelta V3 vs V1: {'+' if delta >= 0 else ''}{delta:.2f}")

    os.makedirs("reports", exist_ok=True)
    comparison = {
        "summaries": {
            "V1-Base":    v1_summary,
            "V2-Rewrite": v2_summary,
            "V3-Clarify": v3_summary,
        },
        "delta_v3_vs_v1": round(delta, 3),
    }
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"V1-Base": v1_results, "V2-Rewrite": v2_results, "V3-Clarify": v3_results},
            f, ensure_ascii=False, indent=2,
        )

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")


if __name__ == "__main__":
    asyncio.run(main())
