"""
Synthetic Data Generator (SDG) for Lab14 AI Evaluation Benchmarking.
Reads all QA pairs from ChromaDB (XanhSM dataset) and uses GPT-4o to generate
50+ diverse test cases including adversarial, edge-case, and multi-turn scenarios.
Each case includes ground_truth_ids for Retrieval Hit Rate evaluation.
"""

import json
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI
from dotenv import load_dotenv
from rag.vectorstore import get_collection

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên gia thiết kế bộ dữ liệu đánh giá (evaluation dataset) cho AI chatbot hỗ trợ khách hàng của Xanh SM - dịch vụ xe công nghệ xanh tại Việt Nam.

Nhiệm vụ: Từ cặp (câu hỏi, câu trả lời) gốc, hãy tạo ra các test case đa dạng theo loại được yêu cầu.

Mỗi test case phải có định dạng JSON với các trường:
- question: câu hỏi của người dùng (tiếng Việt, tự nhiên)
- expected_answer: câu trả lời kỳ vọng ngắn gọn, chính xác (dựa trên nội dung gốc)
- context: đoạn context liên quan (trích từ câu trả lời gốc, tối đa 300 ký tự)
- ground_truth_ids: list các ID tài liệu liên quan (dùng ID được cung cấp)
- metadata: object chứa difficulty (easy/medium/hard), type (fact-check/adversarial/edge-case/multi-turn/out-of-scope), user_type

Chỉ trả về JSON array, không có text thêm."""

GENERATION_PROMPTS = {
    "fact_check": """Từ cặp QA sau, tạo {n} câu hỏi kiểm tra sự thật (fact-check) với độ khó khác nhau.
Câu hỏi phải hỏi về thông tin cụ thể trong câu trả lời gốc.

ID tài liệu: {doc_id}
Câu hỏi gốc: {question}
Câu trả lời gốc: {answer}

Tạo {n} test cases dạng JSON array.""",

    "adversarial": """Từ cặp QA sau, tạo {n} câu hỏi adversarial (tấn công/lừa AI):
- Prompt injection: cố lừa AI bỏ qua context
- Câu hỏi mơ hồ/đánh lừa
- Yêu cầu thông tin sai lệch so với tài liệu

ID tài liệu: {doc_id}
Câu hỏi gốc: {question}
Câu trả lời gốc: {answer}

Tạo {n} test cases dạng JSON array.""",

    "edge_case": """Từ cặp QA sau, tạo {n} edge cases:
- Câu hỏi ngoài phạm vi tài liệu (AI phải nói "không biết")
- Câu hỏi thiếu thông tin/mơ hồ
- Câu hỏi về tình huống đặc biệt/hiếm gặp

ID tài liệu: {doc_id}
Câu hỏi gốc: {question}
Câu trả lời gốc: {answer}

Tạo {n} test cases dạng JSON array.""",
}

# ── Core generation functions ─────────────────────────────────────────────────

async def generate_cases_for_doc(doc_id: str, question: str, answer: str,
                                  user_type: str, case_type: str, n: int = 2) -> list[dict]:
    prompt_template = GENERATION_PROMPTS.get(case_type, GENERATION_PROMPTS["fact_check"])
    user_prompt = prompt_template.format(
        doc_id=doc_id, question=question, answer=answer, n=n
    )

    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)

        # Handle both {"cases": [...]} and direct array wrapped in object
        if isinstance(parsed, dict):
            cases = next((v for v in parsed.values() if isinstance(v, list)), [])
        else:
            cases = parsed

        # Normalise and validate each case
        valid = []
        for c in cases:
            if not isinstance(c, dict):
                continue
            if not c.get("question") or not c.get("expected_answer"):
                continue
            # Ensure required fields
            c.setdefault("context", answer[:300])
            c.setdefault("ground_truth_ids", [doc_id])
            c.setdefault("metadata", {})
            c["metadata"].setdefault("difficulty", "medium")
            c["metadata"].setdefault("type", case_type.replace("_", "-"))
            c["metadata"].setdefault("user_type", user_type)
            valid.append(c)
        return valid

    except Exception as e:
        print(f"  ⚠️  Error generating {case_type} for {doc_id}: {e}")
        return []


async def generate_out_of_scope_cases(n: int = 5) -> list[dict]:
    """Generate questions completely outside XanhSM's domain."""
    prompt = f"""Tạo {n} câu hỏi hoàn toàn ngoài phạm vi dịch vụ Xanh SM (xe công nghệ).
Ví dụ: hỏi về nấu ăn, thời tiết, lịch sử, v.v.
AI phải trả lời "Tôi không có thông tin về vấn đề này" hoặc tương tự.

Trả về JSON object với key "cases" chứa array, mỗi phần tử có:
- question, expected_answer, context, ground_truth_ids (empty list), metadata"""

    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        cases = next((v for v in parsed.values() if isinstance(v, list)), [])
        for c in cases:
            c.setdefault("context", "")
            c.setdefault("ground_truth_ids", [])
            c.setdefault("metadata", {})
            c["metadata"]["difficulty"] = "hard"
            c["metadata"]["type"] = "out-of-scope"
            c["metadata"]["user_type"] = "nguoi_dung"
        return cases
    except Exception as e:
        print(f"  ⚠️  Error generating out-of-scope cases: {e}")
        return []


async def generate_multi_turn_cases(docs: list[dict], n: int = 5) -> list[dict]:
    """Generate multi-turn conversation test cases."""
    import random
    sample = random.sample(docs, min(n, len(docs)))
    cases = []
    for doc in sample:
        prompt = f"""Từ cặp QA sau, tạo 1 test case multi-turn (hội thoại nhiều lượt):
Câu hỏi thứ 2 phải phụ thuộc vào câu trả lời thứ 1.

ID: {doc['id']}
Câu hỏi gốc: {doc['question']}
Câu trả lời gốc: {doc['answer'][:400]}

Trả về JSON object với key "cases" chứa 1 test case có thêm trường:
- follow_up_question: câu hỏi tiếp theo
- follow_up_expected: câu trả lời kỳ vọng cho câu hỏi tiếp theo"""

        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            batch = next((v for v in parsed.values() if isinstance(v, list)), [])
            for c in batch:
                c.setdefault("context", doc["answer"][:300])
                c.setdefault("ground_truth_ids", [doc["id"]])
                c.setdefault("metadata", {})
                c["metadata"]["difficulty"] = "hard"
                c["metadata"]["type"] = "multi-turn"
                c["metadata"]["user_type"] = doc.get("user_type", "nguoi_dung")
                cases.append(c)
        except Exception as e:
            print(f"  ⚠️  Error generating multi-turn for {doc['id']}: {e}")
    return cases


# ── Main orchestrator ─────────────────────────────────────────────────────────

async def main():
    print("📚 Loading documents from ChromaDB...")
    collection = get_collection()
    result = collection.get(include=["metadatas", "documents"])

    ids = result["ids"]
    metadatas = result["metadatas"]

    docs = []
    for doc_id, meta in zip(ids, metadatas):
        docs.append({
            "id": doc_id,
            "question": meta.get("question", ""),
            "answer": meta.get("answer", ""),
            "user_type": meta.get("user_type", "nguoi_dung"),
        })

    print(f"  ✅ Loaded {len(docs)} documents from ChromaDB")

    # Select a diverse sample to generate from (spread across user types)
    import random
    random.seed(42)

    # Pick ~20 docs spread across user types for generation
    by_type: dict[str, list] = {}
    for d in docs:
        by_type.setdefault(d["user_type"], []).append(d)

    selected = []
    for ut, group in by_type.items():
        n = min(6, len(group))
        selected.extend(random.sample(group, n))

    print(f"  📋 Selected {len(selected)} source docs for generation")

    all_cases: list[dict] = []

    # 1. Fact-check cases (easy/medium) — 2 per doc
    print("\n🔨 Generating fact-check cases...")
    tasks = [
        generate_cases_for_doc(d["id"], d["question"], d["answer"], d["user_type"], "fact_check", n=2)
        for d in selected
    ]
    batches = await asyncio.gather(*tasks)
    fact_cases = [c for b in batches for c in b]
    print(f"  ✅ {len(fact_cases)} fact-check cases")
    all_cases.extend(fact_cases)

    # 2. Adversarial cases — 1 per doc (subset)
    print("\n⚔️  Generating adversarial cases...")
    adv_docs = random.sample(selected, min(10, len(selected)))
    tasks = [
        generate_cases_for_doc(d["id"], d["question"], d["answer"], d["user_type"], "adversarial", n=1)
        for d in adv_docs
    ]
    batches = await asyncio.gather(*tasks)
    adv_cases = [c for b in batches for c in b]
    print(f"  ✅ {len(adv_cases)} adversarial cases")
    all_cases.extend(adv_cases)

    # 3. Edge cases — 1 per doc (subset)
    print("\n🔬 Generating edge cases...")
    edge_docs = random.sample(selected, min(10, len(selected)))
    tasks = [
        generate_cases_for_doc(d["id"], d["question"], d["answer"], d["user_type"], "edge_case", n=1)
        for d in edge_docs
    ]
    batches = await asyncio.gather(*tasks)
    edge_cases = [c for b in batches for c in b]
    print(f"  ✅ {len(edge_cases)} edge cases")
    all_cases.extend(edge_cases)

    # 4. Out-of-scope cases
    print("\n🚫 Generating out-of-scope cases...")
    oos_cases = await generate_out_of_scope_cases(n=5)
    print(f"  ✅ {len(oos_cases)} out-of-scope cases")
    all_cases.extend(oos_cases)

    # 5. Multi-turn cases
    print("\n💬 Generating multi-turn cases...")
    mt_cases = await generate_multi_turn_cases(docs, n=5)
    print(f"  ✅ {len(mt_cases)} multi-turn cases")
    all_cases.extend(mt_cases)

    print(f"\n📊 Total generated: {len(all_cases)} cases")

    # Deduplicate by question text
    seen_q: set[str] = set()
    unique_cases = []
    for c in all_cases:
        q = c.get("question", "").strip()
        if q and q not in seen_q:
            seen_q.add(q)
            unique_cases.append(c)

    print(f"📊 After dedup: {len(unique_cases)} unique cases")

    # Ensure at least 50
    if len(unique_cases) < 50:
        print(f"⚠️  Only {len(unique_cases)} cases — generating more fact-check cases...")
        extra_docs = random.sample(docs, min(20, len(docs)))
        tasks = [
            generate_cases_for_doc(d["id"], d["question"], d["answer"], d["user_type"], "fact_check", n=2)
            for d in extra_docs
        ]
        batches = await asyncio.gather(*tasks)
        for c in [c for b in batches for c in b]:
            q = c.get("question", "").strip()
            if q and q not in seen_q:
                seen_q.add(q)
                unique_cases.append(c)
        print(f"  ✅ Now {len(unique_cases)} cases")

    # Save
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for case in unique_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(unique_cases)} test cases to data/golden_set.jsonl")

    # Summary breakdown
    by_type_count: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    for c in unique_cases:
        t = c.get("metadata", {}).get("type", "unknown")
        d = c.get("metadata", {}).get("difficulty", "unknown")
        by_type_count[t] = by_type_count.get(t, 0) + 1
        by_diff[d] = by_diff.get(d, 0) + 1

    print("\n📈 Breakdown by type:")
    for t, cnt in sorted(by_type_count.items()):
        print(f"  {t}: {cnt}")
    print("\n📈 Breakdown by difficulty:")
    for d, cnt in sorted(by_diff.items()):
        print(f"  {d}: {cnt}")


if __name__ == "__main__":
    asyncio.run(main())
