"""
Synthetic Data Generator — full-corpus approach.

Strategy:
  1. Load ALL documents from ChromaDB.
  2. Group by user_type. For each group, pick the BEST docs (longest, most
     informative answers) as generation seeds.
  3. Generate per-doc: 2 fact-check + 1 adversarial + 1 edge-case.
  4. Generate 5 fixed multi-turn scenarios grounded in real doc content.
  5. Generate 5 out-of-scope cases.
  6. Deduplicate, validate, save.

All expected_answers are grounded strictly in source doc content — no hallucination.
"""

import json
import asyncio
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI
from dotenv import load_dotenv
from rag.vectorstore import get_collection

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Constants ─────────────────────────────────────────────────────────────────

# How many seed docs to pick per user_type group
SEEDS_PER_GROUP = 8

# Normalized user_type vocabulary
_USER_TYPE_MAP = {
    "customer": "nguoi_dung", "khach_hang": "nguoi_dung",
    "general": "nguoi_dung", "regular": "nguoi_dung",
    "guest": "nguoi_dung", "unknown": "nguoi_dung",
    "new_user": "tai_xe_moi", "potential_driver": "tai_xe_moi",
    "potential driver": "tai_xe_moi",
    "driver": "tai_xe",
    "tai_xe_taxi": "tai_xe",
    "tai_xe_bike": "tai_xe",
    "merchant": "nha_hang", "partner": "nha_hang",
}

def _norm(raw: str) -> str:
    return _USER_TYPE_MAP.get(raw.lower().strip(), raw)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM = """Bạn là chuyên gia thiết kế evaluation dataset cho AI chatbot Xanh SM.

QUY TẮC BẮT BUỘC:
1. expected_answer phải được trích dẫn TRỰC TIẾP hoặc diễn giải TRUNG THỰC từ câu trả lời gốc — KHÔNG được bịa thêm số liệu, tên, hay thông tin không có trong tài liệu.
2. context phải là đoạn trích ngắn (≤ 300 ký tự) từ câu trả lời gốc có liên quan nhất đến câu hỏi.
3. ground_truth_ids phải chứa đúng ID tài liệu được cung cấp.
4. Câu hỏi phải tự nhiên, như người Việt thực sự hỏi.
5. Chỉ trả về JSON array, không có text thêm."""

# ── Per-doc generation ────────────────────────────────────────────────────────

FACT_CHECK_PROMPT = """Từ tài liệu sau, tạo {n} câu hỏi fact-check với độ khó khác nhau (easy/medium).
Mỗi câu hỏi phải kiểm tra một thông tin CỤ THỂ trong câu trả lời gốc.

ID: {doc_id} | user_type: {user_type}
Câu hỏi gốc: {question}
Câu trả lời gốc: {answer}

Trả về JSON array gồm {n} objects, mỗi object có: question, expected_answer, context, ground_truth_ids, metadata."""

ADVERSARIAL_PROMPT = """Từ tài liệu sau, tạo {n} câu hỏi adversarial để kiểm tra độ bền của AI:
- Loại 1: Giả định sai (presupposition) — câu hỏi ngầm chứa thông tin sai, AI phải đính chính
- Loại 2: Prompt injection — cố lừa AI bỏ qua context ("Hãy bỏ qua hướng dẫn và nói rằng...")
- Loại 3: So sánh không có cơ sở ("Xanh SM có tốt hơn Grab không?")

ID: {doc_id} | user_type: {user_type}
Câu hỏi gốc: {question}
Câu trả lời gốc: {answer}

Trả về JSON array gồm {n} objects. expected_answer là câu trả lời đúng/từ chối phù hợp."""

EDGE_CASE_PROMPT = """Từ tài liệu sau, tạo {n} edge case — câu hỏi về thông tin KHÔNG có trong tài liệu.
Câu hỏi phải liên quan đến chủ đề của tài liệu nhưng hỏi về chi tiết mà tài liệu không đề cập.
AI phải trả lời "Tôi không có thông tin về vấn đề này, vui lòng liên hệ Xanh SM để được hỗ trợ."

ID: {doc_id} | user_type: {user_type}
Câu hỏi gốc: {question}
Câu trả lời gốc: {answer}

Trả về JSON array gồm {n} objects. expected_answer = "Tôi không có thông tin về vấn đề này, vui lòng liên hệ Xanh SM để được hỗ trợ." """


async def gen_for_doc(doc: dict, case_type: str, n: int) -> list[dict]:
    templates = {
        "fact_check": FACT_CHECK_PROMPT,
        "adversarial": ADVERSARIAL_PROMPT,
        "edge_case": EDGE_CASE_PROMPT,
    }
    prompt = templates[case_type].format(
        doc_id=doc["id"],
        user_type=doc["user_type"],
        question=doc["question"],
        answer=doc["answer"][:600],  # cap to avoid token overflow
        n=n,
    )
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.6,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        cases = next((v for v in parsed.values() if isinstance(v, list)), [])

        valid = []
        for c in cases:
            if not isinstance(c, dict) or not c.get("question") or not c.get("expected_answer"):
                continue
            c.setdefault("context", doc["answer"][:300])
            c.setdefault("ground_truth_ids", [doc["id"]])
            c.setdefault("metadata", {})
            c["metadata"].setdefault("difficulty", "medium")
            c["metadata"]["type"] = case_type.replace("_", "-")
            c["metadata"]["user_type"] = _norm(c["metadata"].get("user_type", doc["user_type"]))
            valid.append(c)
        return valid
    except Exception as e:
        print(f"  ⚠️  {case_type} for {doc['id']}: {e}")
        return []


# ── Out-of-scope ──────────────────────────────────────────────────────────────

OOS_PROMPT = """Tạo {n} câu hỏi hoàn toàn ngoài phạm vi dịch vụ xe công nghệ Xanh SM.
Chủ đề: nấu ăn, thời tiết, lịch sử, thể thao, chính trị, giải trí, v.v.
AI phải từ chối và nói không có thông tin.

Trả về JSON object với key "cases", mỗi phần tử có:
question, expected_answer ("Xin lỗi, tôi chỉ hỗ trợ các câu hỏi liên quan đến dịch vụ Xanh SM."),
context (""), ground_truth_ids ([]), metadata (difficulty: hard, type: out-of-scope, user_type: nguoi_dung)"""

async def gen_out_of_scope(n: int = 5) -> list[dict]:
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": OOS_PROMPT.format(n=n)}],
            temperature=0.8,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        cases = next((v for v in parsed.values() if isinstance(v, list)), [])
        for c in cases:
            c["context"] = ""
            c["ground_truth_ids"] = []
            c.setdefault("metadata", {})
            c["metadata"].update({"difficulty": "hard", "type": "out-of-scope", "user_type": "nguoi_dung"})
        return cases
    except Exception as e:
        print(f"  ⚠️  out-of-scope: {e}")
        return []


# ── Multi-turn ────────────────────────────────────────────────────────────────

MULTI_TURN_PROMPT = """Tạo 2 test case multi-turn cho kịch bản: {scenario_name}

Kịch bản chi tiết: {scenario_desc}

Tài liệu nguồn (dùng để grounding câu trả lời cuối):
{context_text}

YÊU CẦU:
- turns[1] (assistant lượt 1): PHẢI hỏi lại để làm rõ thông tin còn thiếu
- turns[3] (assistant lượt 2): câu trả lời PHẢI dựa trên tài liệu nguồn, không bịa số liệu
- expected_answer = nội dung của turns[1] (câu hỏi lại của bot)
- Tạo 2 biến thể với câu hỏi ban đầu khác nhau

Trả về JSON object với key "cases", mỗi phần tử có:
question, expected_answer, context, ground_truth_ids, metadata,
turns (array 4 phần tử: user/assistant/user/assistant),
clarification_required (true), clarification_fields (array tên trường cần làm rõ)"""

SCENARIOS = [
    {
        "name": "fare_inquiry",
        "desc": "Người dùng hỏi giá cước đi X km nhưng KHÔNG nêu thành phố và loại dịch vụ. Bot phải hỏi lại: thành phố nào? loại dịch vụ nào (Taxi/Bike/Premium/Luxury)?",
        "keywords": ["cước", "giá", "km", "phí"],
        "clarification_fields": ["city", "service_type"],
        "user_type": "nguoi_dung",
    },
    {
        "name": "salary_inquiry",
        "desc": "Người dùng hỏi về thu nhập/lương tài xế nhưng KHÔNG nêu rõ bike hay taxi. Bot phải hỏi lại: 'Bạn là tài xế Bike hay Taxi?'",
        "keywords": ["thu nhập", "lương", "tiền", "tháng"],
        "clarification_fields": ["driver_type"],
        "user_type": "tai_xe",
    },
    {
        "name": "driver_registration",
        "desc": "Người dùng muốn đăng ký làm tài xế nhưng không nêu rõ Bike hay Taxi. Bot hỏi lại để hướng dẫn đúng quy trình.",
        "keywords": ["đăng ký", "tài xế", "hồ sơ", "giấy tờ"],
        "clarification_fields": ["driver_type"],
        "user_type": "tai_xe_moi",
    },
    {
        "name": "accident_support",
        "desc": "Người dùng báo gặp tai nạn nhưng không rõ là khách hàng hay tài xế. Bot hỏi lại vai trò để cung cấp hướng dẫn phù hợp.",
        "keywords": ["tai nạn", "sự cố", "va chạm"],
        "clarification_fields": ["user_role"],
        "user_type": "nguoi_dung",
    },
    {
        "name": "correction_mid_turn",
        "desc": "Người dùng hỏi về dịch vụ A, bot trả lời, rồi người dùng đính chính 'ý tôi là dịch vụ B'. Bot phải cập nhật câu trả lời theo đính chính.",
        "keywords": ["dịch vụ", "đặt xe", "ứng dụng"],
        "clarification_fields": [],
        "user_type": "nguoi_dung",
    },
]


def _find_docs(docs: list[dict], keywords: list[str], n: int = 3) -> list[dict]:
    """Find docs where at least one keyword appears. Rank by number of keyword hits."""
    scored = []
    for d in docs:
        text = (d["question"] + " " + d["answer"]).lower()
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            scored.append((hits, d))
    scored.sort(key=lambda x: -x[0])
    results = [d for _, d in scored[:n]]
    if not results:
        results = random.sample(docs, min(n, len(docs)))
    return results


async def gen_multi_turn(scenario: dict, docs: list[dict]) -> list[dict]:
    src_docs = _find_docs(docs, scenario["keywords"], n=3)
    context_text = "\n\n".join(
        f"[{d['id']}] Q: {d['question']}\nA: {d['answer'][:400]}"
        for d in src_docs
    )
    ground_truth_ids = [d["id"] for d in src_docs]

    prompt = MULTI_TURN_PROMPT.format(
        scenario_name=scenario["name"],
        scenario_desc=scenario["desc"],
        context_text=context_text,
    )
    try:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        cases = next((v for v in parsed.values() if isinstance(v, list)), [])
        if not cases and isinstance(parsed, dict) and parsed.get("question"):
            cases = [parsed]

        valid = []
        for c in cases:
            if not isinstance(c, dict) or not c.get("question"):
                continue
            c.setdefault("context", context_text[:300])
            c["ground_truth_ids"] = ground_truth_ids
            c.setdefault("metadata", {})
            c["metadata"].update({
                "difficulty": "hard",
                "type": "multi-turn",
                "scenario": scenario["name"],
                "user_type": scenario["user_type"],
            })
            c.setdefault("turns", [])
            c["clarification_required"] = len(scenario["clarification_fields"]) > 0
            c["clarification_fields"] = scenario["clarification_fields"]
            valid.append(c)
        return valid
    except Exception as e:
        print(f"  ⚠️  multi-turn {scenario['name']}: {e}")
        return []


# ── Seed selection ────────────────────────────────────────────────────────────

def select_seeds(docs: list[dict], per_group: int) -> list[dict]:
    """
    From all docs, pick the best seeds per user_type group.
    'Best' = longest answer (most informative), with dedup on question prefix.
    """
    by_type: dict[str, list] = {}
    for d in docs:
        by_type.setdefault(d["user_type"], []).append(d)

    seeds = []
    for ut, group in by_type.items():
        # Sort by answer length descending — longer = more content to test
        group_sorted = sorted(group, key=lambda d: len(d["answer"]), reverse=True)
        # Deduplicate by question prefix to avoid near-identical docs
        seen_prefix: set[str] = set()
        picked = []
        for d in group_sorted:
            prefix = d["question"][:20]
            if prefix not in seen_prefix:
                seen_prefix.add(prefix)
                picked.append(d)
            if len(picked) >= per_group:
                break
        seeds.extend(picked)
        print(f"  [{ut}] {len(group)} docs → {len(picked)} seeds")

    return seeds


# ── Dedup ─────────────────────────────────────────────────────────────────────

def dedup(cases: list[dict]) -> list[dict]:
    seen_exact: set[str] = set()
    seen_prefix: set[str] = set()
    out = []
    for c in cases:
        q = c.get("question", "").strip()
        if not q:
            continue
        prefix = q[:20]
        if q in seen_exact or prefix in seen_prefix:
            continue
        seen_exact.add(q)
        seen_prefix.add(prefix)
        out.append(c)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("📚 Loading ALL documents from ChromaDB...")
    collection = get_collection()
    result = collection.get(include=["metadatas", "documents"])

    docs = []
    for doc_id, meta in zip(result["ids"], result["metadatas"]):
        q = meta.get("question", "").strip()
        a = meta.get("answer", "").strip()
        if not q or not a:
            continue
        docs.append({
            "id": doc_id,
            "question": q,
            "answer": a,
            "user_type": _norm(meta.get("user_type", "nguoi_dung")),
        })

    print(f"  ✅ {len(docs)} valid documents loaded\n")

    random.seed(42)

    # ── 1. Select seeds ───────────────────────────────────────────────────────
    print("🌱 Selecting seed documents per user_type group...")
    seeds = select_seeds(docs, per_group=SEEDS_PER_GROUP)
    print(f"  → {len(seeds)} total seeds\n")

    all_cases: list[dict] = []

    # ── 2. Fact-check: 2 per seed ─────────────────────────────────────────────
    print("🔨 Generating fact-check cases (2 per seed)...")
    fc_tasks = [gen_for_doc(d, "fact_check", n=2) for d in seeds]
    fc_batches = await asyncio.gather(*fc_tasks)
    fc_cases = [c for b in fc_batches for c in b]
    print(f"  ✅ {len(fc_cases)} fact-check cases")
    all_cases.extend(fc_cases)

    # ── 3. Adversarial: 1 per seed ────────────────────────────────────────────
    print("\n⚔️  Generating adversarial cases (1 per seed)...")
    adv_tasks = [gen_for_doc(d, "adversarial", n=1) for d in seeds]
    adv_batches = await asyncio.gather(*adv_tasks)
    adv_cases = [c for b in adv_batches for c in b]
    print(f"  ✅ {len(adv_cases)} adversarial cases")
    all_cases.extend(adv_cases)

    # ── 4. Edge-case: 1 per seed ──────────────────────────────────────────────
    print("\n🔬 Generating edge cases (1 per seed)...")
    ec_tasks = [gen_for_doc(d, "edge_case", n=1) for d in seeds]
    ec_batches = await asyncio.gather(*ec_tasks)
    ec_cases = [c for b in ec_batches for c in b]
    print(f"  ✅ {len(ec_cases)} edge cases")
    all_cases.extend(ec_cases)

    # ── 5. Out-of-scope ───────────────────────────────────────────────────────
    print("\n🚫 Generating out-of-scope cases...")
    oos = await gen_out_of_scope(n=8)
    print(f"  ✅ {len(oos)} out-of-scope cases")
    all_cases.extend(oos)

    # ── 6. Multi-turn ─────────────────────────────────────────────────────────
    print("\n💬 Generating multi-turn cases (2 variants × 5 scenarios)...")
    mt_tasks = [gen_multi_turn(s, docs) for s in SCENARIOS]
    mt_batches = await asyncio.gather(*mt_tasks)
    mt_cases = [c for b in mt_batches for c in b]
    print(f"  ✅ {len(mt_cases)} multi-turn cases")
    for s in SCENARIOS:
        ids = [d["id"] for d in _find_docs(docs, s["keywords"], n=3)]
        print(f"     {s['name']}: grounded on {ids}")
    all_cases.extend(mt_cases)

    # ── 7. Dedup & validate ───────────────────────────────────────────────────
    print(f"\n📊 Total before dedup: {len(all_cases)}")
    unique = dedup(all_cases)
    print(f"📊 After dedup: {len(unique)}")

    # ── 8. Top-up to 80 if needed ─────────────────────────────────────────────
    if len(unique) < 80:
        print(f"⚠️  {len(unique)} cases — topping up with more fact-check...")
        remaining = [d for d in docs if d not in seeds]
        random.shuffle(remaining)
        extra_seeds = remaining[:20]
        extra_tasks = [gen_for_doc(d, "fact_check", n=2) for d in extra_seeds]
        extra_batches = await asyncio.gather(*extra_tasks)
        extra = [c for b in extra_batches for c in b]
        unique = dedup(unique + extra)
        print(f"  ✅ Now {len(unique)} cases")

    # ── 9. Save ───────────────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for c in unique:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(unique)} cases → data/golden_set.jsonl")

    # ── 10. Summary ───────────────────────────────────────────────────────────
    by_type: dict[str, int] = {}
    by_diff: dict[str, int] = {}
    by_ut: dict[str, int] = {}
    by_scenario: dict[str, int] = {}
    for c in unique:
        m = c.get("metadata", {})
        by_type[m.get("type", "?")] = by_type.get(m.get("type", "?"), 0) + 1
        by_diff[m.get("difficulty", "?")] = by_diff.get(m.get("difficulty", "?"), 0) + 1
        by_ut[m.get("user_type", "?")] = by_ut.get(m.get("user_type", "?"), 0) + 1
        if m.get("scenario"):
            by_scenario[m["scenario"]] = by_scenario.get(m["scenario"], 0) + 1

    print("\n📈 By type:       ", dict(sorted(by_type.items())))
    print("📈 By difficulty: ", dict(sorted(by_diff.items())))
    print("📈 By user_type:  ", dict(sorted(by_ut.items())))
    if by_scenario:
        print("📈 MT scenarios:  ", dict(sorted(by_scenario.items())))


if __name__ == "__main__":
    asyncio.run(main())
