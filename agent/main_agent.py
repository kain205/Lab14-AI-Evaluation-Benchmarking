import asyncio
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from openai import AsyncOpenAI
from rag.vectorstore import get_collection
from rag.tools.query_rewriter import rewrite_query
from config import OPENAI_API_KEY, OPENAI_MODEL

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SYSTEM_FULL = """\
Bạn là Trợ lý AI Hỗ trợ của Xanh SM. Ưu tiên sự chính xác hơn sự đầy đủ.

Quy tắc:
- Trả lời bằng ngôn ngữ người dùng đã sử dụng.
- Trả lời dựa trên thông tin trong <context>. Không dùng kiến thức bên ngoài.
- Nếu không có thông tin liên quan trong context, hãy nói rõ bạn không tìm thấy.
- Từ chối câu hỏi không liên quan đến dịch vụ XanhSM.
{clarification_rule}
<context>
{context}
</context>"""

# V3 dùng full XanhSM production prompt (bỏ feedback_policy và tool rule)
SYSTEM_XANHSM = """\
<persona>
Bạn là Trợ lý AI Hỗ trợ của Xanh SM.
Mục tiêu chính của bạn là cung cấp thông tin chính xác, an toàn và cập nhật nhất.
Bạn LUÔN ưu tiên sự chính xác hơn sự đầy đủ.
Nếu bạn không chắc chắn, hãy nói rõ và chuyển lên cấp trên xử lý.
</persona>

<rules>
- Trả lời bằng ngôn ngữ người dùng đã sử dụng trong câu hỏi.
- Nếu câu hỏi không rõ, hãy hỏi lại khách hàng để làm rõ câu hỏi.
- Nếu câu trả lời có sự khác biệt giữa tài xế bike và tài xế taxi (ví dụ: lương, chính sách, quyền lợi), hãy hỏi người dùng họ là tài xế bike hay tài xế taxi trước khi trả lời.
- Trả lời câu hỏi dựa trên thông tin được cung cấp trong phần <context> bên dưới. Không sử dụng kiến thức bên ngoài phần này.
</rules>

<constraints>
- Nếu không tìm thấy thông tin liên quan trong phần context, hãy trả lời rằng bạn không tìm thấy thông tin, không bịa câu trả lời.
- Từ chối mọi câu hỏi không liên quan đến dịch vụ của XanhSM (VD: viết code, làm bài tập, tư vấn tài chính, chính trị).
</constraints>

<context>
{context}
</context>"""

CLARIFICATION_RULE = """\
- Nếu câu hỏi mơ hồ hoặc thiếu thông tin cần thiết để trả lời chính xác \
(ví dụ: không rõ loại dịch vụ, không rõ thành phố, không rõ vai trò tài xế/hành khách), \
hãy hỏi lại người dùng để làm rõ trước khi trả lời.
"""


async def _retrieve(query: str, top_k: int = 5) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    chunks = []
    if results and results["metadatas"]:
        for doc_id, meta in zip(results["ids"][0], results["metadatas"][0]):
            chunks.append({
                "id": doc_id,
                "question": meta.get("question", ""),
                "answer": meta.get("answer", ""),
            })
    return chunks


def _build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(Không tìm thấy thông tin liên quan.)"
    return "\n\n".join(f"Q: {c['question']}\nA: {c['answer']}" for c in chunks)


async def _call_llm(messages: list[dict]) -> str:
    import random
    max_retries = 5
    for attempt in range(max_retries):
        try:
            resp = await _client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries - 1:
                # Extract wait time from error if available
                wait_time = 2 ** attempt * 5 + random.uniform(0, 2)  # Add jitter
                print(f"⏳ Rate limit (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            else:
                raise


class AgentV1:
    """V1 — Query rewrite + full system prompt."""

    name = "XanhSM-V1-Rewrite"

    async def query(self, question: str, history: list = None, **_) -> dict:
        rag_query = await rewrite_query(question, history or [])
        chunks = await _retrieve(rag_query, top_k=5)
        system = SYSTEM_FULL.format(
            context=_build_context(chunks),
            clarification_rule="",
        )
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        answer = await _call_llm(messages)
        return {
            "answer": answer,
            "contexts": [c["question"] for c in chunks],
            "retrieved_ids": [c["id"] for c in chunks],
            "metadata": {"agent": self.name, "chunks_used": len(chunks), "rewritten_query": rag_query},
        }


class AgentV3:
    """V3 — Query rewrite + full prompt + ask for clarification on ambiguous questions."""

    name = "XanhSM-V3-Clarify"

    async def query(self, question: str, history: list = None, **_) -> dict:
        rag_query = await rewrite_query(question, history or [])
        chunks = await _retrieve(rag_query, top_k=5)
        system = SYSTEM_FULL.format(
            context=_build_context(chunks),
            clarification_rule=CLARIFICATION_RULE,
        )
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        answer = await _call_llm(messages)
        return {
            "answer": answer,
            "contexts": [c["question"] for c in chunks],
            "retrieved_ids": [c["id"] for c in chunks],
            "metadata": {"agent": self.name, "chunks_used": len(chunks), "rewritten_query": rag_query},
        }


def MainAgent() -> AgentV1:
    return AgentV1()


if __name__ == "__main__":
    async def _test():
        for AgentClass in [AgentV1, AgentV3]:
            agent = AgentClass()
            resp = await agent.query("Làm thế nào để đăng ký làm tài xế Xanh SM?")
            print(f"\n[{agent.name}]")
            print(f"Answer: {resp['answer'][:200]}")
            print(f"Chunks: {resp['metadata']['chunks_used']}")

    asyncio.run(_test())
