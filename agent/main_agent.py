import asyncio
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from openai import AsyncOpenAI
from rag.vectorstore import get_collection
from rag.tools.query_rewriter import rewrite_query
from config import OPENAI_API_KEY, OPENAI_MODEL

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

SYSTEM_BASIC = """\
Bạn là trợ lý AI Xanh SM. Trả lời câu hỏi dựa trên thông tin trong <context>.
Nếu không có thông tin liên quan, hãy nói rõ bạn không biết. Không bịa câu trả lời.

<context>
{context}
</context>"""

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
    for attempt in range(4):
        try:
            resp = await _client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if "429" in str(e) and attempt < 3:
                await asyncio.sleep(2 ** attempt * 5)
            else:
                raise


class AgentV1:
    """V1 — Basic RAG: direct ChromaDB query, no rewrite, minimal prompt."""

    name = "XanhSM-V1-Base"

    async def query(self, question: str, history: list = None, **_) -> dict:
        chunks = await _retrieve(question, top_k=5)
        messages = [{"role": "system", "content": SYSTEM_BASIC.format(context=_build_context(chunks))}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        answer = await _call_llm(messages)
        return {
            "answer": answer,
            "contexts": [c["question"] for c in chunks],
            "retrieved_ids": [c["id"] for c in chunks],
            "metadata": {"agent": self.name, "chunks_used": len(chunks)},
        }


class AgentV2:
    """V2 — Query rewrite + full system prompt."""

    name = "XanhSM-V2-Rewrite"

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
        for AgentClass in [AgentV1, AgentV2, AgentV3]:
            agent = AgentClass()
            resp = await agent.query("Làm thế nào để đăng ký làm tài xế Xanh SM?")
            print(f"\n[{agent.name}]")
            print(f"Answer: {resp['answer'][:200]}")
            print(f"Chunks: {resp['metadata']['chunks_used']}")

    asyncio.run(_test())
