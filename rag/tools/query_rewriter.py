"""
Query rewriter: viết lại câu hỏi của user thành query độc lập, rõ intent,
trước khi đưa vào cosine similarity search trên vector DB.

Chỉ gọi LLM khi cần (phát hiện follow-up / câu mơ hồ / đại từ thay thế).
Trả về câu gốc ngay nếu câu hỏi đã rõ ràng và độc lập.
"""

import logging

from openai import AsyncOpenAI
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

_REWRITE_MODEL = "gpt-4o-mini"   # model nhẹ, chỉ dùng cho rewrite

_SYSTEM_PROMPT = """\
Bạn là một module xử lý ngôn ngữ tự nhiên. Nhiệm vụ của bạn là viết lại câu hỏi \
của người dùng thành một câu truy vấn độc lập, rõ ràng để tìm kiếm thông tin liên \
quan đến dịch vụ Xanh SM.

Quy tắc:
1. Phân tích lịch sử hội thoại để hiểu đúng intent người dùng.
2. Giải quyết đại từ chỉ định ("nó", "đó", "cái đó", "chính sách đó"...) bằng \
   nội dung cụ thể từ lịch sử.
3. Khai triển câu hỏi follow-up ("còn bike thì sao?", "thế Hà Nội?") thành câu \
   hỏi đầy đủ.
4. Nếu câu hỏi ĐÃ đầy đủ và rõ ràng → trả nguyên câu hỏi, KHÔNG thay đổi.
5. Chỉ trả về câu truy vấn đã viết lại. KHÔNG giải thích, KHÔNG thêm gì khác.
"""

# Heuristics: chỉ gọi LLM khi tin nhắn có dấu hiệu cần viết lại
_FOLLOWUP_SIGNALS = (
    "còn ", "thế ", "vậy ", "đó ", "nó ", "cái đó", "chính sách đó",
    "điều đó", "như vậy", "tương tự", "của nó", "với nó",
    "bike thì", "taxi thì", "car thì", "premium thì", "luxury thì",
)


def _needs_rewrite(message: str, history: list[dict]) -> bool:
    """Trả về True nếu câu hỏi có khả năng cần viết lại."""
    if not history:
        return False

    msg_lower = message.strip().lower()

    # Câu quá ngắn → rất có thể là follow-up
    if len(msg_lower) < 30:
        return True

    # Chứa tín hiệu follow-up
    if any(msg_lower.startswith(s) or f" {s}" in msg_lower for s in _FOLLOWUP_SIGNALS):
        return True

    # Không có động từ hỏi rõ ràng + không dấu hỏi → có thể cần bổ sung ngữ cảnh
    if "?" not in message and len(msg_lower) < 60:
        return True

    return False


def _build_context_block(history: list[dict], max_turns: int = 4) -> str:
    """Lấy N turn gần nhất (user + assistant) làm ngữ cảnh."""
    recent = [m for m in history if m["role"] in ("user", "assistant")][-max_turns * 2:]
    lines = []
    for m in recent:
        role = "Người dùng" if m["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {m['content'][:300]}")
    return "\n".join(lines)


async def rewrite_query(user_message: str, history: list[dict]) -> str:
    """
    Viết lại câu hỏi thành RAG query tốt hơn.

    Args:
        user_message: Câu hỏi hiện tại của user.
        history:      Lịch sử hội thoại (list of {role, content}).

    Returns:
        Query đã viết lại, hoặc câu gốc nếu không cần.
    """
    if not _needs_rewrite(user_message, history):
        logger.debug("[REWRITE] skipped (standalone) | msg=%r", user_message[:80])
        return user_message

    context_block = _build_context_block(history)
    user_prompt = (
        f"Lịch sử hội thoại:\n{context_block}\n\n"
        f"Câu hỏi hiện tại: {user_message}\n\n"
        f"Câu truy vấn đã viết lại:"
    )

    try:
        resp = await _client.chat.completions.create(
            model=_REWRITE_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=120,
            temperature=0,
        )
        rewritten = resp.choices[0].message.content.strip()
        logger.info(
            "[REWRITE] original=%r → rewritten=%r",
            user_message[:80], rewritten[:80],
        )
        return rewritten
    except Exception as e:
        logger.warning("[REWRITE] failed (%s), fallback to original", e)
        return user_message
