"""
Phân loại intent của user message bằng LLM (gpt-4o-mini).
Chạy trước pipeline RAG để routing sớm.
"""

import logging
from openai import AsyncOpenAI
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

_SYSTEM_PROMPT = """\
Phân loại intent của tin nhắn người dùng gửi đến chatbot hỗ trợ dịch vụ Xanh SM.

Chỉ trả về đúng một trong các nhãn sau, không giải thích gì thêm:
- driver_registration : người dùng muốn đăng ký / ứng tuyển / nộp đơn làm tài xế xe máy điện hoặc taxi điện Xanh SM
- human_escalation   : người dùng EXPLICITLY yêu cầu gặp nhân viên / tổng đài / người thật, HOẶC bày tỏ thất vọng rõ ràng rằng bot không giúp được — VD: "gặp nhân viên", "gọi hotline cho tôi", "bot vô dụng", "không giải quyết được", "chuyển tôi sang người thật". KHÔNG áp dụng cho câu hỏi thông thường hay yêu cầu giúp đỡ chung chung như "giúp tôi", "hỗ trợ tôi"
- general             : tất cả các trường hợp còn lại (hỏi thông tin, giá cước, chính sách, v.v.)
"""

VALID_INTENTS = ("driver_registration", "human_escalation", "general")


async def detect_intent(message: str) -> str:
    """
    Trả về 'driver_registration', 'human_escalation', hoặc 'general'.
    Fallback về 'general' nếu LLM lỗi.
    """
    try:
        resp = await _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": message},
            ],
            max_tokens=10,
            temperature=0,
        )
        intent = resp.choices[0].message.content.strip().lower()
        logger.info("[INTENT] %r → %s", message[:60], intent)
        return intent if intent in VALID_INTENTS else "general"
    except Exception as e:
        logger.warning("[INTENT] failed (%s), fallback=general", e)
        return "general"
