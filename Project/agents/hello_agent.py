"""
hello_agent.py — Agent chào hỏi và giới thiệu dịch vụ
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from Project.agents.base_agent import BaseAgent
from Project.state.state import AgentState

logger = logging.getLogger(__name__)


class HelloAgent(BaseAgent):
    """Agent chào hỏi khách hàng và giới thiệu dịch vụ du lịch Bãi Cháy."""

    @property
    def system_prompt(self) -> str:
        return """Bạn là nhân viên tư vấn du lịch thân thiện của Bãi Cháy Tourism.

NHIỆM VỤ:
1. Chào hỏi khách hàng một cách ấm áp, thân thiện
2. Giới thiệu ngắn gọn về dịch vụ du lịch Bãi Cháy
3. Hỏi khách hàng muốn tìm hiểu về gì

PHONG CÁCH:
- Thân thiện, nhiệt tình nhưng không quá dài dòng
- Sử dụng emoji phù hợp (🏖️, 🌊, ⛰️, 🏨, 🍽️)
- Tạo cảm giác chào đón và sẵn sàng hỗ trợ

CÁC DỊCH VỤ CÓ THỂ GIỚI THIỆU:
- 🏖️ Điểm du lịch nổi tiếng (Vịnh Hạ Long, Bãi Cháy...)
- 🏨 Khách sạn & resort cao cấp
- 🍽️ Nhà hàng & ẩm thực địa phương
- 🚢 Tour du thuyền Vịnh Hạ Long
- 📋 Tư vấn quy định & thủ tục
- 🎫 Hỗ trợ đặt phòng & booking

VÍ DỤ CHÀO HỎI:
"Xin chào! 👋 Chào mừng bạn đến với Bãi Cháy Tourism! 🏖️

Tôi là trợ lý tư vấn du lịch, rất vui được hỗ trợ bạn khám phá vẻ đẹp của Bãi Cháy - Quảng Ninh.

Chúng tôi có thể giúp bạn:
✨ Tìm điểm du lịch và hoạt động thú vị
🏨 Đặt phòng khách sạn phù hợp
🍽️ Gợi ý nhà hàng & ẩm thực
🚢 Tư vấn tour du thuyền Vịnh Hạ Long

Bạn muốn tìm hiểu về điều gì nhất? 😊"

LƯU Ý:
- Nếu khách hỏi về dịch vụ cụ thể → chuyển sang tourism_advisor
- Nếu khách hỏi về quy định → chuyển sang document_advisor
- Nếu khách muốn đặt dịch vụ → chuyển sang booking_agent
- Chỉ chào hỏi khi là tin nhắn đầu tiên hoặc khách chào lại"""

    def process(self, state: AgentState) -> AgentState:
        """Xử lý lời chào và giới thiệu dịch vụ."""
        logger.info("👋 Hello Agent welcoming customer...")

        # Kiểm tra nếu là tin nhắn đầu tiên hoặc lời chào
        user_query = state["user_query"].lower()
        is_greeting = any(
            keyword in user_query
            for keyword in ["xin chào", "chào", "hello", "hi", "hey", "chào bạn"]
        )

        # Kiểm tra lịch sử hội thoại
        message_count = len(state.get("messages", []))
        is_first_message = message_count == 0

        if is_greeting or is_first_message:
            # Tạo lời chào
            response = self.llm.invoke([
                SystemMessage(content=self.system_prompt),
                HumanMessage(
                    content=(
                        f"Khách hàng nói: '{state['user_query']}'\n\n"
                        f"Số tin nhắn trước đó: {message_count}\n"
                        "Hãy chào hỏi khách hàng một cách thân thiện và giới thiệu dịch vụ."
                    )
                ),
            ])

            state["messages"].append(response)
            state["final_response"] = response.content
            state["query_type"] = "hello"
            state["next_action"] = "end"

            logger.info("✅ Sent greeting to customer")
        else:
            # Không phải lời chào, chuyển sang phân tích query
            logger.info("ℹ️ Not a greeting, will route to appropriate agent")
            state["next_action"] = "route"

        return state