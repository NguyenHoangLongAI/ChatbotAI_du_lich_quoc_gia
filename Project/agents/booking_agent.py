"""
booking_agent.py — Agent xử lý đặt tour/dịch vụ
"""

import logging
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from Project.agents.base_agent import BaseAgent
from Project.state.state import AgentState

logger = logging.getLogger(__name__)


class BookingAgent(BaseAgent):
    """Agent thu thập thông tin và tạo booking."""

    @property
    def system_prompt(self) -> str:
        return """Bạn là chuyên viên đặt tour du lịch Bãi Cháy.

NHIỆM VỤ:
1. Thu thập đầy đủ thông tin:
   - Họ tên khách hàng
   - Số điện thoại
   - ID dịch vụ đã chọn (nếu có từ hội thoại trước)
   - Ngày check-in (YYYY-MM-DD)
   - Ngày check-out (YYYY-MM-DD)

2. Nếu ĐỦ thông tin — trả về JSON:
   {"action": "create_booking", "name": "...", "phone": "...",
    "service_ids": [...], "checkin": "YYYY-MM-DD", "checkout": "YYYY-MM-DD",
    "description": "..."}

3. Nếu THIẾU — hỏi thêm thông tin còn thiếu."""

    def process(self, state: AgentState) -> AgentState:
        """Thu thập thông tin và tạo booking."""
        logger.info("🎫 Booking Agent working...")

        conversation_text = "\n".join(
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in state["messages"][-3:]
            if hasattr(msg, "content")
        )

        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=(
                    f"Lịch sử hội thoại:\n{conversation_text}\n\n"
                    f"Tin nhắn mới: {state['user_query']}\n\n"
                    "Phân tích và xử lý."
                )
            ),
        ])
        response_text = response.content

        # Nếu agent đã đủ thông tin — tạo booking
        if '"action": "create_booking"' in response_text:
            try:
                json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
                if json_match:
                    booking_data = json.loads(json_match.group())
                    result = self.tools.create_customer_booking.invoke(booking_data)
                    state["messages"].append(AIMessage(content=f"Booking result: {result}"))
                    state["final_response"] = f"✅ Đặt hàng thành công!\n\n{result}"
                    state["booking_info"] = json.loads(result)
                else:
                    state["final_response"] = response_text
            except Exception as e:
                logger.error(f"Booking error: {e}")
                state["final_response"] = (
                    f"Xin lỗi, có lỗi khi tạo booking: {e}\n\nVui lòng thử lại."
                )
        else:
            state["final_response"] = response_text

        state["messages"].append(response)
        state["next_action"] = "end"
        logger.info("✅ Booking processing completed")
        return state