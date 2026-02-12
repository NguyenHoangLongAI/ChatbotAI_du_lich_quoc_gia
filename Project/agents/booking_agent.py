"""
booking_agent.py — Agent xử lý đặt tour/dịch vụ (UPDATED: sử dụng customer_info từ HumanAgent)
"""

import logging
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from Project.agents.base_agent import BaseAgent
from Project.state.state import AgentState

logger = logging.getLogger(__name__)


class BookingAgent(BaseAgent):
    """Agent tạo booking từ thông tin đã được HumanAgent extract."""

    @property
    def system_prompt(self) -> str:
        return """Bạn là chuyên viên đặt tour du lịch Bãi Cháy.

NHIỆM VỤ:
1. Kiểm tra thông tin khách hàng đã có:
   - Họ tên
   - Số điện thoại
   - Ngày check-in (YYYY-MM-DD)
   - Ngày check-out (YYYY-MM-DD)
   - ID dịch vụ đã chọn (nếu có)

2. Nếu ĐỦ thông tin — xác nhận và tạo booking:
   - Hiển thị lại thông tin để khách kiểm tra
   - Hỏi xác nhận "Thông tin có chính xác không?"
   - Nếu khách xác nhận → trả về JSON để tạo booking

3. Nếu THIẾU thông tin:
   - Liệt kê thông tin còn thiếu
   - Hướng dẫn khách cung cấp

FORMAT JSON KHI TẠO BOOKING:
{
    "action": "create_booking",
    "name": "...",
    "phone": "...",
    "service_ids": [...],
    "checkin": "YYYY-MM-DD",
    "checkout": "YYYY-MM-DD",
    "description": "..."
}

NGUYÊN TẮC:
- Luôn xác nhận lại thông tin trước khi đặt
- Nếu thiếu thông tin → hướng dẫn rõ ràng
- Thân thiện, tạo cảm giác an tâm cho khách"""

    def process(self, state: AgentState) -> AgentState:
        """Xử lý booking với thông tin từ customer_info."""
        logger.info("🎫 Booking Agent working...")

        # Lấy thông tin khách hàng từ state
        customer_info = state.get("customer_info", {})

        # Build context
        conversation_text = "\n".join(
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in state["messages"][-3:]
            if hasattr(msg, "content")
        )

        # Chuẩn bị thông tin cho LLM
        info_summary = self._format_customer_info(customer_info)

        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=(
                    f"Lịch sử hội thoại:\n{conversation_text}\n\n"
                    f"Thông tin khách hàng hiện có:\n{info_summary}\n\n"
                    f"Tin nhắn mới: {state['user_query']}\n\n"
                    "Phân tích và xử lý booking."
                )
            ),
        ])
        response_text = response.content

        # Kiểm tra nếu agent đã đủ thông tin để tạo booking
        if '"action": "create_booking"' in response_text:
            try:
                # Extract JSON
                json_match = re.search(r"\{[^}]+\}", response_text, re.DOTALL)
                if json_match:
                    booking_data = json.loads(json_match.group())

                    # Điền thông tin từ customer_info nếu thiếu trong JSON
                    if not booking_data.get("name") and customer_info.get("name"):
                        booking_data["name"] = customer_info["name"]
                    if not booking_data.get("phone") and customer_info.get("phone"):
                        booking_data["phone"] = customer_info["phone"]
                    if not booking_data.get("checkin") and customer_info.get("checkin"):
                        booking_data["checkin"] = customer_info["checkin"]
                    if not booking_data.get("checkout") and customer_info.get("checkout"):
                        booking_data["checkout"] = customer_info["checkout"]

                    # Tạo booking
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
            # Chưa đủ thông tin hoặc đang xác nhận
            state["final_response"] = response_text

        state["messages"].append(response)
        state["next_action"] = "end"
        logger.info("✅ Booking processing completed")
        return state

    def _format_customer_info(self, customer_info: dict) -> str:
        """Format customer_info thành text dễ đọc."""
        if not customer_info:
            return "Chưa có thông tin khách hàng"

        lines = []

        # Basic info
        name = customer_info.get("name")
        phone = customer_info.get("phone")
        checkin = customer_info.get("checkin")
        checkout = customer_info.get("checkout")

        lines.append(f"- Tên: {name if name else '❌ Chưa có'}")
        lines.append(f"- SĐT: {phone if phone else '❌ Chưa có'}")
        lines.append(f"- Check-in: {checkin if checkin else '❌ Chưa có'}")
        lines.append(f"- Check-out: {checkout if checkout else '❌ Chưa có'}")

        # Confidence scores
        confidence = customer_info.get("confidence", {})
        if confidence:
            lines.append("\nĐộ tin cậy:")
            for field, score in confidence.items():
                if score > 0:
                    lines.append(f"  - {field}: {score:.0%}")

        # Missing fields
        missing = customer_info.get("missing_fields", [])
        if missing:
            lines.append(f"\n⚠️ Còn thiếu: {', '.join(missing)}")

        return "\n".join(lines)