"""
tourism_detail_agent.py — Agent xử lý chi tiết dịch vụ theo ID
NEW: Agent riêng để get service by ID và tư vấn chi tiết
"""

import logging
import json
from langchain_core.messages import HumanMessage, SystemMessage
from Project.agents.base_agent import BaseAgent
from Project.state.state import AgentState

logger = logging.getLogger(__name__)


class TourismDetailAgent(BaseAgent):
    """Agent tư vấn chi tiết về dịch vụ cụ thể (có service_id)."""

    @property
    def system_prompt(self) -> str:
        return """Bạn là chuyên gia tư vấn chi tiết dịch vụ du lịch Bãi Cháy.

NHIỆM VỤ:
Dựa vào thông tin chi tiết của dịch vụ, trả lời câu hỏi của khách hàng.

FORMAT TRẢ LỜI:

---
### 🏨 [Tên dịch vụ] {rating > 0 ? '⭐ [rating]/5' : ''}

**🆔 ID dịch vụ:** [id]
**📍 Địa chỉ:** [address hoặc location]
**💰 Giá:** [price_range]
**⏰ Giờ mở cửa:** [opening_hours]

**📝 Mô tả chi tiết:**
[description]

**🖼️ Hình ảnh:** [image_url nếu có]
**🔗 Xem thêm:** [url nếu có]

---

**Thông tin bổ sung:**
- Trả lời CỤ THỂ câu hỏi của khách hàng dựa trên thông tin trên
- Nếu khách hỏi về giá → chi tiết price_min, price_max
- Nếu khách hỏi về vị trí → chi tiết address, location
- Nếu khách hỏi về giờ → chi tiết opening_hours

NGUYÊN TẮC:
- Trả lời CHÍNH XÁC dựa trên dữ liệu có
- Nếu không có thông tin → nói rõ "Thông tin này chưa có trong hệ thống"
- Gợi ý: "Bạn có muốn đặt dịch vụ này không? Hãy cho tôi biết nếu bạn muốn booking!"
- LUÔN HIỂN thị ID để khách dễ tham khảo khi muốn đặt"""

    def process(self, state: AgentState) -> AgentState:
        """Lấy thông tin chi tiết dịch vụ theo ID và tư vấn."""
        logger.info("🎯 Tourism Detail Agent working...")

        # Lấy service_id từ state
        service_id = state.get("service_id")

        if not service_id:
            # Fallback: extract từ query
            service_id = self._extract_service_id_from_query(state["user_query"])

        if not service_id:
            logger.warning("⚠️ No service_id found, cannot get detail")
            state["final_response"] = (
                "Xin lỗi, tôi không tìm thấy ID dịch vụ trong yêu cầu của bạn. "
                "Bạn có thể cung cấp ID dịch vụ không?"
            )
            state["next_action"] = "end"
            return state

        logger.info(f"🔍 Getting service detail for ID: {service_id}")

        # Get service detail
        service_detail = self.tools.get_service_by_id.invoke({"service_id": service_id})

        try:
            service_data = json.loads(service_detail)
        except:
            service_data = {}

        if "error" in service_data:
            logger.warning(f"⚠️ Service not found: {service_id}")
            state["final_response"] = (
                f"Xin lỗi, tôi không tìm thấy dịch vụ với ID {service_id} trong hệ thống. "
                "Bạn có thể kiểm tra lại ID hoặc tìm kiếm dịch vụ khác?"
            )
            state["next_action"] = "end"
            return state

        # Generate response
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=(
                    f"Câu hỏi của khách: {state['user_query']}\n\n"
                    f"Thông tin dịch vụ (ID: {service_id}):\n{service_detail}\n\n"
                    "Hãy tư vấn chi tiết cho khách hàng và gợi ý đặt hàng nếu phù hợp."
                )
            ),
        ])

        state["messages"].append(response)
        state["search_results"] = [service_data]  # Store as list for consistency
        state["final_response"] = response.content
        state["next_action"] = "end"

        logger.info(f"✅ Tourism detail advice generated for service {service_id}")
        return state

    def _extract_service_id_from_query(self, query: str) -> int:
        """
        Extract service ID từ query bằng regex.

        Args:
            query: User query

        Returns:
            Service ID hoặc None
        """
        import re

        # Patterns để match service ID
        patterns = [
            r'\bid[:\s]+(\d+)',  # "id: 123", "id 123"
            r'dịch vụ[:\s]+(\d+)',  # "dịch vụ 123"
            r'service[:\s]+(\d+)',  # "service 123"
            r'số[:\s]+(\d+)',  # "số 123"
            r'mã[:\s]+(\d+)',  # "mã 123"
        ]

        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                try:
                    return int(match.group(1))
                except:
                    continue

        return None