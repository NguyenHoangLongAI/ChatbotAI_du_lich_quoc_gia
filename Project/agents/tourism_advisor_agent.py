"""
tourism_advisor_agent.py — Agent tư vấn dịch vụ du lịch
"""

import logging
import json
from langchain_core.messages import HumanMessage, SystemMessage
from Project.agents.base_agent import BaseAgent
from Project.state.state import AgentState

logger = logging.getLogger(__name__)


class TourismAdvisorAgent(BaseAgent):
    """Agent tư vấn dịch vụ du lịch kèm image_url và url bài viết."""

    @property
    def system_prompt(self) -> str:
        return """Bạn là chuyên gia tư vấn du lịch Bãi Cháy - Quảng Ninh.

NHIỆM VỤ:
Dựa vào kết quả tìm kiếm, tư vấn cho khách hàng về các dịch vụ du lịch.

FORMAT TRẢ LỜI BẮT BUỘC (mỗi dịch vụ một khối):

---
### 🏨 [Tên dịch vụ] {rating > 0 ? '⭐ [rating]/5' : ''}

**📍 Địa chỉ:** [address hoặc location]
**💰 Giá:** [price_range]
**📝 Mô tả:** [Tóm tắt description, khoảng 4-5 câu]
**🖼️ Hình ảnh:** [image_url nếu có, nguyên URL]
**🔗 Xem chi tiết:** [url nếu có, nguyên URL]
**🆔 ID để đặt:** [id]
---

NGUYÊN TẮC QUAN TRỌNG:
1. ✅ LUÔN LUÔN hiển thị image_url nếu có
2. ✅ LUÔN LUÔN hiển thị url bài viết nếu có
3. ✅ Sắp xếp theo similarity_score cao nhất trước
4. ✅ Kết thúc bằng: "Bạn có muốn đặt dịch vụ nào không? Hãy cho tôi biết ID, tôi sẽ tiến hành booking."

KHÔNG ĐƯỢC:
- ❌ Bỏ qua image_url hoặc url nếu có
- ❌ Gộp chung nhiều dịch vụ vào một mục"""

    def process(self, state: AgentState) -> AgentState:
        """Tư vấn dịch vụ du lịch kèm image_url và url bài viết."""
        logger.info("🏖️ Tourism Advisor Agent working...")

        search_results = self.tools.search_tourism_services(
            query=state["user_query"], top_k=5
        )

        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=(
                    f"Câu hỏi: {state['user_query']}\n\n"
                    f"Kết quả tìm kiếm:\n{search_results}\n\n"
                    "Hãy tư vấn cho khách hàng."
                )
            ),
        ])

        state["messages"].append(response)
        state["search_results"] = json.loads(search_results)
        state["final_response"] = response.content
        state["next_action"] = "end"
        logger.info("✅ Tourism advice generated")
        return state