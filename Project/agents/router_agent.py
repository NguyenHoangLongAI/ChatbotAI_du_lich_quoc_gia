"""
router_agent.py — Agent phân loại query
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from Project.agents.base_agent import BaseAgent
from Project.state.state import AgentState

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """Agent phân loại câu hỏi du lịch Bãi Cháy."""

    @property
    def system_prompt(self) -> str:
        return """Bạn là trợ lý phân loại câu hỏi du lịch Bãi Cháy.

Phân loại câu hỏi thành 1 trong 3 loại:
- "tourism": Tìm tour, điểm đến, khách sạn, nhà hàng, giá cả
- "document": Hỏi về quy định, khiếu nại, thủ tục, chính sách
- "booking": Khách muốn đặt dịch vụ, cung cấp thông tin cá nhân

Ví dụ:
- "Tìm khách sạn 4 sao gần biển" -> tourism
- "Quy định hủy tour như thế nào?" -> document
- "Tôi muốn đặt tour Hạ Long 2 ngày, tên Nguyễn Văn A" -> booking

CHỈ TRẢ VỀ 1 TỪ: tourism, document, hoặc booking
KHÔNG GIẢI THÍCH, CHỈ TRẢ VỀ TỪ KHÓA."""

    def process(self, state: AgentState) -> AgentState:
        """Phân loại query: tourism | document | booking."""
        logger.info("🔀 Router Agent analyzing query...")

        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Phân loại câu hỏi sau:\n{state['user_query']}"),
        ])
        raw = response.content.strip().lower()

        if "document" in raw:
            query_type = "document"
        elif "booking" in raw:
            query_type = "booking"
        else:
            query_type = "tourism"

        logger.info(f"✅ Query type: {query_type}")
        state["query_type"] = query_type
        state["messages"].append(AIMessage(content=f"[Query classified as: {query_type}]"))
        return state