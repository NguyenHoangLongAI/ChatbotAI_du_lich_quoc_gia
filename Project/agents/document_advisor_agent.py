"""
document_advisor_agent.py — Agent giải đáp quy định & tài liệu
UPDATED: Sử dụng contextualized_query từ RouterAgent
"""

import logging
import json
from langchain_core.messages import HumanMessage, SystemMessage
from Project.agents.base_agent import BaseAgent
from Project.state.state import AgentState

logger = logging.getLogger(__name__)


class DocumentAdvisorAgent(BaseAgent):
    """Agent giải đáp quy định & tài liệu du lịch."""

    @property
    def system_prompt(self) -> str:
        return """Bạn là chuyên gia tư vấn quy định du lịch Bãi Cháy.

NHIỆM VỤ:
1. Đọc kỹ nội dung tài liệu tìm được
2. Trả lời chính xác dựa trên tài liệu
3. Trích dẫn nguồn (document_id) nếu có

NGUYÊN TẮC:
- Chỉ trả lời dựa trên tài liệu tìm được
- Nếu không tìm thấy: "Tôi chưa tìm thấy thông tin này trong tài liệu"
- Trình bày rõ ràng, dễ hiểu
- Gợi ý liên hệ hotline nếu cần"""

    def process(self, state: AgentState) -> AgentState:
        """Giải đáp quy định & tài liệu."""
        logger.info("📚 Document Advisor Agent working...")

        # Sử dụng contextualized_query nếu có
        search_query = state.get("contextualized_query", state["user_query"])

        # Log context info
        context_info = state.get("context_info", {})
        if context_info.get("is_followup"):
            logger.info(f"🔍 Using contextualized query: {search_query}")
            logger.info(f"   Context: {context_info.get('context_summary')}")

        # Search với contextualized query
        search_results = self.tools.search_documents.invoke(
            {"query": search_query, "top_k": 3}
        )

        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=(
                    f"Câu hỏi gốc: {state['user_query']}\n"
                    f"Câu hỏi đã làm rõ: {search_query}\n\n"
                    f"Tài liệu tìm được:\n{search_results}\n\n"
                    "Hãy trả lời câu hỏi."
                )
            ),
        ])

        state["messages"].append(response)
        state["search_results"] = json.loads(search_results)
        state["final_response"] = response.content
        state["next_action"] = "end"
        logger.info("✅ Document advice generated")
        return state