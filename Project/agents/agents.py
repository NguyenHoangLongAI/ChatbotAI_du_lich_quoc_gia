"""
agents.py — Các agent chuyên biệt: Router, TourismAdvisor, DocumentAdvisor, BookingAgent
"""

import json
import re
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from Project.state.state import AgentState
from Project.llm.llm import OpenAILLMWrapper
from Project.tools.tools import RAGTools

logger = logging.getLogger(__name__)


class TourismAgents:
    """Tập hợp tất cả các agent node cho LangGraph workflow."""

    # ------------------------------------------------------------------ #
    # System prompts                                                       #
    # ------------------------------------------------------------------ #

    _ROUTER_SYSTEM = """Bạn là trợ lý phân loại câu hỏi du lịch Bãi Cháy.

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

    _TOURISM_SYSTEM = """Bạn là chuyên gia tư vấn du lịch Bãi Cháy - Quảng Ninh.

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

    _DOCUMENT_SYSTEM = """Bạn là chuyên gia tư vấn quy định du lịch Bãi Cháy.

NHIỆM VỤ:
1. Đọc kỹ nội dung tài liệu tìm được
2. Trả lời chính xác dựa trên tài liệu
3. Trích dẫn nguồn (document_id) nếu có

NGUYÊN TẮC:
- Chỉ trả lời dựa trên tài liệu tìm được
- Nếu không tìm thấy: "Tôi chưa tìm thấy thông tin này trong tài liệu"
- Trình bày rõ ràng, dễ hiểu
- Gợi ý liên hệ hotline nếu cần"""

    _BOOKING_SYSTEM = """Bạn là chuyên viên đặt tour du lịch Bãi Cháy.

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

    # ------------------------------------------------------------------ #
    # Init                                                                 #
    # ------------------------------------------------------------------ #

    def __init__(self, tools: RAGTools, openai_model: str = "gpt-4o"):
        self.tools = tools
        self.llm = OpenAILLMWrapper(model=openai_model, temperature=0.1)

    # ------------------------------------------------------------------ #
    # Nodes (mỗi method là 1 node trong LangGraph)                        #
    # ------------------------------------------------------------------ #

    def router_agent(self, state: AgentState) -> AgentState:
        """Phân loại query: tourism | document | booking."""
        logger.info("🔀 Router Agent analyzing query...")

        response = self.llm.invoke([
            SystemMessage(content=self._ROUTER_SYSTEM),
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

    # ------------------------------------------------------------------

    def tourism_advisor_agent(self, state: AgentState) -> AgentState:
        """Tư vấn dịch vụ du lịch kèm image_url và url bài viết."""
        logger.info("🏖️ Tourism Advisor Agent working...")

        search_results = self.tools.search_tourism_services(
            query=state["user_query"], top_k=5
        )

        response = self.llm.invoke([
            SystemMessage(content=self._TOURISM_SYSTEM),
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

    # ------------------------------------------------------------------

    def document_advisor_agent(self, state: AgentState) -> AgentState:
        """Giải đáp quy định & tài liệu."""
        logger.info("📚 Document Advisor Agent working...")

        search_results = self.tools.search_documents.invoke(
            {"query": state["user_query"], "top_k": 3}
        )

        response = self.llm.invoke([
            SystemMessage(content=self._DOCUMENT_SYSTEM),
            HumanMessage(
                content=(
                    f"Câu hỏi: {state['user_query']}\n\n"
                    f"Tài liệu tìm được:\n{search_results}\n\n"
                    "Hãy trả lời câu hỏi."
                )
            ),
        ])

        state["messages"].append(response)
        state["search_results"] = json.loads(search_results)
        state["final_response"] = response.content
        state["next_action"] = "end"
        return state

    # ------------------------------------------------------------------

    def booking_agent(self, state: AgentState) -> AgentState:
        """Thu thập thông tin và tạo booking."""
        logger.info("🎫 Booking Agent working...")

        conversation_text = "\n".join(
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in state["messages"][-3:]
            if hasattr(msg, "content")
        )

        response = self.llm.invoke([
            SystemMessage(content=self._BOOKING_SYSTEM),
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
        return state