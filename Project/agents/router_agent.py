"""
router_agent.py — Agent phân loại query với context processing
UPDATED: Context-aware query rewriting cho follow-up questions
"""

import logging
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from Project.agents.base_agent import BaseAgent
from Project.state.state import AgentState

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """Agent phân loại câu hỏi du lịch Bãi Cháy với context processing."""

    @property
    def context_processing_prompt(self) -> str:
        """System prompt cho context processing."""
        return """Bạn là chuyên gia phân tích ngữ cảnh hội thoại.

NHIỆM VỤ:
Phân tích câu hỏi hiện tại và lịch sử hội thoại để:
1. Xác định xem có phải câu hỏi follow-up không
2. Làm rõ câu hỏi với đầy đủ ngữ cảnh

YÊU CẦU QUAN TRỌNG:
- Phân tích xem câu hỏi có phải follow-up (tiếp theo cuộc trò chuyện trước) không
- Truy vết lịch sử để xác định chính xác đối tượng được nhắc tới
- Đặc biệt chú ý các cụm từ:
  * Đại từ: "nó", "ý trên", "cái đó", "phần này", "thành phần thứ X"
  * Xác nhận: "OK", "có", "được", "đồng ý"
  * Yêu cầu tiếp: "chi tiết", "hãy hướng dẫn", "tiếp tục", "nói thêm"
  * Chỉ định: "cái thứ nhất", "option 2", "số 3"
  
- Nếu lịch sử có DANH SÁCH ĐÁNH SỐ → ánh xạ theo ĐÚNG THỨ TỰ
- Nếu có yêu cầu hành động không cụ thể → dựa vào lịch sử hội thoại làm rõ yêu cầu
- Viết lại câu hỏi (contextualized_question) bằng TIẾNG VIỆT ĐẦY ĐỦ – RÕ NGHĨA – CÓ NGỮ CẢNH

Đảm bảo câu hỏi được làm rõ (contextualized_question) phải có:
- ĐỐI TƯỢNG cụ thể là gì (tên khách sạn, tour, địa điểm...)
- HÀNH ĐỘNG cụ thể là gì (tìm, đặt, hỏi về...)
- Trong NGỮ CẢNH cụ thể là gì (giá, vị trí, thời gian...)

Nếu không phải follow-up: 
- contextualized_question = câu hỏi gốc
- context_summary = "Câu hỏi độc lập"

OUTPUT FORMAT (JSON):
{{
  "is_followup": true hoặc false,
  "contextualized_question": "Câu hỏi đã được làm rõ rất cụ thể hoặc câu hỏi gốc",
  "context_summary": "Tóm tắt ngắn gọn ngữ cảnh BẰNG TIẾNG VIỆT",
  "detected_references": {{
    "pronouns": ["nó", "cái đó"...],
    "numbers": ["thứ 1", "số 2"...],
    "actions": ["chi tiết", "đặt luôn"...]
  }},
  "resolved_entities": {{
    "hotel_name": "Tên khách sạn nếu có",
    "tour_name": "Tên tour nếu có",
    "service_id": "ID dịch vụ nếu xác định được"
  }}
}}

VÍ DỤ:

Example 1 - Follow-up với đại từ:
Input:
  Question: "Nó giá bao nhiêu?"
  History: "Assistant: Đây là khách sạn Mường Thanh 4 sao..."
Output:
{{
  "is_followup": true,
  "contextualized_question": "Khách sạn Mường Thanh 4 sao giá bao nhiêu?",
  "context_summary": "Hỏi giá khách sạn Mường Thanh được nhắc ở tin nhắn trước",
  "detected_references": {{"pronouns": ["nó"]}},
  "resolved_entities": {{"hotel_name": "Mường Thanh"}}
}}

Example 2 - Follow-up với số thứ tự:
Input:
  Question: "Cái thứ 2 đi"
  History: "Assistant: Có 3 tour: 1. Tour Hạ Long 1 ngày, 2. Tour Hạ Long 2 ngày 1 đêm, 3. Tour..."
Output:
{{
  "is_followup": true,
  "contextualized_question": "Cho tôi thông tin chi tiết về Tour Hạ Long 2 ngày 1 đêm",
  "context_summary": "Chọn tour thứ 2 trong danh sách được đề xuất",
  "detected_references": {{"numbers": ["thứ 2"]}},
  "resolved_entities": {{"tour_name": "Tour Hạ Long 2 ngày 1 đêm"}}
}}

Example 3 - Follow-up với xác nhận:
Input:
  Question: "OK, đặt luôn"
  History: "User: Tôi muốn khách sạn gần biển. Assistant: Khách sạn Novotel..."
Output:
{{
  "is_followup": true,
  "contextualized_question": "Đặt phòng khách sạn Novotel gần biển",
  "context_summary": "Xác nhận đặt khách sạn Novotel được giới thiệu",
  "detected_references": {{"actions": ["đặt luôn"]}},
  "resolved_entities": {{"hotel_name": "Novotel"}}
}}

Example 4 - Câu hỏi độc lập:
Input:
  Question: "Tìm khách sạn 4 sao gần biển"
  History: ""
Output:
{{
  "is_followup": false,
  "contextualized_question": "Tìm khách sạn 4 sao gần biển",
  "context_summary": "Câu hỏi độc lập",
  "detected_references": {{}},
  "resolved_entities": {{}}
}}

NGUYÊN TẮC:
- LUÔN LUÔN trả về JSON hợp lệ
- contextualized_question PHẢI rõ ràng, có thể search được
- Nếu không chắc chắn → is_followup = false
- Ưu tiên thông tin gần nhất trong lịch sử"""

    @property
    def classification_prompt(self) -> str:
        """System prompt cho classification."""
        return """Bạn là trợ lý phân loại câu hỏi du lịch Bãi Cháy.

Phân loại câu hỏi thành 1 trong 5 loại:
- "hello": Lời chào, chào hỏi, giới thiệu ban đầu
- "human": Khách cung cấp thông tin cá nhân (tên, SĐT, ngày check-in/out)
- "tourism": Tìm tour, điểm đến, khách sạn, nhà hàng, giá cả
- "document": Hỏi về quy định, khiếu nại, thủ tục, chính sách
- "booking": Khách muốn đặt dịch vụ (sau khi đã có đủ thông tin)

Ví dụ:
- "Xin chào" → hello
- "Tên tôi là Nguyễn Văn A" → human
- "Tìm khách sạn 4 sao gần biển" → tourism
- "Khách sạn Mường Thanh giá bao nhiêu?" → tourism
- "Quy định hủy tour như thế nào?" → document
- "Đặt luôn tour này" → booking

CHỈ TRẢ VỀ 1 TỪ: hello, human, tourism, document, hoặc booking
KHÔNG GIẢI THÍCH, CHỈ TRẢ VỀ TỪ KHÓA."""

    @property
    def system_prompt(self) -> str:
        """Backward compatibility."""
        return self.classification_prompt

    def process(self, state: AgentState) -> AgentState:
        """
        Phân loại query với context processing:
        1. Phân tích context và làm rõ câu hỏi
        2. Phân loại query type
        3. Update state với contextualized question
        """
        logger.info("🔀 Router Agent analyzing query...")

        # Step 1: Context Processing
        context_result = self._process_context(state)

        if context_result:
            # Update state với contextualized question
            original_query = state["user_query"]
            contextualized_query = context_result.get("contextualized_question", original_query)

            # Log context analysis
            if context_result.get("is_followup"):
                logger.info(f"📝 Follow-up detected!")
                logger.info(f"   Original: {original_query}")
                logger.info(f"   Contextualized: {contextualized_query}")
                logger.info(f"   Summary: {context_result.get('context_summary')}")

            # Store context info in state
            state["contextualized_query"] = contextualized_query
            state["context_info"] = context_result

            # Use contextualized query for classification
            query_for_classification = contextualized_query
        else:
            # No context processing, use original
            query_for_classification = state["user_query"]
            state["contextualized_query"] = state["user_query"]
            state["context_info"] = {
                "is_followup": False,
                "context_summary": "Câu hỏi độc lập"
            }

        # Step 2: Build customer info context
        customer_info = state.get("customer_info", {})
        context_info_text = ""

        if customer_info:
            has_name = customer_info.get("name") is not None
            has_phone = customer_info.get("phone") is not None
            has_checkin = customer_info.get("checkin") is not None
            has_checkout = customer_info.get("checkout") is not None

            context_info_text = f"""
Thông tin khách hàng hiện có:
- Tên: {"Có" if has_name else "Chưa có"}
- SĐT: {"Có" if has_phone else "Chưa có"}
- Check-in: {"Có" if has_checkin else "Chưa có"}
- Check-out: {"Có" if has_checkout else "Chưa có"}
"""

        # Step 3: Classification
        response = self.llm.invoke([
            SystemMessage(content=self.classification_prompt),
            HumanMessage(
                content=(
                    f"Phân loại câu hỏi sau:\n{query_for_classification}\n\n"
                    f"{context_info_text}"
                )
            ),
        ])
        raw = response.content.strip().lower()

        # Determine query type
        if "hello" in raw:
            query_type = "hello"
        elif "human" in raw:
            query_type = "human"
        elif "document" in raw:
            query_type = "document"
        elif "booking" in raw:
            query_type = "booking"
        else:
            query_type = "tourism"

        logger.info(f"✅ Query type: {query_type}")
        state["query_type"] = query_type
        state["messages"].append(AIMessage(
            content=f"[Query classified as: {query_type}]"
        ))

        return state

    def _process_context(self, state: AgentState) -> dict:
        """
        Xử lý context và làm rõ câu hỏi follow-up.

        Returns:
            Dict với is_followup, contextualized_question, context_summary
        """
        try:
            # Build conversation history
            history_text = self._build_history_text(state)

            # If no history, no context needed
            if not history_text or len(history_text.strip()) < 10:
                return {
                    "is_followup": False,
                    "contextualized_question": state["user_query"],
                    "context_summary": "Câu hỏi độc lập"
                }

            # Invoke LLM for context analysis
            response = self.llm.invoke([
                SystemMessage(content=self.context_processing_prompt),
                HumanMessage(
                    content=(
                        f"Đầu vào:\n"
                        f"Câu hỏi hiện tại: \"{state['user_query']}\"\n"
                        f"Lịch sử hội thoại:\n{history_text}\n\n"
                        "Hãy phân tích và trả lời theo định dạng JSON."
                    )
                ),
            ])

            # Parse JSON response
            result = self._parse_context_response(response.content)

            if result:
                return result
            else:
                # Fallback
                logger.warning("⚠️ Context processing failed, using original query")
                return {
                    "is_followup": False,
                    "contextualized_question": state["user_query"],
                    "context_summary": "Không thể phân tích context"
                }

        except Exception as e:
            logger.error(f"❌ Context processing error: {e}")
            return {
                "is_followup": False,
                "contextualized_question": state["user_query"],
                "context_summary": f"Lỗi xử lý context: {e}"
            }

    def _build_history_text(self, state: AgentState, max_turns: int = 3) -> str:
        """
        Build conversation history text từ messages.

        Args:
            state: Agent state
            max_turns: Số lượt hội thoại tối đa (default: 3)

        Returns:
            Formatted history text
        """
        messages = state.get("messages", [])

        if not messages:
            return ""

        # Take last N messages
        recent_messages = messages[-(max_turns * 2):]  # *2 vì có cả user và assistant

        history_lines = []
        for msg in recent_messages:
            if hasattr(msg, "content"):
                # Skip internal messages
                if msg.content.startswith("[") and msg.content.endswith("]"):
                    continue

                role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"

                # Truncate long messages
                content = msg.content
                if len(content) > 500:
                    content = content[:497] + "..."

                history_lines.append(f"{role}: {content}")

        return "\n".join(history_lines)

    def _parse_context_response(self, response_text: str) -> dict:
        """
        Parse JSON response từ LLM.

        Returns:
            Parsed dict hoặc None nếu invalid
        """
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group())

                # Validate required fields
                if "is_followup" in result and "contextualized_question" in result:
                    return result

            # Try direct parse
            result = json.loads(response_text)

            if "is_followup" in result and "contextualized_question" in result:
                return result

            return None

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse error: {e}")
            return None