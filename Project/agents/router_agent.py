"""
router_agent.py — Agent phân loại query với context processing
ENHANCED: Nhận diện service ID và booking intent
"""

import logging
import json
import re
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from Project.agents.base_agent import BaseAgent
from Project.state.state import AgentState

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """Agent phân loại câu hỏi du lịch Bãi Cháy với context processing và service ID detection."""

    @property
    def context_processing_prompt(self) -> str:
        """System prompt cho context processing."""
        return """Bạn là chuyên gia phân tích ngữ cảnh hội thoại.

NHIỆM VỤ:
Phân tích câu hỏi hiện tại và lịch sử hội thoại để:
1. Xác định xem có phải câu hỏi follow-up không
2. Làm rõ câu hỏi với đầy đủ ngữ cảnh
3. **Trích xuất service ID nếu có**
4. **Nhận diện intent đặt hàng**

YÊU CẦU QUAN TRỌNG:
- Phân tích xem câu hỏi có phải follow-up (tiếp theo cuộc trò chuyện trước) không
- Truy vết lịch sử để xác định chính xác đối tượng được nhắc tới
- **ĐẶC BIỆT CHÚ Ý**:
  * Nếu có số ID dịch vụ (VD: "id: 123", "dịch vụ 456", "số 789") → trích xuất
  * Nếu có intent đặt hàng ("đặt luôn", "booking", "book ngay", "tôi muốn đặt", "chốt") → đánh dấu
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
    "actions": ["chi tiết", "đặt luôn"...],
    "service_id": null hoặc số ID (int)
  }},
  "resolved_entities": {{
    "hotel_name": "Tên khách sạn nếu có",
    "tour_name": "Tên tour nếu có",
    "service_id": null hoặc số ID (int)
  }},
  "booking_intent": {{
    "has_intent": true/false,
    "confidence": 0.0-1.0,
    "keywords": ["đặt", "booking"...]
  }}
}}

VÍ DỤ:

Example 1 - Hỏi chi tiết về service ID:
Input:
  Question: "Cho tôi biết thêm về dịch vụ id 20123"
  History: ""
Output:
{{
  "is_followup": false,
  "contextualized_question": "Cho tôi biết thông tin chi tiết về dịch vụ có ID 20123",
  "context_summary": "Yêu cầu thông tin chi tiết dịch vụ",
  "detected_references": {{"service_id": 20123}},
  "resolved_entities": {{"service_id": 20123}},
  "booking_intent": {{"has_intent": false, "confidence": 0.0, "keywords": []}}
}}

Example 2 - Booking intent với ID:
Input:
  Question: "Tôi muốn đặt dịch vụ 30045"
  History: ""
Output:
{{
  "is_followup": false,
  "contextualized_question": "Tôi muốn đặt dịch vụ có ID 30045",
  "context_summary": "Yêu cầu đặt dịch vụ cụ thể",
  "detected_references": {{"service_id": 30045, "actions": ["đặt"]}},
  "resolved_entities": {{"service_id": 30045}},
  "booking_intent": {{"has_intent": true, "confidence": 0.95, "keywords": ["đặt"]}}
}}

Example 3 - Follow-up với đại từ + booking:
Input:
  Question: "Đặt luôn cái đó"
  History: "Assistant: Đây là tour Hạ Long 2 ngày 1 đêm (ID: 20045)..."
Output:
{{
  "is_followup": true,
  "contextualized_question": "Đặt tour Hạ Long 2 ngày 1 đêm (ID: 20045)",
  "context_summary": "Xác nhận đặt tour được giới thiệu trước đó",
  "detected_references": {{"pronouns": ["cái đó"], "actions": ["đặt luôn"], "service_id": 20045}},
  "resolved_entities": {{"tour_name": "Tour Hạ Long 2 ngày 1 đêm", "service_id": 20045}},
  "booking_intent": {{"has_intent": true, "confidence": 1.0, "keywords": ["đặt luôn"]}}
}}

Example 4 - Hỏi giá về service đã nhắc:
Input:
  Question: "Giá của nó bao nhiêu?"
  History: "Assistant: Tour Vịnh Hạ Long 1 ngày (ID: 20012) là..."
Output:
{{
  "is_followup": true,
  "contextualized_question": "Giá của Tour Vịnh Hạ Long 1 ngày (ID: 20012) bao nhiêu?",
  "context_summary": "Hỏi giá tour được nhắc ở tin nhắn trước",
  "detected_references": {{"pronouns": ["nó"], "service_id": 20012}},
  "resolved_entities": {{"tour_name": "Tour Vịnh Hạ Long 1 ngày", "service_id": 20012}},
  "booking_intent": {{"has_intent": false, "confidence": 0.0, "keywords": []}}
}}

NGUYÊN TẮC:
- LUÔN LUÔN trả về JSON hợp lệ
- contextualized_question PHẢI rõ ràng, có thể search được
- **service_id** luôn là số nguyên hoặc null
- **booking_intent.has_intent** = true nếu có từ khóa đặt hàng
- Nếu không chắc chắn → is_followup = false
- Ưu tiên thông tin gần nhất trong lịch sử"""

    @property
    def classification_prompt(self) -> str:
        """System prompt cho classification."""
        return """Bạn là trợ lý phân loại câu hỏi du lịch Bãi Cháy.

Phân loại câu hỏi thành 1 trong 6 loại:
- "hello": Lời chào, chào hỏi, giới thiệu ban đầu
- "human": Khách cung cấp thông tin cá nhân (tên, SĐT, ngày check-in/out)
- "tourism": Tìm tour, điểm đến, khách sạn, nhà hàng, giá cả
- "tourism_detail": Hỏi chi tiết về dịch vụ CỤ THỂ (có service_id hoặc tên rõ ràng)
- "document": Hỏi về quy định, khiếu nại, thủ tục, chính sách
- "booking": Khách muốn đặt dịch vụ (có booking intent HOẶC đã đủ thông tin)

**QUAN TRỌNG - Phân biệt tourism vs tourism_detail:**
- "tourism": Tìm kiếm CHUNG ("tìm khách sạn gần biển", "tour Hạ Long")
- "tourism_detail": Hỏi về dịch vụ CỤ THỂ ("thông tin về id 20123", "giá của tour Mường Thanh")

**QUAN TRỌNG - Nhận diện booking:**
- Có từ khóa: "đặt", "book", "booking", "chốt", "đặt luôn", "tôi muốn đặt"
- Hoặc đã có đủ thông tin: tên + SĐT + ngày + service_id

Ví dụ:
- "Xin chào" → hello
- "Tên tôi là Nguyễn Văn A" → human
- "Tìm khách sạn 4 sao gần biển" → tourism
- "Cho tôi biết về dịch vụ id 20123" → tourism_detail
- "Giá của khách sạn Mường Thanh bao nhiêu?" → tourism_detail
- "Quy định hủy tour như thế nào?" → document
- "Đặt luôn tour này" → booking
- "Tôi muốn đặt dịch vụ 30045" → booking

CHỈ TRẢ VỀ 1 TỪ: hello, human, tourism, tourism_detail, document, hoặc booking
KHÔNG GIẢI THÍCH, CHỈ TRẢ VỀ TỪ KHÓA."""

    @property
    def system_prompt(self) -> str:
        """Backward compatibility."""
        return self.classification_prompt

    def process(self, state: AgentState) -> AgentState:
        """
        Phân loại query với context processing:
        1. Phân tích context và làm rõ câu hỏi
        2. Extract service_id và booking_intent
        3. Phân loại query type
        4. Update state
        """
        logger.info("🔀 Router Agent analyzing query...")

        # Step 1: Context Processing
        context_result = self._process_context(state)

        if context_result:
            # Update state với contextualized question
            original_query = state["user_query"]
            contextualized_query = context_result.get("contextualized_question", original_query)

            # Extract service_id
            service_id = context_result.get("resolved_entities", {}).get("service_id")
            if not service_id:
                service_id = context_result.get("detected_references", {}).get("service_id")

            # Extract booking intent
            booking_intent = context_result.get("booking_intent", {})
            has_booking_intent = booking_intent.get("has_intent", False)

            # Log context analysis
            if context_result.get("is_followup"):
                logger.info(f"📝 Follow-up detected!")
                logger.info(f"   Original: {original_query}")
                logger.info(f"   Contextualized: {contextualized_query}")
                logger.info(f"   Summary: {context_result.get('context_summary')}")

            if service_id:
                logger.info(f"🎯 Service ID detected: {service_id}")

            if has_booking_intent:
                logger.info(f"🎫 Booking intent detected (confidence: {booking_intent.get('confidence', 0):.2f})")

            # Store context info in state
            state["contextualized_query"] = contextualized_query
            state["context_info"] = context_result
            state["service_id"] = service_id  # ⭐ NEW

            # Use contextualized query for classification
            query_for_classification = contextualized_query
        else:
            # No context processing, use original
            query_for_classification = state["user_query"]
            state["contextualized_query"] = state["user_query"]
            state["context_info"] = {
                "is_followup": False,
                "context_summary": "Câu hỏi độc lập",
                "booking_intent": {"has_intent": False, "confidence": 0.0}
            }
            state["service_id"] = None  # ⭐ NEW

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
                    f"Service ID: {state.get('service_id', 'Không có')}\n"
                    f"Booking intent: {state.get('context_info', {}).get('booking_intent', {}).get('has_intent', False)}\n\n"
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
        elif "tourism_detail" in raw:
            query_type = "tourism_detail"  # ⭐ NEW
        elif "document" in raw:
            query_type = "document"
        elif "booking" in raw:
            query_type = "booking"
        else:
            query_type = "tourism"

        # Override với booking nếu có booking intent mạnh
        if state.get("context_info", {}).get("booking_intent", {}).get("confidence", 0) >= 0.8:
            logger.info("🎫 Strong booking intent → overriding to booking")
            query_type = "booking"

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
            Dict với is_followup, contextualized_question, context_summary, service_id, booking_intent
        """
        try:
            # Build conversation history
            history_text = self._build_history_text(state)

            # If no history, no context needed
            if not history_text or len(history_text.strip()) < 10:
                return {
                    "is_followup": False,
                    "contextualized_question": state["user_query"],
                    "context_summary": "Câu hỏi độc lập",
                    "detected_references": {},
                    "resolved_entities": {},
                    "booking_intent": {"has_intent": False, "confidence": 0.0}
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
                    "context_summary": "Không thể phân tích context",
                    "detected_references": {},
                    "resolved_entities": {},
                    "booking_intent": {"has_intent": False, "confidence": 0.0}
                }

        except Exception as e:
            logger.error(f"❌ Context processing error: {e}")
            return {
                "is_followup": False,
                "contextualized_question": state["user_query"],
                "context_summary": f"Lỗi xử lý context: {e}",
                "detected_references": {},
                "resolved_entities": {},
                "booking_intent": {"has_intent": False, "confidence": 0.0}
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