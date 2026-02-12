"""
human_agent.py — Agent phân tích context và extract thông tin khách hàng
"""

import logging
import re
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from Project.agents.base_agent import BaseAgent
from Project.state.state import AgentState

logger = logging.getLogger(__name__)


class HumanAgent(BaseAgent):
    """Agent phân tích context hội thoại và extract thông tin khách hàng."""

    @property
    def system_prompt(self) -> str:
        return """Bạn là trợ lý thông minh phân tích thông tin khách hàng từ hội thoại.

NHIỆM VỤ:
Phân tích hội thoại và extract các thông tin sau (nếu có):

1. **Họ tên khách hàng**: Tìm trong các mẫu câu:
   - "Tôi tên là...", "Tên tôi là...", "Mình tên..."
   - "Đặt cho...", "Đặt tên..."
   - Tên xuất hiện sau "anh", "chị", "bác", "cô"

2. **Số điện thoại**: Các định dạng:
   - 10 số: 0123456789
   - Có dấu cách/gạch: 012 345 6789, 012-345-6789
   - Có +84: +84 123 456 789

3. **Ngày check-in**: Tìm trong các mẫu:
   - Ngày cụ thể: "15/03", "15/03/2024", "15-03-2024"
   - Tương đối: "ngày mai", "tuần sau", "cuối tuần này"
   - Mô tả: "từ ngày...", "check-in ngày..."

4. **Ngày check-out**: Tìm trong các mẫu:
   - Ngày cụ thể: "17/03", "17/03/2024"
   - Tương đối: "2 ngày sau", "3 ngày 2 đêm"
   - Mô tả: "đến ngày...", "check-out ngày..."

OUTPUT FORMAT (JSON):
Trả về JSON với cấu trúc:
{
    "name": "Họ tên khách hàng hoặc null",
    "phone": "Số điện thoại hoặc null",
    "checkin": "YYYY-MM-DD hoặc null",
    "checkout": "YYYY-MM-DD hoặc null",
    "confidence": {
        "name": 0.0-1.0,
        "phone": 0.0-1.0,
        "checkin": 0.0-1.0,
        "checkout": 0.0-1.0
    },
    "raw_info": {
        "name_context": "Câu chứa tên",
        "phone_context": "Câu chứa SĐT",
        "checkin_context": "Câu chứa ngày checkin",
        "checkout_context": "Câu chứa ngày checkout"
    },
    "missing_fields": ["danh sách field còn thiếu"],
    "interpretation_notes": "Ghi chú về cách hiểu ngày tháng nếu có"
}

RULES:
- Chỉ extract thông tin RÕ RÀNG, không đoán
- Ngày tháng tương đối → chuyển thành ngày cụ thể (dựa vào ngày hôm nay: {today})
- Nếu chỉ có "3 ngày 2 đêm" mà không có ngày bắt đầu → để null
- Confidence cao (0.8-1.0) nếu thông tin rõ ràng
- Confidence thấp (0.3-0.6) nếu không chắc chắn
- Luôn trả về JSON hợp lệ

VÍ DỤ:

Input: "Tôi tên Nguyễn Văn A, SĐT 0901234567, muốn đặt phòng từ 15/03 đến 17/03"
Output:
{{
    "name": "Nguyễn Văn A",
    "phone": "0901234567",
    "checkin": "2024-03-15",
    "checkout": "2024-03-17",
    "confidence": {{
        "name": 1.0,
        "phone": 1.0,
        "checkin": 1.0,
        "checkout": 1.0
    }},
    "raw_info": {{
        "name_context": "Tôi tên Nguyễn Văn A",
        "phone_context": "SĐT 0901234567",
        "checkin_context": "từ 15/03",
        "checkout_context": "đến 17/03"
    }},
    "missing_fields": [],
    "interpretation_notes": "Giả định năm 2024"
}}

Input: "Đặt cho anh Minh nhé, 098 765 4321, check-in ngày mai"
Output:
{{
    "name": "anh Minh",
    "phone": "0987654321",
    "checkin": "{tomorrow}",
    "checkout": null,
    "confidence": {{
        "name": 0.8,
        "phone": 1.0,
        "checkin": 0.9,
        "checkout": 0.0
    }},
    "raw_info": {{
        "name_context": "Đặt cho anh Minh",
        "phone_context": "098 765 4321",
        "checkin_context": "check-in ngày mai",
        "checkout_context": null
    }},
    "missing_fields": ["checkout"],
    "interpretation_notes": "Ngày mai = {tomorrow}"
}}"""

    def __init__(self, tools, openai_model: str = "gpt-4o"):
        super().__init__(tools, openai_model)
        # Pattern để extract thông tin
        self.phone_pattern = re.compile(r'(?:\+84|0)[\s\-]?[1-9](?:[\s\-]?\d){8}')
        self.date_pattern = re.compile(r'\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?')

    def process(self, state: AgentState) -> AgentState:
        """Phân tích context và extract thông tin khách hàng."""
        logger.info("🧠 Human Agent analyzing context...")

        # Lấy toàn bộ context hội thoại
        conversation_history = self._build_conversation_context(state)

        # Thêm current query
        full_context = conversation_history + f"\n\nTin nhắn mới: {state['user_query']}"

        # Get today's date for context
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Format system prompt với ngày hôm nay
        formatted_prompt = self.system_prompt.format(
            today=today,
            tomorrow=tomorrow
        )

        # Invoke LLM để extract thông tin
        response = self.llm.invoke([
            SystemMessage(content=formatted_prompt),
            HumanMessage(
                content=(
                    f"Phân tích hội thoại sau và extract thông tin khách hàng:\n\n"
                    f"{full_context}\n\n"
                    "Trả về JSON theo format đã chỉ định."
                )
            ),
        ])

        # Parse JSON response
        extracted_info = self._parse_llm_response(response.content)

        # Fallback extraction nếu LLM không trả JSON
        if not extracted_info or "error" in extracted_info:
            logger.warning("⚠️ LLM didn't return valid JSON, using regex fallback")
            extracted_info = self._fallback_extraction(full_context)

        # Merge với customer_info hiện tại (nếu có)
        current_info = state.get("customer_info", {})
        merged_info = self._merge_customer_info(current_info, extracted_info)

        # Update state
        state["customer_info"] = merged_info
        state["messages"].append(AIMessage(
            content=f"[Extracted customer info: {json.dumps(merged_info, ensure_ascii=False)}]"
        ))

        # Log extracted info
        logger.info(f"✅ Extracted info: Name={merged_info.get('name')}, "
                   f"Phone={merged_info.get('phone')}, "
                   f"Check-in={merged_info.get('checkin')}, "
                   f"Check-out={merged_info.get('checkout')}")

        # Determine next action
        missing = merged_info.get("missing_fields", [])
        if not missing or len(missing) == 0:
            logger.info("✅ All info collected, ready for booking")
            state["next_action"] = "booking"
        else:
            logger.info(f"ℹ️ Still missing: {missing}")
            state["next_action"] = "continue"

        return state

    def _build_conversation_context(self, state: AgentState) -> str:
        """Build context từ lịch sử hội thoại."""
        context_lines = []

        for msg in state.get("messages", [])[-5:]:  # Lấy 5 tin nhắn gần nhất
            if hasattr(msg, "content"):
                role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                context_lines.append(f"{role}: {msg.content}")

        return "\n".join(context_lines)

    def _parse_llm_response(self, response_text: str) -> Optional[Dict]:
        """Parse JSON từ LLM response."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            # Try direct parse
            return json.loads(response_text)

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parse error: {e}")
            return None

    def _fallback_extraction(self, text: str) -> Dict:
        """Fallback extraction sử dụng regex khi LLM fails."""
        result = {
            "name": None,
            "phone": None,
            "checkin": None,
            "checkout": None,
            "confidence": {
                "name": 0.0,
                "phone": 0.0,
                "checkin": 0.0,
                "checkout": 0.0
            },
            "raw_info": {},
            "missing_fields": [],
            "interpretation_notes": "Extracted using regex fallback"
        }

        # Extract phone
        phone_match = self.phone_pattern.search(text)
        if phone_match:
            phone = re.sub(r'[\s\-]', '', phone_match.group())
            result["phone"] = phone
            result["confidence"]["phone"] = 0.8
            result["raw_info"]["phone_context"] = phone_match.group()

        # Extract dates
        dates = self.date_pattern.findall(text)
        if dates:
            # Try to parse dates
            parsed_dates = []
            for date_str in dates:
                try:
                    # Try different formats
                    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d/%m", "%d-%m"]:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            # If year not specified, assume current year
                            if fmt in ["%d/%m", "%d-%m"]:
                                dt = dt.replace(year=datetime.now().year)
                            parsed_dates.append(dt)
                            break
                        except ValueError:
                            continue
                except:
                    continue

            if parsed_dates:
                parsed_dates.sort()
                result["checkin"] = parsed_dates[0].strftime("%Y-%m-%d")
                result["confidence"]["checkin"] = 0.7
                if len(parsed_dates) > 1:
                    result["checkout"] = parsed_dates[1].strftime("%Y-%m-%d")
                    result["confidence"]["checkout"] = 0.7

        # Extract name (simple heuristic)
        name_patterns = [
            r'(?:tên|tên là|tên tôi là|mình tên)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)',
            r'(?:đặt cho|cho)\s+(anh|chị|bác|cô)\s+([A-ZÀ-Ỹ][a-zà-ỹ]+)',
        ]

        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["name"] = match.group(1) if len(match.groups()) == 1 else match.group(2)
                result["confidence"]["name"] = 0.6
                result["raw_info"]["name_context"] = match.group()
                break

        # Determine missing fields
        missing = []
        for field in ["name", "phone", "checkin", "checkout"]:
            if not result[field]:
                missing.append(field)
        result["missing_fields"] = missing

        return result

    def _merge_customer_info(
        self,
        current: Dict,
        new: Dict
    ) -> Dict:
        """Merge thông tin mới vào thông tin hiện tại (ưu tiên info mới có confidence cao hơn)."""
        merged = current.copy()

        for field in ["name", "phone", "checkin", "checkout"]:
            new_value = new.get(field)
            new_confidence = new.get("confidence", {}).get(field, 0.0)
            current_confidence = current.get("confidence", {}).get(field, 0.0)

            # Update nếu:
            # 1. Chưa có giá trị cũ
            # 2. Có giá trị mới và confidence mới cao hơn
            if new_value:
                if not merged.get(field) or new_confidence > current_confidence:
                    merged[field] = new_value

        # Merge confidence
        if "confidence" not in merged:
            merged["confidence"] = {}
        for field, conf in new.get("confidence", {}).items():
            if conf > merged["confidence"].get(field, 0.0):
                merged["confidence"][field] = conf

        # Merge raw_info
        if "raw_info" not in merged:
            merged["raw_info"] = {}
        merged["raw_info"].update(new.get("raw_info", {}))

        # Update missing fields
        missing = []
        for field in ["name", "phone", "checkin", "checkout"]:
            if not merged.get(field):
                missing.append(field)
        merged["missing_fields"] = missing

        # Merge interpretation notes
        notes = []
        if current.get("interpretation_notes"):
            notes.append(current["interpretation_notes"])
        if new.get("interpretation_notes"):
            notes.append(new["interpretation_notes"])
        if notes:
            merged["interpretation_notes"] = "; ".join(notes)

        return merged