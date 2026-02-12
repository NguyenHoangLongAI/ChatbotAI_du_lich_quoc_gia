"""
state.py — Định nghĩa AgentState cho hệ thống multi-agent RAG Bãi Cháy
UPDATED: Thêm stream_messages, stream_system_prompt cho streaming và customer_info cho HumanAgent
"""

from typing import TypedDict, Annotated, List, Dict, Optional
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Trạng thái dùng chung giữa tất cả các agent trong workflow."""

    messages: Annotated[List[BaseMessage], operator.add]  # Lịch sử hội thoại
    user_query: str                                        # Query gốc từ user
    contextualized_query: str                              # Query đã được làm rõ với context (by RouterAgent)
    context_info: Optional[Dict]                           # Thông tin context analysis
    query_type: str                                        # tourism | document | booking | hello | unknown
    search_results: Optional[Dict]                         # Kết quả tìm kiếm từ vector DB
    selected_services: List[Dict]                          # Dịch vụ khách hàng đã chọn
    booking_info: Optional[Dict]                           # Thông tin đặt hàng
    customer_info: Optional[Dict]                          # Thông tin khách hàng (extracted by HumanAgent)
    next_action: str                                       # Action tiếp theo
    final_response: str                                    # Response cuối cùng
    stream_messages: Optional[List[BaseMessage]]           # Messages cho streaming LLM
    stream_system_prompt: Optional[str]                    # System prompt cho streaming