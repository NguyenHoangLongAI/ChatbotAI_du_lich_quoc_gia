"""
state.py — Định nghĩa AgentState cho hệ thống multi-agent RAG Bãi Cháy
"""

from typing import TypedDict, Annotated, List, Dict, Optional
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Trạng thái dùng chung giữa tất cả các agent trong workflow."""

    messages: Annotated[List[BaseMessage], operator.add]  # Lịch sử hội thoại
    user_query: str                                        # Query gốc từ user
    query_type: str                                        # tourism | document | booking | unknown
    search_results: Optional[Dict]                         # Kết quả tìm kiếm từ vector DB
    selected_services: List[Dict]                          # Dịch vụ khách hàng đã chọn
    booking_info: Optional[Dict]                           # Thông tin đặt hàng
    customer_info: Optional[Dict]                          # Thông tin khách hàng
    next_action: str                                       # Action tiếp theo
    final_response: str                                    # Response cuối cùng