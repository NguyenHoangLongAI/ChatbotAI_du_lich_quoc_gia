"""
workflow.py — Build và compile LangGraph StateGraph cho hệ thống RAG Bãi Cháy
UPDATED: Thêm TourismDetailAgent vào workflow
"""

import logging
from langgraph.graph import StateGraph, END

from Project.state.state import AgentState
from Project.tools.tools import RAGTools
from Project.agents import (
    RouterAgent,
    HelloAgent,
    HumanAgent,
    TourismAdvisorAgent,
    DocumentAdvisorAgent,
    BookingAgent,
)
from Project.agents.tourism_detail_agent import TourismDetailAgent  # NEW

logger = logging.getLogger(__name__)


def _route_query(state: AgentState) -> str:
    """Hàm routing: chuyển từ router node sang agent phù hợp."""
    query_type = state.get("query_type", "tourism")

    if query_type == "hello":
        return "hello"
    if query_type == "human":
        return "human"
    if query_type == "tourism_detail":  # ⭐ NEW
        return "tourism_detail"
    if query_type == "document":
        return "document_advisor"
    if query_type == "booking":
        return "booking_agent"
    return "tourism_advisor"


def _route_after_human(state: AgentState) -> str:
    """Routing sau HumanAgent: tiếp tục collect info hoặc chuyển booking."""
    next_action = state.get("next_action", "continue")

    if next_action == "booking":
        # Đã đủ thông tin → chuyển booking
        return "booking_agent"
    else:
        # Vẫn còn thiếu info → end (để user tiếp tục cung cấp)
        return "end"


def build_rag_workflow(
    openai_model: str = "gpt-4o",
    milvus_host: str = "localhost",
    milvus_port: str = "19530",
):
    """
    Khởi tạo tools + agents rồi compile StateGraph.

    Workflow:
        router → {
            hello,
            human,
            tourism_advisor,      # Tìm kiếm chung
            tourism_detail,       # ⭐ NEW - Chi tiết dịch vụ theo ID
            document_advisor,
            booking_agent
        }
        human → booking_agent | end

    Returns:
        Compiled LangGraph (CompiledGraph).
    """
    # Initialize tools
    tools = RAGTools(milvus_host=milvus_host, milvus_port=milvus_port)

    # Initialize modular agents
    router = RouterAgent(tools=tools, openai_model=openai_model)
    hello = HelloAgent(tools=tools, openai_model=openai_model)
    human = HumanAgent(tools=tools, openai_model=openai_model)
    tourism_advisor = TourismAdvisorAgent(tools=tools, openai_model=openai_model)
    tourism_detail = TourismDetailAgent(tools=tools, openai_model=openai_model)  # ⭐ NEW
    document_advisor = DocumentAdvisorAgent(tools=tools, openai_model=openai_model)
    booking = BookingAgent(tools=tools, openai_model=openai_model)

    # Build graph
    graph = StateGraph(AgentState)

    # Add nodes - sử dụng method process của mỗi agent
    graph.add_node("router", router.process)
    graph.add_node("hello", hello.process)
    graph.add_node("human", human.process)
    graph.add_node("tourism_advisor", tourism_advisor.process)
    graph.add_node("tourism_detail", tourism_detail.process)  # ⭐ NEW
    graph.add_node("document_advisor", document_advisor.process)
    graph.add_node("booking_agent", booking.process)

    # Entry point
    graph.set_entry_point("router")

    # Conditional edges từ router
    graph.add_conditional_edges(
        "router",
        _route_query,
        {
            "hello": "hello",
            "human": "human",
            "tourism_advisor": "tourism_advisor",
            "tourism_detail": "tourism_detail",  # ⭐ NEW
            "document_advisor": "document_advisor",
            "booking_agent": "booking_agent",
        },
    )

    # Conditional edges từ human agent
    graph.add_conditional_edges(
        "human",
        _route_after_human,
        {
            "booking_agent": "booking_agent",
            "end": END,
        },
    )

    # Tất cả agent khác đều kết thúc workflow
    graph.add_edge("hello", END)
    graph.add_edge("tourism_advisor", END)
    graph.add_edge("tourism_detail", END)  # ⭐ NEW
    graph.add_edge("document_advisor", END)
    graph.add_edge("booking_agent", END)

    logger.info("✅ LangGraph workflow compiled with TourismDetailAgent")
    return graph.compile()