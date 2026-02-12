"""
workflow.py — Build và compile LangGraph StateGraph cho hệ thống RAG Bãi Cháy
UPDATED: Sử dụng modular agents thay vì TourismAgents class
"""

import logging
from langgraph.graph import StateGraph, END

from Project.state.state import AgentState
from Project.tools.tools import RAGTools
from Project.agents import (
    RouterAgent,
    TourismAdvisorAgent,
    DocumentAdvisorAgent,
    BookingAgent,
)

logger = logging.getLogger(__name__)


def _route_query(state: AgentState) -> str:
    """Hàm routing: chuyển từ router node sang agent phù hợp."""
    query_type = state.get("query_type", "tourism")
    if query_type == "document":
        return "document_advisor"
    if query_type == "booking":
        return "booking_agent"
    return "tourism_advisor"


def build_rag_workflow(
    openai_model: str = "gpt-4o",
    milvus_host: str = "localhost",
    milvus_port: str = "19530",
):
    """
    Khởi tạo tools + agents rồi compile StateGraph.

    Returns:
        Compiled LangGraph (CompiledGraph).
    """
    # Initialize tools
    tools = RAGTools(milvus_host=milvus_host, milvus_port=milvus_port)

    # Initialize modular agents
    router = RouterAgent(tools=tools, openai_model=openai_model)
    tourism_advisor = TourismAdvisorAgent(tools=tools, openai_model=openai_model)
    document_advisor = DocumentAdvisorAgent(tools=tools, openai_model=openai_model)
    booking = BookingAgent(tools=tools, openai_model=openai_model)

    # Build graph
    graph = StateGraph(AgentState)

    # Add nodes - sử dụng method process của mỗi agent
    graph.add_node("router", router.process)
    graph.add_node("tourism_advisor", tourism_advisor.process)
    graph.add_node("document_advisor", document_advisor.process)
    graph.add_node("booking_agent", booking.process)

    # Entry point
    graph.set_entry_point("router")

    # Conditional edges từ router
    graph.add_conditional_edges(
        "router",
        _route_query,
        {
            "tourism_advisor": "tourism_advisor",
            "document_advisor": "document_advisor",
            "booking_agent": "booking_agent",
        },
    )

    # Tất cả agent đều kết thúc workflow
    graph.add_edge("tourism_advisor", END)
    graph.add_edge("document_advisor", END)
    graph.add_edge("booking_agent", END)

    logger.info("✅ LangGraph workflow compiled with modular agents")
    return graph.compile()