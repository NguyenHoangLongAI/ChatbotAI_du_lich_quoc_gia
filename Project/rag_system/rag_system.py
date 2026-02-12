"""
rag_system.py — Giao diện chính của hệ thống RAG Bãi Cháy.

Sử dụng:
    from rag_system.rag_system import BaiChayRAGSystem

    rag = BaiChayRAGSystem()
    result = rag.process_query("Tìm khách sạn 4 sao gần biển")
    print(result["response"])
"""

import logging
from typing import Dict, List, Optional

from Project.workflow.workflow import build_rag_workflow

logger = logging.getLogger(__name__)


class BaiChayRAGSystem:
    """
    Facade cho toàn bộ hệ thống multi-agent RAG Bãi Cháy.

    Workflow bên trong:
        router  →  tourism_advisor  |  document_advisor  |  booking_agent
    """

    def __init__(
        self,
        openai_model: str = "gpt-4o",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
    ):
        logger.info("🚀 Initializing Bãi Cháy RAG System...")
        logger.info(f"   Model: {openai_model}")

        self.workflow = build_rag_workflow(
            openai_model=openai_model,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
        )
        logger.info("✅ RAG System ready!")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_query(
        self,
        user_query: str,
        conversation_history: Optional[List] = None,
    ) -> Dict:
        """
        Xử lý câu hỏi qua workflow multi-agent.

        Args:
            user_query: Câu hỏi của người dùng.
            conversation_history: Lịch sử hội thoại (list LangChain messages).

        Returns:
            Dict với keys: response, query_type, messages.
        """
        initial_state = {
            "messages": conversation_history or [],
            "user_query": user_query,
            "query_type": "unknown",
            "search_results": None,
            "selected_services": [],
            "booking_info": None,
            "customer_info": None,
            "next_action": "",
            "final_response": "",
        }

        logger.info(f"🔄 Processing query: {user_query}")
        final_state = self.workflow.invoke(initial_state)

        return {
            "response": final_state.get("final_response", "Xin lỗi, tôi chưa hiểu câu hỏi."),
            "query_type": final_state.get("query_type"),
            "messages": final_state.get("messages", []),
        }

    def question(
        self,
        question: str,
        history: Optional[List] = None,
    ) -> Dict:
        """Alias của process_query — tương thích với endpoint /chat."""
        return self.process_query(user_query=question, conversation_history=history)