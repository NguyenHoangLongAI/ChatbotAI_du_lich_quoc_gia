"""
rag_system.py — Giao diện chính của hệ thống RAG Bãi Cháy với streaming support.
UPDATED: Sử dụng modular agents
"""

import logging
import json
import asyncio
import os
from typing import Dict, List, Optional, AsyncGenerator

from Project.workflow.workflow import build_rag_workflow
from Project.state.state import AgentState
from Project.tools.tools import RAGTools
from Project.agents.base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class BaiChayRAGSystem:
    """
    Facade cho toàn bộ hệ thống multi-agent RAG Bãi Cháy với streaming support.

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
        logger.info("   ✅ Streaming support enabled")

        self.openai_model = openai_model
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

        # Non-streaming workflow
        self.workflow = build_rag_workflow(
            openai_model=openai_model,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
        )

        # Streaming workflow - sẽ build khi cần
        self._streaming_workflow = None
        self._streaming_llm = None
        self._tools = None

        logger.info("✅ RAG System ready! (non-stream + stream)")

    def _build_streaming_workflow(self):
        """Build streaming workflow với modular agents"""
        if self._streaming_workflow is None:
            logger.info("🔄 Building streaming workflow...")

            # Initialize tools
            self._tools = RAGTools(
                milvus_host=self.milvus_host,
                milvus_port=self.milvus_port
            )

            # Import streaming versions
            from Project.agents import (
                RouterAgent,
                TourismAdvisorAgent,
                DocumentAdvisorAgent,
                BookingAgent
            )

            # Create custom streaming agents
            class StreamingTourismAdvisor(TourismAdvisorAgent):
                """Tourism advisor cho streaming - không invoke LLM"""
                def process(self, state: AgentState) -> AgentState:
                    logger.info("🏖️ [STREAM] Tourism Advisor: searching only...")
                    search_results = self.tools.search_tourism_services(
                        query=state["user_query"], top_k=5
                    )

                    try:
                        state["search_results"] = json.loads(search_results)
                    except:
                        state["search_results"] = {}

                    llm_messages = [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=f"Câu hỏi: {state['user_query']}\n\nKết quả tìm kiếm:\n{search_results}\n\nHãy tư vấn cho khách hàng.")
                    ]

                    state["stream_messages"] = llm_messages
                    state["stream_system_prompt"] = self.system_prompt
                    state["final_response"] = ""
                    state["next_action"] = "stream"
                    return state

            class StreamingDocumentAdvisor(DocumentAdvisorAgent):
                """Document advisor cho streaming - không invoke LLM"""
                def process(self, state: AgentState) -> AgentState:
                    logger.info("📚 [STREAM] Document Advisor: searching only...")
                    search_results = self.tools.search_documents.invoke({
                        "query": state["user_query"], "top_k": 3
                    })

                    try:
                        state["search_results"] = json.loads(search_results)
                    except:
                        state["search_results"] = {}

                    llm_messages = [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=f"Câu hỏi: {state['user_query']}\n\nTài liệu tìm được:\n{search_results}\n\nHãy trả lời câu hỏi.")
                    ]

                    state["stream_messages"] = llm_messages
                    state["stream_system_prompt"] = self.system_prompt
                    state["final_response"] = ""
                    state["next_action"] = "stream"
                    return state

            class StreamingBookingAgent(BookingAgent):
                """Booking agent cho streaming - không invoke LLM"""
                def process(self, state: AgentState) -> AgentState:
                    logger.info("🎫 [STREAM] Booking Agent working...")
                    conversation_text = "\n".join([
                        f"{msg.__class__.__name__}: {msg.content}"
                        for msg in state["messages"][-3:] if hasattr(msg, 'content')
                    ])

                    llm_messages = [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=f"Lịch sử hội thoại:\n{conversation_text}\n\nTin nhắn mới: {state['user_query']}\n\nPhân tích và xử lý.")
                    ]

                    state["stream_messages"] = llm_messages
                    state["stream_system_prompt"] = self.system_prompt
                    state["final_response"] = ""
                    state["next_action"] = "stream"
                    return state

            # Initialize streaming agents
            router = RouterAgent(tools=self._tools, openai_model=self.openai_model)
            tourism_advisor = StreamingTourismAdvisor(tools=self._tools, openai_model=self.openai_model)
            document_advisor = StreamingDocumentAdvisor(tools=self._tools, openai_model=self.openai_model)
            booking = StreamingBookingAgent(tools=self._tools, openai_model=self.openai_model)

            # Build workflow
            workflow = StateGraph(AgentState)
            workflow.add_node("router", router.process)
            workflow.add_node("tourism_advisor", tourism_advisor.process)
            workflow.add_node("document_advisor", document_advisor.process)
            workflow.add_node("booking_agent", booking.process)

            workflow.set_entry_point("router")

            def route_query(state: AgentState) -> str:
                query_type = state.get("query_type", "tourism")
                if query_type == "document":
                    return "document_advisor"
                elif query_type == "booking":
                    return "booking_agent"
                else:
                    return "tourism_advisor"

            workflow.add_conditional_edges(
                "router",
                route_query,
                {
                    "tourism_advisor": "tourism_advisor",
                    "document_advisor": "document_advisor",
                    "booking_agent": "booking_agent"
                }
            )

            workflow.add_edge("tourism_advisor", END)
            workflow.add_edge("document_advisor", END)
            workflow.add_edge("booking_agent", END)

            self._streaming_workflow = workflow.compile()

            # LLM riêng để stream
            api_key = os.getenv("OPENAI_API_KEY")
            self._streaming_llm = ChatOpenAI(
                model=self.openai_model,
                temperature=0.1,
                streaming=True,
                api_key=api_key
            )

            logger.info("✅ Streaming workflow ready with modular agents")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_query(
        self,
        user_query: str,
        conversation_history: Optional[List] = None,
    ) -> Dict:
        """
        Xử lý câu hỏi qua workflow multi-agent (non-streaming).

        Args:
            user_query: Câu hỏi của người dùng.
            conversation_history: Lịch sử hội thoại (list LangChain messages).

        Returns:
            Dict với keys: response, query_type, messages.
        """
        initial_state = {
            "messages": conversation_history or [],
            "user_query": user_query,
            "contextualized_query": user_query,  # ADD THIS
            "context_info": None,  # ADD THIS
            "query_type": "unknown",
            "search_results": None,
            "selected_services": [],
            "booking_info": None,
            "customer_info": None,
            "next_action": "",
            "final_response": "",
            "stream_messages": None,
            "stream_system_prompt": None,
        }

        logger.info(f"🔄 [NON-STREAM] Processing query: {user_query}")
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

    async def astream_query(
        self,
        user_query: str,
        conversation_history: Optional[List] = None
    ) -> AsyncGenerator[str, None]:
        """
        ⭐ ASYNC STREAMING: Chạy workflow để search, sau đó async stream LLM response.

        Yield từng JSON string theo format:
            {"type": "start",  "content": null, "references": null, "status": "processing"}
            {"type": "chunk",  "content": "text...", "references": null, "status": null}
            {"type": "end",    "content": null, "references": null, "status": "done"}
            {"type": "error",  "content": "msg", "references": null, "status": "error"}
        """
        # Lazy build streaming workflow
        self._build_streaming_workflow()

        initial_state = {
            "messages": conversation_history or [],
            "user_query": user_query,
            "contextualized_query": user_query,  # ADD THIS
            "context_info": None,  # ADD THIS
            "query_type": "unknown",
            "search_results": None,
            "selected_services": [],
            "booking_info": None,
            "customer_info": None,
            "next_action": "",
            "final_response": "",
            "stream_messages": None,
            "stream_system_prompt": None,
        }

        # Bước 1: Start chunk
        yield json.dumps({
            "type": "start",
            "content": None,
            "references": None,
            "status": "processing"
        })

        # Bước 2: Chạy workflow trong thread pool
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None,
            lambda: self._streaming_workflow.invoke(initial_state)
        )

        stream_messages = final_state.get("stream_messages")
        query_type = final_state.get("query_type", "tourism")

        if not stream_messages:
            logger.error("❌ [ASTREAM] No stream_messages found in state")
            yield json.dumps({
                "type": "error",
                "content": "Xin lỗi, có lỗi xảy ra.",
                "references": None,
                "status": "error"
            })
            return

        logger.info(f"✅ [ASTREAM] Workflow done (type={query_type}), starting async LLM stream...")

        # Bước 3: Async stream LLM realtime
        chunk_count = 0
        try:
            async for chunk in self._streaming_llm.astream(stream_messages):
                if chunk.content:
                    chunk_count += 1
                    yield json.dumps({
                        "type": "chunk",
                        "content": chunk.content,
                        "references": None,
                        "status": None
                    })
        except Exception as e:
            logger.error(f"❌ [ASTREAM] LLM streaming error: {e}")
            yield json.dumps({
                "type": "error",
                "content": str(e),
                "references": None,
                "status": "error"
            })
            return

        logger.info(f"✅ [ASTREAM] Done, streamed {chunk_count} chunks")

        # Bước 4: End chunk
        yield json.dumps({
            "type": "end",
            "content": None,
            "references": None,
            "status": "done"
        })