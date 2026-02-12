"""
rag_system.py — Giao diện chính của hệ thống RAG Bãi Cháy với streaming support.
UPDATED: Thêm astream_query cho real-time streaming
FIXED: Không phụ thuộc vào rag_multi_agent_system

Sử dụng:
    from rag_system.rag_system import BaiChayRAGSystem

    # Non-streaming
    rag = BaiChayRAGSystem()
    result = rag.process_query("Tìm khách sạn 4 sao gần biển")
    print(result["response"])

    # Streaming
    async for json_str in rag.astream_query("Tìm khách sạn 4 sao gần biển"):
        print(json_str)  # JSON string cho mỗi chunk
"""

import logging
import json
import asyncio
import os
from typing import Dict, List, Optional, AsyncGenerator

from Project.workflow.workflow import build_rag_workflow
from Project.state.state import AgentState
from Project.tools.tools import RAGTools
from Project.agents.agents import TourismAgents
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

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
        """
        Build streaming workflow - không invoke LLM trong agent nodes
        """
        if self._streaming_workflow is None:
            logger.info("🔄 Building streaming workflow...")

            # Initialize tools
            self._tools = RAGTools(
                milvus_host=self.milvus_host,
                milvus_port=self.milvus_port
            )

            # Create streaming agents (không invoke LLM)
            class StreamingAgents(TourismAgents):
                """
                Agents cho streaming - chỉ search, không invoke LLM
                """

                def tourism_advisor_agent(self, state: AgentState) -> AgentState:
                    """Tourism advisor: chỉ search, KHÔNG invoke LLM"""
                    logger.info("🏖️ [STREAM] Tourism Advisor: searching only...")
                    user_query = state["user_query"]

                    # Search only
                    search_results = self.tools.search_tourism_services(
                        query=user_query, top_k=5
                    )

                    try:
                        state["search_results"] = json.loads(search_results)
                    except:
                        state["search_results"] = {}

                    # Prepare messages for streaming (không invoke)
                    from langchain_core.messages import SystemMessage, HumanMessage

                    system_prompt = """Bạn là chuyên gia tư vấn du lịch Bãi Cháy - Quảng Ninh.

NHIỆM VỤ:
Dựa vào kết quả tìm kiếm, tư vấn cho khách hàng về các dịch vụ du lịch.

FORMAT TRẢ LỜI BẮT BUỘC:
Với mỗi dịch vụ, trình bày theo cấu trúc sau:

---
### 🏨 [Tên dịch vụ] {rating > 0 ? '⭐ [rating]/5' : ''}

**📍 Địa chỉ:** [address hoặc location]
**💰 Giá:** [price_range]
**📝 Mô tả:** [Tóm tắt description, khoảng 4-5 câu]
**🖼️ Hình ảnh:** {image_url có giá trị ? hiển thị URL : "Chưa có hình ảnh"}
**🔗 Xem chi tiết:** {url có giá trị ? hiển thị URL : "Liên hệ để biết thêm"}
**🆔 ID để đặt:** [id]
---

NGUYÊN TẮC QUAN TRỌNG:
1. ✅ LUÔN LUÔN hiển thị image_url nếu có
2. ✅ LUÔN LUÔN hiển thị url bài viết nếu có
3. ✅ Sắp xếp theo similarity_score (cao nhất trước)
4. ✅ Kết thúc bằng câu hỏi booking

PHONG CÁCH: Thân thiện, nhiệt tình, chuyên nghiệp."""

                    llm_messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Câu hỏi: {user_query}\n\nKết quả tìm kiếm:\n{search_results}\n\nHãy tư vấn cho khách hàng.")
                    ]

                    state["stream_messages"] = llm_messages
                    state["stream_system_prompt"] = system_prompt
                    state["final_response"] = ""
                    state["next_action"] = "stream"

                    logger.info(f"✅ [STREAM] Search done, ready for streaming")
                    return state

                def document_advisor_agent(self, state: AgentState) -> AgentState:
                    """Document advisor: chỉ search, KHÔNG invoke LLM"""
                    logger.info("📚 [STREAM] Document Advisor: searching only...")
                    user_query = state["user_query"]

                    search_results = self.tools.search_documents.invoke({
                        "query": user_query, "top_k": 3
                    })

                    try:
                        state["search_results"] = json.loads(search_results)
                    except:
                        state["search_results"] = {}

                    from langchain_core.messages import SystemMessage, HumanMessage

                    system_prompt = """Bạn là chuyên gia tư vấn quy định du lịch Bãi Cháy.

NHIỆM VỤ:
1. Đọc kỹ nội dung tài liệu tìm được
2. Trả lời chính xác dựa trên tài liệu
3. Trích dẫn nguồn (document_id) nếu có

NGUYÊN TẮC:
- Chỉ trả lời dựa trên tài liệu tìm được
- Nếu không tìm thấy: "Tôi chưa tìm thấy thông tin này trong tài liệu"
- Trình bày rõ ràng, dễ hiểu
- Gợi ý liên hệ hotline nếu cần"""

                    llm_messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Câu hỏi: {user_query}\n\nTài liệu tìm được:\n{search_results}\n\nHãy trả lời câu hỏi.")
                    ]

                    state["stream_messages"] = llm_messages
                    state["stream_system_prompt"] = system_prompt
                    state["final_response"] = ""
                    state["next_action"] = "stream"

                    return state

                def booking_agent(self, state: AgentState) -> AgentState:
                    """Booking agent: chuẩn bị context, KHÔNG invoke LLM"""
                    logger.info("🎫 [STREAM] Booking Agent working...")
                    user_query = state["user_query"]
                    messages_history = state["messages"]

                    conversation_text = "\n".join([
                        f"{msg.__class__.__name__}: {msg.content}"
                        for msg in messages_history[-3:] if hasattr(msg, 'content')
                    ])

                    from langchain_core.messages import SystemMessage, HumanMessage

                    system_prompt = """Bạn là chuyên viên đặt tour du lịch Bãi Cháy.

NHIỆM VỤ:
1. Thu thập đầy đủ thông tin:
   - Họ tên khách hàng
   - Số điện thoại
   - ID dịch vụ đã chọn
   - Ngày check-in (YYYY-MM-DD)
   - Ngày check-out (YYYY-MM-DD)

2. Nếu ĐỦ thông tin: Trả về JSON
3. Nếu THIẾU: Hỏi thêm thông tin"""

                    llm_messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Lịch sử hội thoại:\n{conversation_text}\n\nTin nhắn mới: {user_query}\n\nPhân tích và xử lý.")
                    ]

                    state["stream_messages"] = llm_messages
                    state["stream_system_prompt"] = system_prompt
                    state["final_response"] = ""
                    state["next_action"] = "stream"

                    return state

            # Build workflow
            agents = StreamingAgents(self._tools, openai_model=self.openai_model)

            workflow = StateGraph(AgentState)
            workflow.add_node("router", agents.router_agent)
            workflow.add_node("tourism_advisor", agents.tourism_advisor_agent)
            workflow.add_node("document_advisor", agents.document_advisor_agent)
            workflow.add_node("booking_agent", agents.booking_agent)

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

            logger.info("✅ Streaming workflow ready")

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

        Usage:
            async for json_str in rag_system.astream_query("..."):
                yield f"data: {json_str}\\n\\n"
        """
        # Lazy build streaming workflow
        self._build_streaming_workflow()

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

        # Bước 2: Chạy workflow trong thread pool (LangGraph invoke là sync)
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