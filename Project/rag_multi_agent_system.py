"""
RAG Multi-Agent System with LangGraph for Bãi Cháy Tourism
UPDATED: Tích hợp đầy đủ image_url và url bài viết vào câu trả lời
UPDATED v2: Thêm Real LLM Streaming support

Hệ thống đa tác nhân với workflow thông minh xử lý:
1. Tư vấn dịch vụ du lịch (có hình ảnh và link bài viết)
2. Giải đáp quy định & tài liệu
3. Hướng dẫn đặt hàng & hoàn thành booking
"""

from typing import TypedDict, Annotated, List, Dict, Optional, Generator, AsyncGenerator
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import operator
from datetime import datetime
import json
import logging
import os

# Import your existing DAOs
import sys
sys.path.append('/mnt/user-data/uploads')
from crawl_baichay_service.tourism_dao import BaiChayTourismDAO
from document_db.tourism_document_dao import TourismDocumentDAO
from baichay_db.customer_dao import CustomerDAO
from document_api.embedding_service import EmbeddingService
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# OPENAI LLM WRAPPER
# ============================================================================

class OpenAILLMWrapper:
    """Wrapper for OpenAI Chat models (GPT-4o)"""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        streaming: bool = False
    ):
        self.model = model
        self.temperature = temperature
        self.streaming = streaming

        logger.info(f"🤖 Initializing OpenAI LLM: {model}")

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            api_key=api_key
        )

        self.output_parser = StrOutputParser()
        logger.info(f"✅ OpenAI {model} initialized successfully")

    def invoke(self, messages: list, **kwargs) -> AIMessage:
        """Non-streaming invoke"""
        try:
            response = self.llm.invoke(messages, **kwargs)
            return AIMessage(content=response.content)

        except Exception as e:
            logger.error(f"❌ OpenAI invoke error: {e}", exc_info=True)
            return AIMessage(content=f"Lỗi xử lý OpenAI: {str(e)}")

    def stream(self, messages: list, **kwargs):
        """Streaming (sync) - yields real chunks từ LLM"""
        try:
            for chunk in self.llm.stream(messages, **kwargs):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"❌ OpenAI streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi streaming: {str(e)}]"

    async def astream(self, messages: list, **kwargs):
        """Async streaming - yields real chunks từ LLM"""
        try:
            async for chunk in self.llm.astream(messages, **kwargs):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"❌ OpenAI async streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi streaming: {str(e)}]"

    def bind_tools(self, tools: list):
        """Bind tools to LLM"""
        self.llm = self.llm.bind_tools(tools)
        return self


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """Trạng thái của hệ thống multi-agent"""
    messages: Annotated[List, operator.add]  # Lịch sử hội thoại
    user_query: str                           # Query gốc từ user
    query_type: str                           # tourism | document | booking | unknown
    search_results: Optional[Dict]            # Kết quả tìm kiếm từ vector DB
    selected_services: List[Dict]             # Dịch vụ khách hàng đã chọn
    booking_info: Optional[Dict]              # Thông tin đặt hàng
    customer_info: Optional[Dict]             # Thông tin khách hàng
    next_action: str                          # Action tiếp theo
    final_response: str                       # Response cuối cùng
    # ⭐ Thêm field để lưu messages cần streaming
    stream_messages: Optional[List]           # Messages cho LLM streaming (cuối workflow)
    stream_system_prompt: Optional[str]       # System prompt cho streaming


# ============================================================================
# TOOLS - Kết nối với Database
# ============================================================================

class RAGTools:
    """Tools để tương tác với Milvus collections"""

    def __init__(self, milvus_host="localhost", milvus_port="19530"):
        self.tourism_dao = BaiChayTourismDAO(host=milvus_host, port=milvus_port)
        self.document_dao = TourismDocumentDAO(host=milvus_host, port=milvus_port)
        self.customer_dao = CustomerDAO(host=milvus_host, port=milvus_port)
        self.embedding_service = EmbeddingService()

        logger.info("✅ RAG Tools initialized")

    def search_tourism_services(self, query: str, top_k: int = 5) -> str:
        """
        Tìm kiếm dịch vụ du lịch (tour, điểm đến, khách sạn, nhà hàng...)
        UPDATED: Bao gồm image_url và url bài viết
        """
        try:
            query_vector = self.embedding_service.get_embedding(query)
            results = self.tourism_dao.search_by_description(
                query_vector=query_vector,
                top_k=top_k
            )

            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id"),
                    "name": result.get("name"),
                    "type": result.get("type"),
                    "sub_type": result.get("sub_type"),
                    "location": result.get("location"),
                    "address": result.get("address"),
                    "description": result.get("description", "")[:500],
                    "price_range": result.get("price_range"),
                    "price_min": result.get("price_min"),
                    "price_max": result.get("price_max"),
                    "rating": result.get("rating"),
                    "opening_hours": result.get("opening_hours"),
                    "image_url": result.get("image_url", ""),
                    "url": result.get("url", ""),
                    "similarity_score": round(result.get("score", 0), 3)
                })

            logger.info(f"✅ Found {len(formatted_results)} services")
            return json.dumps(formatted_results, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"❌ Search tourism error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def search_documents(self, query: str, top_k: int = 3) -> str:
        """Tìm kiếm tài liệu quy định, hướng dẫn du lịch"""
        try:
            query_vector = self.embedding_service.get_embedding(query)
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 100}
            }
            results = self.document_dao.doc_collection.search(
                data=[query_vector],
                anns_field="description_vector",
                param=search_params,
                limit=top_k,
                output_fields=["document_id", "description"]
            )
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "document_id": hit.entity.get("document_id"),
                        "content": hit.entity.get("description"),
                        "similarity_score": hit.score
                    })
            return json.dumps(formatted_results, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Search document error: {e}")
            return json.dumps({"error": str(e)})

    @tool
    def get_service_by_id(self, service_id: int) -> str:
        """Lấy thông tin chi tiết dịch vụ theo ID"""
        try:
            result = self.tourism_dao.get_by_id(service_id)
            if result:
                return json.dumps(result, ensure_ascii=False, indent=2)
            else:
                return json.dumps({"error": "Service not found"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def create_customer_booking(
        self,
        name: str,
        phone: str,
        service_ids: List[int],
        service_descriptions: str,
        checkin_date: str,
        checkout_date: str
    ) -> str:
        """Tạo booking cho khách hàng"""
        try:
            checkin = datetime.strptime(checkin_date, "%Y-%m-%d")
            checkout = datetime.strptime(checkout_date, "%Y-%m-%d")
            description = f"Đặt dịch vụ du lịch Bãi Cháy. Dịch vụ: {service_descriptions}. IDs: {service_ids}"
            description_vector = self.embedding_service.get_embedding(description)
            customer_data = {
                "name": name,
                "phone": phone,
                "checkin_time": checkin,
                "checkout_time": checkout,
                "description": description,
                "description_vector": description_vector
            }
            customer_id = self.customer_dao.insert_customer(customer_data)
            result = {
                "status": "success",
                "customer_id": customer_id,
                "name": name,
                "phone": phone,
                "checkin": checkin_date,
                "checkout": checkout_date,
                "services": service_ids,
                "message": "Booking đã được tạo thành công!"
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Create booking error: {e}")
            return json.dumps({"status": "error", "message": str(e)})


# ============================================================================
# SYSTEM PROMPTS (tách ra để tái dùng cho cả invoke và stream)
# ============================================================================

TOURISM_SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn du lịch Bãi Cháy - Quảng Ninh.

NHIỆM VỤ:
Dựa vào kết quả tìm kiếm, tư vấn cho khách hàng về các dịch vụ du lịch.

FORMAT TRẢ LỜI BẮT BUỘC:
Với mỗi dịch vụ, trình bày theo cấu trúc sau:

---
### 🏨 [Tên dịch vụ] {rating > 0 ? '⭐ [rating]/5' : ''}

**📍 Địa chỉ:** [address hoặc location]

**💰 Giá:** [price_range]

**📝 Mô tả:** [Tóm tắt description, khoảng 4-5 câu]

**🖼️ Hình ảnh:**
{image_url có giá trị ? hiển thị URL : "Chưa có hình ảnh"}

**🔗 Xem chi tiết:** {url có giá trị ? hiển thị URL : "Liên hệ để biết thêm"}

**🆔 ID để đặt:** [id]

---

NGUYÊN TẮC QUAN TRỌNG:
1. ✅ LUÔN LUÔN hiển thị image_url nếu có (không được bỏ qua)
2. ✅ LUÔN LUÔN hiển thị url bài viết nếu có
3. ✅ Format URL rõ ràng, dễ click (markdown link hoặc plain URL)
4. ✅ Sắp xếp theo similarity_score (cao nhất trước)
5. ✅ Kết thúc bằng câu hỏi: "Bạn có muốn đặt dịch vụ nào không? Hãy cho tôi biết ID dịch vụ, tôi sẽ tiến hành booking cho bạn."

PHONG CÁCH: Thân thiện, nhiệt tình, chuyên nghiệp.

KHÔNG ĐƯỢC:
- ❌ Bỏ qua image_url hoặc url nếu có
- ❌ Gộp chung nhiều dịch vụ vào một mục"""

DOCUMENT_SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn quy định du lịch Bãi Cháy.

NHIỆM VỤ:
1. Đọc kỹ nội dung tài liệu tìm được
2. Trả lời chính xác dựa trên tài liệu
3. Trích dẫn nguồn (document_id) nếu có

NGUYÊN TẮC:
- Chỉ trả lời dựa trên tài liệu tìm được
- Nếu không tìm thấy: "Tôi chưa tìm thấy thông tin này trong tài liệu"
- Trình bày rõ ràng, dễ hiểu
- Gợi ý liên hệ hotline nếu cần

PHONG CÁCH: Chuyên nghiệp, chính xác, hữu ích."""

BOOKING_SYSTEM_PROMPT = """Bạn là chuyên viên đặt tour du lịch Bãi Cháy.

NHIỆM VỤ:
1. Thu thập đầy đủ thông tin:
   - Họ tên khách hàng
   - Số điện thoại
   - ID dịch vụ đã chọn (nếu có từ hội thoại trước)
   - Ngày check-in (YYYY-MM-DD)
   - Ngày check-out (YYYY-MM-DD)

2. Phân tích xem đã đủ thông tin chưa:
   - Nếu ĐỦ: Trả về JSON với format:
     {"action": "create_booking", "name": "...", "phone": "...", "service_ids": [...], "checkin": "YYYY-MM-DD", "checkout": "YYYY-MM-DD", "description": "..."}
   
   - Nếu THIẾU: Hỏi thêm thông tin còn thiếu

PHONG CÁCH: Chuyên nghiệp, thân thiện, xác nhận lại thông tin trước khi đặt."""


# ============================================================================
# AGENT NODES
# ============================================================================

class TourismAgents:
    """Các agent chuyên biệt trong hệ thống"""

    def __init__(
            self,
            tools: RAGTools,
            openai_model: str = "gpt-4o"
    ):
        self.tools = tools

        self.llm = OpenAILLMWrapper(
            model=openai_model,
            temperature=0.1
        )

        self.tool_list = [
            tools.search_tourism_services,
            tools.search_documents,
            tools.get_service_by_id,
            tools.create_customer_booking
        ]

    def router_agent(self, state: AgentState) -> AgentState:
        """Agent phân loại: Xác định loại query"""
        logger.info("🔀 Router Agent analyzing query...")

        user_query = state["user_query"]

        system_prompt = """Bạn là trợ lý phân loại câu hỏi du lịch Bãi Cháy.

Phân loại câu hỏi thành 1 trong 3 loại:
- "tourism": Tìm tour, điểm đến, khách sạn, nhà hàng, giá cả
- "document": Hỏi về quy định, khiếu nại, thủ tục, chính sách
- "booking": Khách muốn đặt dịch vụ, cung cấp thông tin cá nhân

CHỈ TRẢ VỀ 1 TỪ: tourism, document, hoặc booking
KHÔNG GIẢI THÍCH, CHỈ TRẢ VỀ TỪ KHÓA."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Phân loại câu hỏi sau:\n{user_query}")
        ]

        response = self.llm.invoke(messages)
        query_type = response.content.strip().lower()

        if "tourism" in query_type:
            query_type = "tourism"
        elif "document" in query_type:
            query_type = "document"
        elif "booking" in query_type:
            query_type = "booking"
        else:
            query_type = "tourism"

        logger.info(f"✅ Query type: {query_type}")

        state["query_type"] = query_type
        state["messages"].append(AIMessage(content=f"[Query classified as: {query_type}]"))

        return state

    def tourism_advisor_agent(self, state: AgentState) -> AgentState:
        """
        Agent tư vấn dịch vụ du lịch
        ⭐ Khi streaming=False: dùng invoke như cũ
        ⭐ Khi streaming=True: lưu messages vào state, KHÔNG invoke LLM ở đây
        """
        logger.info("🏖️ Tourism Advisor Agent working...")

        user_query = state["user_query"]

        # Bước 1: Search (luôn chạy, không phụ thuộc streaming)
        logger.info("📞 Calling search_tourism_services tool...")
        search_results = self.tools.search_tourism_services(query=user_query, top_k=5)

        # Lưu search results
        try:
            state["search_results"] = json.loads(search_results)
        except:
            state["search_results"] = {}

        # Bước 2: Chuẩn bị messages cho LLM
        llm_messages = [
            SystemMessage(content=TOURISM_SYSTEM_PROMPT),
            HumanMessage(content=f"Câu hỏi: {user_query}\n\nKết quả tìm kiếm:\n{search_results}\n\nHãy tư vấn cho khách hàng.")
        ]

        # ⭐ Lưu messages vào state để streaming endpoint dùng
        state["stream_messages"] = llm_messages
        state["stream_system_prompt"] = TOURISM_SYSTEM_PROMPT

        # Bước 3: Invoke non-streaming (dùng cho /chat endpoint bình thường)
        response = self.llm.invoke(llm_messages)
        state["messages"].append(response)
        state["final_response"] = response.content
        state["next_action"] = "end"

        logger.info("✅ Tourism advice generated")
        return state

    def document_advisor_agent(self, state: AgentState) -> AgentState:
        """Agent giải đáp quy định & tài liệu"""
        logger.info("📚 Document Advisor Agent working...")

        user_query = state["user_query"]

        # Bước 1: Search
        logger.info("📞 Calling search_documents tool...")
        search_results = self.tools.search_documents.invoke({"query": user_query, "top_k": 3})

        try:
            state["search_results"] = json.loads(search_results)
        except:
            state["search_results"] = {}

        # Bước 2: Chuẩn bị messages
        llm_messages = [
            SystemMessage(content=DOCUMENT_SYSTEM_PROMPT),
            HumanMessage(content=f"Câu hỏi: {user_query}\n\nTài liệu tìm được:\n{search_results}\n\nHãy trả lời câu hỏi.")
        ]

        state["stream_messages"] = llm_messages
        state["stream_system_prompt"] = DOCUMENT_SYSTEM_PROMPT

        # Bước 3: Invoke non-streaming
        response = self.llm.invoke(llm_messages)
        state["messages"].append(response)
        state["search_results"] = json.loads(search_results) if isinstance(search_results, str) else search_results
        state["final_response"] = response.content
        state["next_action"] = "end"

        return state

    def booking_agent(self, state: AgentState) -> AgentState:
        """Agent xử lý đặt hàng"""
        logger.info("🎫 Booking Agent working...")

        user_query = state["user_query"]
        messages_history = state["messages"]

        conversation_text = "\n".join([
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in messages_history[-3:] if hasattr(msg, 'content')
        ])

        llm_messages = [
            SystemMessage(content=BOOKING_SYSTEM_PROMPT),
            HumanMessage(content=f"Lịch sử hội thoại:\n{conversation_text}\n\nTin nhắn mới: {user_query}\n\nPhân tích và xử lý.")
        ]

        state["stream_messages"] = llm_messages
        state["stream_system_prompt"] = BOOKING_SYSTEM_PROMPT

        response = self.llm.invoke(llm_messages)
        response_text = response.content

        # Check if response contains booking action
        if '"action": "create_booking"' in response_text or "'action': 'create_booking'" in response_text:
            try:
                import re
                json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
                if json_match:
                    booking_data = json.loads(json_match.group())
                    logger.info("📞 Creating customer booking...")
                    result = self.tools.create_customer_booking.invoke(booking_data)
                    state["messages"].append(AIMessage(content=f"Booking result: {result}"))
                    state["final_response"] = f"✅ Đặt hàng thành công!\n\n{result}"
                    state["booking_info"] = json.loads(result)
                else:
                    state["final_response"] = response_text
            except Exception as e:
                logger.error(f"Booking error: {e}")
                state["final_response"] = f"Xin lỗi, có lỗi khi tạo booking: {e}\n\nVui lòng thử lại."
        else:
            state["final_response"] = response_text

        state["messages"].append(response)
        state["next_action"] = "end"

        return state


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def build_rag_workflow(openai_model: str = "gpt-4o") -> StateGraph:
    """Build the multi-agent RAG workflow"""

    tools = RAGTools()
    agents = TourismAgents(tools, openai_model=openai_model)

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

    return workflow.compile()


# ============================================================================
# STREAMING WORKFLOW BUILDER
# ⭐ Workflow này: router + search chạy bình thường, KHÔNG invoke LLM cuối
# ============================================================================

def build_rag_workflow_for_streaming(openai_model: str = "gpt-4o"):
    """
    Build workflow phục vụ streaming.
    Workflow chạy đến khi search xong, lưu messages vào state.
    Sau đó caller sẽ stream LLM bên ngoài workflow.
    """
    tools = RAGTools()

    # Tạo agents nhưng override để KHÔNG invoke LLM cuối trong advisor nodes
    class StreamingAgents(TourismAgents):

        def tourism_advisor_agent(self, state: AgentState) -> AgentState:
            """Tourism advisor: chỉ search, KHÔNG invoke LLM"""
            logger.info("🏖️ [STREAM] Tourism Advisor: searching only...")
            user_query = state["user_query"]

            search_results = self.tools.search_tourism_services(query=user_query, top_k=5)

            try:
                state["search_results"] = json.loads(search_results)
            except:
                state["search_results"] = {}

            llm_messages = [
                SystemMessage(content=TOURISM_SYSTEM_PROMPT),
                HumanMessage(content=f"Câu hỏi: {user_query}\n\nKết quả tìm kiếm:\n{search_results}\n\nHãy tư vấn cho khách hàng.")
            ]

            state["stream_messages"] = llm_messages
            state["stream_system_prompt"] = TOURISM_SYSTEM_PROMPT
            state["final_response"] = ""  # Sẽ được fill bởi streaming
            state["next_action"] = "stream"

            logger.info(f"✅ [STREAM] Search done, {len(state['search_results'])} results ready for streaming")
            return state

        def document_advisor_agent(self, state: AgentState) -> AgentState:
            """Document advisor: chỉ search, KHÔNG invoke LLM"""
            logger.info("📚 [STREAM] Document Advisor: searching only...")
            user_query = state["user_query"]

            search_results = self.tools.search_documents.invoke({"query": user_query, "top_k": 3})

            try:
                state["search_results"] = json.loads(search_results)
            except:
                state["search_results"] = {}

            llm_messages = [
                SystemMessage(content=DOCUMENT_SYSTEM_PROMPT),
                HumanMessage(content=f"Câu hỏi: {user_query}\n\nTài liệu tìm được:\n{search_results}\n\nHãy trả lời câu hỏi.")
            ]

            state["stream_messages"] = llm_messages
            state["stream_system_prompt"] = DOCUMENT_SYSTEM_PROMPT
            state["final_response"] = ""
            state["next_action"] = "stream"

            return state

        def booking_agent(self, state: AgentState) -> AgentState:
            """Booking agent: chỉ chuẩn bị context, KHÔNG invoke LLM (trừ khi cần tạo booking)"""
            logger.info("🎫 [STREAM] Booking Agent working...")
            user_query = state["user_query"]
            messages_history = state["messages"]

            conversation_text = "\n".join([
                f"{msg.__class__.__name__}: {msg.content}"
                for msg in messages_history[-3:] if hasattr(msg, 'content')
            ])

            llm_messages = [
                SystemMessage(content=BOOKING_SYSTEM_PROMPT),
                HumanMessage(content=f"Lịch sử hội thoại:\n{conversation_text}\n\nTin nhắn mới: {user_query}\n\nPhân tích và xử lý.")
            ]

            state["stream_messages"] = llm_messages
            state["stream_system_prompt"] = BOOKING_SYSTEM_PROMPT
            state["final_response"] = ""
            state["next_action"] = "stream"

            return state

    agents = StreamingAgents(tools, openai_model=openai_model)

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

    return workflow.compile(), tools


# ============================================================================
# MAIN INTERFACE
# ============================================================================

class BaiChayRAGSystem:
    """Main interface for the RAG system with OpenAI GPT-4o"""

    def __init__(self, openai_model: str = "gpt-4o"):
        logger.info("🚀 Initializing Bãi Cháy RAG System with OpenAI GPT-4o...")
        logger.info(f"   Model: {openai_model}")
        logger.info("   ✅ Streaming support enabled")

        self.openai_model = openai_model

        # Workflow cho non-streaming (invoke như cũ)
        self.workflow = build_rag_workflow(openai_model=openai_model)

        # Workflow + LLM cho streaming
        self.streaming_workflow, self._tools = build_rag_workflow_for_streaming(openai_model=openai_model)

        # LLM riêng để stream
        api_key = os.getenv("OPENAI_API_KEY")
        self._streaming_llm = ChatOpenAI(
            model=openai_model,
            temperature=0.1,
            streaming=True,
            api_key=api_key
        )

        logger.info("✅ RAG System ready! (non-stream + stream)")

    def question(self, question: str, history: List = None) -> Dict:
        """Alias method cho /chat endpoint"""
        return self.process_query(user_query=question, conversation_history=history)

    def process_query(self, user_query: str, conversation_history: List = None) -> Dict:
        """
        Non-streaming: chạy workflow đầy đủ, trả về response string.
        Giữ nguyên behavior cũ.
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
            "messages": final_state.get("messages", [])
        }

    def stream_query(self, user_query: str, conversation_history: List = None) -> Generator[str, None, None]:
        """
        ⭐ SYNC STREAMING: Chạy workflow để search, sau đó stream LLM response real-time.

        Yield từng JSON string theo format:
            {"type": "start",  "content": null, "references": null, "status": "processing"}
            {"type": "chunk",  "content": "text...", "references": null, "status": null}
            {"type": "end",    "content": null, "references": null, "status": "done"}
            {"type": "error",  "content": "msg", "references": null, "status": "error"}

        Usage:
            for line in rag_system.stream_query("..."):
                print(line)  # mỗi line là 1 JSON string (chưa wrap SSE)
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

        # Bước 1: Start chunk
        yield json.dumps({"type": "start", "content": None, "references": None, "status": "processing"})

        # Bước 2: Chạy workflow (router + search)
        logger.info(f"🔄 [STREAM] Running RAG workflow for: {user_query}")
        final_state = self.streaming_workflow.invoke(initial_state)

        stream_messages = final_state.get("stream_messages")
        query_type = final_state.get("query_type", "tourism")

        if not stream_messages:
            logger.error("❌ [STREAM] No stream_messages found in state")
            yield json.dumps({"type": "error", "content": "Xin lỗi, có lỗi xảy ra.", "references": None, "status": "error"})
            return

        logger.info(f"✅ [STREAM] Workflow done (type={query_type}), starting LLM stream...")

        # Bước 3: Stream LLM realtime - mỗi chunk là token thật từ OpenAI
        chunk_count = 0
        try:
            for chunk in self._streaming_llm.stream(stream_messages):
                if chunk.content:
                    chunk_count += 1
                    yield json.dumps({"type": "chunk", "content": chunk.content, "references": None, "status": None})
        except Exception as e:
            logger.error(f"❌ [STREAM] LLM streaming error: {e}")
            yield json.dumps({"type": "error", "content": str(e), "references": None, "status": "error"})
            return

        logger.info(f"✅ [STREAM] Done, streamed {chunk_count} chunks")

        # Bước 4: End chunk
        yield json.dumps({"type": "end", "content": None, "references": None, "status": "done"})

    async def astream_query(
        self,
        user_query: str,
        conversation_history: List = None
    ) -> AsyncGenerator[str, None]:
        """
        ⭐ ASYNC STREAMING: Chạy workflow để search, sau đó async stream LLM response.

        Yield từng JSON string theo format:
            {"type": "start",  "content": null, "references": null, "status": "processing"}
            {"type": "chunk",  "content": "text...", "references": null, "status": null}
            {"type": "end",    "content": null, "references": null, "status": "done"}
            {"type": "error",  "content": "msg", "references": null, "status": "error"}

        Dùng cho FastAPI StreamingResponse với asyncio.

        Usage:
            async for json_str in rag_system.astream_query("..."):
                # json_str là JSON string, caller wrap thành SSE: f"data: {json_str}\\n\\n"
        """
        import asyncio

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
        yield json.dumps({"type": "start", "content": None, "references": None, "status": "processing"})

        # Bước 2: Chạy workflow trong thread pool (LangGraph invoke là sync)
        loop = asyncio.get_event_loop()
        final_state = await loop.run_in_executor(
            None,
            lambda: self.streaming_workflow.invoke(initial_state)
        )

        stream_messages = final_state.get("stream_messages")
        query_type = final_state.get("query_type", "tourism")

        if not stream_messages:
            logger.error("❌ [ASTREAM] No stream_messages found in state")
            yield json.dumps({"type": "error", "content": "Xin lỗi, có lỗi xảy ra.", "references": None, "status": "error"})
            return

        logger.info(f"✅ [ASTREAM] Workflow done (type={query_type}), starting async LLM stream...")

        # Bước 3: Async stream LLM realtime - mỗi chunk là token thật từ OpenAI
        chunk_count = 0
        try:
            async for chunk in self._streaming_llm.astream(stream_messages):
                if chunk.content:
                    chunk_count += 1
                    yield json.dumps({"type": "chunk", "content": chunk.content, "references": None, "status": None})
        except Exception as e:
            logger.error(f"❌ [ASTREAM] LLM streaming error: {e}")
            yield json.dumps({"type": "error", "content": str(e), "references": None, "status": "error"})
            return

        logger.info(f"✅ [ASTREAM] Done, streamed {chunk_count} chunks")

        # Bước 4: End chunk
        yield json.dumps({"type": "end", "content": None, "references": None, "status": "done"})


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    rag_system = BaiChayRAGSystem(openai_model="gpt-4o")

    # Test non-streaming
    print("=" * 80)
    print("TEST NON-STREAMING:")
    result = rag_system.process_query("Tìm khách sạn 4 sao gần biển Bãi Cháy")
    print(f"Query type: {result['query_type']}")
    print(f"Response: {result['response'][:300]}...")

    # Test sync streaming
    print("\n" + "=" * 80)
    print("TEST SYNC STREAMING:")
    for chunk in rag_system.stream_query("Gợi ý nhà hàng hải sản ngon"):
        print(chunk, end="", flush=True)
    print()