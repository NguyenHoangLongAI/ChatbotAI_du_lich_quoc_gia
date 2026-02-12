"""
FastAPI Server for Bãi Cháy RAG Multi-Agent System
Port: 8503
Using OpenAI GPT-4o
UPDATED: Hỗ trợ streaming LLM realtime

Streaming usage:
  POST /chat           body: {"question": "...", "stream": false}  → JSON response (cũ)
  POST /chat           body: {"question": "...", "stream": true}   → SSE stream (mới)
  GET  /chat/stream?question=...                                   → SSE stream (mới, GET)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
import uvicorn
import logging
from datetime import datetime
import os
import json
from dotenv import load_dotenv
import asyncio
from rag_multi_agent_system import BaiChayRAGSystem

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bãi Cháy Tourism RAG API (OpenAI GPT-4o)",
    version="3.1.0",
    description="""
Multi-Agent RAG System using OpenAI GPT-4o with real-time streaming support.

## Streaming

Thêm `"stream": true` vào body để nhận response dạng SSE stream:

```
POST /chat
{"question": "Tìm khách sạn gần biển", "stream": true}
```

Response là Server-Sent Events stream:
- Dòng đầu: `[META]{...}` chứa query_type
- Các dòng tiếp theo: text chunks từ LLM (realtime)
"""
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.error("❌ OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY is required")

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(default=None)



class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = Field(default=None)
    session_id: Optional[str] = None
    stream: Optional[bool] = Field(
        default=True,
        description="Mặc định stream SSE. Set false nếu muốn JSON response."
    )

class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question", min_length=1)
    conversation_history: Optional[List[ChatMessage]] = Field(default=None)
    session_id: Optional[str] = Field(default=None)


class QueryResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    query_type: str = Field(..., description="tourism | document | booking")
    timestamp: str = Field(..., description="Response timestamp")
    session_id: Optional[str] = Field(default=None)
    metadata: Optional[Dict] = Field(default=None)


# ============================================================================
# GLOBAL RAG SYSTEM
# ============================================================================

rag_system: Optional[BaiChayRAGSystem] = None


@app.on_event("startup")
async def startup_event():
    global rag_system
    try:
        logger.info("🚀 Starting Bãi Cháy RAG API with OpenAI GPT-4o...")
        logger.info(f"   Model: {OPENAI_MODEL}")

        rag_system = BaiChayRAGSystem(openai_model=OPENAI_MODEL)

        logger.info("✅ RAG System initialized (non-stream + stream ready)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise


# ============================================================================
# HELPER: Build history từ ChatMessage list
# ============================================================================

def build_langchain_history(history: Optional[List[ChatMessage]]):
    """Convert ChatMessage list thành LangChain messages"""
    from langchain_core.messages import HumanMessage, AIMessage

    lc_history = []
    if history:
        for msg in history:
            if msg.role == "user":
                lc_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_history.append(AIMessage(content=msg.content))
    return lc_history


# ============================================================================
# HELPER: SSE Generator
# ============================================================================

async def sse_generator(user_question: str, history, session_id: Optional[str] = None):
    """
    Async generator cho SSE streaming response.

    Format mỗi SSE event (data: ...):
        {"type": "start",  "content": null, "references": null, "status": "processing"}
        {"type": "chunk",  "content": "text token", "references": null, "status": null}
        {"type": "end",    "content": null, "references": null, "status": "done"}
        {"type": "error",  "content": "msg", "references": null, "status": "error"}

    Client: đọc đến type=="end" hoặc type=="error" thì dừng.
    """
    if not rag_system:
        error_json = json.dumps({"type": "error", "content": "RAG system not initialized", "references": None, "status": "error"})
        yield f"data: {error_json}\n\n"
        return

    try:
        async for json_str in rag_system.astream_query(
            user_query=user_question,
            conversation_history=history
        ):
            # json_str là JSON string từ astream_query, wrap thành SSE event
            yield f"data: {json_str}\n\n"
            await asyncio.sleep(0.001)  # yield control cho event loop

    except Exception as e:
        logger.error(f"❌ SSE streaming error: {e}", exc_info=True)
        error_json = json.dumps({"type": "error", "content": str(e), "references": None, "status": "error"})
        yield f"data: {error_json}\n\n"


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "Bãi Cháy Tourism RAG API",
        "version": "3.1.0",
        "status": "running",
        "port": 8503,
        "llm": {
            "provider": "OpenAI",
            "model": OPENAI_MODEL,
            "streaming": "supported"
        },
        "streaming_usage": {
            "description": "Thêm 'stream: true' vào POST /chat để nhận SSE stream",
            "example_body": {"question": "Tìm khách sạn gần biển", "stream": True},
            "sse_format": {
                "meta": "data: [META]{\"query_type\": \"tourism\"}",
                "chunk": "data: text chunk from LLM",
                "done": "data: [DONE]"
            }
        },
        "endpoints": {
            "chat": "POST /chat (stream=false → JSON, stream=true → SSE)",
            "chat_stream_get": "GET /chat/stream?question=... (SSE)",
            "health": "GET /api/v1/health",
            "stats": "GET /api/v1/stats",
            "examples": "GET /api/v1/examples"
        }
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint hỗ trợ cả streaming và non-streaming.

    - `stream: false` (mặc định): Trả về JSON QueryResponse như cũ.
    - `stream: true`: Trả về SSE stream. Response là text/event-stream.
      Client đọc từng `data:` event cho đến khi gặp `data: [DONE]`.
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    history = build_langchain_history(request.history)

    # ==================== STREAMING PATH ====================
    if request.stream:
        logger.info(f"📨 [STREAM] Question: {request.question}")

        return StreamingResponse(
            sse_generator(
                user_question=request.question,
                history=history,
                session_id=request.session_id
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",        # Tắt nginx buffering
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )

    # ==================== NON-STREAMING PATH (cũ) ====================
    try:
        logger.info(f"📨 [NON-STREAM] Question: {request.question}")

        result = rag_system.question(
            question=request.question,
            history=history
        )

        return QueryResponse(
            response=result["response"],
            query_type=result.get("query_type", "unknown"),
            timestamp=datetime.now().isoformat(),
            session_id=request.session_id,
            metadata={
                "model": OPENAI_MODEL,
                "stream": False,
                "message_count": len(result.get("messages", []))
            }
        )

    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/stream")
async def chat_stream_get(
    question: str = Query(..., min_length=1, description="Câu hỏi của người dùng"),
    session_id: Optional[str] = Query(default=None)
):
    """
    GET endpoint cho SSE streaming (tiện cho browser/EventSource).

    Ví dụ dùng với JavaScript EventSource:
    ```javascript
    const evtSource = new EventSource('/chat/stream?question=Tìm+khách+sạn');
    evtSource.onmessage = (e) => {
        if (e.data === '[DONE]') { evtSource.close(); return; }
        if (e.data.startsWith('[META]')) { /* parse metadata */ return; }
        if (e.data.startsWith('[ERROR]')) { /* handle error */ return; }
        responseDiv.textContent += e.data.replace(/\\\\n/g, '\\n');
    };
    ```
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    logger.info(f"📨 [STREAM GET] Question: {question}")

    return StreamingResponse(
        sse_generator(
            user_question=question,
            history=[],
            session_id=session_id
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.get("/api/v1/health")
async def health_check():
    try:
        rag_status = rag_system is not None
        openai_status = bool(OPENAI_API_KEY)

        milvus_status = False
        try:
            if rag_system and rag_system.workflow:
                milvus_status = True
        except:
            pass

        return {
            "status": "healthy" if (rag_status and openai_status and milvus_status) else "degraded",
            "service": "rag-multi-agent-api",
            "version": "3.1.0",
            "port": 8503,
            "llm": {
                "provider": "OpenAI",
                "model": OPENAI_MODEL,
                "api_key_configured": openai_status,
                "streaming": "enabled"
            },
            "components": {
                "rag_system": "ready" if rag_status else "not_ready",
                "openai": "configured" if openai_status else "not_configured",
                "milvus": "connected" if milvus_status else "disconnected",
                "agents": {
                    "router": "active",
                    "tourism_advisor": "active",
                    "document_advisor": "active",
                    "booking_agent": "active"
                }
            },
            "databases": {
                "tourism_data": "bai_chay_data",
                "documents": "document_tour",
                "customers": "customers"
            }
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "rag-multi-agent-api",
            "error": str(e)
        }


@app.get("/api/v1/stats")
async def get_stats():
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG system not initialized")

        from Project.crawl_baichay_service.tourism_dao import BaiChayTourismDAO
        from Project.document_db.tourism_document_dao import TourismDocumentDAO
        from Project.baichay_db.customer_dao import CustomerDAO

        tourism_dao = BaiChayTourismDAO()
        customer_dao = CustomerDAO()

        tourism_stats = tourism_dao.get_statistics()
        customer_stats = customer_dao.get_statistics()

        return {
            "status": "success",
            "llm_provider": "OpenAI",
            "model": OPENAI_MODEL,
            "streaming_support": True,
            "statistics": {
                "tourism_services": {
                    "total_count": tourism_stats["collection"]["total_count"],
                    "collection": tourism_stats["collection"]["name"]
                },
                "documents": {
                    "collection": "document_tour",
                    "status": "active"
                },
                "customers": {
                    "total_count": customer_stats["total_customers"],
                    "collection": customer_stats["collection_name"]
                }
            },
            "workflow": {
                "nodes": [
                    "router",
                    "tourism_advisor",
                    "document_advisor",
                    "booking_agent"
                ],
                "tools": [
                    "search_tourism_services",
                    "search_documents",
                    "get_service_by_id",
                    "create_customer_booking"
                ]
            }
        }

    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/examples")
async def get_examples():
    return {
        "non_streaming_example": {
            "method": "POST",
            "url": "/chat",
            "body": {
                "question": "Tìm khách sạn 4 sao gần biển Bãi Cháy",
                "stream": False
            },
            "response_type": "application/json"
        },
        "streaming_example": {
            "method": "POST",
            "url": "/chat",
            "body": {
                "question": "Tìm khách sạn 4 sao gần biển Bãi Cháy",
                "stream": True
            },
            "response_type": "text/event-stream",
            "sse_events": [
                "data: {\"type\": \"start\", \"content\": null, \"references\": null, \"status\": \"processing\"}",
                "data: {\"type\": \"chunk\", \"content\": \"**Khách sạn Mường Thanh**\", \"references\": null, \"status\": null}",
                "data: {\"type\": \"chunk\", \"content\": \" là một trong những...\", \"references\": null, \"status\": null}",
                "data: {\"type\": \"end\", \"content\": null, \"references\": null, \"status\": \"done\"}"
            ]
        },
        "streaming_get_example": {
            "method": "GET",
            "url": "/chat/stream?question=T%C3%ACm+kh%C3%A1ch+s%E1%BA%A1n+g%E1%BA%A7n+bi%E1%BB%83n",
            "response_type": "text/event-stream"
        },
        "javascript_client_example": """
// JavaScript SSE client example
async function streamChat(question) {
    const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, stream: true })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\\n');
        buffer = lines.pop(); // giữ lại dòng chưa hoàn chỉnh

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6); // bỏ 'data: '

            const parsed = JSON.parse(data);
            if (parsed.type === 'start') { /* show loading */ continue; }
            if (parsed.type === 'end')   { evtSource && evtSource.close(); return; }
            if (parsed.type === 'error') { console.error(parsed.content); return; }
            if (parsed.type === 'chunk') {
                document.getElementById('response').textContent += parsed.content;
            }
        }
    }
}
""",
        "tourism_queries": [
            "Tìm khách sạn 4 sao gần biển Bãi Cháy",
            "Gợi ý tour du lịch Hạ Long 2 ngày 1 đêm",
            "Nhà hàng hải sản ngon ở Bãi Cháy"
        ],
        "document_queries": [
            "Quy định hủy tour như thế nào?",
            "Chính sách hoàn tiền khi hủy đặt phòng"
        ],
        "booking_queries": [
            "Tôi muốn đặt khách sạn Mường Thanh, tên Nguyễn Văn A, SĐT 0901234567, từ 15/03 đến 17/03"
        ]
    }


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starting Bãi Cháy Tourism RAG API v3.1.0")
    logger.info(f"LLM: OpenAI {OPENAI_MODEL}")
    logger.info("Streaming: ENABLED")
    logger.info("=" * 70)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8503,
        log_level="info"
    )