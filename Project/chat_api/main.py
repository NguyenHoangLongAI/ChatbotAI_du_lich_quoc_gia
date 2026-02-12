"""
main.py — FastAPI Server cho Bãi Cháy RAG Multi-Agent System
Port: 8503  |  LLM: OpenAI GPT-4o
UPDATED: Hỗ trợ streaming LLM realtime

Streaming usage:
  POST /chat           body: {"question": "...", "stream": false}  → JSON response (cũ)
  POST /chat           body: {"question": "...", "stream": true}   → SSE stream (mới)
  GET  /chat/stream?question=...                                   → SSE stream (mới, GET)
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import json
import asyncio

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

# ── Import RAG system từ package mới ─────────────────────────────────────────
import sys
sys.path.append('/mnt/user-data/uploads')

from Project.rag_system.rag_system import BaiChayRAGSystem

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required — set it in your .env file")

# ── FastAPI app ───────────────────────────────────────────────────────────────
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
- Event type "start": bắt đầu xử lý
- Event type "chunk": text chunks từ LLM (realtime)
- Event type "end": kết thúc
- Event type "error": lỗi xảy ra
""",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = None
    session_id: Optional[str] = None
    stream: Optional[bool] = Field(
        default=True,
        description="Mặc định stream SSE. Set false nếu muốn JSON response."
    )


class QueryResponse(BaseModel):
    response: str
    query_type: str
    timestamp: str
    session_id: Optional[str] = None
    metadata: Optional[Dict] = None


# ── Global RAG system ─────────────────────────────────────────────────────────
rag_system: Optional[BaiChayRAGSystem] = None


@app.on_event("startup")
async def startup_event():
    global rag_system
    logger.info("🚀 Starting Bãi Cháy RAG API...")
    logger.info(f"   Model: {OPENAI_MODEL}")

    rag_system = BaiChayRAGSystem(
        openai_model=OPENAI_MODEL,
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
    )
    logger.info("✅ RAG System ready (non-stream + stream)")


# ── Helper functions ──────────────────────────────────────────────────────────

def build_langchain_history(history: Optional[List[ChatMessage]]):
    """Convert ChatMessage list thành LangChain messages"""
    lc_history = []
    if history:
        for msg in history:
            if msg.role == "user":
                lc_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_history.append(AIMessage(content=msg.content))
    return lc_history


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
        error_json = json.dumps({
            "type": "error",
            "content": "RAG system not initialized",
            "references": None,
            "status": "error"
        })
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
        error_json = json.dumps({
            "type": "error",
            "content": str(e),
            "references": None,
            "status": "error"
        })
        yield f"data: {error_json}\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

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
                "start": 'data: {"type": "start", "content": null, "references": null, "status": "processing"}',
                "chunk": 'data: {"type": "chunk", "content": "text...", "references": null, "status": null}',
                "end": 'data: {"type": "end", "content": null, "references": null, "status": "done"}'
            }
        },
        "endpoints": {
            "chat": "/chat (POST) - stream=false → JSON, stream=true → SSE",
            "chat_stream_get": "/chat/stream (GET) - SSE streaming",
            "health": "/api/v1/health (GET)",
            "stats": "/api/v1/stats (GET)",
            "examples": "/api/v1/examples (GET)"
        },
    }


@app.post("/chat", response_model=None)
async def chat(request: ChatRequest):
    """
    Chat endpoint hỗ trợ cả streaming và non-streaming.

    - `stream: false`: Trả về JSON QueryResponse như cũ.
    - `stream: true` (mặc định): Trả về SSE stream. Response là text/event-stream.
      Client đọc từng `data:` event cho đến khi gặp type="end" hoặc type="error".
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
        const data = JSON.parse(e.data);
        if (data.type === 'end') { evtSource.close(); return; }
        if (data.type === 'error') { console.error(data.content); return; }
        if (data.type === 'chunk') {
            responseDiv.textContent += data.content;
        }
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
    openai_ok = bool(OPENAI_API_KEY)
    rag_ok = rag_system is not None

    return {
        "status": "healthy" if (openai_ok and rag_ok) else "degraded",
        "service": "rag-multi-agent-api",
        "version": "3.1.0",
        "port": 8503,
        "llm": {
            "provider": "OpenAI",
            "model": OPENAI_MODEL,
            "api_key_configured": openai_ok,
            "streaming": "enabled"
        },
        "components": {
            "rag_system": "ready" if rag_ok else "not_ready",
            "openai": "configured" if openai_ok else "not_configured",
            "agents": ["router", "tourism_advisor", "document_advisor", "booking_agent"],
        },
    }


@app.get("/api/v1/stats")
async def get_stats():
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        from Project.crawl_baichay_service.tourism_dao import BaiChayTourismDAO
        from Project.baichay_db.customer_dao import CustomerDAO

        tourism_stats = BaiChayTourismDAO(
            host=MILVUS_HOST, port=MILVUS_PORT
        ).get_statistics()
        customer_stats = CustomerDAO(
            host=MILVUS_HOST, port=MILVUS_PORT
        ).get_statistics()

        return {
            "status": "success",
            "llm_provider": "OpenAI",
            "model": OPENAI_MODEL,
            "streaming_support": True,
            "statistics": {
                "tourism_services": {
                    "total_count": tourism_stats["collection"]["total_count"],
                    "collection": tourism_stats["collection"]["name"],
                },
                "documents": {"collection": "document_tour", "status": "active"},
                "customers": {
                    "total_count": customer_stats["total_customers"],
                    "collection": customer_stats["collection_name"],
                },
            },
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
                '{"type": "start", "content": null, "references": null, "status": "processing"}',
                '{"type": "chunk", "content": "**Khách sạn Mường Thanh**", "references": null, "status": null}',
                '{"type": "chunk", "content": " là một trong những...", "references": null, "status": null}',
                '{"type": "end", "content": null, "references": null, "status": "done"}'
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
            if (parsed.type === 'end')   { return; }
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


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starting Bãi Cháy Tourism RAG API v3.1.0")
    logger.info(f"LLM: OpenAI {OPENAI_MODEL}")
    logger.info("Streaming: ENABLED")
    logger.info("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8503, log_level="info")