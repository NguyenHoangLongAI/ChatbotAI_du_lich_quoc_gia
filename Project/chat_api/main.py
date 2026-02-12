"""
main.py — FastAPI Server cho Bãi Cháy RAG Multi-Agent System
Port: 8503  |  LLM: OpenAI GPT-4o
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
import logging

# ── Import RAG system từ package mới ─────────────────────────────────────────
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
    title="Bãi Cháy Tourism RAG API",
    version="3.0.0",
    description="Multi-Agent RAG System using OpenAI GPT-4o",
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
    rag_system = BaiChayRAGSystem(
        openai_model=OPENAI_MODEL,
        milvus_host=MILVUS_HOST,
        milvus_port=MILVUS_PORT,
    )
    logger.info("✅ RAG System ready")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Bãi Cháy Tourism RAG API",
        "version": "3.0.0",
        "status": "running",
        "port": 8503,
        "llm": {"provider": "OpenAI", "model": OPENAI_MODEL},
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/api/v1/health (GET)",
            "stats": "/api/v1/stats (GET)",
            "examples": "/api/v1/examples (GET)",
        },
    }


@app.post("/chat", response_model=QueryResponse)
async def chat(request: ChatRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    logger.info(f"📨 Question: {request.question}")

    # Convert history to LangChain messages
    history = []
    if request.history:
        for msg in request.history:
            if msg.role == "user":
                history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                history.append(AIMessage(content=msg.content))

    try:
        result = rag_system.question(question=request.question, history=history)
    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        response=result["response"],
        query_type=result.get("query_type", "unknown"),
        timestamp=datetime.now().isoformat(),
        session_id=request.session_id,
        metadata={
            "model": OPENAI_MODEL,
            "message_count": len(result.get("messages", [])),
        },
    )


@app.get("/api/v1/health")
async def health_check():
    openai_ok = bool(OPENAI_API_KEY)
    rag_ok = rag_system is not None

    return {
        "status": "healthy" if (openai_ok and rag_ok) else "degraded",
        "service": "rag-multi-agent-api",
        "version": "3.0.0",
        "port": 8503,
        "llm": {
            "provider": "OpenAI",
            "model": OPENAI_MODEL,
            "api_key_configured": openai_ok,
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
        "tourism_queries": [
            "Tìm khách sạn 4 sao gần biển Bãi Cháy",
            "Gợi ý tour du lịch Hạ Long 2 ngày 1 đêm",
            "Nhà hàng hải sản ngon ở Bãi Cháy",
        ],
        "document_queries": [
            "Quy định hủy tour như thế nào?",
            "Chính sách hoàn tiền khi hủy đặt phòng",
        ],
        "booking_queries": [
            "Tôi muốn đặt khách sạn Mường Thanh, tên Nguyễn Văn A, SĐT 0901234567, từ 15/03 đến 17/03",
        ],
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8503, log_level="info")