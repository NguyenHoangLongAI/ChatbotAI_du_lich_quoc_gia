"""
llm.py — Wrapper cho OpenAI Chat models (GPT-4o) tích hợp LangChain
"""

import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class OpenAILLMWrapper:
    """Wrapper đồng nhất cho OpenAI ChatOpenAI (GPT-4o)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        streaming: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.streaming = streaming

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        logger.info(f"🤖 Initializing OpenAI LLM: {model}")
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            api_key=api_key,
        )
        self.output_parser = StrOutputParser()
        logger.info(f"✅ OpenAI {model} initialized successfully")

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def invoke(self, messages: list, **kwargs) -> AIMessage:
        """Non-streaming invoke, trả về AIMessage."""
        try:
            response = self.llm.invoke(messages, **kwargs)
            return AIMessage(content=response.content)
        except Exception as e:
            logger.error(f"❌ OpenAI invoke error: {e}", exc_info=True)
            return AIMessage(content=f"Lỗi xử lý OpenAI: {str(e)}")

    def stream(self, messages: list, **kwargs):
        """Sync streaming — yield từng chunk nội dung."""
        try:
            for chunk in self.llm.stream(messages, **kwargs):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"❌ OpenAI streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi streaming: {str(e)}]"

    async def astream(self, messages: list, **kwargs):
        """Async streaming — yield từng chunk nội dung."""
        try:
            async for chunk in self.llm.astream(messages, **kwargs):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"❌ OpenAI async streaming error: {e}", exc_info=True)
            yield f"\n\n[Lỗi streaming: {str(e)}]"

    def bind_tools(self, tools: list) -> "OpenAILLMWrapper":
        """Bind tools vào LLM và trả về self để chain."""
        self.llm = self.llm.bind_tools(tools)
        return self