"""
base_agent.py — Base class cho tất cả các agent
"""

import logging
from abc import ABC, abstractmethod
from Project.state.state import AgentState
from Project.llm.llm import OpenAILLMWrapper
from Project.tools.tools import RAGTools

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class cho tất cả các agent trong hệ thống."""

    def __init__(self, tools: RAGTools, openai_model: str = "gpt-4o"):
        """
        Args:
            tools: RAGTools instance
            openai_model: OpenAI model name
        """
        self.tools = tools
        self.llm = OpenAILLMWrapper(model=openai_model, temperature=0.1)
        logger.info(f"✅ {self.__class__.__name__} initialized with {openai_model}")

    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        """
        Xử lý state và trả về state mới.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt cho agent."""
        pass