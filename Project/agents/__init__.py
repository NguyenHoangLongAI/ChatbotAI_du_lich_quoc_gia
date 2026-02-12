"""
__init__.py — Export all agents for easy import
"""

from Project.agents.base_agent import BaseAgent
from Project.agents.router_agent import RouterAgent
from Project.agents.tourism_advisor_agent import TourismAdvisorAgent
from Project.agents.document_advisor_agent import DocumentAdvisorAgent
from Project.agents.booking_agent import BookingAgent

__all__ = [
    "BaseAgent",
    "RouterAgent",
    "TourismAdvisorAgent",
    "DocumentAdvisorAgent",
    "BookingAgent",
]