"""ADAPT-AI Agent implementations.

This module provides both:
1. Original agent implementations (base_agent, primary_agent, etc.)
2. Claude Agent SDK compatible implementations (sdk_agents)
"""

from src.agents.base_agent import BaseAgent, AgentResponse
from src.agents.primary_agent import PrimaryAgent
from src.agents.compliance_agent import ComplianceAgent
from src.agents.quality_agent import QualityAgent

# SDK-based agents
from src.agents.sdk_agents import (
    SDKAgent,
    ClinicalAgent,
    ComplianceValidatorAgent,
    QualityAssuranceAgent,
    AgentOrchestrator,
    create_agent,
    AGENT_SDK_AVAILABLE
)

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentResponse",
    # Original agents
    "PrimaryAgent",
    "ComplianceAgent",
    "QualityAgent",
    # SDK agents
    "SDKAgent",
    "ClinicalAgent",
    "ComplianceValidatorAgent",
    "QualityAssuranceAgent",
    "AgentOrchestrator",
    "create_agent",
    "AGENT_SDK_AVAILABLE"
]
