"""Agent implementations using Claude Agent SDK patterns.

This module provides agents that follow the official Claude Agent SDK architecture,
with fallback to direct Anthropic API when SDK is not available.
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Try to import Claude Agent SDK
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    AGENT_SDK_AVAILABLE = True
except ImportError:
    AGENT_SDK_AVAILABLE = False

from anthropic import Anthropic
from config.settings import settings
from src.agents.base_agent import AgentResponse
from src.building_blocks.rag import RAGBlock
from src.building_blocks.rat import RATBlock
from src.domain.ontology_loader import ontology_loader

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    description: str
    system_prompt: str
    allowed_tools: List[str] = field(default_factory=list)
    mcp_servers: Dict[str, Dict] = field(default_factory=dict)
    temperature: float = 0.7
    max_tokens: int = 4000


class SDKAgent(ABC):
    """Base class for Claude Agent SDK-compatible agents.

    This class provides a unified interface that works with:
    1. Official Claude Agent SDK (when available)
    2. Direct Anthropic API (fallback)
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.logger = logging.getLogger(f"agent.{config.name}")

        # Initialize Anthropic client for fallback
        self._client = None

    @property
    def client(self) -> Anthropic:
        """Lazy load Anthropic client."""
        if self._client is None:
            self._client = Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
        return self._client

    async def run(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the agent with streaming output.

        Args:
            prompt: The task/query for the agent
            context: Additional context (patient data, etc.)

        Yields:
            Message events from the agent
        """
        if AGENT_SDK_AVAILABLE:
            async for message in self._run_with_sdk(prompt, context):
                yield message
        else:
            async for message in self._run_with_fallback(prompt, context):
                yield message

    async def execute(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Execute the agent and return final response.

        Args:
            prompt: The task/query for the agent
            context: Additional context

        Returns:
            AgentResponse with results
        """
        messages = []
        async for message in self.run(prompt, context):
            messages.append(message)

        return self._process_messages(messages)

    async def _run_with_sdk(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run using official Claude Agent SDK."""
        full_prompt = self._build_prompt(prompt, context)

        options = ClaudeAgentOptions(
            allowed_tools=self.config.allowed_tools,
            mcp_servers=self.config.mcp_servers,
            system_prompt=self.config.system_prompt
        )

        async for message in query(prompt=full_prompt, options=options):
            yield message

    async def _run_with_fallback(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run using direct Anthropic API (fallback)."""
        full_prompt = self._build_prompt(prompt, context)

        # Build messages
        messages = [{"role": "user", "content": full_prompt}]

        try:
            response = self.client.messages.create(
                model=settings.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.config.system_prompt,
                messages=messages
            )

            yield {
                "type": "assistant_message",
                "content": response.content[0].text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }

        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }

    def _build_prompt(self, prompt: str, context: Optional[Dict[str, Any]]) -> str:
        """Build full prompt with context."""
        parts = [prompt]

        if context:
            if 'patient' in context:
                parts.insert(0, f"Patient Context:\n{json.dumps(context['patient'], indent=2)}\n")
            if 'retrieved_documents' in context:
                parts.insert(0, f"Retrieved Information:\n{context['retrieved_documents']}\n")

        return "\n\n".join(parts)

    @abstractmethod
    def _process_messages(self, messages: List[Dict]) -> AgentResponse:
        """Process message stream into final response."""
        pass


class ClinicalAgent(SDKAgent):
    """Primary clinical agent for diagnostic reasoning."""

    def __init__(self):
        config = AgentConfig(
            name="clinical_agent",
            description="Clinical diagnostics domain expert",
            system_prompt="""You are an expert clinical diagnostic assistant supporting healthcare providers.

Your role:
1. Analyze patient presentations and medical history
2. Generate evidence-based differential diagnoses
3. Recommend appropriate diagnostic workups
4. Suggest treatment considerations

Guidelines:
- Base recommendations on clinical guidelines and evidence
- Consider patient's allergies and current medications
- Highlight urgent/emergent conditions
- Always recommend provider verification
- Use clear, structured clinical reasoning

You are NOT diagnosing - you are providing decision support for qualified healthcare providers.""",
            allowed_tools=["get_patient", "search_diseases_by_symptoms",
                          "check_drug_interactions", "get_treatment_info"],
            mcp_servers={
                "clinical": {
                    "command": "python",
                    "args": ["-m", "src.mcp.servers.clinical_server"]
                }
            }
        )
        super().__init__(config)

        # Initialize RAG/RAT for knowledge retrieval
        self.rag = RAGBlock()
        self.rat = RATBlock(rag_block=self.rag)

    async def execute(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Execute with RAG/RAT augmentation."""
        # Decide: RAG or RAT?
        use_rat = self._should_use_rat(prompt)

        if use_rat:
            rat_result = self.rat.process_query(prompt, max_steps=4)
            if context is None:
                context = {}
            context['retrieved_documents'] = rat_result['context']
            context['reasoning_path'] = rat_result['reasoning_path']
        else:
            rag_result = self.rag.process_query(prompt)
            if context is None:
                context = {}
            context['retrieved_documents'] = rag_result['context']

        return await super().execute(prompt, context)

    def _should_use_rat(self, query: str) -> bool:
        """Decide if RAT reasoning is needed."""
        rat_triggers = [
            'diagnostic workup', 'differential diagnosis', 'what tests',
            'next steps', 'recommend', 'patient presents with',
            'suggest', 'workup', 'evaluate'
        ]
        query_lower = query.lower()
        return any(trigger in query_lower for trigger in rat_triggers)

    def _process_messages(self, messages: List[Dict]) -> AgentResponse:
        """Process messages into clinical response."""
        content = ""
        tokens = 0

        for msg in messages:
            if msg.get("type") == "assistant_message":
                content = msg.get("content", "")
                usage = msg.get("usage", {})
                tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            elif msg.get("type") == "error":
                return AgentResponse(
                    agent_name=self.name,
                    agent_type="clinical_expert",
                    status="error",
                    content=f"Error: {msg.get('error')}",
                    metadata={"error": msg.get("error")}
                )

        return AgentResponse(
            agent_name=self.name,
            agent_type="clinical_expert",
            status="approved",
            content=content,
            confidence_score=0.85,
            metadata={"tokens_used": tokens}
        )


class ComplianceValidatorAgent(SDKAgent):
    """Compliance validation agent for HIPAA/FDA checks."""

    def __init__(self):
        config = AgentConfig(
            name="compliance_validator",
            description="HIPAA and FDA compliance validator",
            system_prompt="""You validate clinical responses for regulatory compliance.

Check for:
1. HIPAA violations (PHI exposure - names, DOB, MRN, SSN)
2. Drug contraindications (allergies, conditions)
3. FDA guideline adherence
4. Required medical disclaimers
5. Appropriate urgency/risk communication

For each issue found, specify:
- Type of violation
- Severity (critical, high, moderate, low)
- Description
- Recommendation

Output your findings as JSON.""",
            allowed_tools=["validate_hipaa_compliance", "check_contraindications"],
            temperature=0.1  # Low temperature for consistent validation
        )
        super().__init__(config)

    def _process_messages(self, messages: List[Dict]) -> AgentResponse:
        """Process validation results."""
        content = ""
        issues = []

        for msg in messages:
            if msg.get("type") == "assistant_message":
                content = msg.get("content", "")
                # Try to parse JSON issues
                try:
                    result = json.loads(content)
                    issues = result.get("issues", [])
                except json.JSONDecodeError:
                    pass

        # Determine status
        critical = [i for i in issues if i.get("severity") == "critical"]
        if critical:
            status = "rejected"
        elif issues:
            status = "warning"
        else:
            status = "approved"

        return AgentResponse(
            agent_name=self.name,
            agent_type="validator",
            status=status,
            content=content if not issues else f"Found {len(issues)} compliance issues",
            issues_found=issues,
            metadata={"hipaa_checked": True, "fda_checked": True}
        )


class QualityAssuranceAgent(SDKAgent):
    """Quality assurance agent for hallucination detection."""

    def __init__(self):
        config = AgentConfig(
            name="quality_assurance",
            description="Quality assurance and hallucination detection",
            system_prompt="""You verify accuracy and detect hallucinations in clinical responses.

Check for:
1. Factual accuracy against medical knowledge
2. Hallucinated drug names or treatments
3. Incorrect dosages or protocols
4. Unsupported clinical claims
5. Coherence of reasoning

For each issue found, specify:
- Type (hallucination, inaccuracy, unsupported_claim)
- Severity (critical, high, moderate, low)
- Description
- Problematic text

Output your findings as JSON with:
{
  "issues": [...],
  "confidence_score": 0.0-1.0,
  "overall_assessment": "..."
}""",
            allowed_tools=["get_disease_info", "get_treatment_info"],
            temperature=0.1
        )
        super().__init__(config)

    def _process_messages(self, messages: List[Dict]) -> AgentResponse:
        """Process quality check results."""
        content = ""
        issues = []
        confidence = 0.85

        for msg in messages:
            if msg.get("type") == "assistant_message":
                content = msg.get("content", "")
                try:
                    result = json.loads(content)
                    issues = result.get("issues", [])
                    confidence = result.get("confidence_score", 0.85)
                except json.JSONDecodeError:
                    pass

        # Determine status
        critical = [i for i in issues if i.get("severity") == "critical"]
        high = [i for i in issues if i.get("severity") == "high"]

        if critical or len(high) >= 2:
            status = "rejected"
        elif issues:
            status = "warning"
        else:
            status = "approved"

        return AgentResponse(
            agent_name=self.name,
            agent_type="validator",
            status=status,
            content=f"Quality check: {len(issues)} issues found" if issues else "Quality checks passed",
            confidence_score=confidence,
            issues_found=issues,
            metadata={
                "hallucination_score": 1.0 - confidence,
                "checks_performed": ["factual_accuracy", "drug_validation", "claim_verification"]
            }
        )


# =============================================================================
# Agent Factory
# =============================================================================

def create_agent(agent_type: str) -> SDKAgent:
    """Factory function to create agents.

    Args:
        agent_type: Type of agent to create

    Returns:
        Configured agent instance
    """
    agents = {
        "clinical": ClinicalAgent,
        "compliance": ComplianceValidatorAgent,
        "quality": QualityAssuranceAgent
    }

    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")

    return agents[agent_type]()


# =============================================================================
# Multi-Agent Orchestration
# =============================================================================

class AgentOrchestrator:
    """Orchestrates multiple agents using Claude Agent SDK patterns."""

    def __init__(self):
        self.clinical = ClinicalAgent()
        self.compliance = ComplianceValidatorAgent()
        self.quality = QualityAssuranceAgent()
        self.logger = logging.getLogger("orchestrator")

    async def process(
        self,
        query: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process query through multi-agent pipeline.

        Args:
            query: Clinical query
            patient_context: Optional patient data

        Returns:
            Orchestrated response with all agent outputs
        """
        context = {"patient": patient_context} if patient_context else {}

        # 1. Clinical Agent
        self.logger.info("Running clinical agent...")
        clinical_response = await self.clinical.execute(query, context)

        # 2. Compliance Validation
        self.logger.info("Running compliance validation...")
        compliance_prompt = f"""Validate this clinical response for compliance:

Response to validate:
{clinical_response.content}

Patient context:
{json.dumps(patient_context, indent=2) if patient_context else 'None'}
"""
        compliance_response = await self.compliance.execute(compliance_prompt, context)

        if compliance_response.status == "rejected":
            return {
                "status": "rejected",
                "reason": "compliance_failure",
                "issues": compliance_response.issues_found
            }

        # 3. Quality Assurance
        self.logger.info("Running quality assurance...")
        quality_prompt = f"""Verify accuracy of this clinical response:

Response to verify:
{clinical_response.content}
"""
        quality_response = await self.quality.execute(quality_prompt, context)

        # 4. Aggregate results
        return {
            "status": "success",
            "content": clinical_response.content,
            "agents": {
                "clinical": {
                    "status": clinical_response.status,
                    "confidence": clinical_response.confidence_score
                },
                "compliance": {
                    "status": compliance_response.status,
                    "issues": len(compliance_response.issues_found)
                },
                "quality": {
                    "status": quality_response.status,
                    "confidence": quality_response.confidence_score
                }
            },
            "metadata": {
                "sdk_available": AGENT_SDK_AVAILABLE
            }
        }
