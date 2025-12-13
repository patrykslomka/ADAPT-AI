"""SDK-based MCP Orchestrator using official Claude Agent SDK and MCP patterns.

This orchestrator provides:
1. Claude Agent SDK integration for agentic workflows
2. MCP server connection for clinical domain tools
3. Multi-agent coordination with proper tool use
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

# Try to import Claude Agent SDK
try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    AGENT_SDK_AVAILABLE = True
except ImportError:
    AGENT_SDK_AVAILABLE = False

# Try to import MCP SDK
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_SDK_AVAILABLE = True
except ImportError:
    MCP_SDK_AVAILABLE = False

from anthropic import Anthropic
from config.settings import settings
from src.agents.base_agent import AgentResponse
from src.domain.patient_handler import PatientHandler
from src.building_blocks.rag import RAGBlock
from src.building_blocks.rat import RATBlock
from src.llmops.metrics_collector import metrics_collector, QueryMetrics
from src.llmops.tracer import tracer, TraceContext, SpanContext

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent execution."""
    agent_name: str
    status: str
    content: str
    confidence: float
    issues: List[Dict]
    tokens_used: int
    execution_time: float


class SDKMCPOrchestrator:
    """Orchestrator using Claude Agent SDK and MCP protocol.

    Architecture:
    1. Clinical Agent (primary) - Uses RAG/RAT + MCP tools for diagnosis
    2. Compliance Agent - Validates HIPAA/FDA compliance
    3. Quality Agent - Detects hallucinations and validates accuracy

    All agents can connect to the MCP clinical server for domain tools.
    """

    # System prompts for each agent role
    CLINICAL_SYSTEM_PROMPT = """You are an expert clinical diagnostic assistant.

Your role:
1. Analyze patient presentations and medical history
2. Generate evidence-based differential diagnoses
3. Recommend appropriate diagnostic workups
4. Consider drug interactions and contraindications

Use the available clinical tools to:
- Retrieve patient data (get_patient)
- Check drug interactions (check_drug_interactions)
- Search diseases by symptoms (search_diseases_by_symptoms)
- Validate drug safety (check_contraindications)

Always recommend that healthcare providers verify your recommendations."""

    COMPLIANCE_SYSTEM_PROMPT = """You are a healthcare compliance validator.

Check responses for:
1. HIPAA violations (patient names, DOB, MRN, SSN in output)
2. Drug contraindications with patient allergies
3. FDA guideline adherence
4. Required medical disclaimers

Use validate_hipaa_compliance tool when checking text.
Output findings as JSON with issues array."""

    QUALITY_SYSTEM_PROMPT = """You are a medical accuracy validator.

Check responses for:
1. Hallucinated drug names or treatments
2. Incorrect dosages or protocols
3. Unsupported clinical claims
4. Factual accuracy against medical knowledge

Use get_disease_info and get_treatment_info tools to verify claims.
Output: {"issues": [...], "confidence_score": 0.0-1.0, "assessment": "..."}"""

    def __init__(self):
        """Initialize the SDK-based orchestrator."""
        self.patient_handler = PatientHandler()
        self.rag = RAGBlock()
        self.rat = RATBlock(rag_block=self.rag)

        # Anthropic client for direct API calls (fallback)
        self._client = None

        # MCP server configuration
        self.mcp_server_config = {
            "clinical": {
                "command": "python",
                "args": ["-m", "src.mcp.servers.clinical_server"]
            }
        }

        logger.info(f"SDK Orchestrator initialized (SDK: {AGENT_SDK_AVAILABLE}, MCP: {MCP_SDK_AVAILABLE})")

    @property
    def client(self) -> Anthropic:
        """Lazy load Anthropic client."""
        if self._client is None:
            self._client = Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
        return self._client

    async def process_query(
        self,
        query: str,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process query through multi-agent SDK workflow.

        Args:
            query: Clinical query
            patient_id: Optional patient ID for context
            session_id: Optional session tracking ID

        Returns:
            Complete response with agent outputs and metrics
        """
        query_id = str(uuid.uuid4())
        start_time = time.time()

        # Start trace
        with TraceContext(tracer, query_id, "process_query", {"query": query[:100]}) as root_span:
            try:
                # 1. Load patient context
                patient_context = None
                if patient_id:
                    patient_context = self.patient_handler.get_patient(patient_id)
                    tracer.add_event(root_span, "patient_loaded", {"patient_id": patient_id})

                # 2. Get RAG/RAT context
                with SpanContext(tracer, query_id, "retrieval", root_span.span_id) as rag_span:
                    if self._should_use_rat(query):
                        rat_result = self.rat.process_query(query, max_steps=4)
                        retrieved_context = rat_result['context']
                        reasoning_path = rat_result.get('reasoning_path', [])
                        tracer.set_attribute(rag_span, "method", "RAT")
                    else:
                        rag_result = self.rag.process_query(query)
                        retrieved_context = rag_result['context']
                        reasoning_path = None
                        tracer.set_attribute(rag_span, "method", "RAG")

                # 3. Run Clinical Agent
                with SpanContext(tracer, query_id, "clinical_agent", root_span.span_id) as clinical_span:
                    clinical_result = await self._run_clinical_agent(
                        query, patient_context, retrieved_context
                    )
                    tracer.set_attribute(clinical_span, "status", clinical_result.status)

                # 4. Run Compliance Agent
                with SpanContext(tracer, query_id, "compliance_agent", root_span.span_id) as compliance_span:
                    compliance_result = await self._run_compliance_agent(
                        clinical_result.content, patient_context
                    )
                    tracer.set_attribute(compliance_span, "status", compliance_result.status)

                # Handle compliance rejection
                if compliance_result.status == 'rejected':
                    return self._build_rejection_response(
                        query_id, "compliance_failure", compliance_result, start_time
                    )

                # 5. Run Quality Agent
                with SpanContext(tracer, query_id, "quality_agent", root_span.span_id) as quality_span:
                    quality_result = await self._run_quality_agent(
                        clinical_result.content, retrieved_context
                    )
                    tracer.set_attribute(quality_span, "status", quality_result.status)

                # Handle quality rejection with revision
                if quality_result.status == 'rejected':
                    # Revision attempt
                    clinical_result = await self._run_clinical_agent(
                        query, patient_context, retrieved_context,
                        quality_feedback=quality_result.issues
                    )
                    quality_result = await self._run_quality_agent(
                        clinical_result.content, retrieved_context
                    )

                # 6. Build final response
                total_time = time.time() - start_time

                final_content = self._aggregate_response(
                    clinical_result, compliance_result, quality_result
                )

                # Record metrics
                await self._record_metrics(
                    query_id, query, patient_id,
                    clinical_result, compliance_result, quality_result,
                    total_time, reasoning_path
                )

                return {
                    'query_id': query_id,
                    'status': 'success',
                    'content': final_content,
                    'agents': {
                        'clinical': {
                            'status': clinical_result.status,
                            'confidence': clinical_result.confidence,
                            'tokens': clinical_result.tokens_used
                        },
                        'compliance': {
                            'status': compliance_result.status,
                            'issues': len(compliance_result.issues)
                        },
                        'quality': {
                            'status': quality_result.status,
                            'confidence': quality_result.confidence,
                            'hallucination_risk': 1.0 - quality_result.confidence
                        }
                    },
                    'metadata': {
                        'query_id': query_id,
                        'patient_id': patient_id,
                        'session_id': session_id,
                        'response_time': round(total_time, 2),
                        'reasoning_path': reasoning_path,
                        'sdk_used': AGENT_SDK_AVAILABLE,
                        'mcp_available': MCP_SDK_AVAILABLE
                    }
                }

            except Exception as e:
                logger.error(f"Orchestration error: {e}", exc_info=True)
                tracer.add_event(root_span, "error", {"error": str(e)})
                return {
                    'query_id': query_id,
                    'status': 'error',
                    'error': str(e),
                    'response_time': time.time() - start_time
                }

    async def _run_clinical_agent(
        self,
        query: str,
        patient_context: Optional[Dict],
        retrieved_context: str,
        quality_feedback: Optional[List[Dict]] = None
    ) -> AgentResult:
        """Run clinical agent using SDK or fallback."""
        start_time = time.time()

        # Build prompt
        prompt_parts = [query]

        if patient_context:
            prompt_parts.insert(0, f"**Patient Context:**\n{json.dumps(patient_context, indent=2)}")

        if retrieved_context:
            prompt_parts.insert(0, f"**Medical Knowledge:**\n{retrieved_context}")

        if quality_feedback:
            feedback_text = "\n".join([f"- {i.get('description', '')}" for i in quality_feedback])
            prompt_parts.append(f"\n**Previous issues to address:**\n{feedback_text}")

        full_prompt = "\n\n".join(prompt_parts)

        if AGENT_SDK_AVAILABLE:
            return await self._run_agent_with_sdk(
                "clinical", self.CLINICAL_SYSTEM_PROMPT, full_prompt, start_time
            )
        else:
            return await self._run_agent_with_api(
                "clinical", self.CLINICAL_SYSTEM_PROMPT, full_prompt, start_time
            )

    async def _run_compliance_agent(
        self,
        content_to_validate: str,
        patient_context: Optional[Dict]
    ) -> AgentResult:
        """Run compliance validation agent."""
        start_time = time.time()

        prompt = f"""Validate this clinical response for compliance:

**Response to validate:**
{content_to_validate}

**Patient context:**
{json.dumps(patient_context, indent=2) if patient_context else 'None provided'}

Check for HIPAA violations, drug contraindications, and required disclaimers.
Output your findings as JSON."""

        if AGENT_SDK_AVAILABLE:
            return await self._run_agent_with_sdk(
                "compliance", self.COMPLIANCE_SYSTEM_PROMPT, prompt, start_time
            )
        else:
            return await self._run_agent_with_api(
                "compliance", self.COMPLIANCE_SYSTEM_PROMPT, prompt, start_time,
                parse_json=True
            )

    async def _run_quality_agent(
        self,
        content_to_validate: str,
        reference_context: str
    ) -> AgentResult:
        """Run quality assurance agent."""
        start_time = time.time()

        prompt = f"""Verify accuracy of this clinical response:

**Response to verify:**
{content_to_validate}

**Reference medical knowledge:**
{reference_context[:2000] if reference_context else 'None available'}

Check for hallucinations, inaccuracies, and unsupported claims.
Output: {{"issues": [...], "confidence_score": 0.0-1.0, "assessment": "..."}}"""

        if AGENT_SDK_AVAILABLE:
            return await self._run_agent_with_sdk(
                "quality", self.QUALITY_SYSTEM_PROMPT, prompt, start_time
            )
        else:
            return await self._run_agent_with_api(
                "quality", self.QUALITY_SYSTEM_PROMPT, prompt, start_time,
                parse_json=True
            )

    async def _run_agent_with_sdk(
        self,
        agent_name: str,
        system_prompt: str,
        prompt: str,
        start_time: float
    ) -> AgentResult:
        """Run agent using Claude Agent SDK."""
        try:
            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                mcp_servers=self.mcp_server_config,
                allowed_tools=["get_patient", "check_drug_interactions",
                              "search_diseases_by_symptoms", "validate_hipaa_compliance",
                              "get_disease_info", "get_treatment_info"]
            )

            content = ""
            tokens = 0

            async for message in query(prompt=prompt, options=options):
                if message.get("type") == "assistant_message":
                    content = message.get("content", "")
                    usage = message.get("usage", {})
                    tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            return self._parse_agent_result(agent_name, content, tokens, start_time)

        except Exception as e:
            logger.error(f"SDK agent error: {e}")
            # Fallback to API
            return await self._run_agent_with_api(
                agent_name, system_prompt, prompt, start_time
            )

    async def _run_agent_with_api(
        self,
        agent_name: str,
        system_prompt: str,
        prompt: str,
        start_time: float,
        parse_json: bool = False
    ) -> AgentResult:
        """Run agent using direct Anthropic API."""
        try:
            response = self.client.messages.create(
                model=settings.model_name,
                max_tokens=settings.max_tokens,
                temperature=0.7 if agent_name == "clinical" else 0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens

            return self._parse_agent_result(
                agent_name, content, tokens, start_time, parse_json
            )

        except Exception as e:
            logger.error(f"API agent error: {e}")
            return AgentResult(
                agent_name=agent_name,
                status="error",
                content=str(e),
                confidence=0.0,
                issues=[{"type": "error", "description": str(e)}],
                tokens_used=0,
                execution_time=time.time() - start_time
            )

    def _parse_agent_result(
        self,
        agent_name: str,
        content: str,
        tokens: int,
        start_time: float,
        parse_json: bool = False
    ) -> AgentResult:
        """Parse agent output into structured result."""
        issues = []
        confidence = 0.85
        status = "approved"

        if parse_json or agent_name in ["compliance", "quality"]:
            try:
                # Try to extract JSON from content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    issues = data.get("issues", [])
                    confidence = data.get("confidence_score", 0.85)
            except (json.JSONDecodeError, AttributeError):
                pass

        # Determine status
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        if critical_issues:
            status = "rejected"
        elif issues:
            status = "warning"

        return AgentResult(
            agent_name=agent_name,
            status=status,
            content=content,
            confidence=confidence,
            issues=issues,
            tokens_used=tokens,
            execution_time=time.time() - start_time
        )

    def _should_use_rat(self, query: str) -> bool:
        """Decide if RAT multi-step reasoning is needed."""
        rat_triggers = [
            'diagnostic workup', 'differential diagnosis', 'what tests',
            'next steps', 'recommend', 'patient presents',
            'suggest', 'workup', 'evaluate', 'treatment plan'
        ]
        query_lower = query.lower()
        return any(trigger in query_lower for trigger in rat_triggers)

    def _aggregate_response(
        self,
        clinical: AgentResult,
        compliance: AgentResult,
        quality: AgentResult
    ) -> str:
        """Aggregate agent responses into final output."""
        parts = [clinical.content]

        # Add compliance warnings
        if compliance.status == 'warning' and compliance.issues:
            warnings = "\n".join([f"- {i.get('description', '')}" for i in compliance.issues])
            parts.append(f"\n**Compliance Considerations:**\n{warnings}")

        # Add quality warnings
        if quality.status == 'warning':
            parts.append(
                f"\n**Note:** Response flagged for review (confidence: {quality.confidence:.0%})"
            )

        # Medical disclaimer
        parts.append(
            "\n---\n"
            "*This is AI-generated clinical decision support. "
            "Healthcare providers must verify all recommendations.*"
        )

        return "\n".join(parts)

    def _build_rejection_response(
        self,
        query_id: str,
        reason: str,
        result: AgentResult,
        start_time: float
    ) -> Dict[str, Any]:
        """Build rejection response."""
        return {
            'query_id': query_id,
            'status': 'rejected',
            'reason': reason,
            'issues': result.issues,
            'response_time': time.time() - start_time
        }

    async def _record_metrics(
        self,
        query_id: str,
        query: str,
        patient_id: Optional[str],
        clinical: AgentResult,
        compliance: AgentResult,
        quality: AgentResult,
        total_time: float,
        reasoning_path: Optional[List]
    ):
        """Record metrics for the query."""
        total_tokens = clinical.tokens_used + quality.tokens_used

        # Estimate cost (Haiku pricing)
        cost = total_tokens * 0.8 / 1_000_000 + total_tokens * 0.2 * 4.0 / 1_000_000

        metrics = QueryMetrics(
            query_id=query_id,
            timestamp=datetime.now().isoformat(),
            query_text=query[:200],
            patient_id=patient_id,
            total_response_time=total_time,
            mcp_time=0.1,
            primary_agent_time=clinical.execution_time,
            compliance_agent_time=compliance.execution_time,
            quality_agent_time=quality.execution_time,
            rag_time=None,
            rat_time=clinical.execution_time if reasoning_path else None,
            total_input_tokens=int(total_tokens * 0.8),
            total_output_tokens=int(total_tokens * 0.2),
            primary_agent_tokens=clinical.tokens_used,
            compliance_agent_tokens=0,
            quality_agent_tokens=quality.tokens_used,
            total_cost=cost,
            compliance_passed=compliance.status != 'rejected',
            quality_passed=quality.status != 'rejected',
            hallucination_score=1.0 - quality.confidence,
            confidence_score=quality.confidence,
            response_generated=True,
            error_occurred=False,
            error_message=None
        )

        metrics_collector.record_metrics(metrics)

    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'status': 'operational',
            'sdk_available': AGENT_SDK_AVAILABLE,
            'mcp_available': MCP_SDK_AVAILABLE,
            'agents': ['clinical', 'compliance', 'quality'],
            'mcp_servers': list(self.mcp_server_config.keys()),
            'metrics': metrics_collector.get_metrics_summary(hours=24)
        }


# Factory function for backward compatibility
def create_orchestrator(use_sdk: bool = True) -> SDKMCPOrchestrator:
    """Create an orchestrator instance.

    Args:
        use_sdk: Whether to prefer SDK-based orchestration

    Returns:
        Configured orchestrator
    """
    return SDKMCPOrchestrator()
