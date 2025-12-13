"""MCP Orchestrator - Central coordination hub."""
from typing import Dict, List, Optional, Any
from src.agents.primary_agent import PrimaryAgent
from src.agents.compliance_agent import ComplianceAgent
from src.agents.quality_agent import QualityAgent
from src.agents.base_agent import AgentResponse
from src.domain.patient_handler import PatientHandler
from src.llmops.metrics_collector import metrics_collector, QueryMetrics
from datetime import datetime
import uuid
import time
import logging

logger = logging.getLogger(__name__)


class MCPOrchestrator:
    """Model Context Protocol orchestrator for multi-agent coordination."""

    def __init__(self):
        # Initialize agents
        self.primary_agent = PrimaryAgent()
        self.compliance_agent = ComplianceAgent()
        self.quality_agent = QualityAgent()

        self.patient_handler = PatientHandler()

        logger.info("MCP Orchestrator initialized with 3 agents")

    async def process_query(
        self,
        query: str,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process query through multi-agent workflow.

        Args:
            query: User query
            patient_id: Optional patient ID for context
            session_id: Optional session ID

        Returns:
            Complete response with agent outputs and metrics
        """
        query_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Processing query {query_id}: {query[:50]}...")

        try:
            # 1. Load patient context if provided
            context = {}
            if patient_id:
                patient = self.patient_handler.get_patient(patient_id)
                if patient:
                    context['patient'] = patient
                    logger.info(f"Loaded patient context: {patient_id}")

            # 2. Primary Agent - Generate clinical response
            primary_start = time.time()
            primary_response = await self.primary_agent.process(
                query, context, previous_responses=None
            )
            primary_time = time.time() - primary_start

            logger.info(f"Primary agent responded: {primary_response.status}")

            # 3. Compliance Agent - Validate
            compliance_start = time.time()
            compliance_response = await self.compliance_agent.process(
                query, context, previous_responses=[primary_response]
            )
            compliance_time = time.time() - compliance_start

            logger.info(f"Compliance check: {compliance_response.status}")

            # 4. Handle compliance rejection
            if compliance_response.status == 'rejected':
                # Return error, don't proceed to quality
                total_time = time.time() - start_time

                # Record metrics
                await self._record_metrics(
                    query_id, query, patient_id,
                    primary_time, compliance_time, 0, total_time,
                    primary_response, compliance_response, None,
                    False, True, "Compliance rejection"
                )

                return {
                    'query_id': query_id,
                    'status': 'rejected',
                    'reason': 'compliance_failure',
                    'compliance_issues': compliance_response.issues_found,
                    'suggestions': compliance_response.suggestions,
                    'response_time': total_time
                }

            # 5. Quality Agent - Final validation
            quality_start = time.time()
            quality_response = await self.quality_agent.process(
                query, context, previous_responses=[primary_response]
            )
            quality_time = time.time() - quality_start

            logger.info(f"Quality check: {quality_response.status}")

            # 6. Handle quality rejection
            if quality_response.status == 'rejected':
                # Feedback loop: Send back to primary agent
                logger.warning("Quality check failed, initiating revision...")

                # Add quality feedback to context
                context['quality_feedback'] = quality_response.issues_found

                # Re-process with primary agent
                revision_start = time.time()
                revised_response = await self.primary_agent.process(
                    query, context, previous_responses=[primary_response, quality_response]
                )
                revision_time = time.time() - revision_start
                primary_time += revision_time

                # Re-check quality (one retry only)
                quality_recheck = await self.quality_agent.process(
                    query, context, previous_responses=[revised_response]
                )

                if quality_recheck.status == 'rejected':
                    logger.error("Quality check failed after revision")
                    # Return with warning
                    final_response = revised_response
                    final_quality = quality_recheck
                else:
                    logger.info("Quality check passed after revision")
                    final_response = revised_response
                    final_quality = quality_recheck
            else:
                final_response = primary_response
                final_quality = quality_response

            # 7. Aggregate final response
            total_time = time.time() - start_time

            final_content = self._aggregate_response(
                final_response,
                compliance_response,
                final_quality
            )

            # 8. Record metrics
            await self._record_metrics(
                query_id, query, patient_id,
                primary_time, compliance_time, quality_time, total_time,
                final_response, compliance_response, final_quality,
                True, False, None
            )

            # 9. Return complete result
            return {
                'query_id': query_id,
                'status': 'success',
                'content': final_content,
                'agents': {
                    'primary': {
                        'status': final_response.status,
                        'confidence': final_response.confidence_score,
                        'reasoning': final_response.reasoning
                    },
                    'compliance': {
                        'status': compliance_response.status,
                        'issues': len(compliance_response.issues_found),
                        'warnings': compliance_response.issues_found if compliance_response.status == 'warning' else []
                    },
                    'quality': {
                        'status': final_quality.status,
                        'confidence': final_quality.confidence_score,
                        'hallucination_risk': 1.0 - final_quality.confidence_score
                    }
                },
                'metadata': {
                    'query_id': query_id,
                    'patient_id': patient_id,
                    'session_id': session_id,
                    'response_time': round(total_time, 2),
                    'reasoning_path': final_response.metadata.get('reasoning_path')
                }
            }

        except Exception as e:
            logger.error(f"MCP orchestration error: {e}", exc_info=True)

            total_time = time.time() - start_time

            return {
                'query_id': query_id,
                'status': 'error',
                'error': str(e),
                'response_time': total_time
            }

    def _aggregate_response(
        self,
        primary: AgentResponse,
        compliance: AgentResponse,
        quality: AgentResponse
    ) -> str:
        """Aggregate agent responses into final output."""

        parts = [primary.content]

        # Add compliance warnings if any
        if compliance.status == 'warning' and compliance.issues_found:
            warnings = "\n".join([
                f"- {issue['description']}"
                for issue in compliance.issues_found
            ])
            parts.append(f"\n**Compliance Considerations:**\n{warnings}")

        # Add quality warnings if any
        if quality.status == 'warning' and quality.issues_found:
            parts.append(
                f"\n**Note:** This response has been flagged for quality review "
                f"(confidence: {quality.confidence_score:.0%})"
            )

        # Always add medical disclaimer
        parts.append(
            "\n---\n"
            "*This is AI-generated clinical decision support. "
            "Healthcare providers must verify all recommendations using their professional judgment.*"
        )

        return "\n".join(parts)

    async def _record_metrics(
        self,
        query_id: str,
        query: str,
        patient_id: Optional[str],
        primary_time: float,
        compliance_time: float,
        quality_time: float,
        total_time: float,
        primary_response: AgentResponse,
        compliance_response: AgentResponse,
        quality_response: Optional[AgentResponse],
        success: bool,
        error: bool,
        error_msg: Optional[str]
    ):
        """Record metrics for this query."""

        # Calculate token usage (from metadata)
        primary_tokens = primary_response.metadata.get('tokens_used', 0)

        # Estimate cost (Claude Haiku pricing)
        # Input: $0.80 / MTok, Output: $4.00 / MTok
        # Estimate 80% input, 20% output
        input_tokens = int(primary_tokens * 0.8)
        output_tokens = int(primary_tokens * 0.2)
        cost = (input_tokens * 0.8 / 1_000_000) + (output_tokens * 4.0 / 1_000_000)

        metrics = QueryMetrics(
            query_id=query_id,
            timestamp=datetime.now().isoformat(),
            query_text=query[:200],  # Truncate for privacy
            patient_id=patient_id,
            total_response_time=total_time,
            mcp_time=0.1,  # Orchestration overhead
            primary_agent_time=primary_time,
            compliance_agent_time=compliance_time,
            quality_agent_time=quality_time,
            rag_time=None,
            rat_time=primary_time if primary_response.metadata.get('used_rat', False) else None,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            primary_agent_tokens=primary_tokens,
            compliance_agent_tokens=0,  # Rule-based, no LLM
            quality_agent_tokens=int(primary_tokens * 0.5),  # Estimate
            total_cost=cost,
            compliance_passed=compliance_response.status != 'rejected',
            quality_passed=quality_response.status != 'rejected' if quality_response else False,
            hallucination_score=1.0 - (quality_response.confidence_score if quality_response else 0.5),
            confidence_score=quality_response.confidence_score if quality_response else 0.5,
            response_generated=success,
            error_occurred=error,
            error_message=error_msg
        )

        metrics_collector.record_metrics(metrics)

    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'status': 'operational',
            'agents': {
                'primary': {'name': self.primary_agent.name, 'type': self.primary_agent.agent_type},
                'compliance': {'name': self.compliance_agent.name, 'type': self.compliance_agent.agent_type},
                'quality': {'name': self.quality_agent.name, 'type': self.quality_agent.agent_type}
            },
            'metrics_summary': metrics_collector.get_metrics_summary(hours=24)
        }
