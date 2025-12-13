"""Primary Clinical Agent - Domain expert for diagnostics."""
from typing import Dict, Any, Optional, List
from anthropic import Anthropic
from config.settings import settings
from src.agents.base_agent import BaseAgent, AgentResponse
from src.building_blocks.rag import RAGBlock
from src.building_blocks.rat import RATBlock
import logging

logger = logging.getLogger(__name__)


class PrimaryAgent(BaseAgent):
    """Clinical diagnostics domain expert."""

    SYSTEM_PROMPT = """You are an expert clinical diagnostic assistant supporting healthcare providers.

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

You are NOT diagnosing - you are providing decision support for qualified healthcare providers."""

    def __init__(self):
        super().__init__(
            name="primary_clinical_agent",
            agent_type="clinical_expert",
            system_prompt=self.SYSTEM_PROMPT
        )

        self.client = Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )

        self.rag = RAGBlock()
        self.rat = RATBlock(rag_block=self.rag)

    async def process(
        self,
        query: str,
        context: Dict[str, Any],
        previous_responses: Optional[List[AgentResponse]] = None
    ) -> AgentResponse:
        """Process clinical query with RAG or RAT."""

        request_id = self._generate_request_id()
        self.logger.info(f"Processing clinical query: {request_id}")

        try:
            # Decide: RAG or RAT?
            use_rat = self._should_use_rat(query, context)

            if use_rat:
                # Complex reasoning needed
                rat_result = self.rat.process_query(query, max_steps=4)
                retrieved_context = rat_result['context']
                reasoning_path = rat_result['reasoning_path']
            else:
                # Simple retrieval sufficient
                rag_result = self.rag.process_query(query)
                retrieved_context = rag_result['context']
                reasoning_path = None

            # Format full context
            full_context = self._format_context(context)
            if retrieved_context:
                full_context += f"\n\n**Medical Knowledge:**\n{retrieved_context}"

            # Check for quality feedback from previous iteration
            if context.get('quality_feedback'):
                full_context += f"\n\n**Quality Review Feedback (Please Address):**\n"
                for issue in context['quality_feedback']:
                    full_context += f"- {issue.get('description', 'Issue identified')}\n"

            # Generate clinical response
            response = self._generate_response(query, full_context, reasoning_path)

            return AgentResponse(
                agent_name=self.name,
                agent_type=self.agent_type,
                status='approved',
                content=response['content'],
                reasoning=response.get('reasoning'),
                confidence_score=response.get('confidence', 0.85),
                metadata={
                    'request_id': request_id,
                    'used_rat': use_rat,
                    'reasoning_path': reasoning_path,
                    'tokens_used': response.get('tokens', 0)
                }
            )

        except Exception as e:
            self.logger.error(f"Primary agent error: {e}")
            return AgentResponse(
                agent_name=self.name,
                agent_type=self.agent_type,
                status='error',
                content=f"Error processing query: {str(e)}",
                metadata={'request_id': request_id, 'error': str(e)}
            )

    def _should_use_rat(self, query: str, context: Dict) -> bool:
        """Decide if RAT reasoning is needed."""
        # Use RAT for:
        # - Diagnostic workup requests
        # - Complex clinical scenarios
        # - Questions requiring multi-step reasoning

        rat_triggers = [
            'diagnostic workup',
            'differential diagnosis',
            'what tests',
            'next steps',
            'recommend',
            'patient presents with',
            'suggest',
            'workup',
            'evaluate'
        ]

        query_lower = query.lower()
        return any(trigger in query_lower for trigger in rat_triggers)

    def _generate_response(
        self,
        query: str,
        context: str,
        reasoning_path: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """Generate clinical response using Claude."""

        # Build prompt
        prompt = f"""{self.system_prompt}

{context}

**Clinical Query:**
{query}

Provide a structured clinical response:
1. **Assessment**: Key findings and clinical reasoning
2. **Differential Diagnoses**: Ranked by likelihood with rationale
3. **Recommended Workup**: Diagnostic tests and priorities
4. **Clinical Pearls**: Important considerations

Remember: This is decision support for healthcare providers. Always recommend verification by a qualified clinician."""

        # Call Claude
        response = self.client.messages.create(
            model=settings.model_name,
            max_tokens=settings.max_tokens,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text

        return {
            'content': content,
            'reasoning': 'RAT multi-step' if reasoning_path else 'RAG retrieval',
            'confidence': 0.85,
            'tokens': response.usage.input_tokens + response.usage.output_tokens
        }
