"""RAT (Retrieval-Augmented Thoughts) Building Block.

Implements multi-step reasoning with iterative retrieval for complex queries.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import logging

from anthropic import Anthropic

from src.building_blocks.rag import RAGBlock
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """Single step in reasoning process."""
    step: int
    type: str  # 'initial_analysis', 'retrieval', 'reasoning_refinement', 'followup_retrieval', 'synthesis'
    thought: str = ""
    query: str = ""
    documents_found: int = 0
    context: str = ""
    confidence: float = 0.0


@dataclass
class RATResult:
    """Result of RAT processing."""
    query: str
    final_context: str
    reasoning_path: List[Dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0
    documents_retrieved: int = 0
    confidence_score: float = 0.0


class RATBlock:
    """Retrieval-Augmented Thoughts building block.

    Implements multi-step reasoning where:
    1. Initial analysis of the query
    2. First retrieval based on analysis
    3. Reasoning refinement based on retrieved information
    4. Follow-up retrievals if needed
    5. Final synthesis

    This approach is more effective than single-shot RAG for complex queries.
    """

    def __init__(
        self,
        rag_block: RAGBlock,
        model_name: str = None,
        api_key: str = None,
        max_steps: int = 4
    ):
        """Initialize RAT block.

        Args:
            rag_block: RAGBlock instance for retrieval
            model_name: Claude model to use for reasoning (defaults to settings.model_name)
            api_key: Anthropic API key
            max_steps: Maximum reasoning steps
        """
        self.rag = rag_block
        self.model_name = model_name or settings.model_name
        self.max_steps = max_steps

        # Initialize Anthropic client if API key provided
        self._client = None
        self._api_key = api_key

        logger.info(f"RAT block initialized with max_steps={max_steps}")

    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            if self._api_key is None:
                # Try to get from settings
                try:
                    from config.settings import settings
                    self._api_key = settings.anthropic_api_key.get_secret_value()
                except Exception:
                    raise ValueError("Anthropic API key required for RAT block")

            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a query using multi-step reasoning.

        Args:
            query: User query
            context: Optional additional context (patient data, etc.)
            max_steps: Override default max steps

        Returns:
            Dict with context, reasoning_path, and metadata
        """
        max_steps = max_steps or self.max_steps
        reasoning_path = []
        all_context = []

        # Step 1: Initial analysis
        step1 = self._initial_analysis(query, context)
        reasoning_path.append({
            "step": 1,
            "type": "initial_analysis",
            "thought": step1.thought,
            "confidence": step1.confidence
        })

        # Step 2: First retrieval
        step2 = self._first_retrieval(query, step1.thought)
        reasoning_path.append({
            "step": 2,
            "type": "first_retrieval",
            "query": step2.query,
            "documents_found": step2.documents_found,
            "context_preview": step2.context[:200] + "..." if len(step2.context) > 200 else step2.context
        })
        all_context.append(step2.context)

        # Step 3: Reasoning refinement
        step3 = self._reasoning_refinement(query, step1.thought, step2.context)
        reasoning_path.append({
            "step": 3,
            "type": "reasoning_refinement",
            "thought": step3.thought,
            "confidence": step3.confidence
        })

        # Step 4: Follow-up retrieval if needed (low confidence or gaps identified)
        if step3.confidence < 0.8 and max_steps >= 4:
            step4 = self._followup_retrieval(query, step3.thought)
            reasoning_path.append({
                "step": 4,
                "type": "followup_retrieval",
                "query": step4.query,
                "documents_found": step4.documents_found
            })
            if step4.context:
                all_context.append(step4.context)

        # Combine all context
        final_context = "\n\n---\n\n".join(all_context)

        # Calculate final confidence
        final_confidence = self._calculate_confidence(reasoning_path)

        result = {
            "query": query,
            "context": final_context,
            "reasoning_path": reasoning_path,
            "total_steps": len(reasoning_path),
            "confidence_score": final_confidence,
            "documents_retrieved": sum(
                step.get("documents_found", 0)
                for step in reasoning_path
                if "documents_found" in step
            )
        }

        logger.info(
            f"RAT completed: {result['total_steps']} steps, "
            f"{result['documents_retrieved']} docs, "
            f"confidence={result['confidence_score']:.2f}"
        )

        return result

    def _initial_analysis(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningStep:
        """Analyze query to determine retrieval strategy.

        Args:
            query: User query
            context: Additional context

        Returns:
            ReasoningStep with analysis
        """
        context_str = ""
        if context:
            if "patient" in context:
                patient = context["patient"]
                context_str = f"""
Patient Context:
- Age: {patient.get('demographics', {}).get('age', 'Unknown')}
- Chief Complaint: {patient.get('presenting_complaint', {}).get('chief_complaint', 'Unknown')}
- Conditions: {', '.join(patient.get('medical_history', {}).get('chronic_conditions', []))}
- Allergies: {', '.join([a['substance'] for a in patient.get('medical_history', {}).get('allergies', [])])}
"""

        prompt = f"""Analyze this clinical query to determine the best retrieval strategy.

Query: {query}
{context_str}

Identify:
1. Key clinical concepts to search for
2. What type of information is needed (diagnosis, treatment, drug info, etc.)
3. Any specific conditions or symptoms mentioned
4. Initial confidence level (0-1) in understanding the query

Respond with a brief analysis (2-3 sentences) and confidence score."""

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            thought = response.content[0].text

            # Extract confidence from response (simple heuristic)
            confidence = 0.7  # Default
            if "high confidence" in thought.lower():
                confidence = 0.9
            elif "low confidence" in thought.lower() or "unclear" in thought.lower():
                confidence = 0.5

            return ReasoningStep(
                step=1,
                type="initial_analysis",
                thought=thought,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Initial analysis failed: {e}")
            return ReasoningStep(
                step=1,
                type="initial_analysis",
                thought=f"Analysis failed: {str(e)}",
                confidence=0.5
            )

    def _first_retrieval(self, query: str, analysis: str) -> ReasoningStep:
        """Perform first retrieval based on analysis.

        Args:
            query: Original query
            analysis: Initial analysis

        Returns:
            ReasoningStep with retrieval results
        """
        # Use RAG to retrieve
        result = self.rag.process_query(query, n_results=5)

        return ReasoningStep(
            step=2,
            type="first_retrieval",
            query=query,
            documents_found=result["num_results"],
            context=result["context"]
        )

    def _reasoning_refinement(
        self,
        query: str,
        initial_thought: str,
        retrieved_context: str
    ) -> ReasoningStep:
        """Refine reasoning based on retrieved information.

        Args:
            query: Original query
            initial_thought: Initial analysis
            retrieved_context: Context from first retrieval

        Returns:
            ReasoningStep with refined reasoning
        """
        prompt = f"""Based on the initial analysis and retrieved information, refine the reasoning.

Original Query: {query}

Initial Analysis:
{initial_thought}

Retrieved Information:
{retrieved_context[:2000]}

Questions to consider:
1. Does the retrieved information adequately address the query?
2. Are there gaps that need additional retrieval?
3. What is the confidence level in the current information (0-1)?

Provide a brief assessment (2-3 sentences) and identify any follow-up queries needed."""

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            thought = response.content[0].text

            # Determine confidence
            confidence = 0.75
            if "sufficient" in thought.lower() or "adequate" in thought.lower():
                confidence = 0.85
            elif "gaps" in thought.lower() or "missing" in thought.lower():
                confidence = 0.6
            elif "need more" in thought.lower():
                confidence = 0.5

            return ReasoningStep(
                step=3,
                type="reasoning_refinement",
                thought=thought,
                confidence=confidence
            )
        except Exception as e:
            logger.error(f"Reasoning refinement failed: {e}")
            return ReasoningStep(
                step=3,
                type="reasoning_refinement",
                thought=f"Refinement failed: {str(e)}",
                confidence=0.6
            )

    def _followup_retrieval(self, query: str, refinement: str) -> ReasoningStep:
        """Perform follow-up retrieval for gaps.

        Args:
            query: Original query
            refinement: Refinement analysis

        Returns:
            ReasoningStep with follow-up results
        """
        # Generate follow-up query based on refinement
        prompt = f"""Based on this analysis, generate a specific follow-up search query to fill information gaps.

Analysis: {refinement}
Original Query: {query}

Generate a single, focused follow-up query (just the query, nothing else):"""

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=100,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            followup_query = response.content[0].text.strip()

            # Perform retrieval
            result = self.rag.process_query(followup_query, n_results=3)

            return ReasoningStep(
                step=4,
                type="followup_retrieval",
                query=followup_query,
                documents_found=result["num_results"],
                context=result["context"]
            )
        except Exception as e:
            logger.error(f"Follow-up retrieval failed: {e}")
            return ReasoningStep(
                step=4,
                type="followup_retrieval",
                query="",
                documents_found=0,
                context=""
            )

    def _calculate_confidence(self, reasoning_path: List[Dict]) -> float:
        """Calculate overall confidence from reasoning path.

        Args:
            reasoning_path: List of reasoning steps

        Returns:
            Overall confidence score (0-1)
        """
        confidences = [
            step.get("confidence", 0.7)
            for step in reasoning_path
            if "confidence" in step
        ]

        if not confidences:
            return 0.7

        # Weighted average, giving more weight to later steps
        weights = list(range(1, len(confidences) + 1))
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        total_weight = sum(weights)

        return round(weighted_sum / total_weight, 2)
