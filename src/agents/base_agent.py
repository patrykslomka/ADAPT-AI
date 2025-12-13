"""Base agent interface for all AI agents."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Standardized agent response."""
    agent_name: str
    agent_type: str
    status: str  # 'approved', 'rejected', 'warning', 'error'
    content: str
    reasoning: Optional[str] = None
    confidence_score: float = 0.0
    issues_found: List[Dict] = None
    suggestions: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.issues_found is None:
            self.issues_found = []
        if self.suggestions is None:
            self.suggestions = []
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """Abstract base class for all AI agents."""

    def __init__(self, name: str, agent_type: str, system_prompt: str):
        self.name = name
        self.agent_type = agent_type
        self.system_prompt = system_prompt
        self.request_count = 0
        self.logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    async def process(
        self,
        query: str,
        context: Dict[str, Any],
        previous_responses: Optional[List[AgentResponse]] = None
    ) -> AgentResponse:
        """Process query and return response.

        Args:
            query: User query
            context: Context dict (patient data, retrieved docs, etc.)
            previous_responses: Responses from previous agents

        Returns:
            AgentResponse with status and content
        """
        pass

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self.request_count += 1
        return f"{self.name}-{uuid.uuid4().hex[:8]}-{self.request_count}"

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for LLM prompt."""
        parts = []

        if 'patient' in context:
            patient = context['patient']
            parts.append(f"**Patient Context:**\n{self._format_patient(patient)}")

        if 'retrieved_documents' in context:
            parts.append(f"**Retrieved Information:**\n{context['retrieved_documents']}")

        if 'reasoning_path' in context:
            parts.append(f"**Reasoning Path:**\n{context['reasoning_path']}")

        return "\n\n".join(parts)

    def _format_patient(self, patient: Dict) -> str:
        """Format patient data for prompt."""
        parts = [
            f"ID: {patient.get('demographics', {}).get('patient_id', 'Unknown')}",
            f"Age: {patient.get('demographics', {}).get('age', 'Unknown')}",
            f"Gender: {patient.get('demographics', {}).get('gender', 'Unknown')}"
        ]

        # Allergies (critical!)
        allergies = patient.get('medical_history', {}).get('allergies', [])
        if allergies:
            allergy_list = [f"{a['substance']} ({a['reaction']})" for a in allergies]
            parts.append(f"**ALLERGIES:** {', '.join(allergy_list)}")

        # Chronic conditions
        conditions = patient.get('medical_history', {}).get('chronic_conditions', [])
        if conditions:
            parts.append(f"Chronic Conditions: {', '.join(conditions)}")

        # Current medications
        meds = patient.get('current_medications', [])
        if meds:
            med_list = [f"{m['name']} {m['dose']}" for m in meds]
            parts.append(f"Current Medications: {', '.join(med_list)}")

        # Presenting complaint
        complaint = patient.get('presenting_complaint', {})
        if complaint:
            parts.append(f"Chief Complaint: {complaint.get('chief_complaint', '')}")
            parts.append(f"HPI: {complaint.get('hpi', '')}")

        # Vital signs
        vitals = patient.get('vital_signs', {})
        if vitals:
            vital_str = (
                f"T: {vitals.get('temperature', 'N/A')}F, "
                f"HR: {vitals.get('heart_rate', 'N/A')}, "
                f"BP: {vitals.get('blood_pressure_systolic', 'N/A')}/{vitals.get('blood_pressure_diastolic', 'N/A')}, "
                f"RR: {vitals.get('respiratory_rate', 'N/A')}, "
                f"SpO2: {vitals.get('oxygen_saturation', 'N/A')}%"
            )
            parts.append(f"Vital Signs: {vital_str}")

        return "\n".join(parts)
