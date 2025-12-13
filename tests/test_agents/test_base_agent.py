"""Tests for base agent class."""
import pytest
from src.agents.base_agent import BaseAgent, AgentResponse


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing."""

    async def process(self, query, context, previous_responses=None):
        return AgentResponse(
            agent_name=self.name,
            agent_type=self.agent_type,
            status='approved',
            content=f"Processed: {query}"
        )


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""

    def test_response_creation(self):
        """Test basic response creation."""
        response = AgentResponse(
            agent_name='test_agent',
            agent_type='clinical',
            status='approved',
            content='Test content'
        )

        assert response.agent_name == 'test_agent'
        assert response.agent_type == 'clinical'
        assert response.status == 'approved'
        assert response.content == 'Test content'

    def test_response_defaults(self):
        """Test default values are set."""
        response = AgentResponse(
            agent_name='test',
            agent_type='test',
            status='approved',
            content='content'
        )

        assert response.issues_found == []
        assert response.suggestions == []
        assert response.metadata == {}
        assert response.confidence_score == 0.0

    def test_response_with_issues(self):
        """Test response with issues."""
        issues = [{'type': 'error', 'description': 'Test error'}]
        response = AgentResponse(
            agent_name='test',
            agent_type='test',
            status='rejected',
            content='Error',
            issues_found=issues
        )

        assert len(response.issues_found) == 1
        assert response.issues_found[0]['type'] == 'error'


class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = ConcreteAgent(
            name='test_agent',
            agent_type='clinical',
            system_prompt='Test prompt'
        )

        assert agent.name == 'test_agent'
        assert agent.agent_type == 'clinical'
        assert agent.system_prompt == 'Test prompt'
        assert agent.request_count == 0

    def test_generate_request_id(self):
        """Test request ID generation."""
        agent = ConcreteAgent(
            name='test_agent',
            agent_type='clinical',
            system_prompt='Test'
        )

        id1 = agent._generate_request_id()
        id2 = agent._generate_request_id()

        assert id1 != id2
        assert 'test_agent' in id1
        assert agent.request_count == 2

    def test_format_context_empty(self):
        """Test formatting empty context."""
        agent = ConcreteAgent(
            name='test',
            agent_type='test',
            system_prompt='Test'
        )

        result = agent._format_context({})
        assert result == ''

    def test_format_context_with_patient(self):
        """Test formatting context with patient."""
        agent = ConcreteAgent(
            name='test',
            agent_type='test',
            system_prompt='Test'
        )

        context = {
            'patient': {
                'demographics': {
                    'patient_id': 'P-0001',
                    'age': 45,
                    'gender': 'Male'
                },
                'medical_history': {
                    'allergies': [{'substance': 'Penicillin', 'reaction': 'rash'}],
                    'chronic_conditions': ['Diabetes']
                },
                'current_medications': [
                    {'name': 'Metformin', 'dose': '500mg'}
                ],
                'presenting_complaint': {
                    'chief_complaint': 'Chest pain',
                    'hpi': 'Started 2 hours ago'
                }
            }
        }

        result = agent._format_context(context)

        assert 'P-0001' in result
        assert '45' in result
        assert 'Male' in result
        assert 'Penicillin' in result
        assert 'Metformin' in result
        assert 'Chest pain' in result

    def test_format_patient(self):
        """Test patient formatting."""
        agent = ConcreteAgent(
            name='test',
            agent_type='test',
            system_prompt='Test'
        )

        patient = {
            'demographics': {
                'patient_id': 'P-0001',
                'age': 65,
                'gender': 'Female'
            },
            'medical_history': {
                'allergies': [
                    {'substance': 'Sulfa', 'reaction': 'anaphylaxis'}
                ],
                'chronic_conditions': ['Hypertension', 'Diabetes']
            },
            'current_medications': [
                {'name': 'Lisinopril', 'dose': '10mg'}
            ],
            'presenting_complaint': {
                'chief_complaint': 'Shortness of breath',
                'hpi': 'Progressive over 2 weeks'
            },
            'vital_signs': {
                'temperature': 98.6,
                'heart_rate': 88,
                'blood_pressure_systolic': 140,
                'blood_pressure_diastolic': 90,
                'respiratory_rate': 22,
                'oxygen_saturation': 94
            }
        }

        result = agent._format_patient(patient)

        assert 'P-0001' in result
        assert '65' in result
        assert 'Female' in result
        assert 'ALLERGIES' in result
        assert 'Sulfa' in result
        assert 'Lisinopril' in result
        assert 'Shortness of breath' in result
        assert '140/90' in result


@pytest.mark.asyncio
async def test_agent_process():
    """Test agent processing."""
    agent = ConcreteAgent(
        name='test_agent',
        agent_type='clinical',
        system_prompt='Test'
    )

    response = await agent.process(
        query='Test query',
        context={}
    )

    assert response.status == 'approved'
    assert 'Test query' in response.content
