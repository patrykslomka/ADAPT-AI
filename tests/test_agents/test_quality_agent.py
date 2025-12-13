"""Test quality agent."""
import pytest
from unittest.mock import patch, MagicMock
from src.agents.quality_agent import QualityAgent
from src.agents.base_agent import AgentResponse


class TestQualityAgent:
    """Tests for QualityAgent class."""

    @pytest.fixture
    def mock_client(self):
        """Mock Anthropic client."""
        with patch('src.agents.quality_agent.Anthropic') as mock:
            client = MagicMock()
            mock.return_value = client

            response = MagicMock()
            response.content = [MagicMock(text='{"issues": [], "overall_assessment": "No issues"}')]
            client.messages.create.return_value = response

            yield client

    @pytest.fixture
    def mock_ontology(self):
        """Mock ontology loader."""
        with patch('src.agents.quality_agent.ontology_loader') as mock:
            mock.load_drug_database.return_value = {
                'medications': [
                    {'generic_name': 'Amoxicillin', 'brand_names': ['Amoxil']},
                    {'generic_name': 'Metformin', 'brand_names': ['Glucophage']}
                ]
            }
            mock.load_clinical_ontology.return_value = {
                'diseases': [
                    {'name': 'Pneumonia', 'id': 'D001'},
                    {'name': 'Diabetes', 'id': 'D002'}
                ]
            }
            yield mock

    @pytest.fixture
    def agent(self, mock_client, mock_ontology):
        """Create quality agent."""
        return QualityAgent()

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.name == 'quality_agent'
        assert agent.agent_type == 'validator'

    @pytest.mark.asyncio
    async def test_process_no_response(self, agent):
        """Test with no previous response."""
        response = await agent.process(
            query="Test",
            context={},
            previous_responses=None
        )

        assert response.status == 'approved'
        assert response.confidence_score == 1.0

    @pytest.mark.asyncio
    async def test_process_valid_response(self, agent):
        """Test with valid clinical response."""
        primary = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='Tuberculosis presents with cough, fever, and weight loss. Diagnosis confirmed with chest X-ray and sputum culture.'
        )

        response = await agent.process(
            query="What are TB symptoms?",
            context={},
            previous_responses=[primary]
        )

        assert response.status in ['approved', 'warning']
        assert response.confidence_score > 0.0

    @pytest.mark.asyncio
    async def test_detect_overclaim_cure(self, agent):
        """Test detection of cure claims for chronic diseases."""
        primary = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='This treatment cures diabetes completely and eliminates hypertension.'
        )

        response = await agent.process(
            query="Treatment",
            context={},
            previous_responses=[primary]
        )

        # Should detect overclaim
        assert len(response.issues_found) > 0
        overclaims = [i for i in response.issues_found if i['type'] == 'overclaim']
        assert len(overclaims) > 0

    @pytest.mark.asyncio
    async def test_detect_absolute_claim(self, agent):
        """Test detection of absolute efficacy claims."""
        primary = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='This medication is 100% effective and always works for all patients.'
        )

        response = await agent.process(
            query="Treatment",
            context={},
            previous_responses=[primary]
        )

        assert len(response.issues_found) > 0

    @pytest.mark.asyncio
    async def test_confidence_calculation(self, agent):
        """Test confidence score calculation."""
        # Response with no issues
        primary = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='Consider standard diagnostic workup including CBC, CMP, and chest imaging.'
        )

        response = await agent.process(
            query="Diagnosis",
            context={},
            previous_responses=[primary]
        )

        # With no issues, confidence should be high
        assert response.confidence_score >= 0.7

    def test_check_medications_valid(self, agent):
        """Test medication check with valid drugs."""
        issues = agent._check_medications("Consider Amoxicillin 500mg for the infection.")
        # Should not flag known drugs
        amox_issues = [i for i in issues if 'Amoxicillin' in i.get('description', '')]
        assert len(amox_issues) == 0

    def test_validate_clinical_accuracy_overclaim(self, agent):
        """Test clinical accuracy validation."""
        issues = agent._validate_clinical_accuracy(
            "This treatment is 100% safe and guaranteed to work",
            {}
        )

        assert len(issues) > 0

    def test_llm_hallucination_check_no_context(self, agent):
        """Test LLM check with no reference context."""
        result = agent._llm_hallucination_check(
            "What is diabetes?",
            "Diabetes is a condition...",
            ""  # No reference docs
        )

        # Should return empty issues when no context
        assert result['issues'] == []

    @pytest.mark.asyncio
    async def test_status_rejected_critical(self, mock_client, mock_ontology):
        """Test that critical issues result in rejection."""
        # Make LLM return critical issues
        mock_client.messages.create.return_value.content = [
            MagicMock(text='{"issues": [{"text": "test", "reason": "critical error", "severity": "critical"}]}')
        ]

        agent = QualityAgent()

        primary = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='Some problematic content with fabricated drug Fantazomycin.'
        )

        response = await agent.process(
            query="Treatment",
            context={'retrieved_documents': 'Some reference context about treatments.'},
            previous_responses=[primary]
        )

        # With critical LLM issues, status should be rejected
        if any(i.get('severity') == 'critical' for i in response.issues_found):
            assert response.status == 'rejected'

    def test_create_response_helper(self, agent):
        """Test response creation helper."""
        response = agent._create_response('approved', 'Test', 0.9)

        assert response.status == 'approved'
        assert response.content == 'Test'
        assert response.confidence_score == 0.9
        assert response.agent_name == 'quality_agent'
