"""Test primary clinical agent."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.primary_agent import PrimaryAgent
from src.agents.base_agent import AgentResponse


class TestPrimaryAgent:
    """Tests for PrimaryAgent class."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Anthropic client."""
        with patch('src.agents.primary_agent.Anthropic') as mock:
            client = MagicMock()
            mock.return_value = client

            # Mock response
            response = MagicMock()
            response.content = [MagicMock(text="Test clinical response")]
            response.usage.input_tokens = 100
            response.usage.output_tokens = 50
            client.messages.create.return_value = response

            yield client

    @pytest.fixture
    def mock_rag(self):
        """Create mock RAG block."""
        with patch('src.agents.primary_agent.RAGBlock') as mock:
            rag = MagicMock()
            rag.process_query.return_value = {
                'context': 'Retrieved medical knowledge',
                'documents': [],
                'num_results': 3
            }
            mock.return_value = rag
            yield rag

    @pytest.fixture
    def mock_rat(self):
        """Create mock RAT block."""
        with patch('src.agents.primary_agent.RATBlock') as mock:
            rat = MagicMock()
            rat.process_query.return_value = {
                'context': 'Complex reasoning context',
                'reasoning_path': [{'step': 1, 'thought': 'Analysis'}],
                'total_steps': 3,
                'confidence_score': 0.85
            }
            mock.return_value = rat
            yield rat

    def test_primary_agent_initialization(self, mock_client, mock_rag, mock_rat):
        """Test agent initialization."""
        agent = PrimaryAgent()

        assert agent.name == 'primary_clinical_agent'
        assert agent.agent_type == 'clinical_expert'
        assert 'clinical diagnostic assistant' in agent.system_prompt.lower()

    def test_should_use_rat_diagnostic(self, mock_client, mock_rag, mock_rat):
        """Test RAT decision for diagnostic queries."""
        agent = PrimaryAgent()

        # Should use RAT
        assert agent._should_use_rat("What is the diagnostic workup for chest pain?", {})
        assert agent._should_use_rat("Suggest differential diagnosis", {})
        assert agent._should_use_rat("What tests should we order?", {})
        assert agent._should_use_rat("Patient presents with fever and cough", {})

    def test_should_use_rag_simple(self, mock_client, mock_rag, mock_rat):
        """Test RAG decision for simple queries."""
        agent = PrimaryAgent()

        # Should use RAG (simpler)
        assert not agent._should_use_rat("What are the symptoms of TB?", {})
        assert not agent._should_use_rat("Define hypertension", {})

    @pytest.mark.asyncio
    async def test_process_simple_query(self, mock_client, mock_rag, mock_rat):
        """Test processing a simple query."""
        agent = PrimaryAgent()

        response = await agent.process(
            query="What are symptoms of TB?",
            context={}
        )

        assert response.status == 'approved'
        assert response.agent_name == 'primary_clinical_agent'
        assert response.content is not None
        assert response.metadata.get('used_rat') is False

    @pytest.mark.asyncio
    async def test_process_with_patient_context(self, mock_client, mock_rag, mock_rat):
        """Test processing with patient context."""
        agent = PrimaryAgent()

        patient = {
            'demographics': {
                'patient_id': 'P-TEST',
                'age': 45,
                'gender': 'Male'
            },
            'medical_history': {
                'chronic_conditions': ['Diabetes'],
                'allergies': [{'substance': 'Penicillin', 'reaction': 'rash'}]
            },
            'presenting_complaint': {
                'chief_complaint': 'Chest pain',
                'hpi': 'Acute onset crushing chest pain'
            }
        }

        response = await agent.process(
            query="Suggest diagnostic workup",
            context={'patient': patient}
        )

        assert response.status == 'approved'
        assert response.metadata.get('used_rat') is True

    @pytest.mark.asyncio
    async def test_process_with_quality_feedback(self, mock_client, mock_rag, mock_rat):
        """Test processing with quality feedback in context."""
        agent = PrimaryAgent()

        response = await agent.process(
            query="What is hypertension?",
            context={
                'quality_feedback': [
                    {'description': 'Missing source citation'}
                ]
            }
        )

        assert response.status == 'approved'

    @pytest.mark.asyncio
    async def test_process_error_handling(self, mock_client, mock_rag, mock_rat):
        """Test error handling."""
        agent = PrimaryAgent()

        # Make client raise exception
        mock_client.messages.create.side_effect = Exception("API Error")

        response = await agent.process(
            query="Test query",
            context={}
        )

        assert response.status == 'error'
        assert 'error' in response.metadata

    def test_generate_response(self, mock_client, mock_rag, mock_rat):
        """Test response generation."""
        agent = PrimaryAgent()

        result = agent._generate_response(
            query="What is diabetes?",
            context="Context about diabetes",
            reasoning_path=None
        )

        assert 'content' in result
        assert 'reasoning' in result
        assert 'confidence' in result
        assert 'tokens' in result
