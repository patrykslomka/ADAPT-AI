"""End-to-end integration tests."""
import pytest
import asyncio
from unittest.mock import patch, MagicMock


class TestMCPOrchestrator:
    """Integration tests for MCP Orchestrator."""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client for all agents."""
        with patch('src.agents.primary_agent.Anthropic') as primary_mock, \
             patch('src.agents.quality_agent.Anthropic') as quality_mock:

            # Mock primary agent response
            primary_response = MagicMock()
            primary_response.content = [MagicMock(text="""
## Assessment
Based on the clinical presentation, this appears to be a case of community-acquired pneumonia.

## Differential Diagnoses
1. Community-acquired pneumonia (most likely)
2. Acute bronchitis
3. Influenza

## Recommended Workup
- Chest X-ray
- CBC with differential
- Basic metabolic panel
- Sputum culture if productive cough

## Clinical Pearls
Healthcare providers should verify these recommendations.
""")]
            primary_response.usage.input_tokens = 500
            primary_response.usage.output_tokens = 200

            primary_client = MagicMock()
            primary_client.messages.create.return_value = primary_response
            primary_mock.return_value = primary_client

            # Mock quality agent response
            quality_response = MagicMock()
            quality_response.content = [MagicMock(text='{"issues": [], "overall_assessment": "No issues found"}')]
            quality_client = MagicMock()
            quality_client.messages.create.return_value = quality_response
            quality_mock.return_value = quality_client

            yield primary_mock, quality_mock

    @pytest.fixture
    def mock_rag(self):
        """Mock RAG block."""
        with patch('src.agents.primary_agent.RAGBlock') as mock:
            rag = MagicMock()
            rag.process_query.return_value = {
                'context': 'Medical knowledge about respiratory conditions.',
                'documents': [],
                'num_results': 3
            }
            mock.return_value = rag
            yield rag

    @pytest.fixture
    def mock_rat(self):
        """Mock RAT block."""
        with patch('src.agents.primary_agent.RATBlock') as mock:
            rat = MagicMock()
            rat.process_query.return_value = {
                'context': 'Detailed reasoning context.',
                'reasoning_path': [{'step': 1, 'thought': 'Analysis'}],
                'total_steps': 3,
                'confidence_score': 0.85
            }
            mock.return_value = rat
            yield rat

    @pytest.fixture
    def mock_ontology(self):
        """Mock ontology loader."""
        with patch('src.agents.compliance_agent.ontology_loader') as comp_mock, \
             patch('src.agents.quality_agent.ontology_loader') as qual_mock:

            mock_data = {
                'diseases': [{'name': 'Pneumonia', 'id': 'D001'}],
                'medications': [{'generic_name': 'Amoxicillin', 'brand_names': ['Amoxil']}],
                'allergy_groups': []
            }

            for mock in [comp_mock, qual_mock]:
                mock.load_clinical_ontology.return_value = mock_data
                mock.load_drug_database.return_value = mock_data
                mock.load_hipaa_rules.return_value = {'rules': []}
                mock.load_fda_guidelines.return_value = {'guidelines': []}

            yield comp_mock

    @pytest.mark.asyncio
    async def test_simple_clinical_query(self, mock_anthropic, mock_rag, mock_rat, mock_ontology):
        """Test simple clinical query with RAG."""
        from src.mcp.orchestrator import MCPOrchestrator

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="What are the typical symptoms of pneumonia?",
            patient_id=None
        )

        assert result['status'] == 'success'
        assert 'content' in result
        assert result['agents']['primary']['status'] == 'approved'

    @pytest.mark.asyncio
    async def test_patient_specific_query(self, mock_anthropic, mock_rag, mock_rat, mock_ontology):
        """Test query with patient context."""
        from src.mcp.orchestrator import MCPOrchestrator

        with patch('src.mcp.orchestrator.PatientHandler') as handler_mock:
            handler = MagicMock()
            handler.get_patient.return_value = {
                'demographics': {'patient_id': 'P-0001', 'age': 45, 'gender': 'Male'},
                'medical_history': {'allergies': [], 'chronic_conditions': ['COPD']},
                'presenting_complaint': {'chief_complaint': 'Cough', 'hpi': 'Productive cough for 5 days'}
            }
            handler_mock.return_value = handler

            orchestrator = MCPOrchestrator()
            result = await orchestrator.process_query(
                query="Suggest diagnostic workup for this patient",
                patient_id="P-0001"
            )

            assert result['status'] == 'success'
            assert result['metadata']['patient_id'] == "P-0001"

    @pytest.mark.asyncio
    async def test_compliance_check_runs(self, mock_anthropic, mock_rag, mock_rat, mock_ontology):
        """Test that compliance check runs."""
        from src.mcp.orchestrator import MCPOrchestrator

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="Test query",
            patient_id=None
        )

        # Check compliance was evaluated
        assert 'compliance' in result['agents']
        assert result['agents']['compliance']['status'] in ['approved', 'warning', 'rejected']

    @pytest.mark.asyncio
    async def test_quality_check_runs(self, mock_anthropic, mock_rag, mock_rat, mock_ontology):
        """Test that quality check runs."""
        from src.mcp.orchestrator import MCPOrchestrator

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="What is tuberculosis?",
            patient_id=None
        )

        # Check quality was evaluated
        assert 'quality' in result['agents']
        assert result['agents']['quality']['status'] in ['approved', 'warning', 'rejected']

    @pytest.mark.asyncio
    async def test_response_time_recorded(self, mock_anthropic, mock_rag, mock_rat, mock_ontology):
        """Test that response time is recorded."""
        from src.mcp.orchestrator import MCPOrchestrator

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="Simple query",
            patient_id=None
        )

        assert 'response_time' in result['metadata']
        # Response time can be 0.0 for mocked tests, just check it exists and is a number
        assert isinstance(result['metadata']['response_time'], (int, float))

    @pytest.mark.asyncio
    async def test_query_id_generated(self, mock_anthropic, mock_rag, mock_rat, mock_ontology):
        """Test that query ID is generated."""
        from src.mcp.orchestrator import MCPOrchestrator

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="Test query",
            patient_id=None
        )

        assert 'query_id' in result
        assert len(result['query_id']) > 0

    @pytest.mark.asyncio
    async def test_disclaimer_in_response(self, mock_anthropic, mock_rag, mock_rat, mock_ontology):
        """Test that medical disclaimer is added."""
        from src.mcp.orchestrator import MCPOrchestrator

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="Treatment recommendations",
            patient_id=None
        )

        # Disclaimer should be in content
        assert 'Healthcare providers' in result['content'] or 'AI-generated' in result['content']


class TestPatientHandler:
    """Integration tests for Patient Handler."""

    def test_load_patients(self):
        """Test loading patients from file."""
        from src.domain.patient_handler import PatientHandler

        handler = PatientHandler()
        patients = handler.load_patients()

        assert len(patients) > 0

    def test_get_patient(self):
        """Test getting specific patient."""
        from src.domain.patient_handler import PatientHandler

        handler = PatientHandler()
        patient = handler.get_patient("P-0001")

        assert patient is not None
        assert patient['demographics']['patient_id'] == 'P-0001'

    def test_get_patient_not_found(self):
        """Test getting non-existent patient."""
        from src.domain.patient_handler import PatientHandler

        handler = PatientHandler()
        patient = handler.get_patient("P-9999")

        assert patient is None

    def test_list_patients(self):
        """Test listing patients."""
        from src.domain.patient_handler import PatientHandler

        handler = PatientHandler()
        summaries = handler.list_patients()

        assert len(summaries) > 0
        assert 'patient_id' in summaries[0]
        assert 'chief_complaint' in summaries[0]


class TestMetricsCollection:
    """Integration tests for metrics collection."""

    def test_metrics_summary(self):
        """Test getting metrics summary."""
        from src.llmops.metrics_collector import metrics_collector

        summary = metrics_collector.get_metrics_summary(hours=24)

        assert 'total_queries' in summary
        assert 'avg_response_time' in summary
        assert 'compliance_pass_rate' in summary


class TestSDKAgents:
    """Integration tests for SDK-based agents."""

    @pytest.fixture
    def mock_anthropic_sdk(self):
        """Mock for SDK agents."""
        with patch('src.agents.sdk_agents.Anthropic') as mock:
            client = MagicMock()

            response = MagicMock()
            response.content = [MagicMock(text="SDK agent response")]
            response.usage.input_tokens = 100
            response.usage.output_tokens = 50
            client.messages.create.return_value = response

            mock.return_value = client
            yield client

    @pytest.fixture
    def mock_rag_sdk(self):
        """Mock RAG for SDK agents."""
        with patch('src.agents.sdk_agents.RAGBlock') as mock:
            rag = MagicMock()
            rag.process_query.return_value = {
                'context': 'Retrieved context',
                'documents': [],
                'num_results': 2
            }
            mock.return_value = rag
            yield rag

    @pytest.fixture
    def mock_rat_sdk(self):
        """Mock RAT for SDK agents."""
        with patch('src.agents.sdk_agents.RATBlock') as mock:
            rat = MagicMock()
            rat.process_query.return_value = {
                'context': 'Reasoning context',
                'reasoning_path': [],
                'total_steps': 2,
                'confidence_score': 0.8
            }
            mock.return_value = rat
            yield rat

    @pytest.mark.asyncio
    async def test_clinical_agent_execute(self, mock_anthropic_sdk, mock_rag_sdk, mock_rat_sdk):
        """Test clinical agent execution."""
        from src.agents.sdk_agents import ClinicalAgent

        agent = ClinicalAgent()
        response = await agent.execute("What are symptoms of diabetes?")

        assert response.status == 'approved'
        assert response.agent_name == 'clinical_agent'

    @pytest.mark.asyncio
    async def test_agent_orchestrator(self, mock_anthropic_sdk, mock_rag_sdk, mock_rat_sdk):
        """Test agent orchestrator."""
        from src.agents.sdk_agents import AgentOrchestrator

        orchestrator = AgentOrchestrator()
        result = await orchestrator.process("What is hypertension?")

        assert 'status' in result
        assert 'agents' in result
