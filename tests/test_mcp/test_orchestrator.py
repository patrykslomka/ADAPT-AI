"""Tests for MCP Orchestrator."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from src.agents.base_agent import AgentResponse


class TestMCPOrchestrator:
    """Tests for MCPOrchestrator class."""

    @pytest.fixture
    def mock_agents(self):
        """Mock all agents."""
        with patch('src.mcp.orchestrator.PrimaryAgent') as primary_mock, \
             patch('src.mcp.orchestrator.ComplianceAgent') as compliance_mock, \
             patch('src.mcp.orchestrator.QualityAgent') as quality_mock, \
             patch('src.mcp.orchestrator.PatientHandler') as handler_mock:

            # Primary agent
            primary = MagicMock()
            primary.process = AsyncMock(return_value=AgentResponse(
                agent_name='primary',
                agent_type='clinical',
                status='approved',
                content='Clinical response',
                confidence_score=0.85,
                metadata={'tokens_used': 500, 'used_rat': False}
            ))
            primary_mock.return_value = primary

            # Compliance agent
            compliance = MagicMock()
            compliance.process = AsyncMock(return_value=AgentResponse(
                agent_name='compliance',
                agent_type='validator',
                status='approved',
                content='Compliance passed',
                issues_found=[]
            ))
            compliance_mock.return_value = compliance

            # Quality agent
            quality = MagicMock()
            quality.process = AsyncMock(return_value=AgentResponse(
                agent_name='quality',
                agent_type='validator',
                status='approved',
                content='Quality passed',
                confidence_score=0.9,
                issues_found=[]
            ))
            quality_mock.return_value = quality

            # Patient handler
            handler = MagicMock()
            handler.get_patient.return_value = {
                'demographics': {'patient_id': 'P-0001', 'age': 45},
                'medical_history': {'allergies': []}
            }
            handler_mock.return_value = handler

            yield {
                'primary': primary,
                'compliance': compliance,
                'quality': quality,
                'handler': handler
            }

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, mock_agents):
        """Test orchestrator initialization."""
        from src.mcp.orchestrator import MCPOrchestrator

        orchestrator = MCPOrchestrator()

        assert orchestrator.primary_agent is not None
        assert orchestrator.compliance_agent is not None
        assert orchestrator.quality_agent is not None

    @pytest.mark.asyncio
    async def test_process_query_success(self, mock_agents):
        """Test successful query processing."""
        from src.mcp.orchestrator import MCPOrchestrator

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="Test query",
            patient_id=None
        )

        assert result['status'] == 'success'
        assert 'content' in result
        assert 'query_id' in result

    @pytest.mark.asyncio
    async def test_process_query_with_patient(self, mock_agents):
        """Test query with patient context."""
        from src.mcp.orchestrator import MCPOrchestrator

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="Diagnostic workup",
            patient_id="P-0001"
        )

        assert result['status'] == 'success'
        assert result['metadata']['patient_id'] == 'P-0001'
        mock_agents['handler'].get_patient.assert_called_with('P-0001')

    @pytest.mark.asyncio
    async def test_compliance_rejection(self, mock_agents):
        """Test handling of compliance rejection."""
        from src.mcp.orchestrator import MCPOrchestrator

        # Make compliance reject
        mock_agents['compliance'].process = AsyncMock(return_value=AgentResponse(
            agent_name='compliance',
            agent_type='validator',
            status='rejected',
            content='PHI violation detected',
            issues_found=[{'type': 'phi', 'severity': 'critical', 'description': 'Name found'}]
        ))

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="Test",
            patient_id=None
        )

        assert result['status'] == 'rejected'
        assert result['reason'] == 'compliance_failure'
        assert len(result['compliance_issues']) > 0

    @pytest.mark.asyncio
    async def test_quality_rejection_triggers_revision(self, mock_agents):
        """Test that quality rejection triggers revision."""
        from src.mcp.orchestrator import MCPOrchestrator

        # First quality check fails, second passes
        mock_agents['quality'].process = AsyncMock(side_effect=[
            AgentResponse(
                agent_name='quality',
                agent_type='validator',
                status='rejected',
                content='Issues found',
                confidence_score=0.3,
                issues_found=[{'type': 'hallucination', 'severity': 'critical'}]
            ),
            AgentResponse(
                agent_name='quality',
                agent_type='validator',
                status='approved',
                content='Passed after revision',
                confidence_score=0.85,
                issues_found=[]
            )
        ])

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="Test",
            patient_id=None
        )

        # Primary agent should be called twice (original + revision)
        assert mock_agents['primary'].process.call_count >= 2

    @pytest.mark.asyncio
    async def test_aggregate_response_with_warnings(self, mock_agents):
        """Test response aggregation with compliance warnings."""
        from src.mcp.orchestrator import MCPOrchestrator

        mock_agents['compliance'].process = AsyncMock(return_value=AgentResponse(
            agent_name='compliance',
            agent_type='validator',
            status='warning',
            content='Warnings found',
            issues_found=[{'description': 'Missing disclaimer'}]
        ))

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="Test",
            patient_id=None
        )

        assert result['status'] == 'success'
        assert 'Compliance Considerations' in result['content']

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_agents):
        """Test error handling during processing."""
        from src.mcp.orchestrator import MCPOrchestrator

        mock_agents['primary'].process = AsyncMock(side_effect=Exception("API Error"))

        orchestrator = MCPOrchestrator()

        result = await orchestrator.process_query(
            query="Test",
            patient_id=None
        )

        assert result['status'] == 'error'
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_system_status(self, mock_agents):
        """Test getting system status."""
        from src.mcp.orchestrator import MCPOrchestrator

        orchestrator = MCPOrchestrator()

        status = await orchestrator.get_system_status()

        assert status['status'] == 'operational'
        assert 'agents' in status
        assert 'primary' in status['agents']
