"""Test compliance agent."""
import pytest
from unittest.mock import patch, MagicMock
from src.agents.compliance_agent import ComplianceAgent
from src.agents.base_agent import AgentResponse


class TestComplianceAgent:
    """Tests for ComplianceAgent class."""

    @pytest.fixture
    def mock_ontology(self):
        """Mock ontology loader."""
        with patch('src.agents.compliance_agent.ontology_loader') as mock:
            mock.load_hipaa_rules.return_value = {
                'rules': [
                    {'id': 'HIPAA-001', 'description': 'No patient names'}
                ]
            }
            mock.load_fda_guidelines.return_value = {
                'guidelines': []
            }
            yield mock

    @pytest.fixture
    def agent(self, mock_ontology):
        """Create compliance agent."""
        return ComplianceAgent()

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.name == 'compliance_agent'
        assert agent.agent_type == 'validator'

    @pytest.mark.asyncio
    async def test_process_no_previous_response(self, agent):
        """Test processing with no previous response."""
        response = await agent.process(
            query="Test",
            context={},
            previous_responses=None
        )

        assert response.status == 'approved'
        assert 'No previous response' in response.content

    @pytest.mark.asyncio
    async def test_detect_phi_name(self, agent):
        """Test PHI detection - patient name."""
        primary_response = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='Patient John Smith should take medication.'
        )

        patient = {
            'demographics': {
                'name': 'John Smith',
                'patient_id': 'P-0001'
            }
        }

        response = await agent.process(
            query="Test",
            context={'patient': patient},
            previous_responses=[primary_response]
        )

        assert response.status == 'rejected'
        assert any('name' in issue['description'].lower() for issue in response.issues_found)

    @pytest.mark.asyncio
    async def test_detect_phi_mrn(self, agent):
        """Test PHI detection - MRN."""
        primary_response = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='Patient MRN-123456 has been treated.'
        )

        response = await agent.process(
            query="Test",
            context={'patient': {}},
            previous_responses=[primary_response]
        )

        assert len(response.issues_found) > 0
        assert any('MRN' in issue['description'] or 'record number' in issue['description'].lower()
                   for issue in response.issues_found)

    @pytest.mark.asyncio
    async def test_detect_drug_allergy_penicillin(self, agent):
        """Test drug allergy detection - penicillin class."""
        primary_response = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='Recommend amoxicillin 500mg TID for 7 days.'
        )

        patient = {
            'demographics': {'patient_id': 'P-0001'},
            'medical_history': {
                'allergies': [
                    {'substance': 'Penicillin', 'reaction': 'anaphylaxis'}
                ]
            }
        }

        response = await agent.process(
            query="Treatment",
            context={'patient': patient},
            previous_responses=[primary_response]
        )

        # Should detect penicillin allergy with amoxicillin
        assert len(response.issues_found) > 0
        drug_issues = [i for i in response.issues_found if i['type'] == 'drug_contraindication']
        assert len(drug_issues) > 0

    @pytest.mark.asyncio
    async def test_detect_drug_allergy_sulfa(self, agent):
        """Test drug allergy detection - sulfa class."""
        primary_response = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='Consider Bactrim for UTI treatment.'
        )

        patient = {
            'demographics': {'patient_id': 'P-0001'},
            'medical_history': {
                'allergies': [
                    {'substance': 'Sulfa', 'reaction': 'rash'}
                ]
            }
        }

        response = await agent.process(
            query="Treatment",
            context={'patient': patient},
            previous_responses=[primary_response]
        )

        assert len(response.issues_found) > 0

    @pytest.mark.asyncio
    async def test_missing_disclaimer(self, agent):
        """Test missing disclaimer detection."""
        primary_response = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='Take aspirin 325mg daily.'  # No disclaimer
        )

        response = await agent.process(
            query="Treatment",
            context={},
            previous_responses=[primary_response]
        )

        assert any(i['type'] == 'missing_disclaimer' for i in response.issues_found)

    @pytest.mark.asyncio
    async def test_has_disclaimer(self, agent):
        """Test that proper disclaimer passes."""
        primary_response = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='Consider aspirin. Healthcare provider should verify this recommendation.'
        )

        response = await agent.process(
            query="Treatment",
            context={},
            previous_responses=[primary_response]
        )

        # Should not have missing disclaimer issue
        disclaimer_issues = [i for i in response.issues_found if i['type'] == 'missing_disclaimer']
        assert len(disclaimer_issues) == 0

    @pytest.mark.asyncio
    async def test_fda_problematic_claim(self, agent):
        """Test FDA problematic claim detection."""
        primary_response = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='This treatment is 100% effective and guaranteed to cure the condition.'
        )

        response = await agent.process(
            query="Treatment",
            context={},
            previous_responses=[primary_response]
        )

        fda_issues = [i for i in response.issues_found if i['type'] == 'fda_violation']
        assert len(fda_issues) > 0

    @pytest.mark.asyncio
    async def test_clean_response_passes(self, agent):
        """Test that clean response passes compliance."""
        primary_response = AgentResponse(
            agent_name='primary',
            agent_type='clinical',
            status='approved',
            content='''Based on the clinical presentation, consider the following differential diagnoses.

            Healthcare providers should verify all recommendations with their clinical judgment.
            This is decision support only.'''
        )

        patient = {
            'demographics': {
                'patient_id': 'P-0001'
                # No name = no PHI risk
            },
            'medical_history': {
                'allergies': []
            }
        }

        response = await agent.process(
            query="Diagnosis",
            context={'patient': patient},
            previous_responses=[primary_response]
        )

        # Should pass or only have warnings
        assert response.status in ['approved', 'warning']

    def test_check_phi_exposure_dob(self, agent):
        """Test DOB detection."""
        patient = {
            'demographics': {
                'date_of_birth': '1980-05-15'
            }
        }

        issues = agent._check_phi_exposure(
            "Patient born on 1980-05-15 presents with symptoms.",
            patient
        )

        assert len(issues) > 0
        assert any('birth' in i['description'].lower() for i in issues)

    def test_has_required_disclaimers(self, agent):
        """Test disclaimer detection."""
        assert agent._has_required_disclaimers("Consult your healthcare provider")
        assert agent._has_required_disclaimers("Use medical judgment")
        assert agent._has_required_disclaimers("This is decision support")
        assert agent._has_required_disclaimers("Please verify with physician")
        assert not agent._has_required_disclaimers("Take this medication now")
