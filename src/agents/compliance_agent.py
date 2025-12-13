"""Compliance Agent - HIPAA/FDA validation."""
from typing import Dict, Any, Optional, List
from src.agents.base_agent import BaseAgent, AgentResponse
from src.domain.ontology_loader import ontology_loader
import re
import logging

logger = logging.getLogger(__name__)


class ComplianceAgent(BaseAgent):
    """HIPAA and FDA compliance validator."""

    SYSTEM_PROMPT = """You validate clinical responses for regulatory compliance.

Check for:
1. HIPAA violations (PHI exposure)
2. Drug contraindications (allergies, conditions)
3. FDA guideline adherence
4. Required medical disclaimers
5. Appropriate urgency/risk communication

Flag any compliance issues and suggest corrections."""

    def __init__(self):
        super().__init__(
            name="compliance_agent",
            agent_type="validator",
            system_prompt=self.SYSTEM_PROMPT
        )

        self.hipaa_rules = ontology_loader.load_hipaa_rules()
        self.fda_guidelines = ontology_loader.load_fda_guidelines()

    async def process(
        self,
        query: str,
        context: Dict[str, Any],
        previous_responses: Optional[List[AgentResponse]] = None
    ) -> AgentResponse:
        """Validate compliance of previous agent response."""

        if not previous_responses or len(previous_responses) == 0:
            return self._create_response('approved', 'No previous response to validate')

        primary_response = previous_responses[0]  # Validate primary agent

        issues = []
        suggestions = []

        # 1. Check HIPAA compliance
        phi_issues = self._check_phi_exposure(
            primary_response.content,
            context.get('patient', {})
        )
        issues.extend(phi_issues)

        # 2. Check drug contraindications
        if context.get('patient'):
            drug_issues = self._check_drug_safety(
                primary_response.content,
                context['patient']
            )
            issues.extend(drug_issues)

        # 3. Check disclaimers
        if not self._has_required_disclaimers(primary_response.content):
            issues.append({
                'type': 'missing_disclaimer',
                'severity': 'moderate',
                'description': 'Missing required medical disclaimer'
            })
            suggestions.append(
                "Add: 'This is AI-generated decision support. "
                "Healthcare provider must verify all recommendations.'"
            )

        # 4. Check FDA guidelines adherence
        fda_issues = self._check_fda_compliance(primary_response.content)
        issues.extend(fda_issues)

        # Determine status
        critical_issues = [i for i in issues if i.get('severity') == 'critical']

        if critical_issues:
            status = 'rejected'
            content = f"COMPLIANCE FAILURE: {len(critical_issues)} critical issues found."
        elif issues:
            status = 'warning'
            content = f"Compliance warnings: {len(issues)} issues requiring attention."
        else:
            status = 'approved'
            content = "All compliance checks passed."

        return AgentResponse(
            agent_name=self.name,
            agent_type=self.agent_type,
            status=status,
            content=content,
            issues_found=issues,
            suggestions=suggestions,
            metadata={
                'hipaa_checked': True,
                'drug_safety_checked': True,
                'fda_checked': True
            }
        )

    def _check_phi_exposure(self, content: str, patient: Dict) -> List[Dict]:
        """Check for PHI exposure."""
        issues = []

        # Check for patient name (should never appear)
        if patient.get('demographics', {}).get('name'):
            patient_name = patient['demographics']['name']
            if patient_name.lower() in content.lower():
                issues.append({
                    'type': 'phi_exposure',
                    'severity': 'critical',
                    'description': f'Patient name "{patient_name}" found in response',
                    'hipaa_rule': 'HIPAA-001'
                })

        # Check for MRN exposure
        mrn_pattern = r'MRN-\d{6}'
        if re.search(mrn_pattern, content):
            issues.append({
                'type': 'phi_exposure',
                'severity': 'critical',
                'description': 'Medical record number found in response',
                'hipaa_rule': 'HIPAA-003'
            })

        # Check for DOB exposure
        if patient.get('demographics', {}).get('date_of_birth'):
            dob = patient['demographics']['date_of_birth']
            if dob in content:
                issues.append({
                    'type': 'phi_exposure',
                    'severity': 'critical',
                    'description': 'Date of birth found in response',
                    'hipaa_rule': 'HIPAA-002'
                })

        # Check for SSN patterns
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        if re.search(ssn_pattern, content):
            issues.append({
                'type': 'phi_exposure',
                'severity': 'critical',
                'description': 'Social Security Number pattern found in response',
                'hipaa_rule': 'HIPAA-004'
            })

        return issues

    def _check_drug_safety(self, content: str, patient: Dict) -> List[Dict]:
        """Check for drug contraindications."""
        issues = []

        # Extract mentioned drugs (simple keyword matching)
        # In production, use NER/medical NLP
        allergies = patient.get('medical_history', {}).get('allergies', [])
        allergy_substances = [a['substance'].lower() for a in allergies]

        # Common penicillin-class drugs (cross-reactivity)
        penicillin_class = [
            'penicillin', 'amoxicillin', 'ampicillin', 'piperacillin',
            'augmentin', 'amox-clav', 'nafcillin', 'oxacillin'
        ]

        # Sulfa drugs
        sulfa_class = [
            'bactrim', 'trimethoprim', 'sulfamethoxazole', 'sulfa',
            'septra', 'sulfasalazine'
        ]

        content_lower = content.lower()

        # Check penicillin allergy cross-reactivity
        if 'penicillin' in allergy_substances:
            for drug in penicillin_class:
                if drug in content_lower:
                    issues.append({
                        'type': 'drug_contraindication',
                        'severity': 'critical',
                        'description': f'Drug "{drug}" contraindicated - patient has Penicillin allergy',
                        'recommendation': 'Remove this drug recommendation and consider alternatives'
                    })

        # Check sulfa allergy
        if 'sulfa' in allergy_substances:
            for drug in sulfa_class:
                if drug in content_lower:
                    issues.append({
                        'type': 'drug_contraindication',
                        'severity': 'critical',
                        'description': f'Drug "{drug}" contraindicated - patient has Sulfa allergy',
                        'recommendation': 'Remove this drug recommendation and consider alternatives'
                    })

        # Generic allergy check
        for allergy in allergy_substances:
            if allergy.lower() in content_lower:
                # Check if it's discussing the allergy vs recommending
                allergy_mentions = re.findall(
                    rf'\b{re.escape(allergy)}\b',
                    content_lower
                )
                for _ in allergy_mentions:
                    # Check context - is it a recommendation?
                    if any(rec in content_lower for rec in ['recommend', 'prescribe', 'administer', 'give']):
                        issues.append({
                            'type': 'drug_contraindication',
                            'severity': 'high',
                            'description': f'Potential allergy conflict with "{allergy}"',
                            'recommendation': 'Verify drug is not contraindicated with patient allergy'
                        })
                        break

        return issues

    def _has_required_disclaimers(self, content: str) -> bool:
        """Check if required disclaimers present."""
        required_phrases = [
            'healthcare provider',
            'medical judgment',
            'decision support',
            'verify',
            'clinician',
            'physician'
        ]

        content_lower = content.lower()
        return any(phrase in content_lower for phrase in required_phrases)

    def _check_fda_compliance(self, content: str) -> List[Dict]:
        """Check FDA guideline compliance."""
        issues = []
        content_lower = content.lower()

        # Check for off-label use without disclosure
        off_label_indicators = [
            'off-label',
            'unapproved indication',
            'not fda approved for'
        ]

        # Check for unsubstantiated claims
        problematic_claims = [
            (r'100%\s+(effective|safe|cure)', 'Absolute efficacy claim'),
            (r'guaranteed\s+(cure|treatment|results)', 'Guaranteed outcome claim'),
            (r'no\s+side\s+effects', 'No side effects claim'),
            (r'completely\s+safe', 'Complete safety claim')
        ]

        for pattern, description in problematic_claims:
            if re.search(pattern, content_lower):
                issues.append({
                    'type': 'fda_violation',
                    'severity': 'moderate',
                    'description': f'Potentially problematic claim: {description}',
                    'recommendation': 'Remove or qualify the claim with appropriate caveats'
                })

        return issues

    def _create_response(self, status: str, content: str) -> AgentResponse:
        """Helper to create response."""
        return AgentResponse(
            agent_name=self.name,
            agent_type=self.agent_type,
            status=status,
            content=content
        )
