"""Quality Agent - Hallucination detection and accuracy validation."""
from typing import Dict, Any, Optional, List
from anthropic import Anthropic
from config.settings import settings
from src.agents.base_agent import BaseAgent, AgentResponse
from src.domain.ontology_loader import ontology_loader
import re
import json
import logging

logger = logging.getLogger(__name__)


class QualityAgent(BaseAgent):
    """Quality assurance and hallucination detection."""

    SYSTEM_PROMPT = """You verify accuracy and detect hallucinations in clinical responses.

Check for:
1. Factual accuracy against medical knowledge
2. Hallucinated drug names or treatments
3. Incorrect dosages or protocols
4. Unsupported clinical claims
5. Coherence of reasoning

Flag inaccuracies and assign confidence scores."""

    def __init__(self):
        super().__init__(
            name="quality_agent",
            agent_type="validator",
            system_prompt=self.SYSTEM_PROMPT
        )

        self.client = Anthropic(
            api_key=settings.anthropic_api_key.get_secret_value()
        )
        self.ontology = ontology_loader

    async def process(
        self,
        query: str,
        context: Dict[str, Any],
        previous_responses: Optional[List[AgentResponse]] = None
    ) -> AgentResponse:
        """Validate quality of primary agent response."""

        if not previous_responses:
            return self._create_response('approved', 'No response to validate', 1.0)

        primary_response = previous_responses[0]

        issues = []

        # 1. Check for hallucinated medications
        drug_issues = self._check_medications(primary_response.content)
        issues.extend(drug_issues)

        # 2. Validate against clinical ontology
        clinical_issues = self._validate_clinical_accuracy(
            primary_response.content,
            context
        )
        issues.extend(clinical_issues)

        # 3. Use LLM to detect subtle hallucinations
        llm_check = self._llm_hallucination_check(
            query,
            primary_response.content,
            context.get('retrieved_documents', '')
        )
        issues.extend(llm_check['issues'])

        # Calculate scores
        hallucination_score = min(len(issues) * 0.15, 1.0)  # Cap at 1.0
        confidence_score = max(0.0, 1.0 - hallucination_score)

        # Determine status
        critical_issues = [i for i in issues if i.get('severity') == 'critical']
        high_issues = [i for i in issues if i.get('severity') == 'high']

        if critical_issues:
            status = 'rejected'
            content = f"QUALITY FAILURE: {len(critical_issues)} critical inaccuracies detected."
        elif len(high_issues) >= 2:
            status = 'rejected'
            content = f"QUALITY FAILURE: Multiple high-severity issues detected ({len(high_issues)})."
        elif issues:
            status = 'warning'
            content = f"Quality concerns: {len(issues)} potential issues."
        else:
            status = 'approved'
            content = "Quality checks passed."

        return AgentResponse(
            agent_name=self.name,
            agent_type=self.agent_type,
            status=status,
            content=content,
            confidence_score=confidence_score,
            issues_found=issues,
            metadata={
                'hallucination_score': hallucination_score,
                'checks_performed': ['medication_validation', 'ontology_check', 'llm_verification']
            }
        )

    def _check_medications(self, content: str) -> List[Dict]:
        """Check if mentioned medications exist."""
        issues = []

        # Load drug database
        drug_db = self.ontology.load_drug_database()
        valid_drugs = set()

        for med in drug_db['medications']:
            valid_drugs.add(med['generic_name'].lower())
            valid_drugs.update([b.lower() for b in med.get('brand_names', [])])

        # Expanded list of valid common drugs (extend the ontology)
        common_valid_drugs = {
            'aspirin', 'ibuprofen', 'acetaminophen', 'tylenol', 'advil',
            'metformin', 'lisinopril', 'amlodipine', 'metoprolol', 'atorvastatin',
            'omeprazole', 'losartan', 'gabapentin', 'sertraline', 'levothyroxine',
            'hydrochlorothiazide', 'furosemide', 'prednisone', 'azithromycin',
            'amoxicillin', 'ciprofloxacin', 'doxycycline', 'ceftriaxone',
            'vancomycin', 'piperacillin', 'tazobactam', 'meropenem',
            'heparin', 'warfarin', 'enoxaparin', 'apixaban', 'rivaroxaban',
            'insulin', 'nitroglycerin', 'morphine', 'fentanyl', 'ketamine',
            'propofol', 'midazolam', 'lorazepam', 'diazepam', 'haloperidol',
            'ondansetron', 'diphenhydramine', 'epinephrine', 'norepinephrine',
            'dopamine', 'dobutamine', 'vasopressin', 'phenylephrine'
        }
        valid_drugs.update(common_valid_drugs)

        # Look for potential drug names (words ending in common drug suffixes)
        drug_patterns = [
            r'\b[A-Z][a-z]+(?:mycin|cillin|prazole|olol|pine|statin|pril|sartan|mab|nib)\b',
            r'\b[A-Z][a-z]+(?:ine|ide|ate|one|ol)\b'
        ]

        potential_drugs = set()
        for pattern in drug_patterns:
            matches = re.findall(pattern, content)
            potential_drugs.update(matches)

        for drug in potential_drugs:
            drug_lower = drug.lower()
            if drug_lower not in valid_drugs:
                # Check if it's a common word that's not a drug
                common_non_drugs = {
                    'medicine', 'routine', 'antine', 'antine', 'antine',
                    'antine', 'antine', 'antine', 'antine', 'antine'
                }
                if drug_lower not in common_non_drugs:
                    issues.append({
                        'type': 'potential_hallucination',
                        'severity': 'moderate',
                        'description': f'Mentioned drug "{drug}" not found in knowledge base',
                        'recommendation': 'Verify drug name accuracy'
                    })

        return issues

    def _validate_clinical_accuracy(self, content: str, context: Dict) -> List[Dict]:
        """Validate against clinical ontology."""
        issues = []

        # Load ontology
        ontology = self.ontology.load_clinical_ontology()
        valid_diseases = {d['name'].lower() for d in ontology['diseases']}

        content_lower = content.lower()

        # Check for common hallucination patterns
        hallucination_patterns = [
            (r'(cure|cures|heals|eliminates)\s+(diabetes|hypertension|copd|asthma|heart failure)',
             'Claiming cure for chronic disease'),
            (r'100%\s+(effective|accurate|safe|success)',
             'Absolute efficacy claim'),
            (r'always\s+(works|effective|safe|successful)',
             'Absolute certainty claim'),
            (r'never\s+(fails|causes|has side effects)',
             'Absolute negative claim'),
            (r'guaranteed\s+(to work|results|cure)',
             'Guaranteed outcome claim')
        ]

        for pattern, description in hallucination_patterns:
            if re.search(pattern, content_lower):
                issues.append({
                    'type': 'overclaim',
                    'severity': 'high',
                    'description': f'Potentially inaccurate claim: {description}',
                    'pattern': pattern
                })

        # Check for inconsistent reasoning
        if 'definitely' in content_lower and 'possibly' in content_lower:
            if content_lower.find('definitely') < content_lower.find('possibly'):
                pass  # Legitimate if saying "definitely X, but possibly Y"

        return issues

    def _llm_hallucination_check(
        self,
        query: str,
        response_content: str,
        reference_docs: str
    ) -> Dict:
        """Use LLM to detect hallucinations."""

        # Only perform LLM check if we have reference documents
        if not reference_docs or len(reference_docs) < 100:
            return {'issues': []}

        prompt = f"""You are a medical fact-checker. Compare the clinical response against reference documentation.

**Reference Medical Knowledge:**
{reference_docs[:3000]}

**Clinical Response to Verify:**
{response_content[:2000]}

Identify any statements in the response that:
1. Contradict the reference materials
2. Make unsupported claims not in the references
3. Include potentially fabricated medical information

Respond ONLY in this exact JSON format (no other text):
{{
  "issues": [
    {{
      "text": "specific problematic text",
      "reason": "why it's problematic",
      "severity": "high"
    }}
  ],
  "overall_assessment": "brief assessment"
}}

If there are no issues, return: {{"issues": [], "overall_assessment": "No issues found"}}"""

        try:
            check_response = self.client.messages.create(
                model=settings.model_name,
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = check_response.content[0].text.strip()

            # Try to extract JSON from response
            # Handle case where model adds extra text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {'issues': []}

            # Convert to standard format
            issues = []
            for issue in result.get('issues', []):
                if issue.get('text') and issue.get('reason'):
                    issues.append({
                        'type': 'llm_detected_hallucination',
                        'severity': issue.get('severity', 'moderate'),
                        'description': issue.get('reason', ''),
                        'problematic_text': issue.get('text', '')
                    })

            return {'issues': issues}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM hallucination check response: {e}")
            return {'issues': []}
        except Exception as e:
            logger.error(f"LLM hallucination check failed: {e}")
            return {'issues': []}

    def _create_response(self, status: str, content: str, confidence: float) -> AgentResponse:
        """Helper to create response."""
        return AgentResponse(
            agent_name=self.name,
            agent_type=self.agent_type,
            status=status,
            content=content,
            confidence_score=confidence
        )
