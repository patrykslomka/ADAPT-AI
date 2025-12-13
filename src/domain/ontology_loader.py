"""Ontology and domain knowledge loader."""
import json
from pathlib import Path
from typing import Dict, List, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class OntologyLoader:
    """Load and manage domain ontologies with caching."""

    def __init__(self, base_path: Path = None):
        if base_path is None:
            # Get the directory where this module is located
            base_path = Path(__file__).parent
        self.base_path = base_path
        self.ontology_path = base_path / "ontologies"
        self.compliance_path = base_path / "compliance"

    @lru_cache(maxsize=1)
    def load_clinical_ontology(self) -> Dict:
        """Load clinical ontology with caching.

        Returns:
            Dict containing diseases, symptoms, treatments
        """
        path = self.ontology_path / "clinical_ontology.json"

        with open(path, 'r') as f:
            ontology = json.load(f)

        logger.info(
            f"Loaded clinical ontology: "
            f"{len(ontology['diseases'])} diseases, "
            f"{len(ontology['symptoms'])} symptoms, "
            f"{len(ontology['treatments'])} treatments"
        )

        return ontology

    @lru_cache(maxsize=1)
    def load_drug_database(self) -> Dict:
        """Load drug interaction database.

        Returns:
            Dict containing medications and allergy groups
        """
        path = self.ontology_path / "drug_database.json"

        with open(path, 'r') as f:
            database = json.load(f)

        logger.info(
            f"Loaded drug database: "
            f"{len(database['medications'])} medications, "
            f"{len(database['allergy_groups'])} allergy groups"
        )

        return database

    @lru_cache(maxsize=1)
    def load_hipaa_rules(self) -> Dict:
        """Load HIPAA compliance rules."""
        path = self.compliance_path / "hipaa_rules.json"

        with open(path, 'r') as f:
            rules = json.load(f)

        logger.info("Loaded HIPAA compliance rules")
        return rules

    @lru_cache(maxsize=1)
    def load_fda_guidelines(self) -> Dict:
        """Load FDA guidelines."""
        path = self.compliance_path / "fda_guidelines.json"

        with open(path, 'r') as f:
            guidelines = json.load(f)

        logger.info("Loaded FDA guidelines")
        return guidelines

    def get_disease(self, disease_id: str) -> Optional[Dict]:
        """Get disease information by ID.

        Args:
            disease_id: Disease identifier

        Returns:
            Disease dict or None if not found
        """
        ontology = self.load_clinical_ontology()

        for disease in ontology['diseases']:
            if disease['id'] == disease_id:
                return disease

        logger.warning(f"Disease not found: {disease_id}")
        return None

    def get_symptom(self, symptom_id: str) -> Optional[Dict]:
        """Get symptom information by ID.

        Args:
            symptom_id: Symptom identifier

        Returns:
            Symptom dict or None if not found
        """
        ontology = self.load_clinical_ontology()

        for symptom in ontology['symptoms']:
            if symptom['id'] == symptom_id:
                return symptom

        logger.warning(f"Symptom not found: {symptom_id}")
        return None

    def get_treatment(self, treatment_id: str) -> Optional[Dict]:
        """Get treatment information by ID.

        Args:
            treatment_id: Treatment identifier

        Returns:
            Treatment dict or None if not found
        """
        ontology = self.load_clinical_ontology()

        for treatment in ontology['treatments']:
            if treatment['id'] == treatment_id:
                return treatment

        logger.warning(f"Treatment not found: {treatment_id}")
        return None

    def search_diseases_by_symptoms(self, symptoms: List[str]) -> List[Dict]:
        """Find diseases associated with given symptoms.

        Args:
            symptoms: List of symptom IDs

        Returns:
            List of matching diseases with relevance scores
        """
        ontology = self.load_clinical_ontology()

        results = []
        for disease in ontology['diseases']:
            matching_symptoms = set(symptoms) & set(disease.get('typical_symptoms', []))

            if matching_symptoms:
                relevance = len(matching_symptoms) / len(symptoms)
                results.append({
                    'disease': disease,
                    'matching_symptoms': list(matching_symptoms),
                    'relevance_score': relevance
                })

        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)

        logger.info(
            f"Found {len(results)} diseases matching "
            f"{len(symptoms)} symptoms"
        )

        return results

    def check_drug_interactions(
        self,
        drug_id: str,
        patient_medications: List[str]
    ) -> List[Dict]:
        """Check for drug interactions.

        Args:
            drug_id: Drug to check
            patient_medications: Current patient medications

        Returns:
            List of interactions found
        """
        database = self.load_drug_database()

        # Find drug
        drug = None
        for med in database['medications']:
            if med['id'] == drug_id:
                drug = med
                break

        if not drug:
            logger.warning(f"Drug not found: {drug_id}")
            return []

        # Check interactions
        interactions = []
        for interaction in drug.get('drug_interactions', []):
            if interaction['drug'] in patient_medications:
                interactions.append(interaction)

        if interactions:
            logger.warning(
                f"Found {len(interactions)} drug interactions "
                f"for {drug_id}"
            )

        return interactions

    def check_contraindications(
        self,
        drug_id: str,
        patient_conditions: List[str],
        patient_allergies: List[str]
    ) -> List[Dict]:
        """Check for contraindications.

        Args:
            drug_id: Drug to check
            patient_conditions: Patient's conditions
            patient_allergies: Patient's allergies

        Returns:
            List of contraindications found
        """
        database = self.load_drug_database()

        # Find drug
        drug = None
        for med in database['medications']:
            if med['id'] == drug_id:
                drug = med
                break

        if not drug:
            return []

        violations = []

        # Check contraindications
        for contraindication in drug.get('contraindications', []):
            condition = contraindication['condition']

            # Check allergy groups
            if 'allergy' in condition:
                allergy_type = condition.replace('_allergy', '')
                for allergy_group in database.get('allergy_groups', []):
                    if allergy_type == allergy_group['group_name']:
                        # Check if patient has any allergy in this group
                        for patient_allergy in patient_allergies:
                            if patient_allergy.lower() in [
                                drug.lower() for drug in allergy_group['includes']
                            ]:
                                violations.append(contraindication)
                                break

            # Check conditions
            if condition in patient_conditions:
                violations.append(contraindication)

        if violations:
            logger.warning(
                f"Found {len(violations)} contraindications "
                f"for {drug_id}"
            )

        return violations

    def get_all_disease_ids(self) -> List[str]:
        """Get all disease IDs."""
        ontology = self.load_clinical_ontology()
        return [d['id'] for d in ontology['diseases']]

    def get_all_symptom_ids(self) -> List[str]:
        """Get all symptom IDs."""
        ontology = self.load_clinical_ontology()
        return [s['id'] for s in ontology['symptoms']]

    def get_red_flag_symptoms(self) -> List[Dict]:
        """Get all red flag symptoms that require immediate attention."""
        ontology = self.load_clinical_ontology()
        return [s for s in ontology['symptoms'] if s.get('red_flag', False)]


# Global instance
ontology_loader = OntologyLoader()
