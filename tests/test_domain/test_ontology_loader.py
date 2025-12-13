"""Test ontology loader."""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.domain.ontology_loader import OntologyLoader


@pytest.fixture
def loader():
    """Create ontology loader instance."""
    return OntologyLoader()


class TestClinicalOntology:
    """Test clinical ontology loading."""

    def test_load_clinical_ontology(self, loader):
        """Test loading clinical ontology."""
        ontology = loader.load_clinical_ontology()

        assert 'diseases' in ontology
        assert 'symptoms' in ontology
        assert 'treatments' in ontology
        assert len(ontology['diseases']) >= 5

    def test_get_disease(self, loader):
        """Test getting disease by ID."""
        disease = loader.get_disease('tuberculosis')

        assert disease is not None
        assert disease['name'] == 'Tuberculosis (TB)'
        assert 'hemoptysis' in disease['typical_symptoms']
        assert disease['category'] == 'Infectious Disease'

    def test_get_disease_not_found(self, loader):
        """Test getting non-existent disease."""
        disease = loader.get_disease('nonexistent_disease')
        assert disease is None

    def test_search_diseases_by_symptoms(self, loader):
        """Test disease search by symptoms."""
        results = loader.search_diseases_by_symptoms(
            ['chronic_cough', 'hemoptysis', 'night_sweats']
        )

        assert len(results) > 0
        # TB should be top result for these symptoms
        assert results[0]['disease']['id'] == 'tuberculosis'
        assert results[0]['relevance_score'] > 0

    def test_search_diseases_by_symptoms_no_match(self, loader):
        """Test disease search with no matching symptoms."""
        results = loader.search_diseases_by_symptoms(
            ['nonexistent_symptom_xyz']
        )
        assert len(results) == 0

    def test_get_symptom(self, loader):
        """Test getting symptom by ID."""
        symptom = loader.get_symptom('hemoptysis')

        assert symptom is not None
        assert symptom['name'] == 'Hemoptysis'
        assert symptom['red_flag'] is True

    def test_get_treatment(self, loader):
        """Test getting treatment by ID."""
        treatment = loader.get_treatment('rifampin')

        assert treatment is not None
        assert treatment['name'] == 'Rifampin'
        assert 'tuberculosis' in treatment['indications']


class TestDrugDatabase:
    """Test drug database loading."""

    def test_load_drug_database(self, loader):
        """Test loading drug database."""
        database = loader.load_drug_database()

        assert 'medications' in database
        assert 'allergy_groups' in database
        assert len(database['medications']) >= 1

    def test_check_drug_interactions(self, loader):
        """Test drug interaction checking."""
        interactions = loader.check_drug_interactions(
            'trimethoprim_sulfamethoxazole',
            ['warfarin', 'lisinopril']
        )

        assert len(interactions) >= 1
        assert interactions[0]['drug'] == 'warfarin'
        assert interactions[0]['severity'] == 'major'

    def test_check_drug_interactions_none(self, loader):
        """Test drug interaction with no interactions."""
        interactions = loader.check_drug_interactions(
            'trimethoprim_sulfamethoxazole',
            ['vitamin_c', 'multivitamin']
        )
        assert len(interactions) == 0

    def test_check_contraindications(self, loader):
        """Test contraindication checking."""
        violations = loader.check_contraindications(
            'trimethoprim_sulfamethoxazole',
            ['severe_renal_impairment'],
            []
        )

        assert len(violations) >= 1
        assert violations[0]['severity'] == 'absolute'


class TestComplianceRules:
    """Test compliance rules loading."""

    def test_load_hipaa_rules(self, loader):
        """Test loading HIPAA rules."""
        rules = loader.load_hipaa_rules()

        assert 'phi_protection' in rules
        assert 'validation_rules' in rules['phi_protection']
        assert len(rules['phi_protection']['protected_identifiers']) > 0

    def test_load_fda_guidelines(self, loader):
        """Test loading FDA guidelines."""
        guidelines = loader.load_fda_guidelines()

        assert 'clinical_decision_support' in guidelines
        assert 'disclaimers' in guidelines
        assert 'required_in_output' in guidelines['disclaimers']


class TestHelperMethods:
    """Test helper methods."""

    def test_get_all_disease_ids(self, loader):
        """Test getting all disease IDs."""
        ids = loader.get_all_disease_ids()

        assert len(ids) >= 5
        assert 'tuberculosis' in ids
        assert 'pneumonia_cap' in ids

    def test_get_all_symptom_ids(self, loader):
        """Test getting all symptom IDs."""
        ids = loader.get_all_symptom_ids()

        assert len(ids) >= 1
        assert 'hemoptysis' in ids
        assert 'chest_pain' in ids

    def test_get_red_flag_symptoms(self, loader):
        """Test getting red flag symptoms."""
        red_flags = loader.get_red_flag_symptoms()

        assert len(red_flags) >= 1
        # All returned symptoms should be red flags
        for symptom in red_flags:
            assert symptom['red_flag'] is True


class TestCaching:
    """Test caching behavior."""

    def test_ontology_cached(self, loader):
        """Test that ontology is cached."""
        # Load twice
        ontology1 = loader.load_clinical_ontology()
        ontology2 = loader.load_clinical_ontology()

        # Should be same object (cached)
        assert ontology1 is ontology2

    def test_drug_database_cached(self, loader):
        """Test that drug database is cached."""
        db1 = loader.load_drug_database()
        db2 = loader.load_drug_database()

        assert db1 is db2
