"""Patient data handler.

Healthcare-specific reference implementation of the generic ``subject_id`` hook.
The pipeline itself is domain-agnostic and only ever sees an opaque ``subject_id``
(see ``QueryRequest.subject_id``); this handler is one domain's realisation of that
hook, modelling a clinical patient (vitals, allergies, medications, history). Legal
and finance have no equivalent subject store today - adding one means a sibling
handler, not a change to the agents/orchestrator.
"""
import json
from pathlib import Path
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class PatientHandler:
    """Manage synthetic patient records."""

    def __init__(self, patients_file: Path = None):
        if patients_file is None:
            patients_file = Path(__file__).parent / "synthetic_patients" / "patients.json"
        self.patients_file = patients_file
        self._patients_cache: Optional[Dict] = None

    def load_patients(self) -> Dict:
        """Load all patients from file."""
        if self._patients_cache is None:
            try:
                with open(self.patients_file, 'r') as f:
                    data = json.load(f)
                    self._patients_cache = {
                        p['demographics']['patient_id']: p
                        for p in data['patients']
                    }
                logger.info(f"Loaded {len(self._patients_cache)} patients")
            except FileNotFoundError:
                logger.error(f"Patients file not found: {self.patients_file}")
                self._patients_cache = {}
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in patients file: {e}")
                self._patients_cache = {}

        return self._patients_cache

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Get patient by ID."""
        patients = self.load_patients()
        patient = patients.get(patient_id)

        if patient:
            logger.info(f"Retrieved patient: {patient_id}")
        else:
            logger.warning(f"Patient not found: {patient_id}")

        return patient

    def list_patients(self) -> List[Dict]:
        """List all patients (summary view)."""
        patients = self.load_patients()

        summaries = []
        for patient_id, patient in patients.items():
            demographics = patient.get('demographics', {})
            complaint = patient.get('presenting_complaint', {})

            summaries.append({
                'patient_id': patient_id,
                'name': demographics.get('name', 'Unknown'),
                'age': demographics.get('age'),
                'gender': demographics.get('gender'),
                'chief_complaint': complaint.get('chief_complaint', 'N/A'),
                'suspected_condition': patient.get('suspected_condition', 'Unknown')
            })

        return summaries

    def get_patient_summary(self, patient_id: str) -> Optional[str]:
        """Get a text summary of a patient."""
        patient = self.get_patient(patient_id)
        if not patient:
            return None

        demo = patient.get('demographics', {})
        history = patient.get('medical_history', {})
        complaint = patient.get('presenting_complaint', {})
        vitals = patient.get('vital_signs', {})

        summary_parts = [
            f"Patient: {demo.get('patient_id', 'Unknown')}",
            f"Age: {demo.get('age', 'Unknown')} | Gender: {demo.get('gender', 'Unknown')}",
            "",
            f"Chief Complaint: {complaint.get('chief_complaint', 'N/A')}",
            f"HPI: {complaint.get('hpi', 'N/A')}",
            ""
        ]

        # Chronic conditions
        conditions = history.get('chronic_conditions', [])
        if conditions:
            summary_parts.append(f"Chronic Conditions: {', '.join(conditions)}")

        # Allergies
        allergies = history.get('allergies', [])
        if allergies:
            allergy_list = [f"{a['substance']} ({a['reaction']})" for a in allergies]
            summary_parts.append(f"ALLERGIES: {', '.join(allergy_list)}")

        # Current medications
        meds = patient.get('current_medications', [])
        if meds:
            med_list = [f"{m['name']} {m['dose']}" for m in meds]
            summary_parts.append(f"Current Medications: {', '.join(med_list)}")

        # Vital signs
        if vitals:
            vital_str = (
                f"Vitals: T {vitals.get('temperature', 'N/A')}F, "
                f"HR {vitals.get('heart_rate', 'N/A')}, "
                f"BP {vitals.get('blood_pressure_systolic', 'N/A')}/{vitals.get('blood_pressure_diastolic', 'N/A')}, "
                f"RR {vitals.get('respiratory_rate', 'N/A')}, "
                f"SpO2 {vitals.get('oxygen_saturation', 'N/A')}%"
            )
            summary_parts.append(vital_str)

        return "\n".join(summary_parts)

    def search_patients_by_condition(self, condition: str) -> List[Dict]:
        """Search patients by suspected condition."""
        patients = self.load_patients()
        matching = []

        condition_lower = condition.lower()
        for patient_id, patient in patients.items():
            suspected = patient.get('suspected_condition', '').lower()
            if condition_lower in suspected:
                matching.append({
                    'patient_id': patient_id,
                    'name': patient.get('demographics', {}).get('name', 'Unknown'),
                    'suspected_condition': patient.get('suspected_condition')
                })

        return matching

    def get_patients_with_allergy(self, allergy: str) -> List[Dict]:
        """Find patients with a specific allergy."""
        patients = self.load_patients()
        matching = []

        allergy_lower = allergy.lower()
        for patient_id, patient in patients.items():
            allergies = patient.get('medical_history', {}).get('allergies', [])
            for a in allergies:
                if allergy_lower in a.get('substance', '').lower():
                    matching.append({
                        'patient_id': patient_id,
                        'name': patient.get('demographics', {}).get('name', 'Unknown'),
                        'allergy': a
                    })
                    break

        return matching

    def refresh_cache(self):
        """Force reload of patient data."""
        self._patients_cache = None
        self.load_patients()
        logger.info("Patient cache refreshed")


# Global instance
patient_handler = PatientHandler()
