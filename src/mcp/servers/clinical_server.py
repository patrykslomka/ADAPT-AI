"""Clinical MCP Server - Provides clinical domain tools via MCP protocol.

This MCP server exposes clinical tools for:
- Patient data retrieval
- Drug interaction checking
- Clinical ontology queries
- Compliance validation

Run as: python -m src.mcp.servers.clinical_server
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

# MCP SDK imports - using the official MCP Python SDK pattern
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
        Prompt,
        PromptArgument,
        GetPromptResult,
        PromptMessage,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Fallback for when MCP SDK is not installed
    class Server:
        def __init__(self, name): self.name = name
        def tool(self): return lambda f: f
        def resource(self, uri): return lambda f: f
        def prompt(self): return lambda f: f

from src.domain.patient_handler import PatientHandler
from src.domain.ontology_loader import ontology_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize MCP server
server = Server("adapt-ai-clinical")

# Initialize domain handlers
patient_handler = PatientHandler()


# =============================================================================
# TOOLS - Actions the agent can perform
# =============================================================================

@server.tool()
async def get_patient(patient_id: str) -> Dict[str, Any]:
    """Retrieve patient data by ID.

    Args:
        patient_id: The patient identifier (e.g., P-0001)

    Returns:
        Complete patient record including demographics, history, and vitals
    """
    patient = patient_handler.get_patient(patient_id)
    if patient:
        return {
            "status": "success",
            "patient": patient
        }
    return {
        "status": "error",
        "message": f"Patient {patient_id} not found"
    }


@server.tool()
async def list_patients() -> Dict[str, Any]:
    """List all available patients with summary information.

    Returns:
        List of patient summaries with ID, name, age, chief complaint
    """
    patients = patient_handler.list_patients()
    return {
        "status": "success",
        "count": len(patients),
        "patients": patients
    }


@server.tool()
async def check_drug_interactions(
    drug_id: str,
    patient_medications: List[str]
) -> Dict[str, Any]:
    """Check for drug interactions with patient's current medications.

    Args:
        drug_id: Drug to check (e.g., DRUG-001 or drug name)
        patient_medications: List of current patient medications

    Returns:
        List of interactions found with severity and clinical effects
    """
    interactions = ontology_loader.check_drug_interactions(
        drug_id,
        patient_medications
    )

    return {
        "status": "success",
        "drug_checked": drug_id,
        "interactions_found": len(interactions),
        "interactions": interactions
    }


@server.tool()
async def check_contraindications(
    drug_id: str,
    patient_conditions: List[str],
    patient_allergies: List[str]
) -> Dict[str, Any]:
    """Check for contraindications based on patient conditions and allergies.

    Args:
        drug_id: Drug to check
        patient_conditions: Patient's medical conditions
        patient_allergies: Patient's known allergies

    Returns:
        List of contraindications with severity and recommendations
    """
    violations = ontology_loader.check_contraindications(
        drug_id,
        patient_conditions,
        patient_allergies
    )

    return {
        "status": "success",
        "drug_checked": drug_id,
        "contraindications_found": len(violations),
        "contraindications": violations
    }


@server.tool()
async def search_diseases_by_symptoms(symptoms: List[str]) -> Dict[str, Any]:
    """Find diseases that match given symptoms.

    Args:
        symptoms: List of symptom IDs or names

    Returns:
        Ranked list of matching diseases with relevance scores
    """
    results = ontology_loader.search_diseases_by_symptoms(symptoms)

    return {
        "status": "success",
        "symptoms_checked": symptoms,
        "matches_found": len(results),
        "diseases": results[:10]  # Top 10
    }


@server.tool()
async def get_disease_info(disease_id: str) -> Dict[str, Any]:
    """Get detailed information about a disease.

    Args:
        disease_id: Disease identifier

    Returns:
        Complete disease information including symptoms, treatment, guidelines
    """
    disease = ontology_loader.get_disease(disease_id)
    if disease:
        return {
            "status": "success",
            "disease": disease
        }
    return {
        "status": "error",
        "message": f"Disease {disease_id} not found"
    }


@server.tool()
async def get_treatment_info(treatment_id: str) -> Dict[str, Any]:
    """Get detailed information about a treatment/medication.

    Args:
        treatment_id: Treatment identifier

    Returns:
        Complete treatment information including dosing, contraindications
    """
    treatment = ontology_loader.get_treatment(treatment_id)
    if treatment:
        return {
            "status": "success",
            "treatment": treatment
        }
    return {
        "status": "error",
        "message": f"Treatment {treatment_id} not found"
    }


@server.tool()
async def get_red_flag_symptoms() -> Dict[str, Any]:
    """Get list of red flag symptoms requiring immediate attention.

    Returns:
        List of symptoms marked as red flags with urgency information
    """
    red_flags = ontology_loader.get_red_flag_symptoms()
    return {
        "status": "success",
        "count": len(red_flags),
        "red_flags": red_flags
    }


@server.tool()
async def validate_hipaa_compliance(
    text: str,
    patient_name: Optional[str] = None,
    patient_dob: Optional[str] = None,
    patient_mrn: Optional[str] = None
) -> Dict[str, Any]:
    """Validate text for HIPAA PHI compliance.

    Args:
        text: Text to check for PHI
        patient_name: Patient's name to check for
        patient_dob: Patient's date of birth to check for
        patient_mrn: Patient's MRN to check for

    Returns:
        Compliance status and any violations found
    """
    import re
    violations = []

    # Check for name
    if patient_name and patient_name.lower() in text.lower():
        violations.append({
            "type": "phi_name",
            "severity": "critical",
            "description": "Patient name found in text"
        })

    # Check for DOB
    if patient_dob and patient_dob in text:
        violations.append({
            "type": "phi_dob",
            "severity": "critical",
            "description": "Date of birth found in text"
        })

    # Check for MRN pattern
    mrn_pattern = r'MRN-\d{6}'
    if re.search(mrn_pattern, text):
        violations.append({
            "type": "phi_mrn",
            "severity": "critical",
            "description": "Medical record number found in text"
        })

    # Check for SSN pattern
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    if re.search(ssn_pattern, text):
        violations.append({
            "type": "phi_ssn",
            "severity": "critical",
            "description": "SSN pattern found in text"
        })

    return {
        "status": "compliant" if not violations else "non_compliant",
        "violations_found": len(violations),
        "violations": violations
    }


# =============================================================================
# RESOURCES - Data sources the agent can read
# =============================================================================

@server.resource("clinical://ontology/diseases")
async def get_diseases_resource() -> str:
    """Get all diseases from clinical ontology."""
    ontology = ontology_loader.load_clinical_ontology()
    return json.dumps(ontology.get('diseases', []), indent=2)


@server.resource("clinical://ontology/symptoms")
async def get_symptoms_resource() -> str:
    """Get all symptoms from clinical ontology."""
    ontology = ontology_loader.load_clinical_ontology()
    return json.dumps(ontology.get('symptoms', []), indent=2)


@server.resource("clinical://drugs/database")
async def get_drugs_resource() -> str:
    """Get drug database."""
    drugs = ontology_loader.load_drug_database()
    return json.dumps(drugs, indent=2)


@server.resource("clinical://compliance/hipaa")
async def get_hipaa_resource() -> str:
    """Get HIPAA rules."""
    rules = ontology_loader.load_hipaa_rules()
    return json.dumps(rules, indent=2)


@server.resource("clinical://compliance/fda")
async def get_fda_resource() -> str:
    """Get FDA guidelines."""
    guidelines = ontology_loader.load_fda_guidelines()
    return json.dumps(guidelines, indent=2)


# =============================================================================
# PROMPTS - Pre-defined prompt templates
# =============================================================================

@server.prompt()
async def clinical_assessment_prompt(
    patient_id: str,
    query: str
) -> GetPromptResult:
    """Generate a clinical assessment prompt for a patient.

    Args:
        patient_id: Patient to assess
        query: Clinical question to answer
    """
    patient = patient_handler.get_patient(patient_id)

    if not patient:
        return GetPromptResult(
            description="Error: Patient not found",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Patient {patient_id} not found."
                    )
                )
            ]
        )

    summary = patient_handler.get_patient_summary(patient_id)

    return GetPromptResult(
        description=f"Clinical assessment for patient {patient_id}",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""You are a clinical decision support assistant.

Please analyze the following patient case and answer the clinical question.

## Patient Information
{summary}

## Clinical Question
{query}

Provide a structured response including:
1. Assessment of key findings
2. Differential diagnoses (ranked by likelihood)
3. Recommended diagnostic workup
4. Important clinical considerations

Remember: This is decision support. Healthcare providers must verify all recommendations."""
                )
            )
        ]
    )


@server.prompt()
async def drug_safety_check_prompt(
    drug_name: str,
    patient_id: str
) -> GetPromptResult:
    """Generate a drug safety check prompt.

    Args:
        drug_name: Drug to check
        patient_id: Patient to check against
    """
    patient = patient_handler.get_patient(patient_id)

    if not patient:
        return GetPromptResult(
            description="Error: Patient not found",
            messages=[]
        )

    allergies = patient.get('medical_history', {}).get('allergies', [])
    conditions = patient.get('medical_history', {}).get('chronic_conditions', [])
    medications = patient.get('current_medications', [])

    return GetPromptResult(
        description=f"Drug safety check: {drug_name} for {patient_id}",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Please evaluate the safety of prescribing {drug_name} for this patient.

## Patient Allergies
{json.dumps(allergies, indent=2)}

## Chronic Conditions
{json.dumps(conditions, indent=2)}

## Current Medications
{json.dumps(medications, indent=2)}

Check for:
1. Drug allergies and cross-reactivity
2. Contraindications based on conditions
3. Drug-drug interactions
4. Dosing considerations

Provide a clear recommendation with reasoning."""
                )
            )
        ]
    )


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

async def main():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        logger.error("MCP SDK not installed. Run: pip install mcp")
        return

    logger.info("Starting ADAPT-AI Clinical MCP Server...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
