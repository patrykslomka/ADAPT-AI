#!/usr/bin/env python3
"""Generate synthetic patient data for ADAPT-AI."""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from faker import Faker

# Initialize Faker
fake = Faker()
Faker.seed(42)
random.seed(42)

# Clinical data templates
CHRONIC_CONDITIONS = [
    "Hypertension",
    "Type 2 Diabetes",
    "Hyperlipidemia",
    "COPD",
    "Asthma",
    "Heart Failure",
    "Atrial Fibrillation",
    "Chronic Kidney Disease",
    "Osteoarthritis",
    "Depression",
    "Anxiety",
    "GERD",
    "Hypothyroidism",
    "Obesity"
]

ALLERGIES = [
    {"substance": "Penicillin", "reaction": "rash", "severity": "moderate"},
    {"substance": "Sulfa", "reaction": "anaphylaxis", "severity": "severe"},
    {"substance": "Aspirin", "reaction": "GI upset", "severity": "mild"},
    {"substance": "Codeine", "reaction": "nausea", "severity": "mild"},
    {"substance": "Latex", "reaction": "hives", "severity": "moderate"},
    {"substance": "Iodine contrast", "reaction": "rash", "severity": "moderate"},
    {"substance": "NSAIDs", "reaction": "bronchospasm", "severity": "moderate"},
    {"substance": "ACE inhibitors", "reaction": "angioedema", "severity": "severe"},
    {"substance": "Shellfish", "reaction": "anaphylaxis", "severity": "severe"},
    {"substance": "Eggs", "reaction": "hives", "severity": "mild"}
]

COMMON_MEDICATIONS = [
    {"name": "Lisinopril", "dose": "10mg", "frequency": "daily", "indication": "Hypertension"},
    {"name": "Metformin", "dose": "500mg", "frequency": "BID", "indication": "Diabetes"},
    {"name": "Atorvastatin", "dose": "20mg", "frequency": "daily", "indication": "Hyperlipidemia"},
    {"name": "Metoprolol", "dose": "25mg", "frequency": "BID", "indication": "Hypertension"},
    {"name": "Omeprazole", "dose": "20mg", "frequency": "daily", "indication": "GERD"},
    {"name": "Amlodipine", "dose": "5mg", "frequency": "daily", "indication": "Hypertension"},
    {"name": "Levothyroxine", "dose": "50mcg", "frequency": "daily", "indication": "Hypothyroidism"},
    {"name": "Albuterol inhaler", "dose": "2 puffs", "frequency": "PRN", "indication": "Asthma"},
    {"name": "Warfarin", "dose": "5mg", "frequency": "daily", "indication": "Atrial Fibrillation"},
    {"name": "Aspirin", "dose": "81mg", "frequency": "daily", "indication": "CAD prevention"},
    {"name": "Gabapentin", "dose": "300mg", "frequency": "TID", "indication": "Neuropathy"},
    {"name": "Sertraline", "dose": "50mg", "frequency": "daily", "indication": "Depression"}
]

PRESENTING_SCENARIOS = [
    {
        "chief_complaint": "Persistent cough and night sweats",
        "hpi": "Patient reports 3 weeks of productive cough with occasional blood-tinged sputum, drenching night sweats, and unintentional weight loss of 10 pounds. Recent travel to Southeast Asia 2 months ago.",
        "symptoms": ["chronic_cough", "hemoptysis", "night_sweats", "weight_loss", "fever"],
        "risk_factors": ["endemic_travel"],
        "suspected_condition": "tuberculosis"
    },
    {
        "chief_complaint": "Sudden onset chest pain",
        "hpi": "60-year-old presents with crushing substernal chest pain radiating to left arm, started 45 minutes ago. Associated with diaphoresis and shortness of breath. History of HTN and diabetes.",
        "symptoms": ["chest_pain", "diaphoresis", "dyspnea", "radiation_to_arm"],
        "risk_factors": ["hypertension", "diabetes"],
        "suspected_condition": "myocardial_infarction_stemi"
    },
    {
        "chief_complaint": "Cough, fever, and shortness of breath",
        "hpi": "Patient reports 5 days of productive cough with green sputum, fever up to 102°F, and progressive dyspnea. No recent hospitalizations. Lives independently.",
        "symptoms": ["cough", "fever", "dyspnea", "sputum_production"],
        "risk_factors": ["age_over_65", "chronic_lung_disease"],
        "suspected_condition": "pneumonia_cap"
    },
    {
        "chief_complaint": "Nausea, vomiting, and confusion",
        "hpi": "Type 1 diabetic presents with 2 days of nausea, vomiting, abdominal pain, and increasing confusion. Admits to missing insulin doses due to running out of supplies. Last glucose reading at home was 'HIGH'.",
        "symptoms": ["nausea", "vomiting", "abdominal_pain", "altered_mental_status"],
        "risk_factors": ["type_1_diabetes", "insulin_noncompliance"],
        "suspected_condition": "diabetic_ketoacidosis"
    },
    {
        "chief_complaint": "Sudden weakness and speech difficulty",
        "hpi": "68-year-old brought by family with sudden onset right-sided weakness and slurred speech, noticed 90 minutes ago. Patient has history of atrial fibrillation, not compliant with anticoagulation.",
        "symptoms": ["arm_weakness", "speech_difficulty", "facial_droop"],
        "risk_factors": ["atrial_fibrillation", "hypertension"],
        "suspected_condition": "stroke_ischemic"
    },
    {
        "chief_complaint": "Fatigue and weight loss",
        "hpi": "Patient reports 2 months of progressive fatigue, unintentional 15-pound weight loss, and intermittent low-grade fevers. No cough or respiratory symptoms. Has noticed enlarged lymph nodes in neck.",
        "symptoms": ["fatigue", "weight_loss", "fever"],
        "risk_factors": [],
        "suspected_condition": "malignancy_workup"
    },
    {
        "chief_complaint": "Severe headache and neck stiffness",
        "hpi": "Young adult presents with worst headache of life, sudden onset 4 hours ago. Associated with neck stiffness, photophobia, and one episode of vomiting. No fever.",
        "symptoms": ["severe_headache", "neck_stiffness"],
        "risk_factors": [],
        "suspected_condition": "subarachnoid_hemorrhage_rule_out"
    },
    {
        "chief_complaint": "Leg swelling and shortness of breath",
        "hpi": "Patient with recent 6-hour flight presents with unilateral left leg swelling for 3 days and new onset shortness of breath today. Denies chest pain. Taking oral contraceptives.",
        "symptoms": ["leg_swelling", "dyspnea"],
        "risk_factors": ["recent_travel", "oral_contraceptives"],
        "suspected_condition": "dvt_pe_rule_out"
    },
    {
        "chief_complaint": "Burning with urination",
        "hpi": "Female patient reports 3 days of dysuria, urinary frequency, and suprapubic discomfort. No fever, flank pain, or vaginal discharge. Sexually active with new partner.",
        "symptoms": ["dysuria", "urinary_frequency"],
        "risk_factors": ["female", "sexually_active"],
        "suspected_condition": "urinary_tract_infection"
    },
    {
        "chief_complaint": "Epigastric pain after meals",
        "hpi": "Patient reports 3 weeks of burning epigastric pain that worsens after meals, especially spicy foods. Associated with occasional nausea. Takes NSAIDs regularly for arthritis. No melena or hematemesis.",
        "symptoms": ["epigastric_pain", "nausea"],
        "risk_factors": ["nsaid_use"],
        "suspected_condition": "peptic_ulcer_disease"
    }
]

SURGICAL_HISTORY = [
    "Appendectomy (2010)",
    "Cholecystectomy (2015)",
    "Cesarean section (2012)",
    "Total knee replacement (2019)",
    "CABG x3 (2018)",
    "Hysterectomy (2016)",
    "Hernia repair (2014)",
    "Cataract surgery (2020)",
    "Colonoscopy with polypectomy (2021)",
    "Tonsillectomy (childhood)"
]

SOCIAL_HISTORY_OPTIONS = {
    "smoking": ["Never smoker", "Former smoker (quit 5 years ago)", "Current smoker (1 pack/day)", "Current smoker (half pack/day)"],
    "alcohol": ["None", "Social (1-2 drinks/week)", "Moderate (1 drink/day)", "Heavy (>2 drinks/day)"],
    "drugs": ["Denies illicit drug use", "History of marijuana use", "Remote history of IV drug use (20 years ago)"],
    "occupation": ["Retired", "Office worker", "Healthcare worker", "Construction worker", "Teacher", "Unemployed"],
    "living_situation": ["Lives alone", "Lives with spouse", "Lives with family", "Assisted living facility", "Homeless"]
}


def generate_patient(patient_num: int) -> Dict[str, Any]:
    """Generate a single synthetic patient record."""

    # Demographics
    gender = random.choice(["Male", "Female"])
    age = random.randint(25, 85)

    demographics = {
        "patient_id": f"P-{patient_num:04d}",
        "name": fake.name_male() if gender == "Male" else fake.name_female(),
        "age": age,
        "gender": gender,
        "date_of_birth": (datetime.now() - timedelta(days=age*365)).strftime("%Y-%m-%d"),
        "mrn": f"MRN-{random.randint(100000, 999999)}"
    }

    # Medical history - more conditions for older patients
    num_conditions = min(random.randint(0, 3) + (age // 30), 5)
    chronic_conditions = random.sample(CHRONIC_CONDITIONS, num_conditions) if num_conditions > 0 else []

    # Allergies - some patients have none
    num_allergies = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
    allergies = random.sample(ALLERGIES, num_allergies) if num_allergies > 0 else []

    # Surgical history
    num_surgeries = random.choices([0, 1, 2, 3], weights=[0.3, 0.3, 0.25, 0.15])[0]
    surgeries = random.sample(SURGICAL_HISTORY, num_surgeries) if num_surgeries > 0 else []

    medical_history = {
        "chronic_conditions": chronic_conditions,
        "allergies": allergies,
        "surgical_history": surgeries,
        "family_history": generate_family_history()
    }

    # Current medications based on conditions
    current_medications = []
    for condition in chronic_conditions:
        matching_meds = [m for m in COMMON_MEDICATIONS if condition.lower() in m["indication"].lower()]
        if matching_meds:
            current_medications.append(random.choice(matching_meds))

    # Limit medications
    current_medications = current_medications[:5]

    # Social history
    social_history = {
        "smoking": random.choice(SOCIAL_HISTORY_OPTIONS["smoking"]),
        "alcohol": random.choice(SOCIAL_HISTORY_OPTIONS["alcohol"]),
        "drugs": random.choice(SOCIAL_HISTORY_OPTIONS["drugs"]),
        "occupation": random.choice(SOCIAL_HISTORY_OPTIONS["occupation"]),
        "living_situation": random.choice(SOCIAL_HISTORY_OPTIONS["living_situation"])
    }

    # Select presenting scenario
    scenario = random.choice(PRESENTING_SCENARIOS)

    presenting_complaint = {
        "chief_complaint": scenario["chief_complaint"],
        "hpi": scenario["hpi"],
        "symptoms": scenario["symptoms"],
        "duration": random.choice(["hours", "days", "weeks"]),
        "onset": random.choice(["sudden", "gradual"]),
        "severity": random.choice(["mild", "moderate", "severe"])
    }

    # Vital signs
    vital_signs = generate_vital_signs(scenario.get("suspected_condition", ""), age)

    # Recent lab results (if applicable)
    recent_labs = generate_labs(chronic_conditions, scenario.get("suspected_condition", ""))

    return {
        "demographics": demographics,
        "medical_history": medical_history,
        "current_medications": current_medications,
        "social_history": social_history,
        "presenting_complaint": presenting_complaint,
        "vital_signs": vital_signs,
        "recent_labs": recent_labs,
        "risk_factors": scenario.get("risk_factors", []),
        "suspected_condition": scenario.get("suspected_condition", "unknown"),
        "created_at": datetime.now().isoformat()
    }


def generate_family_history() -> List[str]:
    """Generate family history."""
    options = [
        "Father: MI at age 55",
        "Mother: Type 2 Diabetes",
        "Mother: Breast cancer",
        "Father: Colon cancer",
        "Sister: Hypertension",
        "Brother: CAD",
        "Paternal grandfather: Stroke",
        "No significant family history"
    ]
    return random.sample(options, random.randint(1, 3))


def generate_vital_signs(condition: str, age: int) -> Dict[str, Any]:
    """Generate vital signs based on condition."""
    # Base vitals
    vitals = {
        "temperature": round(random.uniform(97.5, 98.9), 1),
        "heart_rate": random.randint(60, 90),
        "blood_pressure_systolic": random.randint(110, 140),
        "blood_pressure_diastolic": random.randint(70, 90),
        "respiratory_rate": random.randint(12, 18),
        "oxygen_saturation": random.randint(95, 100),
        "recorded_at": datetime.now().isoformat()
    }

    # Adjust based on condition
    if condition == "tuberculosis":
        vitals["temperature"] = round(random.uniform(99.5, 102.0), 1)
    elif condition == "myocardial_infarction_stemi":
        vitals["heart_rate"] = random.randint(90, 120)
        vitals["blood_pressure_systolic"] = random.randint(90, 160)
    elif condition == "pneumonia_cap":
        vitals["temperature"] = round(random.uniform(100.0, 103.0), 1)
        vitals["respiratory_rate"] = random.randint(20, 28)
        vitals["oxygen_saturation"] = random.randint(88, 94)
    elif condition == "diabetic_ketoacidosis":
        vitals["respiratory_rate"] = random.randint(22, 32)  # Kussmaul
        vitals["heart_rate"] = random.randint(100, 130)
    elif condition == "stroke_ischemic":
        vitals["blood_pressure_systolic"] = random.randint(160, 200)

    return vitals


def generate_labs(conditions: List[str], suspected: str) -> Dict[str, Any]:
    """Generate relevant lab values."""
    labs = {
        "recorded_at": (datetime.now() - timedelta(hours=random.randint(1, 24))).isoformat()
    }

    # Basic metabolic panel
    if "Diabetes" in str(conditions) or suspected == "diabetic_ketoacidosis":
        labs["glucose"] = random.randint(250, 500) if suspected == "diabetic_ketoacidosis" else random.randint(100, 180)
        labs["hba1c"] = round(random.uniform(7.0, 10.0), 1)

    if "Chronic Kidney Disease" in conditions:
        labs["creatinine"] = round(random.uniform(1.5, 3.0), 1)
        labs["bun"] = random.randint(25, 50)
    else:
        labs["creatinine"] = round(random.uniform(0.8, 1.2), 1)
        labs["bun"] = random.randint(10, 20)

    # CBC
    labs["wbc"] = round(random.uniform(4.5, 11.0), 1)
    if suspected in ["pneumonia_cap", "tuberculosis"]:
        labs["wbc"] = round(random.uniform(12.0, 20.0), 1)

    labs["hemoglobin"] = round(random.uniform(12.0, 16.0), 1)
    labs["platelets"] = random.randint(150, 350)

    # Cardiac markers for chest pain
    if suspected == "myocardial_infarction_stemi":
        labs["troponin"] = round(random.uniform(0.5, 10.0), 2)

    return labs


def generate_all_patients(num_patients: int = 20) -> Dict[str, Any]:
    """Generate all synthetic patients."""
    patients = []

    for i in range(1, num_patients + 1):
        patient = generate_patient(i)
        patients.append(patient)
        print(f"Generated patient P-{i:04d}: {patient['presenting_complaint']['chief_complaint']}")

    return {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "total_patients": num_patients,
        "patients": patients
    }


def main():
    """Main function to generate and save patients."""
    print("🏥 Generating synthetic patient data for ADAPT-AI...\n")

    # Generate patients
    data = generate_all_patients(20)

    # Save to file
    output_dir = Path(__file__).parent.parent / "adapt_ai" / "domain" / "synthetic_patients"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "patients.json"

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Generated {data['total_patients']} patients")
    print(f"📁 Saved to: {output_file}")

    # Print summary
    print("\n📊 Patient Summary:")
    conditions = {}
    for patient in data['patients']:
        cond = patient.get('suspected_condition', 'unknown')
        conditions[cond] = conditions.get(cond, 0) + 1

    for cond, count in sorted(conditions.items(), key=lambda x: -x[1]):
        print(f"   - {cond}: {count} patients")


if __name__ == "__main__":
    main()
