"""Patient selector component."""
import streamlit as st
from typing import Optional

def render_patient_selector(patient_handler) -> Optional[str]:
    """Render patient selector dropdown with detailed preview."""
    # Get patient list
    patients = patient_handler.list_patients()
    
    if not patients:
        st.warning("No patients found in database.")
        return None

    # Create options
    options = ["No patient selected"] + [
        f"{p['patient_id']} - {p.get('demographics', {}).get('age')}yo {p.get('demographics', {}).get('gender')} - {p.get('presenting_complaint', {}).get('chief_complaint', 'Unknown')}"
        for p in patients
    ]
    
    selected = st.selectbox(
        "Select Patient",
        options,
        index=0,
        help="Choose a patient to provide context for queries"
    )
    
    if selected == "No patient selected":
        return None
    
    # Extract patient ID
    patient_id = selected.split(" - ")[0]
    
    # Show patient details in an expander
    if patient_id:
        patient = patient_handler.get_patient(patient_id)
        if patient:
            with st.expander("📋 Patient Details", expanded=False):
                demographics = patient.get('demographics', {})
                history = patient.get('medical_history', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Age:** {demographics.get('age')}")
                with col2:
                    st.write(f"**Gender:** {demographics.get('gender')}")
                
                # Allergies (important!)
                allergies = history.get('allergies', [])
                if allergies:
                    st.error("**⚠️ ALLERGIES:**")
                    for allergy in allergies:
                        st.write(f"- {allergy['substance']} ({allergy['reaction']})")
                else:
                    st.write("**Allergies:** NKDA")
                
                # Presenting complaint
                complaint = patient.get('presenting_complaint', {})
                if complaint:
                    st.markdown("---")
                    st.write(f"**Chief Complaint:** {complaint.get('chief_complaint')}")
                    st.caption(f"**HPI:** {complaint.get('hpi', '')[:150]}...")
    
    return patient_id