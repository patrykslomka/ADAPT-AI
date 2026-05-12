"""Agent activity monitoring component."""
import streamlit as st
from typing import Dict

def render_agent_monitor(result: Dict):
    """Render real-time agent activity timeline."""
    if 'agents' not in result:
        st.warning("No agent data available")
        return
    
    agents_data = result['agents']
    
    # Display each agent's status
    for agent_name, agent_info in agents_data.items():
        status = agent_info.get('status', 'unknown')
        
        # Status color
        if status == 'approved':
            status_html = '<span class="agent-status-approved">✅ APPROVED</span>'
        elif status == 'warning':
            status_html = '<span class="agent-status-warning">⚠️ WARNING</span>'
        elif status == 'rejected':
            status_html = '<span class="agent-status-rejected">❌ REJECTED</span>'
        else:
            status_html = '<span>❓ UNKNOWN</span>'
        
        # Agent card
        st.markdown(f"""
        <div class="metric-card">
            <strong>{agent_name.replace('_', ' ').title()}</strong><br>
            Status: {status_html}
        </div>
        """, unsafe_allow_html=True)
        
        # Show additional info based on agent type
        if agent_name == 'primary':
            confidence = agent_info.get('confidence', 0)
            st.progress(confidence, text=f"Diagnostic Confidence: {confidence:.0%}")
        
        elif agent_name == 'compliance':
            issues = agent_info.get('issues', 0)
            if issues > 0:
                st.warning(f"⚠️ {issues} compliance issue(s) found")
                warnings = agent_info.get('warnings', [])
                for warning in warnings[:2]: 
                    st.caption(f"- {warning.get('description', '')}")
            else:
                st.caption("No compliance violations detected.")
        
        elif agent_name == 'quality':
            hallucination_risk = agent_info.get('hallucination_risk', 0)
            st.metric("Hallucination Risk", f"{hallucination_risk:.0%}")
    
    # Response time
    response_time = result.get('metadata', {}).get('response_time', 0)
    st.metric("Total Processing Time", f"{response_time:.2f}s")