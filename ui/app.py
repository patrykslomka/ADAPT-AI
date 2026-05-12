"""ADAPT-AI Streamlit Application."""
import streamlit as st
import asyncio
from pathlib import Path
import sys

# Add src to path to allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp.orchestrator import MCPOrchestrator
from src.domain.patient_handler import PatientHandler
from ui.components.patient_selector import render_patient_selector
from ui.components.agent_monitor import render_agent_monitor
from ui.components.reasoning_visualizer import render_reasoning_viz
from ui.components.metrics_dashboard import render_metrics_dashboard

# Page config
st.set_page_config(
    page_title="ADAPT-AI Clinical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical dashboard look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 5px solid #1f77b4;
    }
    .agent-status-approved { color: #28a745; font-weight: bold; }
    .agent-status-warning { color: #ffc107; font-weight: bold; }
    .agent-status-rejected { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = MCPOrchestrator()
    st.session_state.patient_handler = PatientHandler()
    st.session_state.messages = []
    st.session_state.selected_patient = None
    st.session_state.last_result = None
    st.session_state.show_metrics = False

def main():
    """Main application layout."""
    
    # Header
    st.markdown('<div class="main-header">🏥 ADAPT-AI Clinical Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Agent Clinical Decision Support System</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Patient selector
        st.subheader("Patient Context")
        selected_patient = render_patient_selector(st.session_state.patient_handler)
        
        if selected_patient != st.session_state.selected_patient:
            st.session_state.selected_patient = selected_patient
            st.rerun()
        
        st.divider()
        
        # System info
        st.subheader("System Status")
        st.metric("Agents Active", "3/3 ✅")
        st.metric("MCP Status", "Connected")
        
        st.divider()
        
        # Quick actions
        st.subheader("Quick Actions")
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.last_result = None
            st.rerun()
            
    # Main layout: 2 columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.subheader("💬 Clinical Query")
        
        # Display chat history
        for message in st.session_state.messages:
            role_icon = "🧑‍⚕️" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=role_icon):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a clinical question (e.g., 'Suggest diagnostic workup')..."):
            if not st.session_state.selected_patient:
                st.error("Please select a patient from the sidebar first.")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar="🧑‍⚕️"):
                    st.markdown(prompt)
                
                # Process with MCP
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("🤔 Agents collaborating..."):
                        try:
                            # Using asyncio.run to handle the async orchestrator
                            result = asyncio.run(
                                st.session_state.orchestrator.process_query(
                                    query=prompt,
                                    patient_id=st.session_state.selected_patient
                                )
                            )
                            
                            st.session_state.last_result = result
                            
                            if result.get('status') == 'success':
                                content = result.get('content', 'No response generated.')
                                st.markdown(content)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": content
                                })
                            else:
                                error_msg = f"❌ Error: {result.get('error', 'Unknown error')}"
                                st.error(error_msg)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": error_msg
                                })
                        except Exception as e:
                            st.error(f"System Error: {str(e)}")
    
    with col2:
        # Agent activity monitor
        st.subheader("🤖 Agent Activity")
        if st.session_state.last_result:
            render_agent_monitor(st.session_state.last_result)
        else:
            st.info("Awaiting query to visualize agent activity...")
    
    # Bottom panels
    st.divider()
    
    tab1, tab2 = st.tabs(["📊 Metrics Dashboard", "🧠 Reasoning Path"])
    
    with tab1:
        render_metrics_dashboard()
    
    with tab2:
        if st.session_state.last_result:
            render_reasoning_viz(st.session_state.last_result)
        else:
            st.info("No reasoning path available yet")

if __name__ == "__main__":
    main()