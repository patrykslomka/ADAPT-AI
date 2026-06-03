"""ADAPT-AI Streamlit Application."""
import asyncio
import uuid
from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapt_ai.domain.patient_handler import PatientHandler
from ui.components.patient_selector import render_patient_selector
from ui.components.agent_monitor import render_agent_monitor
from ui.components.reasoning_visualizer import render_reasoning_viz
from ui.components.metrics_dashboard import render_metrics_dashboard

st.set_page_config(
    page_title="ADAPT-AI",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header  { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;
                   margin: 0.5rem 0; border-left: 5px solid #1f77b4; }
    .agent-status-approved { color: #28a745; font-weight: bold; }
    .agent-status-warning  { color: #ffc107; font-weight: bold; }
    .agent-status-rejected { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

DOMAINS = ["healthcare", "legal", "finance"]


@st.cache_resource
def _build_pipeline():
    """Build and cache the adapt_ai LangGraph pipeline (one per process)."""
    from adapt_ai.orchestrator.client import build_mcp_client
    from adapt_ai.agents.graph import build_graph
    mcp_client = build_mcp_client()
    return build_graph(mcp_client)


async def _run_query(query: str, subject_id: str | None, domain: str, session_id: str) -> dict:
    pipeline = _build_pipeline()
    initial_state = {
        "query": query,
        "subject_id": subject_id,
        "domain": domain,
        "session_id": session_id,
        "use_rat": False,
        "retrieved_context": "",
        "primary_response": "",
        "compliance_result": {},
        "quality_result": {},
        "final_response": "",
        "revision_count": 0,
        "revision_feedback": "",
        "agent_statuses": {},
        "error": None,
        "llm_usage": None,
    }
    result = await pipeline.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
    )
    return result


# ── Session state init ─────────────────────────────────────────────────────────

if "patient_handler" not in st.session_state:
    st.session_state.patient_handler = PatientHandler()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def main():
    st.markdown('<div class="main-header">⚕ ADAPT-AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Adaptive Multi-Agent System for Regulated Domains</div>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.header("Configuration")

        # Domain selector
        st.subheader("Domain")
        domain = st.selectbox("Active domain", DOMAINS, index=0)

        st.divider()

        # Patient selector (healthcare only)
        if domain == "healthcare":
            st.subheader("Patient Context")
            selected_patient = render_patient_selector(st.session_state.patient_handler)
            st.session_state.selected_patient = selected_patient
        else:
            st.session_state.selected_patient = None

        st.divider()
        st.subheader("System Status")
        st.metric("Pipeline", "LangGraph ✅")
        st.metric("Domain", domain.title())

        st.divider()
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.last_result = None
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Query")
        for message in st.session_state.messages:
            icon = "🧑‍⚕️" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=icon):
                st.markdown(message["content"])

        placeholder = {
            "healthcare": "Ask a clinical question (e.g. 'Suggest diagnostic workup')…",
            "legal": "Ask a legal question…",
            "finance": "Ask a finance question…",
        }.get(domain, "Enter your query…")

        if prompt := st.chat_input(placeholder):
            if domain == "healthcare" and not st.session_state.selected_patient:
                st.error("Please select a patient from the sidebar first.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar="🧑‍⚕️"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Agents collaborating…"):
                        try:
                            raw = asyncio.run(
                                _run_query(
                                    query=prompt,
                                    subject_id=st.session_state.selected_patient,
                                    domain=domain,
                                    session_id=st.session_state.session_id,
                                )
                            )
                            # Normalise to the shape UI components expect
                            result = {
                                "status": "success" if raw.get("final_response") else "rejected",
                                "content": raw.get("final_response") or raw.get("primary_response", ""),
                                "agents": raw.get("agent_statuses", {}),
                                "metadata": {
                                    "response_time": 0,  # not tracked in-process
                                    "use_rat": raw.get("use_rat", False),
                                    "revision_count": raw.get("revision_count", 0),
                                    "domain": domain,
                                },
                                "llm_usage": raw.get("llm_usage"),
                            }
                            st.session_state.last_result = result
                            st.markdown(result["content"] or "*(no response)*")
                            st.session_state.messages.append(
                                {"role": "assistant", "content": result["content"]}
                            )
                        except Exception as e:
                            st.error(f"System error: {e}")

    with col2:
        st.subheader("🤖 Agent Activity")
        if st.session_state.last_result:
            render_agent_monitor(st.session_state.last_result)
        else:
            st.info("Awaiting query…")

    st.divider()
    tab1, tab2 = st.tabs(["📊 Session Metrics", "🧠 Reasoning Path"])
    with tab1:
        render_metrics_dashboard(st.session_state.last_result)
    with tab2:
        if st.session_state.last_result:
            render_reasoning_viz(st.session_state.last_result)
        else:
            st.info("No reasoning path available yet")


if __name__ == "__main__":
    main()
