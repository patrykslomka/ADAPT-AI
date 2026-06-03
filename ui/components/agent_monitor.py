"""Agent activity monitoring component."""
import streamlit as st
from typing import Dict


def render_agent_monitor(result: Dict):
    """Render agent activity panel from adapt_ai AgentState output."""
    agents_data = result.get("agents", {})
    if not agents_data:
        st.warning("No agent data available")
        return

    for agent_name, agent_info in agents_data.items():
        status = agent_info.get("status", "unknown")

        if status == "approved":
            status_html = '<span class="agent-status-approved">✅ APPROVED</span>'
        elif status == "warning":
            status_html = '<span class="agent-status-warning">⚠️ WARNING</span>'
        elif status == "rejected":
            status_html = '<span class="agent-status-rejected">❌ REJECTED</span>'
        else:
            status_html = '<span>❓ UNKNOWN</span>'

        st.markdown(f"""
        <div class="metric-card">
            <strong>{agent_name.replace('_', ' ').title()}</strong><br>
            Status: {status_html}
        </div>
        """, unsafe_allow_html=True)

        if agent_name == "compliance":
            issues = agent_info.get("issues", [])
            if issues:
                st.warning(f"⚠️ {len(issues)} compliance issue(s) found")
                for issue in issues[:3]:
                    msg = issue if isinstance(issue, str) else issue.get("description", str(issue))
                    st.caption(f"- {msg}")
            else:
                st.caption("No compliance violations detected.")

        elif agent_name == "quality":
            score = agent_info.get("score", 0)
            st.progress(min(float(score), 1.0), text=f"Quality score: {score:.0%}")
            issues = agent_info.get("issues", [])
            if issues:
                for issue in issues[:2]:
                    msg = issue if isinstance(issue, str) else issue.get("description", str(issue))
                    st.caption(f"⚠ {msg}")

    response_time = result.get("metadata", {}).get("response_time", 0)
    if response_time:
        st.metric("Processing Time", f"{response_time:.2f}s")

    use_rat = result.get("metadata", {}).get("use_rat", False)
    st.caption(f"Retrieval strategy: {'RAT (multi-step)' if use_rat else 'RAG (single-pass)'}")
