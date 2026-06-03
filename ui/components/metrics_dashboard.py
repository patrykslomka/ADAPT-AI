"""Session metrics dashboard component."""
import streamlit as st
from typing import Optional, Dict


def render_metrics_dashboard(last_result: Optional[Dict]):
    """Render LLM usage metrics from the last pipeline result."""
    if not last_result:
        st.info("Run a query to see session metrics.")
        return

    metadata = last_result.get("metadata", {})
    llm_usage = last_result.get("llm_usage") or {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", last_result.get("status", "—").title())
    col2.metric("Response Time", f"{metadata.get('response_time', 0):.2f}s")
    col3.metric("Revisions", metadata.get("revision_count", 0))
    col4.metric("Domain", metadata.get("domain", "—").title())

    if llm_usage:
        st.markdown("### LLM Usage")
        u1, u2, u3 = st.columns(3)
        u1.metric("Input tokens", llm_usage.get("input_tokens", 0))
        u2.metric("Output tokens", llm_usage.get("output_tokens", 0))
        u3.metric("Est. cost", f"${llm_usage.get('estimated_cost_usd', 0):.5f}")

        calls = llm_usage.get("calls", [])
        if calls:
            with st.expander("Per-agent token breakdown"):
                for call in calls:
                    st.caption(
                        f"**{call.get('agent', '?')}** — "
                        f"in: {call.get('input_tokens', 0)}, "
                        f"out: {call.get('output_tokens', 0)}"
                    )
