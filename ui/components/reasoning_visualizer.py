"""Reasoning path visualization component."""
import streamlit as st
try:
    from streamlit_mermaid import st_mermaid
except ImportError:
    st_mermaid = None
from typing import Dict


_PIPELINE_DIAGRAM = """graph LR
    A([Query]) --> B[intent_and_retrieve]
    B -->|RAT| C[RAT reasoning]
    B -->|RAG| D[RAG retrieval]
    C & D --> E[primary_agent]
    E --> F[compliance_agent]
    E --> G[quality_agent]
    F & G --> H{review_results}
    H -->|pass| I[aggregate_response]
    H -->|quality fail| E
    H -->|compliance fail| J([Rejected])
    I --> K([Response])
"""


def render_reasoning_viz(result: Dict):
    """Render the adapt_ai pipeline graph and per-run routing info."""
    use_rat = result.get("metadata", {}).get("use_rat", False)
    revision_count = result.get("metadata", {}).get("revision_count", 0)

    # Routing summary
    col1, col2 = st.columns(2)
    col1.info(f"Retrieval: {'🧠 RAT (multi-step reasoning)' if use_rat else '📄 RAG (single-pass)'}")
    col2.info(f"Quality revisions: {revision_count}")

    st.markdown("### Pipeline graph")
    if st_mermaid is not None:
        st_mermaid(_PIPELINE_DIAGRAM, height=320)
    else:
        st.code(_PIPELINE_DIAGRAM, language="text")
        st.caption("Install `streamlit-mermaid` for a rendered diagram.")

    # Agent status detail
    agents = result.get("agents", {})
    if agents:
        with st.expander("Agent status detail"):
            st.json(agents)
