"""Reasoning path visualization component."""
import streamlit as st
try:
    from streamlit_mermaid import st_mermaid
except ImportError:
    st_mermaid = None
from typing import Dict

def render_reasoning_viz(result: Dict):
    """Render reasoning path as Mermaid diagram."""
    if st_mermaid is None:
        st.warning("Please install streamlit-mermaid to view the diagram.")
        return

    # Check if RAT was used or just RAG
    reasoning_path = result.get('metadata', {}).get('reasoning_path')
    
    if not reasoning_path:
        st.info("💡 Simple RAG retrieval used - query was straightforward.")
        return
    
    st.success("🧠 Multi-step RAT reasoning engaged for complex query")
    
    # Build Mermaid diagram
    mermaid_code = "graph TD\n"
    mermaid_code += "    Start([Start]) --> S1\n"
    
    for i, step in enumerate(reasoning_path):
        step_num = step.get('step', i+1)
        step_type = step.get('type', 'unknown')
        
        # Format step label
        if step_type == 'initial_analysis':
            label = f"Step {step_num}: Analysis"
            thought = step.get('thought', '')[:30]
            label += f"<br/>{thought}..."
        
        elif step_type == 'first_retrieval':
            label = f"Step {step_num}: Retrieval"
            docs = step.get('documents_found', 0)
            label += f"<br/>Found {docs} docs"
        
        elif step_type == 'reasoning_refinement':
            label = f"Step {step_num}: Refinement"
        
        else:
            label = f"Step {step_num}: {step_type.title()}"
        
        # Add node
        node_id = f"S{step_num}"
        mermaid_code += f'    {node_id}["{label}"]\n'
        
        # Add edge
        if i > 0:
            prev_id = f"S{reasoning_path[i-1]['step']}"
            mermaid_code += f"    {prev_id} --> {node_id}\n"
            
    # Add end node
    last_step = reasoning_path[-1]['step']
    mermaid_code += f"    S{last_step} --> End([Response])\n"
    
    # Render diagram
    st_mermaid(mermaid_code, height=400)
    
    # Detailed steps expander
    with st.expander("View Full Reasoning Logs"):
        st.json(reasoning_path)