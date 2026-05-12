"""Metrics dashboard component."""
import streamlit as st
import plotly.graph_objects as go
from src.llmops.metrics_collector import metrics_collector
from src.llmops.dashboard_backend import dashboard

def render_metrics_dashboard():
    """Render LLMOps metrics dashboard."""
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last Hour", "Last 24 Hours", "Last Week"],
        index=1
    )
    
    hours_map = {"Last Hour": 1, "Last 24 Hours": 24, "Last Week": 168}
    hours = hours_map[time_range]
    
    # Get summary metrics
    try:
        summary = metrics_collector.get_metrics_summary(hours=hours)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", summary.get('total_queries', 0))
        col2.metric("Avg Latency", f"{summary.get('avg_response_time', 0):.2f}s")
        col3.metric("Total Cost", f"${summary.get('total_cost', 0):.4f}")
        col4.metric("Success Rate", f"{summary.get('compliance_pass_rate', 0):.1f}%")
        
        st.markdown("### 📈 Performance Trends")
        
        # Charts
        c1, c2 = st.columns(2)
        
        with c1:
            st.caption("Response Latency")
            time_series = dashboard.get_time_series(hours=hours, metric='total_response_time')
            if time_series['labels']:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time_series['labels'], y=time_series['average'], mode='lines+markers', name='Avg'))
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=200)
                st.plotly_chart(fig, width=stretch)
            else:
                st.info("No data for chart")
        
        with c2:
            st.caption("Cost per Query")
            cost_series = dashboard.get_time_series(hours=hours, metric='total_cost')
            if cost_series['labels']:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=cost_series['labels'], y=cost_series['average'], name='Cost'))
                fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=200)
                st.plotly_chart(fig, width=True)
            else:
                st.info("No data for chart")
                
    except Exception as e:
        st.error(f"Could not load metrics: {e}")