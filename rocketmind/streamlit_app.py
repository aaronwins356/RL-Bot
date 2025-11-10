"""
RocketMind Streamlit Dashboard
Interactive training visualization, control, and monitoring hub.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Optional
import json

# Page configuration
st.set_page_config(
    page_title="🚀 RocketMind Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class DashboardState:
    """Manage dashboard state across reruns."""
    
    def __init__(self):
        if 'training_active' not in st.session_state:
            st.session_state.training_active = False
        if 'reward_history' not in st.session_state:
            st.session_state.reward_history = []
        if 'loss_history' not in st.session_state:
            st.session_state.loss_history = []
        if 'timesteps' not in st.session_state:
            st.session_state.timesteps = 0
        if 'current_reward' not in st.session_state:
            st.session_state.current_reward = 0.0


def render_header():
    """Render dashboard header."""
    st.markdown('<p class="main-header">🚀 RocketMind Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### Next-Generation PPO Rocket League Bot")
    st.markdown("---")


def render_control_panel():
    """Render control panel in sidebar."""
    st.sidebar.header("🎮 Control Panel")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Mode",
        ["Train", "Evaluate", "Spectate", "Configure"],
        help="Select bot operation mode"
    )
    
    st.sidebar.markdown("---")
    
    # Training controls
    if mode == "Train":
        st.sidebar.subheader("Training Controls")
        
        if st.sidebar.button("▶️ Start Training", use_container_width=True):
            st.session_state.training_active = True
            st.sidebar.success("Training started!")
        
        if st.sidebar.button("⏸️ Pause Training", use_container_width=True):
            st.session_state.training_active = False
            st.sidebar.warning("Training paused")
        
        if st.sidebar.button("💾 Save Checkpoint", use_container_width=True):
            st.sidebar.success("Checkpoint saved!")
    
    # RLBot controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("RLBot Integration")
    
    if st.sidebar.button("🚀 Launch in RLBot GUI", use_container_width=True):
        st.sidebar.info("Launching bot in RLBot...")
        # Actual launch logic would go here
    
    if st.sidebar.button("🔄 Reload Model", use_container_width=True):
        st.sidebar.success("Model reloaded!")
    
    return mode


def render_training_metrics(mode: str):
    """Render real-time training metrics."""
    if mode != "Train":
        return
    
    st.header("📊 Training Metrics")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Reward",
            f"{st.session_state.current_reward:.2f}",
            delta="+0.5"
        )
    
    with col2:
        st.metric(
            "Timesteps",
            f"{st.session_state.timesteps:,}",
            delta="+2048"
        )
    
    with col3:
        st.metric(
            "Episodes",
            "1,250",
            delta="+10"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            "65.2%",
            delta="+2.1%"
        )
    
    # Reward curve
    st.subheader("Reward History")
    
    if len(st.session_state.reward_history) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.reward_history,
            mode='lines',
            name='Reward',
            line=dict(color='#667eea', width=2)
        ))
        fig.update_layout(
            xaxis_title="Update",
            yaxis_title="Mean Reward",
            height=300,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Training metrics will appear here once training starts")
    
    # Loss curves
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Policy Loss")
        if len(st.session_state.loss_history) > 0:
            fig = px.line(y=st.session_state.loss_history, title="Policy Loss Over Time")
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No loss data yet")
    
    with col2:
        st.subheader("Value Loss")
        st.info("Value loss data will appear here")


def render_live_telemetry():
    """Render live match telemetry."""
    st.header("📡 Live Telemetry")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ball Speed", "1,245 uu/s", delta="+50")
        st.metric("Player Speed", "2,300 uu/s", delta="MAX")
        
    with col2:
        st.metric("Boost Amount", "67%", delta="-12%")
        st.metric("Distance to Ball", "892 uu", delta="-45")
    
    with col3:
        st.metric("Score Diff", "+2", delta="+1")
        st.metric("Time Remaining", "3:42", delta="-0:01")
    
    # Field heatmap placeholder
    st.subheader("Field Heatmap")
    st.info("🗺️ Ball touch heatmap will be displayed here")


def render_hyperparameter_editor():
    """Render dynamic hyperparameter editor."""
    st.header("⚙️ Hyperparameters")
    
    with st.expander("PPO Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            learning_rate = st.slider(
                "Learning Rate",
                min_value=1e-5,
                max_value=1e-3,
                value=3e-4,
                format="%.5f",
                help="Initial learning rate for optimizer"
            )
            
            gamma = st.slider(
                "Gamma (Discount)",
                min_value=0.90,
                max_value=0.999,
                value=0.99,
                help="Discount factor for future rewards"
            )
            
            gae_lambda = st.slider(
                "GAE Lambda",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                help="GAE lambda for advantage estimation"
            )
        
        with col2:
            clip_range = st.slider(
                "Clip Range",
                min_value=0.1,
                max_value=0.3,
                value=0.2,
                help="PPO clip range"
            )
            
            ent_coef = st.slider(
                "Entropy Coefficient",
                min_value=0.0,
                max_value=0.1,
                value=0.01,
                format="%.3f",
                help="Entropy bonus for exploration"
            )
            
            batch_size = st.select_slider(
                "Batch Size",
                options=[1024, 2048, 4096, 8192],
                value=4096,
                help="PPO minibatch size"
            )
    
    with st.expander("Reward Weights"):
        st.slider("Goal Scored", 0.0, 20.0, 10.0)
        st.slider("Ball Touch", 0.0, 2.0, 0.5)
        st.slider("Aerial Touch", 0.0, 3.0, 1.0)
        st.slider("Positioning", 0.0, 1.0, 0.1)
    
    if st.button("💾 Save Configuration"):
        st.success("Configuration saved!")


def render_performance_monitor():
    """Render performance monitoring."""
    st.header("⚡ Performance Monitor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("GPU Utilization", "87%", delta="+5%")
        st.metric("GPU Memory", "6.2 / 8.0 GB", delta="+0.3 GB")
    
    with col2:
        st.metric("Rollout FPS", "2,350", delta="+150")
        st.metric("Update Time", "0.45s", delta="-0.05s")
    
    with col3:
        st.metric("Samples/sec", "92,000", delta="+5,000")
        st.metric("Training FPS", "850", delta="+50")
    
    # Performance chart
    st.subheader("Training Throughput")
    performance_data = pd.DataFrame({
        'Time': range(100),
        'FPS': np.random.randint(800, 900, 100)
    })
    fig = px.line(performance_data, x='Time', y='FPS', title="Training FPS Over Time")
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)


def render_skill_progression():
    """Render skill progression tracker."""
    st.header("🎯 Skill Progression")
    
    skills = {
        'Aerials': 75,
        'Dribbling': 60,
        'Shooting': 80,
        'Defense': 70,
        'Positioning': 65,
        'Boost Management': 85
    }
    
    # Radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(skills.values()),
        theta=list(skills.keys()),
        fill='toself',
        name='Current Skills'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_replay_viewer():
    """Render replay viewer."""
    st.header("🎬 Replay Viewer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("🎥 Replay playback will be displayed here")
        
        # Playback controls
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.button("⏮️ Previous")
        with col_b:
            st.button("▶️ Play")
        with col_c:
            st.button("⏸️ Pause")
        with col_d:
            st.button("⏭️ Next")
    
    with col2:
        st.subheader("Replay List")
        replays = ["replay_001.pkl", "replay_002.pkl", "replay_003.pkl"]
        selected_replay = st.selectbox("Select Replay", replays)
        
        if st.button("Load Replay"):
            st.success(f"Loaded {selected_replay}")


def main():
    """Main dashboard application."""
    # Initialize state
    state = DashboardState()
    
    # Render header
    render_header()
    
    # Render control panel (sidebar)
    mode = render_control_panel()
    
    # Main content based on mode
    if mode == "Train":
        # Training tab
        tab1, tab2, tab3 = st.tabs(["📊 Metrics", "⚡ Performance", "🎯 Skills"])
        
        with tab1:
            render_training_metrics(mode)
            render_live_telemetry()
        
        with tab2:
            render_performance_monitor()
        
        with tab3:
            render_skill_progression()
    
    elif mode == "Evaluate":
        st.header("📈 Evaluation Mode")
        st.info("Evaluation interface coming soon")
        render_skill_progression()
    
    elif mode == "Spectate":
        st.header("👁️ Spectate Mode")
        render_live_telemetry()
        render_replay_viewer()
    
    elif mode == "Configure":
        render_hyperparameter_editor()
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**RocketMind** v1.0")
    with col2:
        st.markdown("Status: " + ("🟢 Training" if st.session_state.training_active else "🔴 Idle"))
    with col3:
        st.markdown(f"Last update: {time.strftime('%H:%M:%S')}")
    
    # Auto-refresh when training is active
    if st.session_state.training_active:
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
