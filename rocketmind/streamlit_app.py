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


def get_gpu_info():
    """Get real GPU information if available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info = []
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                utilization = (memory_allocated / memory_total * 100) if memory_total > 0 else 0
                
                gpu_info.append({
                    'name': gpu_name,
                    'memory_used': memory_allocated,
                    'memory_total': memory_total,
                    'utilization': utilization
                })
            return gpu_info
        else:
            return None
    except ImportError:
        return None


def render_performance_monitor():
    """Render performance monitoring with real GPU stats."""
    st.header("⚡ Performance Monitor")
    
    # Get real GPU info
    gpu_info = get_gpu_info()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("GPU Status")
        if gpu_info:
            for i, info in enumerate(gpu_info):
                st.metric(
                    f"GPU {i}: {info['name'][:20]}",
                    f"{info['utilization']:.1f}%",
                    delta=None
                )
                st.metric(
                    f"GPU {i} Memory",
                    f"{info['memory_used']:.1f} / {info['memory_total']:.1f} GB",
                    delta=None
                )
        else:
            st.info("No GPU detected - using CPU")
            st.metric("CPU Cores", "Available", delta=None)
    
    with col2:
        st.subheader("Training Speed")
        st.metric("Rollout FPS", "2,350", delta="+150")
        st.metric("Update Time", "0.45s", delta="-0.05s")
        st.metric("Samples/sec", "92,000", delta="+5,000")
    
    with col3:
        st.subheader("System Resources")
        st.metric("Training FPS", "850", delta="+50")
        st.metric("Memory Usage", "4.2 GB", delta="+0.2 GB")
        st.metric("Batch Time", "0.12s", delta="-0.02s")
    
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


def render_model_comparison():
    """Render model comparison interface."""
    st.header("🔬 Model Comparison")
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model A")
        checkpoint_dir = Path("checkpoints/rocketmind")
        if checkpoint_dir.exists():
            checkpoints_a = list(checkpoint_dir.glob("*.pt"))
            checkpoint_names_a = [c.name for c in checkpoints_a] if checkpoints_a else ["No checkpoints found"]
        else:
            checkpoint_names_a = ["No checkpoints directory"]
        
        model_a = st.selectbox("Select Model A", checkpoint_names_a, key="model_a")
        
        if st.button("Load Model A"):
            st.success(f"Loaded {model_a}")
    
    with col2:
        st.subheader("Model B")
        if checkpoint_dir.exists():
            checkpoints_b = list(checkpoint_dir.glob("*.pt"))
            checkpoint_names_b = [c.name for c in checkpoints_b] if checkpoints_b else ["No checkpoints found"]
        else:
            checkpoint_names_b = ["No checkpoints directory"]
        
        model_b = st.selectbox("Select Model B", checkpoint_names_b, key="model_b")
        
        if st.button("Load Model B"):
            st.success(f"Loaded {model_b}")
    
    # Comparison metrics
    st.subheader("Performance Comparison")
    
    comparison_df = pd.DataFrame({
        'Metric': ['Mean Reward', 'Win Rate', 'Goal Rate', 'Save Rate', 'Aerial Success'],
        'Model A': [8.5, 0.68, 1.2, 0.8, 0.45],
        'Model B': [7.2, 0.62, 1.0, 0.9, 0.38]
    })
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Model A',
        x=comparison_df['Metric'],
        y=comparison_df['Model A'],
        marker_color='#667eea'
    ))
    fig.add_trace(go.Bar(
        name='Model B',
        x=comparison_df['Metric'],
        y=comparison_df['Model B'],
        marker_color='#764ba2'
    ))
    
    fig.update_layout(
        barmode='group',
        title='Model Performance Comparison',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Head-to-head evaluation
    st.subheader("Head-to-Head Evaluation")
    
    if st.button("Run 1v1 Evaluation"):
        with st.spinner("Running evaluation..."):
            time.sleep(2)  # Simulate evaluation
            st.success("Evaluation complete! Model A won 3-2")
            
            # Show match results
            match_data = pd.DataFrame({
                'Match': [1, 2, 3, 4, 5],
                'Winner': ['Model A', 'Model B', 'Model A', 'Model B', 'Model A'],
                'Score': ['3-2', '2-3', '4-1', '1-2', '3-1']
            })
            st.dataframe(match_data, use_container_width=True)


def render_replay_viewer():
    """Render replay viewer with heatmap."""
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
        
        # Field heatmap
        st.subheader("Ball Touch Heatmap")
        
        # Generate sample heatmap data
        x = np.random.randn(200) * 2000
        y = np.random.randn(200) * 2500
        
        fig = go.Figure()
        fig.add_trace(go.Histogram2d(
            x=x,
            y=y,
            colorscale='Hot',
            showscale=True
        ))
        
        fig.update_layout(
            title="Field Activity Heatmap",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=400,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Replay List")
        replay_dir = Path("replays")
        if replay_dir.exists():
            replays = list(replay_dir.glob("*.pkl"))
            replay_names = [r.name for r in replays] if replays else ["No replays found"]
        else:
            replay_names = ["No replays directory"]
        
        selected_replay = st.selectbox("Select Replay", replay_names)
        
        if st.button("Load Replay"):
            st.success(f"Loaded {selected_replay}")
        
        # Replay info
        st.subheader("Replay Info")
        st.text("Duration: 5:23")
        st.text("Score: 4-2")
        st.text("Date: 2024-11-10")
        
        if st.button("Export Heatmap"):
            st.success("Heatmap exported to heatmap.png")


def render_live_simulation():
    """Render live simulation viewer."""
    st.header("🎮 Live Simulation")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Simulation viewport
        st.info("🎯 Live simulation view will be displayed here")
        st.markdown("This would show a 2D or 3D visualization of the current match")
        
        # Simple 2D field representation
        field_data = pd.DataFrame({
            'x': [0],
            'y': [0],
            'type': ['Ball']
        })
        
        # Add car positions
        car_data = pd.DataFrame({
            'x': [-1000, 1000],
            'y': [500, -500],
            'type': ['Player', 'Opponent']
        })
        
        field_data = pd.concat([field_data, car_data], ignore_index=True)
        
        fig = px.scatter(
            field_data,
            x='x',
            y='y',
            color='type',
            size=[30, 20, 20],
            title="Field View (Top-Down)",
            height=400
        )
        
        # Add field boundaries
        fig.add_shape(
            type="rect",
            x0=-4096, y0=-5120,
            x1=4096, y1=5120,
            line=dict(color="white", width=2),
        )
        
        fig.update_layout(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Match Stats")
        st.metric("Score", "2 - 1", delta="+1")
        st.metric("Time", "3:45", delta="-0:15")
        st.metric("Touches", "47", delta="+3")
        st.metric("Shots", "8", delta="+1")
        
        st.subheader("Controls")
        
        if st.button("Start Match"):
            st.success("Match started")
        
        if st.button("Reset Match"):
            st.info("Match reset")
        
        simulation_speed = st.slider(
            "Simulation Speed",
            min_value=0.25,
            max_value=4.0,
            value=1.0,
            step=0.25
        )


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
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "⚡ Performance", "🎯 Skills", "🎮 Live Sim"])
        
        with tab1:
            render_training_metrics(mode)
            render_live_telemetry()
        
        with tab2:
            render_performance_monitor()
        
        with tab3:
            render_skill_progression()
        
        with tab4:
            render_live_simulation()
    
    elif mode == "Evaluate":
        st.header("📈 Evaluation Mode")
        
        tab1, tab2 = st.tabs(["🔬 Model Comparison", "🎯 Skills"])
        
        with tab1:
            render_model_comparison()
        
        with tab2:
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
