import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Digital Twin for Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (DARK THEME & MODERN AESTHETICS) ---
st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
    }
    .main {
        background-color: #0b0f19;
        color: #e6edf3;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .status-healthy {
        color: #3fb950;
        font-weight: bold;
    }
    .status-degrading {
        color: #d29922;
        font-weight: bold;
    }
    .status-critical {
        color: #f85149;
        font-weight: bold;
    }
    .status-box {
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 15px;
        margin-top: 10px;
    }
    .box-healthy {
        background: rgba(63, 185, 80, 0.15);
        border: 1px solid #3fb950;
        color: #7ee787;
    }
    .box-degrading {
        background: rgba(210, 153, 34, 0.15);
        border: 1px solid #d29922;
        color: #f2cc60;
    }
    .box-critical {
        background: rgba(248, 81, 73, 0.15);
        border: 1px solid #f85149;
        color: #ff7b72;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("Digital Twin Control Panel")
st.sidebar.markdown("---")

monitoring_mode = st.sidebar.selectbox(
    "Select Asset / Monitoring Mode",
    ["RUL Prediction - Engine Monitoring", "Anomaly Detection - Bearing Monitoring"]
)

data_source = st.sidebar.radio(
    "Telemetry Data Source",
    ["Built-in Simulation Stream", "Upload Custom CSV"]
)

stream_speed = st.sidebar.slider("Stream Speed (Steps / Interval)", min_value=1, max_value=20, value=5)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Digital Twin Predictive Maintenance**\nCombines AI prognostics (RF, LSTM) and Deep Autoencoder anomaly detection.")

# --- SIMULATION DATA GENERATORS ---
def generate_engine_rul_data():
    cycles = []
    # Generate 4 degradation profiles to match the dashboard curves
    for start_rul in [180, 260, 180, 185]:
        drop = np.linspace(start_rul, 10, int(start_rul * 1.0))
        noise = np.random.normal(0, 5, len(drop))
        cycles.extend(drop + noise)
    return np.array(cycles)

def generate_bearing_mse_data():
    mse_stream = []
    # Baseline healthy
    mse_stream.extend(np.random.normal(3, 1.2, 70))
    # Wear initiation
    mse_stream.extend(np.linspace(3, 50, 45) + np.random.normal(0, 4, 45))
    mse_stream.append(115.0) # peak defect impact
    # Replacement / re-stabilization
    mse_stream.extend(np.random.normal(3, 1.0, 50))
    # Secondary wear cycle
    mse_stream.extend(np.linspace(3, 45, 60) + np.random.normal(0, 3, 60))
    mse_stream.append(95.0)
    mse_stream.extend(np.random.normal(3, 1.0, 40))
    # Incipient failure cycle
    mse_stream.extend(np.linspace(3, 35, 50) + np.random.normal(0, 3, 50))
    return np.array(mse_stream)

# --- MODE 1: ENGINE RUL PREDICTION ---
if monitoring_mode == "RUL Prediction - Engine Monitoring":
    st.subheader("🚀 RUL Prediction - Engine Monitoring ✈")
    
    col_plot, col_metric = st.columns([3, 1])
    
    data = None
    if data_source == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload Engine Telemetry CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'RUL' in df.columns:
                    data = df['RUL'].values
                else:
                    data = df.iloc[:, -1].values
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    if data is None:
        data = generate_engine_rul_data()
    
    # Live simulation index
    total_len = len(data)
    step_val = st.sidebar.slider("Simulation Step (Telemetry Ingestion)", 10, total_len, min(70, total_len))
    
    current_rul = float(data[step_val - 1])
    
    # Determine Health Status
    if current_rul > 100:
        status_label = "🟢 Healthy"
        status_class = "box-healthy"
        status_msg = "✅ Machine remains in Healthy Condition."
    elif current_rul > 40:
        status_label = "🟡 Degrading"
        status_class = "box-degrading"
        status_msg = "⚠️ Engine component wear detected. Plan service."
    else:
        status_label = "🔴 Critical"
        status_class = "box-critical"
        status_msg = "🚨 Critical failure imminent! Immediate maintenance required."

    with col_metric:
        st.markdown(f"### Current Condition: {status_label}")
        st.markdown(f"**Predicted RUL**")
        st.markdown(f"# **{current_rul:.2f}** cycles")
        
        st.markdown(f"""
        <div class="status-box {status_class}">
            {status_msg}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Asset Details:**")
        st.markdown("- **Asset ID:** Jet Engine #001")
        st.markdown("- **Model Architecture:** Random Forest + LSTM")
        st.markdown("- **Baseline Operational Cycles:** 250")
        
        # Download Diagnostic Report
        report_df = pd.DataFrame({
            "Step": list(range(step_val)),
            "Predicted_RUL": data[:step_val]
        })
        csv_report = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Telemetry Report",
            data=csv_report,
            file_name="engine_health_report.csv",
            mime="text/csv"
        )

    with col_plot:
        # Top Global Trajectory Chart
        fig_global = go.Figure()
        fig_global.add_trace(go.Scatter(
            x=list(range(total_len)),
            y=data,
            mode='lines',
            name='Engine Fleet Degradation',
            line=dict(color='#00e5ff', width=1.5)
        ))
        fig_global.update_layout(
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font=dict(color='#8b949e'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
            xaxis=dict(showgrid=True, gridcolor='#21262d'),
            yaxis=dict(showgrid=True, gridcolor='#21262d', title="RUL (Cycles)")
        )
        st.plotly_chart(fig_global, use_container_width=True)

        # Bottom Local Step-by-Step Chart
        fig_local = go.Figure()
        sub_steps = list(range(step_val))
        fig_local.add_trace(go.Scatter(
            x=sub_steps,
            y=data[:step_val],
            mode='lines',
            name='RUL',
            line=dict(color='#7986cb', width=2)
        ))
        fig_local.add_trace(go.Scatter(
            x=sub_steps,
            y=np.linspace(0, 50, step_val),
            mode='lines',
            name='step',
            line=dict(color='#1976d2', width=1.5)
        ))
        fig_local.update_layout(
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font=dict(color='#8b949e'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
            xaxis=dict(showgrid=True, gridcolor='#21262d', title="Operating Steps"),
            yaxis=dict(showgrid=True, gridcolor='#21262d')
        )
        st.plotly_chart(fig_local, use_container_width=True)


# --- MODE 2: BEARING ANOMALY DETECTION ---
else:
    st.subheader("⚙ Anomaly Detection - Bearing Monitoring")
    
    col_plot, col_metric = st.columns([3, 1])
    
    data = None
    if data_source == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload Bearing Vibration CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'mse' in df.columns:
                    data = df['mse'].values
                else:
                    data = df.iloc[:, -1].values
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                
    if data is None:
        data = generate_bearing_mse_data()
        
    total_len = len(data)
    step_val = st.sidebar.slider("Simulation Step (Telemetry Ingestion)", 10, total_len, min(37, total_len))
    
    current_mse = float(data[step_val - 1])
    
    # Determine Bearing Anomaly Condition
    if current_mse < 12.0:
        status_label = "🟢 Healthy"
        status_class = "box-healthy"
        status_msg = "✅ Bearing operating within nominal vibration limits."
    elif current_mse < 30.0:
        status_label = "🟡 Degrading"
        status_class = "box-degrading"
        status_msg = "⚠️ Bearing is Deteriorating. Monitor closely."
    else:
        status_label = "🔴 Critical"
        status_class = "box-critical"
        status_msg = "🚨 High vibration anomaly detected! Risk of bearing seizure."

    with col_metric:
        st.markdown(f"### Current Condition: {status_label}")
        st.markdown(f"**Reconstruction Error (MSE)**")
        st.markdown(f"# **{current_mse:.4f}**")
        
        st.markdown(f"""
        <div class="status-box {status_class}">
            {status_msg}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**Bearing Diagnostic Info:**")
        st.markdown("- **Asset:** Deep Groove Ball Bearing #35")
        st.markdown("- **Sampling Rate:** 25.6 kHz")
        st.markdown("- **Model:** Deep Unsupervised Autoencoder")
        st.markdown("- **Baseline Threshold:** 12.0 MSE")
        
        # Download Diagnostic Report
        report_df = pd.DataFrame({
            "Step": list(range(step_val)),
            "Reconstruction_MSE": data[:step_val]
        })
        csv_report = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Anomaly Report",
            data=csv_report,
            file_name="bearing_anomaly_report.csv",
            mime="text/csv"
        )

    with col_plot:
        # Top Global Trajectory Chart
        fig_global = go.Figure()
        fig_global.add_trace(go.Scatter(
            x=list(range(total_len)),
            y=data,
            mode='lines',
            name='Vibration Reconstruction MSE',
            line=dict(color='#7986cb', width=1.5)
        ))
        fig_global.update_layout(
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font=dict(color='#8b949e'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
            xaxis=dict(showgrid=True, gridcolor='#21262d'),
            yaxis=dict(showgrid=True, gridcolor='#21262d', title="Reconstruction Error (MSE)")
        )
        st.plotly_chart(fig_global, use_container_width=True)

        # Bottom Local Step-by-Step Chart
        fig_local = go.Figure()
        sub_steps = list(range(step_val))
        fig_local.add_trace(go.Scatter(
            x=sub_steps,
            y=data[:step_val],
            mode='lines',
            name='mse',
            line=dict(color='#7986cb', width=2)
        ))
        fig_local.add_trace(go.Scatter(
            x=sub_steps,
            y=np.linspace(1, 37, step_val),
            mode='lines',
            name='step',
            line=dict(color='#1976d2', width=1.5)
        ))
        fig_local.update_layout(
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font=dict(color='#8b949e'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
            xaxis=dict(showgrid=True, gridcolor='#21262d', title="Operating Steps"),
            yaxis=dict(showgrid=True, gridcolor='#21262d')
        )
        st.plotly_chart(fig_local, use_container_width=True)
