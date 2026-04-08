import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictIQ — Machine Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: #0a0e1a;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1226 !important;
    border-right: 1px solid #1e2744;
}
[data-testid="stSidebar"] .stRadio label {
    color: #a0aec0 !important;
    font-size: 0.9rem;
}

/* Main content text */
h1, h2, h3, h4, p, label, .stMarkdown {
    color: #e2e8f0 !important;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a2035 0%, #0f1628 100%);
    border: 1px solid #2d3748;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: #4299e1;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.8rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Result cards */
.result-safe {
    background: linear-gradient(135deg, #1a2e1a 0%, #0f1a0f 100%);
    border: 2px solid #48bb78;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    animation: fadeIn 0.6s ease;
}
.result-danger {
    background: linear-gradient(135deg, #2e1a1a 0%, #1a0f0f 100%);
    border: 2px solid #fc8181;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    animation: fadeIn 0.6s ease;
}
.result-warning {
    background: linear-gradient(135deg, #2e2a1a 0%, #1a180f 100%);
    border: 2px solid #f6ad55;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    animation: fadeIn 0.6s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Sliders */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 32px;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s ease, transform 0.1s ease;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* Input fields */
.stSelectbox > div, .stNumberInput > div {
    background: #1a2035 !important;
    border-radius: 10px !important;
    border: 1px solid #2d3748 !important;
}

/* Section header */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #667eea;
    padding-left: 14px;
    margin: 24px 0 16px 0;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-safe   { background: #1a4731; color: #68d391; border: 1px solid #48bb78; }
.badge-danger { background: #4a1a1a; color: #fc8181; border: 1px solid #e53e3e; }
.badge-warn   { background: #4a3a1a; color: #f6ad55; border: 1px solid #d97706; }

/* Logo area */
.logo-area {
    text-align: center;
    padding: 20px 0 30px 0;
    border-bottom: 1px solid #1e2744;
    margin-bottom: 20px;
}
.logo-text {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.logo-sub {
    font-size: 0.75rem;
    color: #4a5568;
    margin-top: 2px;
}

/* Divider */
hr { border-color: #1e2744 !important; }

/* Dataframe */
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("xgboost_binary_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()

FEATURE_NAMES = ["Air temperature", "Process temperature",
                 "Rotational speed", "Torque", "Tool wear"]

# ─── Helpers ──────────────────────────────────────────────────────────────────
# Model feature order (from stored feature_names): Type, Air temperature,
# Process temperature, Rotational speed, Torque, Tool wear  (6 total)
# • Type is label-encoded: H=0, L=1, M=2  (int, not scaled)
# • The 5 numerical columns are passed through the StandardScaler

TYPE_ENCODING = {"H": 0, "L": 1, "M": 2}

def encode_machine_type(mt: str) -> int:
    """Label-encode Machine Type: H→0, L→1, M→2"""
    return TYPE_ENCODING.get(mt.upper().strip(), 2)

def build_feature_row(machine_type, air_temp, proc_temp, rpm, torque, tool_wear):
    # Scale the 5 numerical features using the fitted scaler
    numerical = np.array([[air_temp, proc_temp, rpm, torque, tool_wear]], dtype=float)
    num_scaled = scaler.transform(numerical)          # shape (1, 5)
    # Prepend the integer-encoded Type as column 0
    type_val = np.array([[encode_machine_type(machine_type)]], dtype=float)
    return np.hstack([type_val, num_scaled])          # shape (1, 6)

def predict_single(machine_type, air_temp, proc_temp, rpm, torque, tool_wear):
    x = build_feature_row(machine_type, air_temp, proc_temp, rpm, torque, tool_wear)
    pred  = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    fail_prob   = float(proba[1])
    health_score = round((1 - fail_prob) * 100, 1)
    return int(pred), fail_prob, health_score

def predict_batch(df: pd.DataFrame):
    results = []
    for _, row in df.iterrows():
        try:
            mt = str(row.get("Machine type", row.get("machine_type", "M"))).strip().upper()
            at  = float(row.get("Air temperature",   row.get("air_temperature",   300)))
            pt  = float(row.get("Process temperature", row.get("process_temperature", 310)))
            rs  = float(row.get("Rotational speed",  row.get("rotational_speed",  1500)))
            tq  = float(row.get("Torque",            row.get("torque",            40)))
            tw  = float(row.get("Tool wear",         row.get("tool_wear",         100)))
            pred, fp, hs = predict_single(mt, at, pt, rs, tq, tw)
            results.append({
                "Machine Type": mt,
                "Air Temp (K)": at,
                "Proc Temp (K)": pt,
                "RPM": rs,
                "Torque (Nm)": tq,
                "Tool Wear (min)": tw,
                "Prediction": "⚠️ Failure" if pred == 1 else "✅ Normal",
                "Fail Probability (%)": round(fp * 100, 2),
                "Health Score (%)": hs,
            })
        except Exception as e:
            results.append({"Error": str(e)})
    return pd.DataFrame(results)

def gauge_chart(health_score):
    color = "#48bb78" if health_score >= 70 else "#f6ad55" if health_score >= 40 else "#fc8181"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        delta={"reference": 70, "valueformat": ".1f"},
        title={"text": "Machine Health Score", "font": {"color": "#a0aec0", "size": 14}},
        number={"font": {"color": "#e2e8f0", "size": 40}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5568", "tickfont": {"color": "#718096"}},
            "bar": {"color": color},
            "bgcolor": "#1a2035",
            "bordercolor": "#2d3748",
            "steps": [
                {"range": [0,  40], "color": "#2d1a1a"},
                {"range": [40, 70], "color": "#2d2a1a"},
                {"range": [70,100], "color": "#1a2d1a"},
            ],
            "threshold": {
                "line": {"color": "#667eea", "width": 3},
                "thickness": 0.75,
                "value": 70,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        height=280,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig

def radar_chart(air_temp, proc_temp, rpm, torque, tool_wear):
    means   = scaler.mean_
    scales  = scaler.scale_
    raw     = [air_temp, proc_temp, rpm, torque, tool_wear]
    z_vals  = [(v - m) / s for v, m, s in zip(raw, means, scales)]
    normed  = [min(max((z + 3) / 6, 0), 1) * 100 for z in z_vals]  # map z to 0-100
    categories = FEATURE_NAMES + [FEATURE_NAMES[0]]
    values      = normed + [normed[0]]
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill="toself",
        fillcolor="rgba(102,126,234,0.2)",
        line=dict(color="#667eea", width=2),
        marker=dict(color="#764ba2", size=7),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1a2035",
            radialaxis=dict(visible=True, range=[0,100], tickfont=dict(color="#718096"), gridcolor="#2d3748"),
            angularaxis=dict(tickfont=dict(color="#a0aec0"), gridcolor="#2d3748"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        showlegend=False,
        height=300,
        margin=dict(t=30, b=30, l=40, r=40),
    )
    return fig

def feature_bar_chart(air_temp, proc_temp, rpm, torque, tool_wear):
    means  = scaler.mean_
    scales = scaler.scale_
    raw    = [air_temp, proc_temp, rpm, torque, tool_wear]
    contrib = [abs((v - m) / s) for v, m, s in zip(raw, means, scales)]
    colors = ["#fc8181" if c > 1.5 else "#f6ad55" if c > 0.8 else "#68d391" for c in contrib]
    fig = go.Figure(go.Bar(
        x=FEATURE_NAMES, y=contrib,
        marker_color=colors,
        text=[f"{c:.2f}σ" for c in contrib],
        textposition="outside",
        textfont=dict(color="#a0aec0", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1628",
        font=dict(color="#e2e8f0"),
        xaxis=dict(tickfont=dict(color="#a0aec0"), gridcolor="#1e2744"),
        yaxis=dict(title="Std deviations from mean", tickfont=dict(color="#718096"), gridcolor="#1e2744"),
        height=280,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
    )
    return fig

# ─── Sidebar Nav ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <div class="logo-text">⚙️ PredictIQ</div>
        <div class="logo-sub">Machine Health Monitor</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Home", "🔮  Prediction Dashboard", "📦  Batch Predictions", "📊  Model Info", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<p style="color:#4a5568; font-size:0.75rem; text-align:center;">Powered by XGBoost + Streamlit</p>', unsafe_allow_html=True)

# ─── HOME ────────────────────────────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown("## 🏠 Welcome to PredictIQ")
    st.markdown("**Real-time machine failure prediction using XGBoost — built for industrial IoT monitoring.**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Model", "XGBoost", "Binary Classifier"),
        ("Inputs", "6", "Sensor Parameters"),
        ("Output", "Binary", "Failure / Normal"),
        ("Scaler", "Standard", "Z-score Normalized"),
    ]
    for col, (label, val, sub) in zip([col1,col2,col3,col4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
                <div style="color:#4a5568;font-size:0.75rem;margin-top:4px">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">What does this app do?</div>', unsafe_allow_html=True)
        st.markdown("""
        PredictIQ monitors industrial machine health by analysing real-time sensor data.
        The model was trained on the **AI4I 2020 Predictive Maintenance Dataset** and classifies
        machines into two states:

        - ✅ **Normal** — Machine is operating within safe parameters
        - ⚠️ **Machine Failure** — Maintenance action recommended

        The dashboard provides a **health score**, **failure probability**, and an
        **automated maintenance recommendation** to help operators act proactively.
        """)

    with col_r:
        st.markdown('<div class="section-header">Input Parameters</div>', unsafe_allow_html=True)
        params = {
            "Machine Type": "M / L / H (Low / Medium / High quality)",
            "Air Temperature (K)": "Ambient temperature ~300 K",
            "Process Temperature (K)": "Process heat ~310 K",
            "Rotational Speed (RPM)": "Motor speed ~1500 RPM",
            "Torque (Nm)": "Shaft torque ~40 Nm",
            "Tool Wear (min)": "Cumulative tool wear 0–250 min",
        }
        for k, v in params.items():
            st.markdown(f"- **{k}**: {v}")

    st.markdown("---")
    st.info("👈 Navigate to **Prediction Dashboard** in the sidebar to make predictions.")

# ─── PREDICTION DASHBOARD ────────────────────────────────────────────────────
elif page == "🔮  Prediction Dashboard":
    st.markdown("## 🔮 Prediction Dashboard")
    st.markdown("Enter sensor values below and click **Predict** to assess machine health.")
    st.markdown("---")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🏭 Machine Configuration**")
            machine_type = st.selectbox(
                "Machine Type",
                ["M", "L", "H"],
                help="M = Medium, L = Low, H = High quality variant",
            )
            air_temp = st.slider(
                "Air Temperature (K)", 295.0, 305.0, 300.0, step=0.1,
                help="Ambient air temperature in Kelvin",
            )
            proc_temp = st.slider(
                "Process Temperature (K)", 305.0, 315.0, 310.0, step=0.1,
                help="Process temperature in Kelvin",
            )

        with col2:
            st.markdown("**⚡ Motor Parameters**")
            rpm = st.slider(
                "Rotational Speed (RPM)", 1000, 2800, 1500, step=10,
                help="Motor rotational speed",
            )
            torque = st.slider(
                "Torque (Nm)", 3.0, 80.0, 40.0, step=0.5,
                help="Shaft torque in Newton-metres",
            )

        with col3:
            st.markdown("**🔧 Tool Condition**")
            tool_wear = st.slider(
                "Tool Wear (min)", 0, 250, 100, step=1,
                help="Cumulative tool wear time in minutes",
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#1a2035; border:1px solid #2d3748; border-radius:12px; padding:16px; margin-top:8px;">
                <div style="color:#718096; font-size:0.78rem; text-transform:uppercase; letter-spacing:1px;">Current Inputs</div>
                <div style="color:#a0aec0; font-size:0.85rem; margin-top:8px; line-height:1.8;">
                    Type: <b style="color:#e2e8f0">{machine_type}</b><br>
                    Air: <b style="color:#e2e8f0">{air_temp} K</b><br>
                    Process: <b style="color:#e2e8f0">{proc_temp} K</b><br>
                    Speed: <b style="color:#e2e8f0">{rpm} RPM</b><br>
                    Torque: <b style="color:#e2e8f0">{torque} Nm</b><br>
                    Wear: <b style="color:#e2e8f0">{tool_wear} min</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        submitted = st.form_submit_button("⚡ Predict Machine Health")

    if submitted:
        pred, fail_prob, health_score = predict_single(
            machine_type, air_temp, proc_temp, rpm, torque, tool_wear
        )

        # Determine status
        if pred == 0 and health_score >= 70:
            status, card_class, badge_class, emoji = "Normal Operation", "result-safe", "badge-safe", "✅"
            recommendation = "Machine is healthy. Continue normal operations."
        elif pred == 1 or health_score < 40:
            status, card_class, badge_class, emoji = "Machine Failure Risk", "result-danger", "badge-danger", "⚠️"
            recommendation = "Immediate inspection required. Schedule maintenance within 24 hours."
        else:
            status, card_class, badge_class, emoji = "Caution — Monitor Closely", "result-warning", "badge-warn", "🔶"
            recommendation = "Elevated risk detected. Schedule maintenance within 48–72 hours."

        st.markdown("<br>", unsafe_allow_html=True)

        # Result card
        st.markdown(f"""
        <div class="{card_class}">
            <div style="font-size:3rem; margin-bottom:8px">{emoji}</div>
            <div style="font-size:1.8rem; font-weight:800; color:#e2e8f0; margin-bottom:8px">{status}</div>
            <div style="font-size:1rem; color:#a0aec0; margin-bottom:16px">📋 {recommendation}</div>
            <span class="badge {badge_class}">Failure Probability: {fail_prob*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{health_score}%</div>
                <div class="metric-label">Health Score</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{fail_prob*100:.1f}%</div>
                <div class="metric-label">Failure Probability</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'HIGH' if pred==1 else 'LOW'}</div>
                <div class="metric-label">Risk Level</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-header">Health Gauge</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(health_score), use_container_width=True)
        with c2:
            st.markdown('<div class="section-header">Sensor Radar</div>', unsafe_allow_html=True)
            st.plotly_chart(radar_chart(air_temp, proc_temp, rpm, torque, tool_wear), use_container_width=True)
        with c3:
            st.markdown('<div class="section-header">Deviation from Mean</div>', unsafe_allow_html=True)
            st.plotly_chart(feature_bar_chart(air_temp, proc_temp, rpm, torque, tool_wear), use_container_width=True)

# ─── BATCH PREDICTIONS ───────────────────────────────────────────────────────
elif page == "📦  Batch Predictions":
    st.markdown("## 📦 Batch Predictions")
    st.markdown("Upload a CSV file with sensor data to run predictions on multiple machines at once.")
    st.markdown("---")

    st.markdown("**📋 Required CSV columns:**")
    st.code("Machine type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear", language="text")

    # Sample CSV download
    sample = pd.DataFrame({
        "Machine type":        ["M", "L", "H", "M", "L"],
        "Air temperature":     [298.1, 300.5, 302.0, 299.3, 301.7],
        "Process temperature": [308.6, 310.0, 312.1, 309.4, 311.2],
        "Rotational speed":    [1551, 1408, 1862, 1305, 1500],
        "Torque":              [42.8, 46.3, 28.9, 69.9, 40.0],
        "Tool wear":           [0, 54, 200, 235, 100],
    })
    st.download_button(
        "⬇️ Download Sample CSV",
        sample.to_csv(index=False),
        "sample_sensor_data.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.markdown(f"**Loaded {len(df_in)} rows.** Preview:")
            st.dataframe(df_in.head(), use_container_width=True)

            with st.spinner("Running predictions..."):
                df_out = predict_batch(df_in)

            st.markdown("### 📊 Prediction Results")
            st.dataframe(df_out, use_container_width=True)

            # Summary
            n_fail = (df_out["Prediction"] == "⚠️ Failure").sum() if "Prediction" in df_out.columns else 0
            n_ok   = len(df_out) - n_fail
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Machines", len(df_out))
            with col2:
                st.metric("Normal", n_ok)
            with col3:
                st.metric("Failure Risk", n_fail)

            if "Health Score (%)" in df_out.columns:
                fig = px.histogram(
                    df_out, x="Health Score (%)",
                    nbins=20, color_discrete_sequence=["#667eea"],
                    title="Health Score Distribution",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0f1628",
                    font=dict(color="#e2e8f0"),
                    title_font=dict(color="#e2e8f0"),
                    xaxis=dict(gridcolor="#1e2744"),
                    yaxis=dict(gridcolor="#1e2744"),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                "⬇️ Download Results CSV",
                df_out.to_csv(index=False),
                "predictions.csv",
                "text/csv",
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ─── MODEL INFO ─────────────────────────────────────────────────────────────
elif page == "📊  Model Info":
    st.markdown("## 📊 Model Information")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Model Details</div>', unsafe_allow_html=True)
        details = {
            "Algorithm":    "XGBoost (eXtreme Gradient Boosting)",
            "Task":         "Binary Classification",
            "Output 0":     "Normal Operation",
            "Output 1":     "Machine Failure",
            "Scaler":       "StandardScaler (Z-score normalization)",
            "Numerical Features": ", ".join(FEATURE_NAMES),
            "Categorical Feature": "Machine Type (one-hot: H / L / M)",
            "Total Input Dims": "8 (5 scaled numerical + 3 OHE)",
        }
        for k, v in details.items():
            st.markdown(f"**{k}:** {v}")

    with col2:
        st.markdown('<div class="section-header">Scaler Statistics</div>', unsafe_allow_html=True)
        scaler_df = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Mean":    [f"{m:.3f}" for m in scaler.mean_],
            "Std Dev": [f"{s:.3f}" for s in scaler.scale_],
        })
        st.dataframe(scaler_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Feature Encoding Reference</div>', unsafe_allow_html=True)
    enc_df = pd.DataFrame({
        "Machine Type": ["H (High)", "L (Low)", "M (Medium)"],
        "type_H": [1, 0, 0],
        "type_L": [0, 1, 0],
        "type_M": [0, 0, 1],
    })
    st.dataframe(enc_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Scaler Mean Visualisation</div>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=FEATURE_NAMES,
        y=scaler.mean_,
        marker_color=["#667eea","#764ba2","#48bb78","#f6ad55","#fc8181"],
        text=[f"{m:.1f}" for m in scaler.mean_],
        textposition="outside",
        textfont=dict(color="#a0aec0"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1628",
        font=dict(color="#e2e8f0"),
        xaxis=dict(tickfont=dict(color="#a0aec0"), gridcolor="#1e2744"),
        yaxis=dict(title="Mean Value", tickfont=dict(color="#718096"), gridcolor="#1e2744"),
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── ABOUT ───────────────────────────────────────────────────────────────────
elif page == "ℹ️  About":
    st.markdown("## ℹ️ About PredictIQ")
    st.markdown("---")

    st.markdown("""
    ### 🔧 Project Overview
    **PredictIQ** is an AI-powered predictive maintenance dashboard designed to help industrial operators
    monitor machine health in real-time using sensor data.

    ---
    ### 📐 Technical Architecture
    | Layer | Technology |
    |---|---|
    | Frontend | Streamlit |
    | ML Model | XGBoost Binary Classifier |
    | Preprocessing | Scikit-learn StandardScaler |
    | Serialization | Joblib |
    | Visualisation | Plotly |
    | Deployment | Streamlit Cloud / HuggingFace Spaces / Render |

    ---
    ### 📊 Dataset
    Based on the **AI4I 2020 Predictive Maintenance Dataset** — a realistic synthetic dataset
    reflecting real predictive maintenance data encountered in industry.

    **Target variable:** Machine failure (binary: 0 = Normal, 1 = Failure)

    ---
    ### 🚀 Running Locally
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```

    ---
    ### ☁️ Deployment
    - **Streamlit Cloud**: Push to GitHub → deploy via share.streamlit.io
    - **HuggingFace Spaces**: Use Streamlit SDK in Space settings
    - **Render**: Add `start_command = streamlit run app.py` in render.yaml
    """)
