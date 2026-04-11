import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictIQ — Machine Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Force dark always */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background-color: #07090f !important;
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1120 0%, #090c18 100%) !important;
    border-right: 1px solid #1a2340 !important;
}
[data-testid="stSidebar"] * { color: #a0aec0 !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px; }
h1,h2,h3 { color: #f0f4ff !important; font-weight: 800 !important; letter-spacing: -0.5px; }

/* Page banner */
.page-banner {
    background: linear-gradient(135deg, #0f1535 0%, #1a0a2e 50%, #0a1520 100%);
    border: 1px solid #1e2d55;
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.page-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(102,126,234,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.banner-title {
    font-size: 1.9rem; font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #c084fc, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.banner-sub { color: #64748b; font-size: 0.9rem; }

/* KPI card */
.kpi-card {
    background: linear-gradient(135deg, #111827 0%, #0d1117 100%);
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    transition: all 0.2s ease;
}
.kpi-card:hover { border-color: #3b4fd8; transform: translateY(-2px); }
.kpi-val {
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.7rem; color: #475569;
    text-transform: uppercase; letter-spacing: 1.2px; margin-top: 6px;
}

/* Result banners */
.result-safe {
    background: linear-gradient(135deg, #052e16, #071a12);
    border: 1.5px solid #16a34a; border-radius: 18px;
    padding: 28px 32px; text-align: center; animation: slideUp 0.4s ease;
}
.result-danger {
    background: linear-gradient(135deg, #2d0a0a, #1a0808);
    border: 1.5px solid #dc2626; border-radius: 18px;
    padding: 28px 32px; text-align: center; animation: slideUp 0.4s ease;
}
.result-warning {
    background: linear-gradient(135deg, #2d1f00, #1a1200);
    border: 1.5px solid #d97706; border-radius: 18px;
    padding: 28px 32px; text-align: center; animation: slideUp 0.4s ease;
}
@keyframes slideUp {
    from { opacity:0; transform:translateY(16px); }
    to   { opacity:1; transform:translateY(0); }
}

/* Badges */
.badge { display:inline-block; padding:5px 16px; border-radius:30px; font-size:0.8rem; font-weight:600; }
.badge-safe   { background:rgba(22,163,74,.15);  color:#4ade80; border:1px solid #16a34a44; }
.badge-danger { background:rgba(220,38,38,.15);  color:#f87171; border:1px solid #dc262644; }
.badge-warn   { background:rgba(217,119,6,.15);  color:#fbbf24; border:1px solid #d9770644; }

/* Section header */
.sec-hdr {
    font-size:1.0rem; font-weight:700; color:#94a3b8 !important;
    text-transform:uppercase; letter-spacing:1.5px;
    border-left:3px solid #6366f1; padding-left:12px;
    margin:28px 0 14px 0;
}

/* Inputs */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background:#111827 !important; border:1px solid #1e2d45 !important;
    border-radius:10px !important; color:#e2e8f0 !important; font-size:0.95rem !important;
}
label { color:#94a3b8 !important; font-size:0.82rem !important; font-weight:500 !important; }
textarea { background:#111827 !important; border:1px solid #1e2d45 !important;
           border-radius:10px !important; color:#e2e8f0 !important; }
[data-testid="stDateInput"] input { background:#111827 !important; border:1px solid #1e2d45 !important;
                                    color:#e2e8f0 !important; border-radius:10px !important; }

/* Buttons */
.stFormSubmitButton > button, .stButton > button {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    color:#fff !important; border:none !important; border-radius:12px !important;
    padding:13px 28px !important; font-size:0.95rem !important; font-weight:700 !important;
    width:100% !important; transition:opacity .2s, transform .15s !important;
    box-shadow:0 4px 20px rgba(79,70,229,.35) !important;
}
.stFormSubmitButton > button:hover, .stButton > button:hover {
    opacity:.88 !important; transform:translateY(-1px) !important;
}
[data-testid="stDownloadButton"] > button {
    background:rgba(99,102,241,.12) !important; color:#a5b4fc !important;
    border:1px solid rgba(99,102,241,.3) !important; border-radius:10px !important;
    font-weight:600 !important; width:auto !important; padding:10px 20px !important;
}

/* Cards */
.book-card {
    background:linear-gradient(135deg,#2d0a0a,#1a0808);
    border:2px solid #dc2626; border-radius:20px;
    padding:24px 28px; margin:20px 0; animation:slideUp .5s ease;
}
.confirm-card {
    background:linear-gradient(135deg,#052e16,#071a12);
    border:2px solid #16a34a; border-radius:20px;
    padding:24px 28px; margin-top:16px; animation:slideUp .5s ease;
}

hr { border-color:#1a2340 !important; margin:16px 0 !important; }
.stDataFrame, [data-testid="stDataFrame"] {
    border-radius:14px !important; overflow:hidden !important;
    border:1px solid #1e2d45 !important;
}
[data-testid="stInfo"]  { background:rgba(99,102,241,.08)!important; border-color:#4f46e5!important; border-radius:12px!important; }
[data-testid="stError"] { background:rgba(220,38,38,.08)!important; border-radius:12px!important; }
</style>
""", unsafe_allow_html=True)

# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    return joblib.load("xgboost_binary_model.joblib"), joblib.load("scaler.joblib")

model, scaler = load_artifacts()

FEATURE_NAMES = ["Air temperature","Process temperature","Rotational speed","Torque","Tool wear"]
SHORT_LABELS  = ["Air Temp","Proc Temp","RPM","Torque","Tool Wear"]
PARAM_UNITS   = ["K","K","RPM","Nm","min"]
PARAM_ICONS   = ["🌡️","♨️","⚡","🔩","🔧"]
FAILURE_TYPES = ["No Failure","Power Failure","Overstrain Failure","Heat Dissipation Failure","Tool Wear Failure"]
FAILURE_WEIGHTS = np.array([0.0, 0.38, 0.29, 0.19, 0.14])
TYPE_ENC = {"H":0,"L":1,"M":2}

CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(t=24, b=16, l=8, r=8),
)

# ─── Prediction helpers ───────────────────────────────────────────────────────
def predict_single(machine_type, air_temp, proc_temp, rpm, torque, tool_wear):
    num    = np.array([[air_temp, proc_temp, rpm, torque, tool_wear]], dtype=float)
    scaled = scaler.transform(num)
    x      = np.hstack([[[TYPE_ENC.get(machine_type.upper(), 2)]], scaled])
    pred   = int(model.predict(x)[0])
    proba  = model.predict_proba(x)[0]
    nfp, fp = float(proba[0]), float(proba[1])
    health = round(nfp * 100, 6)
    tp = np.zeros(5)
    tp[0] = nfp
    tp[1:] = FAILURE_WEIGHTS[1:] * fp
    return pred, fp, health, tp

def predict_batch(df):
    rows = []
    for _, row in df.iterrows():
        try:
            mt = str(row.get("Machine type", row.get("machine_type","M"))).strip().upper()
            at = float(row.get("Air temperature",    row.get("air_temperature",    300)))
            pt = float(row.get("Process temperature", row.get("process_temperature",310)))
            rs = float(row.get("Rotational speed",   row.get("rotational_speed",  1500)))
            tq = float(row.get("Torque",             row.get("torque",            40)))
            tw = float(row.get("Tool wear",          row.get("tool_wear",         100)))
            pred, fp, hs, tp = predict_single(mt, at, pt, rs, tq, tw)
            rows.append({
                "Machine Type":mt, "Air Temp (K)":at, "Proc Temp (K)":pt,
                "RPM":rs, "Torque (Nm)":tq, "Tool Wear (min)":tw,
                "Prediction":"⚠️ Failure" if pred==1 else "✅ Normal",
                "No Failure (%)":round(tp[0]*100,8), "Power Failure (%)":round(tp[1]*100,8),
                "Overstrain Failure (%)":round(tp[2]*100,8),
                "Heat Dissipation Failure (%)":round(tp[3]*100,8),
                "Tool Wear Failure (%)":round(tp[4]*100,8),
                "Health Score (%)":hs,
            })
        except Exception as e:
            rows.append({"Error":str(e)})
    return pd.DataFrame(rows)

# ─── Chart helpers ────────────────────────────────────────────────────────────
def gauge_chart(health_score):
    c = "#4ade80" if health_score>=70 else "#fbbf24" if health_score>=40 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=health_score,
        delta={"reference":70,"valueformat":".2f","increasing":{"color":"#4ade80"},"decreasing":{"color":"#f87171"}},
        title={"text":"HEALTH SCORE","font":{"color":"#475569","size":10}},
        number={"font":{"color":"#f0f4ff","size":34,"family":"Inter"},"suffix":"%"},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#1e2d45","tickfont":{"color":"#334155","size":9}},
            "bar":{"color":c,"thickness":0.25},
            "bgcolor":"#111827","bordercolor":"#1e2d45","borderwidth":1,
            "steps":[{"range":[0,40],"color":"#1f0a0a"},{"range":[40,70],"color":"#1a1200"},{"range":[70,100],"color":"#052e16"}],
            "threshold":{"line":{"color":"#6366f1","width":2},"thickness":0.8,"value":70},
        },
    ))
    fig.update_layout(**CHART_BASE, height=260, margin=dict(t=50,b=16,l=16,r=16))
    return fig

def failure_bar(type_probs):
    colors = ["#4ade80","#f87171","#fbbf24","#60a5fa","#c084fc"]
    fig = go.Figure(go.Bar(
        x=FAILURE_TYPES, y=[p*100 for p in type_probs],
        marker_color=colors, opacity=0.85,
        text=[f"{p*100:.6f}%" for p in type_probs],
        textposition="outside", textfont=dict(color="#94a3b8", size=8),
    ))
    fig.update_layout(**CHART_BASE, height=280,
        xaxis=dict(tickfont=dict(color="#94a3b8",size=9), gridcolor="#1a2340"),
        yaxis=dict(title="Probability (%)", titlefont=dict(color="#475569",size=10),
                   tickfont=dict(color="#475569",size=9), gridcolor="#1a2340"))
    return fig

def radar_overlay(u_vals, h_vals, s_min, s_max):
    def norm(vals):
        return [min(max((v-mn)/(mx-mn) if mx!=mn else 0.5, 0),1)*100
                for v,mn,mx in zip(vals,s_min,s_max)]
    un, hn = norm(u_vals), norm(h_vals)
    cats = SHORT_LABELS + [SHORT_LABELS[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=hn+[hn[0]], theta=cats, fill="toself",
        fillcolor="rgba(74,222,128,0.08)", line=dict(color="#4ade80",width=1.5,dash="dash"),
        name="✅ Healthy", marker=dict(size=5,color="#4ade80")))
    fig.add_trace(go.Scatterpolar(r=un+[un[0]], theta=cats, fill="toself",
        fillcolor="rgba(99,102,241,0.15)", line=dict(color="#818cf8",width=2),
        name="🔵 Your Machine", marker=dict(size=6,color="#c084fc")))
    fig.update_layout(**CHART_BASE, height=340,
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(visible=True,range=[0,100],tickfont=dict(color="#334155",size=9),gridcolor="#1e2d45"),
            angularaxis=dict(tickfont=dict(color="#64748b",size=10),gridcolor="#1e2d45"),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#94a3b8",size=10)),
        showlegend=True, margin=dict(t=20,b=20,l=30,r=30))
    return fig

def grouped_bar(u_vals, h_vals, bar_colors):
    u_pct = [u/h*100 if h!=0 else 100 for u,h in zip(u_vals,h_vals)]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="✅ Healthy", x=SHORT_LABELS, y=[100]*5,
        marker_color="#1e3a2e", marker_line_color="#16a34a", marker_line_width=1,
        opacity=0.8, text=["100%"]*5, textposition="outside",
        textfont=dict(color="#475569",size=9)))
    fig.add_trace(go.Bar(name="🔵 Your Machine", x=SHORT_LABELS, y=u_pct,
        marker_color=bar_colors, opacity=0.9,
        text=[f"{v:.1f}%" for v in u_pct], textposition="outside",
        textfont=dict(color="#e2e8f0",size=9)))
    fig.update_layout(**CHART_BASE, barmode="group", height=300,
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#94a3b8",size=10)),
        xaxis=dict(tickfont=dict(color="#94a3b8",size=10),gridcolor="#1a2340"),
        yaxis=dict(title="% of healthy reference", titlefont=dict(color="#475569",size=10),
                   tickfont=dict(color="#475569",size=9), gridcolor="#1a2340"))
    return fig

def line_compare(u_vals, h_vals):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=SHORT_LABELS, y=h_vals, mode="lines+markers", name="✅ Healthy",
        line=dict(color="#4ade80",width=2,dash="dash"),
        marker=dict(size=8,color="#4ade80",line=dict(color="#052e16",width=2))))
    fig.add_trace(go.Scatter(x=SHORT_LABELS, y=u_vals, mode="lines+markers", name="🔵 Your Machine",
        line=dict(color="#818cf8",width=2.5),
        marker=dict(size=9,color="#c084fc",line=dict(color="#1e1b4b",width=2))))
    fig.update_layout(**CHART_BASE, height=300,
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#94a3b8",size=10)),
        xaxis=dict(tickfont=dict(color="#94a3b8",size=10),gridcolor="#1a2340"),
        yaxis=dict(title="Raw value", titlefont=dict(color="#475569",size=10),
                   tickfont=dict(color="#475569",size=9), gridcolor="#1a2340"))
    return fig

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 8px 20px; text-align:center; border-bottom:1px solid #1a2340; margin-bottom:20px;">
        <div style="font-size:2rem; margin-bottom:6px;">⚙️</div>
        <div style="font-size:1.15rem; font-weight:800;
                    background:linear-gradient(135deg,#a5b4fc,#c084fc);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.3px;">
            PredictIQ
        </div>
        <div style="font-size:0.7rem; color:#334155; margin-top:3px; text-transform:uppercase; letter-spacing:1px;">
            Machine Health Monitor
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("nav",
        ["🔮  Prediction Dashboard", "📦  Batch Predictions"],
        index=0, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#0d1117; border:1px solid #1a2340; border-radius:14px; padding:16px; margin-top:8px;">
        <div style="font-size:0.68rem; color:#334155; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;">Quick Reference</div>
        <div style="font-size:0.78rem; color:#64748b; line-height:2.1;">
            🌡️ Air Temp &nbsp;<b style="color:#94a3b8">~300 K</b><br>
            ♨️ Proc Temp &nbsp;<b style="color:#94a3b8">~310 K</b><br>
            ⚡ Speed &nbsp;<b style="color:#94a3b8">~1500 RPM</b><br>
            🔩 Torque &nbsp;<b style="color:#94a3b8">~40 Nm</b><br>
            🔧 Tool Wear &nbsp;<b style="color:#94a3b8">~100 min</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮  Prediction Dashboard":

    st.markdown("""
    <div class="page-banner">
        <div class="banner-title">🔮 Prediction Dashboard</div>
        <div class="banner-sub">Enter sensor readings to assess machine health and predict failure risk in real-time</div>
    </div>
    """, unsafe_allow_html=True)

    # Input form
    with st.form("pred_form", border=False):
        st.markdown("""
        <div style="background:#0d1117; border:1px solid #1a2340; border-radius:18px;
                    padding:22px 26px; margin-bottom:20px;">
            <div style="font-size:0.7rem; color:#475569; text-transform:uppercase;
                        letter-spacing:1.5px; margin-bottom:18px; font-weight:600;">
                ⚙️ &nbsp; Sensor Input Parameters
            </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns([1.2,1,1,1,1,1])
        with c1: machine_type = st.selectbox("🏭 Machine Type", ["M","L","H"],
                    help="M=Medium · L=Low · H=High quality")
        with c2: air_temp  = st.number_input("🌡️ Air Temp (K)",    value=300.0, step=0.1, format="%.2f")
        with c3: proc_temp = st.number_input("♨️ Process Temp (K)", value=310.0, step=0.1, format="%.2f")
        with c4: rpm       = st.number_input("⚡ Speed (RPM)",       value=1500,  step=1)
        with c5: torque    = st.number_input("🔩 Torque (Nm)",       value=40.0,  step=0.1, format="%.2f")
        with c6: tool_wear = st.number_input("🔧 Tool Wear (min)",   value=100,   step=1)

        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡  Analyse Machine Health", use_container_width=True)

    if submitted:
        with st.spinner("Running prediction..."):
            pred, fail_prob, health_score, type_probs = predict_single(
                machine_type, air_temp, proc_temp, rpm, torque, tool_wear)

        dominant_idx  = int(np.argmax(type_probs))
        dominant_type = FAILURE_TYPES[dominant_idx]

        if pred == 0 and health_score >= 70:
            status, card_cls, badge_cls, emoji = "No Failure Detected", "result-safe",    "badge-safe",   "✅"
            rec = "Machine is operating within healthy parameters. Continue monitoring."
        elif pred == 1 or health_score < 40:
            status, card_cls, badge_cls, emoji = f"Failure Detected · {dominant_type}", "result-danger", "badge-danger", "🚨"
            rec = "Immediate inspection required. Schedule maintenance within 24 hours."
        else:
            status, card_cls, badge_cls, emoji = "Caution — Elevated Risk", "result-warning","badge-warn","⚠️"
            rec = "Elevated risk detected. Schedule maintenance within 48–72 hours."

        # Result banner
        st.markdown(f"""
        <div class="{card_cls}">
            <div style="font-size:2.6rem; margin-bottom:8px;">{emoji}</div>
            <div style="font-size:1.55rem; font-weight:800; color:#f0f4ff; margin-bottom:6px;">{status}</div>
            <div style="color:#94a3b8; font-size:0.88rem; margin-bottom:16px;">📋 {rec}</div>
            <span class="badge {badge_cls}">Overall Failure Probability: {fail_prob*100:.10f}%</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # KPI row
        k1,k2,k3,k4 = st.columns(4)
        for col, lbl, val in [
            (k1,"Health Score",      f"{health_score:.4f}%"),
            (k2,"Failure Prob",      f"{fail_prob*100:.8f}%"),
            (k3,"Risk Level",        "HIGH 🔴" if pred==1 else "LOW 🟢"),
            (k4,"Dominant Failure",  dominant_type if pred==1 else "None"),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-val" style="font-size:1.05rem;">{val}</div>
                    <div class="kpi-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Failure probability table ─────────────────────────────────────────
        st.markdown('<div class="sec-hdr">Failure Type Probability Breakdown</div>', unsafe_allow_html=True)
        ft_colors = ["#4ade80","#f87171","#fbbf24","#60a5fa","#c084fc"]
        rows_html = ""
        for i,(ft,tp,ch) in enumerate(zip(FAILURE_TYPES, type_probs, ft_colors)):
            bar_pct = min(max(tp*500, 1), 100)
            bold = "font-weight:700;" if i==dominant_idx else ""
            rows_html += f"""
            <tr style="border-bottom:1px solid #1a2340;">
                <td style="padding:11px 16px; color:#e2e8f0; {bold} white-space:nowrap; width:220px;">{ft}</td>
                <td style="padding:11px 16px; width:220px;">
                    <div style="background:#111827; border-radius:6px; height:8px;">
                        <div style="background:{ch}; border-radius:6px; height:8px; width:{bar_pct:.1f}%;
                                    box-shadow:0 0 6px {ch}55;"></div>
                    </div>
                </td>
                <td style="padding:11px 16px; color:{ch}; font-family:monospace; font-size:0.81rem; {bold} white-space:nowrap;">{tp:.12f}</td>
                <td style="padding:11px 16px; color:{ch}; font-family:monospace; font-size:0.81rem; {bold} white-space:nowrap;">{tp*100:.10f}%</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#0d1117; border:1px solid #1a2340; border-radius:16px; overflow:hidden;">
            <table style="width:100%; border-collapse:collapse;">
                <thead>
                    <tr style="background:#111827; border-bottom:1px solid #1e2d45;">
                        <th style="padding:11px 16px; color:#475569; font-size:0.69rem; text-transform:uppercase; letter-spacing:1.2px; text-align:left;">Failure Type</th>
                        <th style="padding:11px 16px; color:#475569; font-size:0.69rem; text-transform:uppercase; letter-spacing:1.2px; text-align:left;">Probability Bar</th>
                        <th style="padding:11px 16px; color:#475569; font-size:0.69rem; text-transform:uppercase; letter-spacing:1.2px; text-align:left;">Raw Probability</th>
                        <th style="padding:11px 16px; color:#475569; font-size:0.69rem; text-transform:uppercase; letter-spacing:1.2px; text-align:left;">Percentage</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # ── Chart row 1: gauge + failure bar ─────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns([1, 1.6])
        with ch1:
            st.markdown('<div class="sec-hdr">Health Score Gauge</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(health_score), use_container_width=True)
        with ch2:
            st.markdown('<div class="sec-hdr">Failure Type Distribution</div>', unsafe_allow_html=True)
            st.plotly_chart(failure_bar(type_probs), use_container_width=True)

        # ── vs Healthy comparison ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="sec-hdr">📊 Your Machine vs Healthy Machine Benchmark</div>', unsafe_allow_html=True)

        HEALTHY   = list(scaler.mean_)
        USER_VALS = [air_temp, proc_temp, float(rpm), torque, float(tool_wear)]
        S_MIN     = [scaler.mean_[i] - 1.5*scaler.scale_[i] for i in range(5)]
        S_MAX     = [scaler.mean_[i] + 1.5*scaler.scale_[i] for i in range(5)]

        bar_colors = []
        for i,u in enumerate(USER_VALS):
            if S_MIN[i] <= u <= S_MAX[i]:                    bar_colors.append("#4ade80")
            elif abs(u-HEALTHY[i]) <= 2*scaler.scale_[i]:   bar_colors.append("#fbbf24")
            else:                                            bar_colors.append("#f87171")

        # 5 parameter deviation mini-cards
        pc = st.columns(5)
        for i,col in enumerate(pc):
            u,h   = USER_VALS[i], HEALTHY[i]
            dev   = (u-h)/h*100 if h!=0 else 0
            in_r  = S_MIN[i] <= u <= S_MAX[i]
            caut  = not in_r and abs(u-h) <= 2*scaler.scale_[i]
            if in_r:  sc,st_txt,bg = "#4ade80","Normal",  "#052e16"
            elif caut:sc,st_txt,bg = "#fbbf24","Caution", "#1a1200"
            else:     sc,st_txt,bg = "#f87171","Critical","#2d0a0a"
            with col:
                st.markdown(f"""
                <div style="background:{bg}; border:1px solid {sc}33;
                            border-left:3px solid {sc}; border-radius:14px;
                            padding:16px 10px; text-align:center; margin-bottom:6px;">
                    <div style="font-size:1.4rem; margin-bottom:4px;">{PARAM_ICONS[i]}</div>
                    <div style="color:#64748b; font-size:0.67rem; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px;">{SHORT_LABELS[i]}</div>
                    <div style="color:#f0f4ff; font-size:1.2rem; font-weight:800; margin-bottom:2px;">{u:.1f}</div>
                    <div style="color:#334155; font-size:0.7rem; margin-bottom:6px;">Healthy: {h:.1f} {PARAM_UNITS[i]}</div>
                    <div style="color:{sc}; font-size:0.73rem; font-weight:700;">{dev:+.2f}% · {st_txt}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Grouped bar + line
        cb1, cb2 = st.columns(2)
        with cb1:
            st.markdown('<div class="sec-hdr">Grouped Bar — % of Healthy Reference</div>', unsafe_allow_html=True)
            st.plotly_chart(grouped_bar(USER_VALS, HEALTHY, bar_colors), use_container_width=True)
        with cb2:
            st.markdown('<div class="sec-hdr">Line Overlay — Absolute Values</div>', unsafe_allow_html=True)
            st.plotly_chart(line_compare(USER_VALS, HEALTHY), use_container_width=True)

        # Radar
        st.markdown('<div class="sec-hdr">Radar Overlay — Normalised Parameter Profile</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_overlay(USER_VALS, HEALTHY, S_MIN, S_MAX), use_container_width=True)

        # ── Technician Booking ────────────────────────────────────────────────
        if pred == 1 or health_score < 70:
            st.markdown("---")
            st.markdown("""
            <div class="book-card">
                <div style="display:flex; align-items:center; gap:14px;">
                    <div style="font-size:2rem;">🚨</div>
                    <div>
                        <div style="font-size:1.2rem; font-weight:800; color:#f87171;">Maintenance Required</div>
                        <div style="color:#64748b; font-size:0.86rem; margin-top:3px;">
                            Your machine shows signs of failure. Book a certified technician below.
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("tech_form", border=False):
                st.markdown("""
                <div style="background:#0d1117; border:1px solid #1a2340; border-radius:18px;
                            padding:22px 26px; margin-bottom:20px;">
                    <div style="font-size:0.7rem; color:#475569; text-transform:uppercase;
                                letter-spacing:1.5px; margin-bottom:18px; font-weight:600;">
                        📅 &nbsp; Appointment Details
                    </div>
                """, unsafe_allow_html=True)

                f1, f2 = st.columns(2)
                with f1:
                    t_name     = st.text_input("👤 Your Name",        placeholder="e.g. Rajesh Kumar")
                    t_email    = st.text_input("📧 Email",             placeholder="e.g. rajesh@factory.com")
                    t_phone    = st.text_input("📞 Phone",             placeholder="e.g. +91 98765 43210")
                    t_location = st.text_input("📍 Plant / Location",  placeholder="e.g. Unit 3 — Indore Plant")
                with f2:
                    t_machine  = st.text_input("🏭 Machine ID",        placeholder="e.g. CNC-07 / Lathe-3")
                    t_priority = st.selectbox("🚦 Priority Level", [
                        "🔴 Critical — Within 24 hrs",
                        "🟠 High — Within 48 hrs",
                        "🟡 Medium — Within 72 hrs",
                    ])
                    t_date = st.date_input("📅 Preferred Date")
                    t_time = st.selectbox("🕐 Time Slot", [
                        "08:00 – 10:00","10:00 – 12:00",
                        "12:00 – 14:00","14:00 – 16:00","16:00 – 18:00",
                    ])
                    t_failure = st.selectbox("⚠️ Failure Type Detected", FAILURE_TYPES[1:],
                        index=max(0, dominant_idx-1))

                t_notes = st.text_area("📝 Notes (optional)",
                    placeholder="Describe unusual sounds, vibrations, or recent events...", height=80)
                st.markdown("</div>", unsafe_allow_html=True)
                book_btn = st.form_submit_button("📅  Confirm & Book Technician", use_container_width=True)

            if book_btn:
                if not t_name or not t_email or not t_phone:
                    st.error("Please provide Name, Email, and Phone to confirm.")
                else:
                    st.markdown(f"""
                    <div class="confirm-card">
                        <div style="font-size:1.35rem; font-weight:800; color:#4ade80; margin-bottom:18px;">
                            ✅ Appointment Confirmed!
                        </div>
                        <div style="display:grid; grid-template-columns:160px 1fr; gap:6px 20px;
                                    font-size:0.87rem; line-height:2.3;">
                            <div style="color:#475569;">👤 Name</div>       <div style="color:#e2e8f0;font-weight:600;">{t_name}</div>
                            <div style="color:#475569;">📧 Email</div>      <div style="color:#e2e8f0;font-weight:600;">{t_email}</div>
                            <div style="color:#475569;">📞 Phone</div>      <div style="color:#e2e8f0;font-weight:600;">{t_phone}</div>
                            <div style="color:#475569;">📍 Location</div>   <div style="color:#e2e8f0;font-weight:600;">{t_location or "—"}</div>
                            <div style="color:#475569;">🏭 Machine</div>    <div style="color:#e2e8f0;font-weight:600;">{t_machine or "—"}</div>
                            <div style="color:#475569;">🚦 Priority</div>   <div style="color:#f87171;font-weight:600;">{t_priority}</div>
                            <div style="color:#475569;">📅 Date & Time</div><div style="color:#e2e8f0;font-weight:600;">{t_date.strftime("%d %B %Y")} · {t_time}</div>
                            <div style="color:#475569;">⚠️ Failure</div>   <div style="color:#fbbf24;font-weight:600;">{t_failure}</div>
                        </div>
                        {f'<div style="margin-top:12px; padding:10px 14px; background:#071a12; border-radius:10px; color:#64748b; font-size:0.82rem;">📝 {t_notes}</div>' if t_notes else ""}
                        <div style="margin-top:16px; padding:14px 16px; background:#071a12; border-radius:12px;
                                    color:#4ade80; font-size:0.84rem; border:1px solid #16a34a33;">
                            📬 Our technician team will contact <b>{t_name}</b> at <b>{t_email}</b> within 2 hours.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦  Batch Predictions":

    st.markdown("""
    <div class="page-banner">
        <div class="banner-title">📦 Batch Predictions</div>
        <div class="banner-sub">Upload a CSV of sensor readings to run predictions on multiple machines at once</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0d1117; border:1px solid #1a2340; border-radius:16px; padding:20px 24px; margin-bottom:20px;">
        <div style="font-size:0.69rem; color:#475569; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px; font-weight:600;">
            📋 &nbsp; Required CSV Columns
        </div>
        <code style="background:#111827; color:#818cf8; padding:8px 14px; border-radius:8px;
                     font-size:0.85rem; display:block; border:1px solid #1e2d45;">
            Machine type &nbsp;·&nbsp; Air temperature &nbsp;·&nbsp; Process temperature
            &nbsp;·&nbsp; Rotational speed &nbsp;·&nbsp; Torque &nbsp;·&nbsp; Tool wear
        </code>
    </div>
    """, unsafe_allow_html=True)

    sample = pd.DataFrame({
        "Machine type":        ["M","L","H","M","L"],
        "Air temperature":     [298.1,300.5,302.0,299.3,301.7],
        "Process temperature": [308.6,310.0,312.1,309.4,311.2],
        "Rotational speed":    [1551,1408,1862,1305,1500],
        "Torque":              [42.8,46.3,28.9,69.9,40.0],
        "Tool wear":           [0,54,200,235,100],
    })
    st.download_button("⬇️ Download Sample CSV", sample.to_csv(index=False),
                       "sample_sensor_data.csv", "text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            st.markdown(f'<div style="color:#4ade80; font-size:0.88rem; margin:8px 0 16px;">✅ &nbsp; Loaded <b>{len(df_in)}</b> rows successfully</div>', unsafe_allow_html=True)
            st.dataframe(df_in.head(), use_container_width=True)

            with st.spinner("Running predictions..."):
                df_out = predict_batch(df_in)

            st.markdown('<div class="sec-hdr">Prediction Results</div>', unsafe_allow_html=True)
            st.dataframe(df_out, use_container_width=True)

            n_fail = (df_out.get("Prediction","")=="⚠️ Failure").sum() if "Prediction" in df_out.columns else 0
            n_ok   = len(df_out) - n_fail
            s1,s2,s3 = st.columns(3)
            for col, lbl, val, color in [
                (s1,"Total Machines",len(df_out),"#818cf8"),
                (s2,"Normal",        n_ok,       "#4ade80"),
                (s3,"Failure Risk",  n_fail,     "#f87171"),
            ]:
                with col:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-val" style="color:{color}; -webkit-text-fill-color:{color};">{val}</div>
                        <div class="kpi-label">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            if "Health Score (%)" in df_out.columns:
                st.markdown('<div class="sec-hdr">Health Score Distribution</div>', unsafe_allow_html=True)
                fig_h = px.histogram(df_out, x="Health Score (%)", nbins=20,
                    color_discrete_sequence=["#6366f1"])
                fig_h.update_layout(**CHART_BASE, height=280,
                    xaxis=dict(gridcolor="#1a2340",tickfont=dict(color="#94a3b8")),
                    yaxis=dict(gridcolor="#1a2340",tickfont=dict(color="#64748b")))
                st.plotly_chart(fig_h, use_container_width=True)

            st.download_button("⬇️ Download Results CSV", df_out.to_csv(index=False),
                               "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
