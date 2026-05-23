import streamlit as st
import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
import gdown
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #030712 !important;
    color: #f1f5f9 !important;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden !important; }
.block-container { padding: 0 2.5rem 4rem !important; max-width: 1400px !important; }
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 900px 500px at 10% 5%,  rgba(16,185,129,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 700px 400px at 90% 90%, rgba(239,68,68,0.05)  0%, transparent 70%);
    pointer-events: none; z-index: 0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0a1628 0%, #06101e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.25rem !important; }
[data-testid="stFileUploader"] section {
    background: rgba(16,185,129,0.03) !important;
    border: 1.5px dashed rgba(16,185,129,0.2) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(16,185,129,0.1) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
    color: #10b981 !important;
    border-radius: 8px !important;
}
.stSpinner > div { border-top-color: #10b981 !important; }
</style>
""", unsafe_allow_html=True)

# ── Feature names and presets ──────────────────────────────────
FEAT_NAMES = [
    "Pkt Len Var","Bwd Pkt Len Mean","Pkt Len Mean","Bwd Pkt Len Max",
    "Fwd IAT Std","Tot Len Fwd Pkts","Fwd Pkt Len Mean","Max Pkt Len",
    "Fwd Pkt Len Max","Fwd Hdr Len","Avg Bwd Seg Size","Bwd IAT Mean",
    "Flow IAT Mean","Bwd IAT Total","Flow IAT Std","Fwd IAT Total",
    "Flow Duration","Fwd IAT Mean","Pkt Len Std","Bwd Pkt Len Std",
    "Avg Fwd Seg Size","Fwd Pkt Len Std","Flow Bytes/s","Bwd Pkts/s",
    "Flow Pkts/s","Tot Fwd Pkts","Tot Bwd Pkts","Fwd Pkt Len Min",
    "Bwd Pkt Len Min","Min Pkt Len","Bwd Hdr Len","Fwd PSH Flags",
    "Init Win Fwd","Init Win Bwd","Act Data Pkt Fwd","Min Seg Fwd",
    "Active Mean","Idle Mean","Fwd Avg Bytes Bulk","Bwd Avg Bytes Bulk",
]

NORMAL = [0.5,800.0,600.0,1200.0,500.0,5000.0,700.0,1400.0,
          1300.0,20.0,800.0,200000.0,150000.0,400000.0,180000.0,
          300000.0,1000000.0,160000.0,400.0,350.0,700.0,380.0,
          50000.0,80.0,120.0,5.0,4.0,200.0,100.0,100.0,
          80.0,0.0,8192.0,8192.0,3.0,20.0,0.0,0.0,0.0,0.0]

ATTACK = [50000.0,0.0,8000.0,65535.0,2000000.0,100000.0,7500.0,
          65535.0,65535.0,32.0,0.0,5000000.0,3000000.0,10000000.0,
          4000000.0,8000000.0,50000000.0,4000000.0,5000.0,4500.0,
          0.0,4800.0,2000000.0,500.0,800.0,100.0,0.0,0.0,0.0,0.0,
          64.0,1.0,1024.0,0.0,0.0,8.0,0.0,0.0,0.0,0.0]

# ── Session state for presets ──────────────────────────────────
if "defaults" not in st.session_state:
    st.session_state.defaults = NORMAL.copy()

# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════
GDRIVE = {
    "models/autoencoder.onnx"     : "1hKPGtRbI1KOjNNCFmg7SJpkVQHkBdeNF",
    "models/isolation_forest.pkl" : "1OGFhLXB68-zpLCCiS2I19HY-zg8kC7og",
    "models/ocsvm.pkl"            : "1e7OFrSKaTQw6p98muXKNTGJGoltmEFtX",
    "models/scaler_ae.pkl"        : "1jznFfgiv0fJ1S0tFsllFpMXKfQSYpbiK",
    "models/config.pkl"           : "1ZSBujLvO1lWAzdSMgzX_BV6JWKtoYNsV",
}

@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)
    for path, fid in GDRIVE.items():
        if not os.path.exists(path):
            gdown.download(
                f"https://drive.google.com/uc?id={fid}",
                path, quiet=False
            )
    sess   = ort.InferenceSession(
        "models/autoencoder.onnx",
        providers=["CPUExecutionProvider"]
    )
    iso    = joblib.load("models/isolation_forest.pkl")
    ocsvm  = joblib.load("models/ocsvm.pkl")
    scaler = joblib.load("models/scaler_ae.pkl")
    config = joblib.load("models/config.pkl")
    return sess, iso, ocsvm, scaler, config

# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════
def predict(features, sess, iso, ocsvm, scaler, config, threshold):
    x        = np.array(features, dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x).astype(np.float32)
    x_scaled = np.clip(x_scaled, -10, 10)

    # ONNX → recon(1,40) + latent(1,8)
    recon, latent = sess.run(None, {"input": x_scaled})

    # AE score
    err      = float(np.mean((x_scaled - recon) ** 2))
    ae_score = float(np.clip(np.log1p(err) / 2.5, 0, 1))

    # IF on 8-dim latent
    if_raw   = iso.decision_function(latent)
    if_score = float(np.clip(1 - (float(if_raw[0]) + 0.15) / 0.30, 0, 1))

    # SVM on 8-dim latent
    svm_raw   = ocsvm.decision_function(latent)
    svm_score = float(np.clip(1 - (float(svm_raw[0]) + 0.5) / 1.0, 0, 1))

    # Ensemble
    w_ae  = config.get('w_ae',  0.7)
    w_if  = config.get('w_if',  0.2)
    w_svm = config.get('w_svm', 0.1)
    ens   = float(w_ae * ae_score + w_if * if_score + w_svm * svm_score)
    conf  = min(abs(ens - threshold) / (threshold + 1e-8) * 100, 100.0)

    return {
        "ensemble"  : ens,
        "ae_score"  : ae_score,
        "if_score"  : if_score,
        "svm_score" : svm_score,
        "is_attack" : ens >= threshold,
        "confidence": conf,
        "ae_error"  : err,
        "latent"    : latent[0].tolist(),
    }

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;
                padding-bottom:1.4rem;border-bottom:1px solid rgba(255,255,255,0.05);
                margin-bottom:1.4rem;">
        <div style="width:38px;height:38px;border-radius:10px;
                    background:linear-gradient(135deg,#10b981,#059669);
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.2rem;">🛡️</div>
        <div>
            <div style="font-size:1rem;font-weight:800;color:#f1f5f9;">NetGuard AI</div>
            <div style="font-size:0.62rem;color:#475569;
                        font-family:'JetBrains Mono',monospace;">
                AE → Latent → IF + SVM
            </div>
        </div>
    </div>
    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;
                color:#1e3a5f;font-family:'JetBrains Mono',monospace;
                margin-bottom:0.8rem;">⬡ Model Performance</div>
    """, unsafe_allow_html=True)

    for k, v, c in [
        ("ROC-AUC",       "0.7168",     "#10b981"),
        ("AUPRC",         "0.5853",     "#10b981"),
        ("Precision",     "82.64%",     "#06b6d4"),
        ("vs Random",     "3×",         "#f59e0b"),
        ("AE Separation", "16.90×",     "#8b5cf6"),
        ("False Alarms",  "2.17%",      "#10b981"),
        ("Dataset",       "CICIDS2017", "#94a3b8"),
        ("Records",       "2,830,743",  "#94a3b8"),
        ("Attack Types",  "14 classes", "#ef4444"),
        ("Method",        "Unsupervised","#f59e0b"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;
                    padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
            <span style="font-size:0.72rem;color:#475569;
                         font-family:'JetBrains Mono',monospace;">{k}</span>
            <span style="font-size:0.75rem;font-weight:600;color:{c};
                         font-family:'JetBrains Mono',monospace;">{v}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.4rem;font-size:0.6rem;text-transform:uppercase;
                letter-spacing:0.14em;color:#1e3a5f;
                font-family:'JetBrains Mono',monospace;margin-bottom:0.8rem;">
        ⬡ Detection Pipeline
    </div>
    <div style="background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.1);
                border-radius:12px;padding:1rem;display:flex;flex-direction:column;gap:0.6rem;">
    """, unsafe_allow_html=True)
    for n, t in [
        ("1","40 network flow features → input"),
        ("2","StandardScaler normalizes features"),
        ("3","AE encodes to 8-dim latent space"),
        ("4","Reconstruction error → AE score"),
        ("5","IF + SVM score in 8-dim latent"),
        ("6","Weighted ensemble → final verdict"),
    ]:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:9px;">
            <div style="width:19px;height:19px;border-radius:50%;flex-shrink:0;
                        background:rgba(16,185,129,0.1);
                        border:1px solid rgba(16,185,129,0.25);
                        display:flex;align-items:center;justify-content:center;
                        font-size:0.58rem;color:#10b981;
                        font-family:'JetBrains Mono',monospace;font-weight:600;">{n}</div>
            <div style="font-size:0.72rem;color:#64748b;line-height:1.5;">{t}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    threshold_ui = st.slider(
        "Detection Threshold", 0.05, 0.60, 0.298, 0.01,
        key="det_threshold",
        help="Lower = more sensitive. Higher = more precise."
    )

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:3rem 0 2rem;">
    <div style="display:inline-flex;align-items:center;gap:8px;
                background:rgba(16,185,129,0.08);
                border:1px solid rgba(16,185,129,0.22);
                border-radius:50px;padding:6px 18px;margin-bottom:1.5rem;">
        <span style="width:7px;height:7px;border-radius:50%;background:#10b981;
                     box-shadow:0 0 10px #10b981;display:inline-block;"></span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.66rem;
                     color:#10b981;letter-spacing:0.12em;text-transform:uppercase;">
            Unsupervised · Zero Attack Labels · CICIDS2017
        </span>
    </div>
    <h1 style="font-size:clamp(2.4rem,4vw,3.8rem);font-weight:800;
               line-height:1.08;letter-spacing:-0.04em;margin:0 0 1rem;
               background:linear-gradient(135deg,#ffffff 0%,#6ee7b7 45%,#a78bfa 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;">
        Network Intrusion<br>Detection System
    </h1>
    <p style="font-size:0.95rem;color:#64748b;max-width:540px;
              margin:0 auto 2.5rem;line-height:1.8;">
        Deep <span style="color:#10b981;font-weight:600;">Autoencoder</span>
        learns normal traffic in 8-dim latent space.
        <span style="color:#f59e0b;font-weight:600;">Isolation Forest</span> +
        <span style="color:#8b5cf6;font-weight:600;">One-Class SVM</span>
        detect anomalies — trained with
        <span style="color:#06b6d4;font-weight:600;">zero attack labels</span>.
    </p>
    <div style="display:inline-grid;grid-template-columns:repeat(4,1fr);
                background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.07);
                border-radius:18px;overflow:hidden;max-width:700px;width:100%;">
        <div style="padding:1.1rem 1rem;border-right:1px solid rgba(255,255,255,0.07);">
            <div style="font-size:1.5rem;font-weight:700;color:#10b981;
                        font-family:'JetBrains Mono',monospace;">0.7168</div>
            <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.1em;margin-top:5px;
                        font-family:'JetBrains Mono',monospace;">ROC-AUC</div>
        </div>
        <div style="padding:1.1rem 1rem;border-right:1px solid rgba(255,255,255,0.07);">
            <div style="font-size:1.5rem;font-weight:700;color:#06b6d4;
                        font-family:'JetBrains Mono',monospace;">82.6%</div>
            <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.1em;margin-top:5px;
                        font-family:'JetBrains Mono',monospace;">Precision</div>
        </div>
        <div style="padding:1.1rem 1rem;border-right:1px solid rgba(255,255,255,0.07);">
            <div style="font-size:1.5rem;font-weight:700;color:#8b5cf6;
                        font-family:'JetBrains Mono',monospace;">16.9×</div>
            <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.1em;margin-top:5px;
                        font-family:'JetBrains Mono',monospace;">AE Sep.</div>
        </div>
        <div style="padding:1.1rem 1rem;">
            <div style="font-size:1.5rem;font-weight:700;color:#f59e0b;
                        font-family:'JetBrains Mono',monospace;">2.83M</div>
            <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.1em;margin-top:5px;
                        font-family:'JetBrains Mono',monospace;">Flows</div>
        </div>
    </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,transparent,
    rgba(16,185,129,0.2),rgba(124,58,237,0.2),transparent);
    margin:0 0 2rem;"></div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🔍 Analyze Traffic",
    "📊 Model Insights",
    "🧬 Architecture"
])

# ─────────────────────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────────────────────
with tab1:
    mode = st.radio("Input Mode",
                    ["Manual Entry", "Upload CSV"],
                    horizontal=True,
                    label_visibility="collapsed")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    features = None

    if mode == "Manual Entry":
        st.markdown("""
        <div style="font-size:0.72rem;color:#475569;
                    font-family:'JetBrains Mono',monospace;margin-bottom:1rem;">
            Load a preset or manually adjust the 40 network flow features below.
        </div>""", unsafe_allow_html=True)

        # ── Preset buttons with session_state ──────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🟢 Normal Traffic Sample", use_container_width=True):
                st.session_state.defaults = NORMAL.copy()
                st.rerun()
        with c2:
            if st.button("🔴 Attack Traffic Sample", use_container_width=True):
                st.session_state.defaults = ATTACK.copy()
                st.rerun()
        with c3:
            if st.button("🎲 Random Sample", use_container_width=True):
                st.session_state.defaults = [
                    round(abs(float(np.random.randn())) * 500, 2)
                    for _ in range(40)
                ]
                st.rerun()

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Feature inputs using session_state defaults ─────────
        cols = st.columns(8)
        vals = []
        for i, name in enumerate(FEAT_NAMES):
            with cols[i % 8]:
                v = st.number_input(
                    name,
                    value=float(st.session_state.defaults[i]),
                    format="%.2f",
                    key=f"feat_{i}",
                    label_visibility="visible"
                )
                vals.append(v)
        features = vals

    else:
        uploaded_csv = st.file_uploader(
            "Upload CSV (40 feature columns, no label)",
            type=["csv"],
            label_visibility="collapsed"
        )
        if uploaded_csv:
            df_up = pd.read_csv(uploaded_csv)
            for col in [' Label', 'Label', 'Class']:
                if col in df_up.columns:
                    df_up = df_up.drop(col, axis=1)
            st.dataframe(df_up.head(3), use_container_width=True)
            if len(df_up) > 1:
                row_idx = st.slider("Row to analyze", 0, len(df_up)-1, 0)
            else:
                row_idx = 0
                st.info("Single row detected — using row 0")
            features = df_up.iloc[row_idx].values[:40].tolist()

    # ── Analyze Button ─────────────────────────────────────────
    if features and st.button(
        "🔍 Analyze Network Traffic",
        use_container_width=True,
        type="primary"
    ):
        with st.spinner("Running inference pipeline..."):
            sess, iso, ocsvm, scaler, config = load_models()
            result = predict(
                features, sess, iso, ocsvm,
                scaler, config, threshold_ui
            )

        is_atk  = result["is_attack"]
        ens_s   = result["ensemble"]
        sc      = "#ef4444" if is_atk else "#10b981"
        verdict = "⚠️ INTRUSION DETECTED" if is_atk else "✅ NORMAL TRAFFIC"
        vbg     = "rgba(239,68,68,0.1)"   if is_atk else "rgba(16,185,129,0.1)"
        vbrd    = "rgba(239,68,68,0.3)"   if is_atk else "rgba(16,185,129,0.3)"

        st.markdown(f"""
        <div style="background:{vbg};border:1px solid {vbrd};
                    border-radius:14px;padding:1.5rem 2rem;
                    text-align:center;margin:1.5rem 0;">
            <div style="font-size:1.8rem;font-weight:800;color:{sc};">{verdict}</div>
            <div style="font-size:0.8rem;color:#64748b;
                        font-family:'JetBrains Mono',monospace;margin-top:0.4rem;">
                Ensemble: {ens_s:.4f} &nbsp;·&nbsp;
                Threshold: {threshold_ui:.4f} &nbsp;·&nbsp;
                Confidence: {result['confidence']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

       c1, c2, c3, c4 = st.columns(4)
       for col, title, score, color, desc in [
           (c1, "AE Recon Error",   result['ae_score'],  "#10b981",
            f"Raw MSE: {result['ae_error']:.6f}"),
           (c2, "Isolation Forest", result['if_score'],  "#f59e0b",
            "8-dim latent space"),
           (c3, "One-Class SVM",    result['svm_score'], "#8b5cf6",
            "8-dim latent space"),
           (c4, "Ensemble Score",   ens_s,               sc,
            "AE×0.7 + IF×0.2 + SVM×0.1"),
       ]:
           pct = min(score * 100, 100)
           col.markdown(f"""
           <div style="background:rgba(255,255,255,0.02);
                       border:1px solid rgba(255,255,255,0.07);
                       border-radius:14px;padding:1.2rem 1.4rem;
                       position:relative;overflow:hidden;">
              <div style="position:absolute;top:0;left:0;right:0;
                          height:2px;background:{color};"></div>
              <div style="font-size:0.6rem;color:#475569;text-transform:uppercase;
                          letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;
                          margin-bottom:0.5rem;">{title}</div>
              <div style="font-size:1.9rem;font-weight:800;color:{color};
                          font-family:'JetBrains Mono',monospace;
                          line-height:1;margin-bottom:0.4rem;">{score:.4f}</div>
              <div style="font-size:0.7rem;color:#475569;
                          margin-bottom:0.8rem;">{desc}</div>
              <div style="height:3px;background:rgba(255,255,255,0.05);
                          border-radius:3px;overflow:hidden;">
                  <div style="height:100%;width:{pct}%;
                              background:{color};border-radius:3px;"></div>
              </div>
           </div>
           """, unsafe_allow_html=True)

        # Latent space bar chart — fixed Plotly API
        latent_vals = result["latent"]
        fig_lat = go.Figure(go.Bar(
            x=[f"z{i+1}" for i in range(8)],
            y=latent_vals,
            marker_color=["#ef4444" if abs(v) > 1.5 else "#10b981"
                          for v in latent_vals],
            marker_line_width=0,
        ))
        fig_lat.update_layout(
            title=dict(
                text="Latent Space Representation (8 dimensions)",
                font=dict(color="white", size=12)
            ),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#94a3b8"),
            xaxis=dict(gridcolor="#1e293b", tickcolor="#475569"),
            yaxis=dict(
                gridcolor="#1e293b",
                tickcolor="#475569",
                title=dict(
                    text="Activation",
                    font=dict(color="#64748b")
                )
            ),
            height=250,
            margin=dict(l=20, r=20, t=45, b=20)
        )
        st.plotly_chart(fig_lat, use_container_width=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = ens_s * 100,
            delta = {"reference": threshold_ui * 100, "valueformat": ".1f"},
            number= {"suffix": "%", "valueformat": ".1f",
                     "font": {"size": 40, "color": sc}},
            gauge = {
                "axis"       : {"range": [0, 100], "tickcolor": "#475569"},
                "bar"        : {"color": sc, "thickness": 0.25},
                "bgcolor"    : "#161b22",
                "bordercolor": "#334155",
                "steps"      : [
                    {"range": [0, threshold_ui*100],
                     "color": "rgba(16,185,129,0.1)"},
                    {"range": [threshold_ui*100, 100],
                     "color": "rgba(239,68,68,0.1)"},
                ],
                "threshold"  : {
                    "line"     : {"color": "#f59e0b", "width": 3},
                    "thickness": 0.75,
                    "value"    : threshold_ui * 100,
                }
            },
            title={"text": "Threat Risk Score",
                   "font": {"color": "#94a3b8", "size": 14}}
        ))
        fig_gauge.update_layout(
            height=280,
            paper_bgcolor="#0d1117",
            font={"color": "#94a3b8"},
            margin=dict(l=30, r=30, t=50, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 2
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;
                color:#334155;font-family:'JetBrains Mono',monospace;
                margin-bottom:1.5rem;">
        Evaluation on 566,149 held-out network flows · CICIDS2017
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem;">', unsafe_allow_html=True)
    for k, v, c, d in [
        ("ROC-AUC",       "0.7168",  "#10b981", "Area under ROC"),
        ("AUPRC",         "0.5853",  "#06b6d4", "3× random baseline"),
        ("Precision",     "82.6%",   "#8b5cf6", "When alert fires"),
        ("F1 Score",      "0.5585",  "#f59e0b", "Harmonic mean P/R"),
        ("Recall",        "42.2%",   "#10b981", "Attacks caught"),
        ("False Alarms",  "2.17%",   "#06b6d4", "Normal flagged"),
        ("AE Separation", "16.90×",  "#8b5cf6", "Normal vs attack"),
        ("TP / Total",    "47K/111K","#ef4444", "At best threshold"),
    ]:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;padding:1.1rem 1.2rem;">
            <div style="font-size:0.58rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;
                        margin-bottom:0.4rem;">{k}</div>
            <div style="font-size:1.5rem;font-weight:700;color:{c};
                        font-family:'JetBrains Mono',monospace;
                        line-height:1;margin-bottom:0.3rem;">{v}</div>
            <div style="font-size:0.7rem;color:#475569;">{d}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="ROC-AUC",
        x=["AE only","IF only","SVM only","Full Ensemble"],
        y=[0.789, 0.718, 0.641, 0.717],
        marker_color="#10b981", marker_line_width=0,
        text=["0.789","0.718","0.641","0.717"],
        textposition="outside", textfont=dict(color="white", size=11)
    ))
    fig_comp.add_trace(go.Bar(
        name="AUPRC",
        x=["AE only","IF only","SVM only","Full Ensemble"],
        y=[0.620, 0.438, 0.529, 0.585],
        marker_color="#f59e0b", marker_line_width=0,
        text=["0.620","0.438","0.529","0.585"],
        textposition="outside", textfont=dict(color="white", size=11)
    ))
    fig_comp.update_layout(
        title=dict(text="Model Comparison — ROC-AUC vs AUPRC",
                   font=dict(color="white", size=13)),
        barmode="group", bargap=0.25, bargroupgap=0.1,
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#94a3b8"),
        legend=dict(bgcolor="#161b22", bordercolor="#334155",
                    font=dict(color="white")),
        yaxis=dict(range=[0, 1.0], gridcolor="#1e293b"),
        height=360, margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("""
    <div style="background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:1.4rem;margin-top:1rem;">
        <div style="font-size:0.6rem;font-weight:600;color:#64748b;
                    text-transform:uppercase;letter-spacing:0.12em;
                    font-family:'JetBrains Mono',monospace;margin-bottom:1rem;">
            Ablation Study
        </div>
    """, unsafe_allow_html=True)

    for m, roc, ap, f1, rec, c, bold in [
        ("Model",                 "ROC-AUC","AUPRC","F1","Recall","#475569",False),
        ("AE only",               "0.7891","0.6200","0.5743","0.4404","#10b981",False),
        ("IF only (latent)",      "0.7183","0.4381","0.5106","0.4629","#f59e0b",False),
        ("SVM only (latent)",     "0.6405","0.5289","0.5619","0.4018","#8b5cf6",False),
        ("AE + IF",               "0.7243","0.5586","0.5516","0.4434","#06b6d4",False),
        ("AE + SVM",              "0.6423","0.5597","0.5764","0.4320","#06b6d4",False),
        ("Full Ensemble ◄ BEST",  "0.7168","0.5853","0.5585","0.4218","#ef4444",True),
    ]:
        fw = "700" if bold else "400"
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1fr;
                    padding:0.55rem 0;border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="font-size:0.74rem;color:{c};
                         font-family:'JetBrains Mono',monospace;
                         font-weight:{fw};">{m}</span>
            <span style="font-size:0.74rem;color:#94a3b8;
                         font-family:'JetBrains Mono',monospace;text-align:center;">{roc}</span>
            <span style="font-size:0.74rem;color:#94a3b8;
                         font-family:'JetBrains Mono',monospace;text-align:center;">{ap}</span>
            <span style="font-size:0.74rem;color:#94a3b8;
                         font-family:'JetBrains Mono',monospace;text-align:center;">{f1}</span>
            <span style="font-size:0.74rem;color:#94a3b8;
                         font-family:'JetBrains Mono',monospace;text-align:center;">{rec}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
        <div style="margin-top:0.8rem;font-size:0.7rem;color:#334155;
                    font-family:'JetBrains Mono',monospace;">
            Key finding: AE alone achieves highest ROC-AUC (0.789), validating
            that 16.90× reconstruction error separation is highly discriminative.
        </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 3
# ─────────────────────────────────────────────────────────────
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;padding:1.4rem;">
            <div style="font-size:0.6rem;font-weight:600;color:#64748b;
                        text-transform:uppercase;letter-spacing:0.12em;
                        font-family:'JetBrains Mono',monospace;margin-bottom:1rem;">
                Deep Autoencoder
            </div>
        """, unsafe_allow_html=True)
        for k, v, c in [
            ("Architecture",  "40→32→24→16→8→16→24→32→40", "#10b981"),
            ("Latent dim",    "8 dimensions",               "#10b981"),
            ("Activation",    "GELU + BatchNorm",           "#06b6d4"),
            ("Dropout",       "0.2 in encoder + decoder",   "#06b6d4"),
            ("Loss",          "MSE + 1e-4·‖z‖²",           "#f59e0b"),
            ("Optimizer",     "AdamW lr=1e-3 wd=1e-4",     "#f59e0b"),
            ("Scheduler",     "CosineAnnealingLR T=60",     "#8b5cf6"),
            ("Train data",    "1,818,477 normal flows",     "#8b5cf6"),
            ("Best loss",     "0.0939",                     "#94a3b8"),
            ("AE Separation", "16.90× normal vs attack",    "#ef4444"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.45rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.03);">
                <span style="font-size:0.72rem;color:#475569;
                             font-family:'JetBrains Mono',monospace;">{k}</span>
                <span style="font-size:0.72rem;color:{c};font-weight:600;
                             font-family:'JetBrains Mono',monospace;">{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;padding:1.4rem;">
            <div style="font-size:0.6rem;font-weight:600;color:#64748b;
                        text-transform:uppercase;letter-spacing:0.12em;
                        font-family:'JetBrains Mono',monospace;margin-bottom:1rem;">
                Anomaly Models + Dataset
            </div>
        """, unsafe_allow_html=True)
        for k, v, c in [
            ("IF Trees",      "300 · contamination=0.01",   "#f59e0b"),
            ("IF Input",      "8-dim latent (not raw)",     "#f59e0b"),
            ("SVM Kernel",    "RBF · nu=0.01 · scale",     "#8b5cf6"),
            ("SVM Input",     "8-dim latent (not raw)",     "#8b5cf6"),
            ("Ensemble",      "AE×0.7 + IF×0.2 + SVM×0.1", "#06b6d4"),
            ("Dataset",       "CICIDS2017 — UNB",           "#10b981"),
            ("Total records", "2,830,743 flows",            "#10b981"),
            ("Attack types",  "14 categories",              "#ef4444"),
            ("Features",      "Top 40 by RF importance",    "#94a3b8"),
            ("Deployment",    "ONNX 33KB — no PyTorch",     "#94a3b8"),
        ]:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.45rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.03);">
                <span style="font-size:0.72rem;color:#475569;
                             font-family:'JetBrains Mono',monospace;">{k}</span>
                <span style="font-size:0.72rem;color:{c};font-weight:600;
                             font-family:'JetBrains Mono',monospace;">{v}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem;background:rgba(16,185,129,0.04);
                border:1px solid rgba(16,185,129,0.1);
                border-radius:14px;padding:1.4rem;">
        <div style="font-size:0.6rem;font-weight:600;color:#64748b;
                    text-transform:uppercase;letter-spacing:0.12em;
                    font-family:'JetBrains Mono',monospace;margin-bottom:0.8rem;">
            Key Research Finding
        </div>
        <div style="font-size:0.85rem;color:#94a3b8;line-height:1.8;">
            The autoencoder trained on
            <span style="color:#10b981;font-weight:600;">1.8M normal flows</span>
            achieves reconstruction error separation of
            <span style="color:#ef4444;font-weight:600;">16.90×</span>
            between benign and attack traffic. Both IF and OCSVM operate in the
            <span style="color:#8b5cf6;font-weight:600;">8-dimensional latent space</span>
            — not raw features — making the ensemble more powerful.
            Core finding: <span style="color:#f59e0b;font-weight:600;">anomalies are
            detectable with zero labeled attack data</span>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div style="height:1px;background:linear-gradient(90deg,transparent,
    rgba(16,185,129,0.15),rgba(139,92,246,0.15),transparent);
    margin:3rem 0 1.5rem;"></div>
<div style="display:flex;justify-content:space-between;padding-bottom:1rem;">
    <span style="font-size:0.62rem;color:#1e293b;
                 font-family:'JetBrains Mono',monospace;letter-spacing:0.06em;">
        NETGUARD AI · AE + IF + SVM · IITP 2025
    </span>
    <span style="font-size:0.62rem;color:#1e293b;
                 font-family:'JetBrains Mono',monospace;">
        ROC-AUC 0.7168 · PRECISION 82.6% · AE SEP 16.90×
    </span>
</div>
""", unsafe_allow_html=True)
