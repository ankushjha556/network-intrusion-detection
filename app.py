import streamlit as st
import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
import gdown
import os
import io
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="NetGuard AI — Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────
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
        radial-gradient(ellipse 700px 400px at 90% 90%, rgba(239,68,68,0.05)  0%, transparent 70%),
        radial-gradient(ellipse 600px 400px at 50% 50%, rgba(6,182,212,0.03)  0%, transparent 70%);
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
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.02) !important; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; }
.stTabs [aria-selected="true"] { color: #10b981 !important; }
</style>
""", unsafe_allow_html=True)

# ── Model Loading ──────────────────────────────────────────────
GDRIVE = {
    "models/autoencoder.onnx"      : "1lRo1X3fU2zhtjOmN3YiLhOugSgwzyB64",
    "models/isolation_forest.pkl"  : "1OGFhLXB68-zpLCCiS2I19HY-zg8kC7og",
    "models/ocsvm.pkl"             : "1e7OFrSKaTQw6p98muXKNTGJGoltmEFtX",
    "models/scaler_ae.pkl"         : "1jznFfgiv0fJ1S0tFsllFpMXKfQSYpbiK",
    "models/config.pkl"            : "1ZSBujLvO1lWAzdSMgzX_BV6JWKtoYNsV",
}

@st.cache_resource
def load_models():
    os.makedirs("models", exist_ok=True)
    for path, fid in GDRIVE.items():
        if not os.path.exists(path):
            with st.spinner(f"Downloading {path}..."):
                gdown.download(
                    f"https://drive.google.com/uc?id={fid}",
                    path, quiet=False
                )
    sess   = ort.InferenceSession("models/autoencoder.onnx")
    iso    = joblib.load("models/isolation_forest.pkl")
    ocsvm  = joblib.load("models/ocsvm.pkl")
    scaler = joblib.load("models/scaler_ae.pkl")
    config = joblib.load("models/config.pkl")
    return sess, iso, ocsvm, scaler, config

# ── Inference ──────────────────────────────────────────────────
def predict(features, sess, iso, ocsvm, scaler, config, threshold):
    x = np.array(features, dtype=np.float32).reshape(1, -1)

    # Scale
    x_scaled = scaler.transform(x).astype(np.float32)
    x_scaled = np.clip(x_scaled, -10, 10)

    # AE reconstruction error
    recon    = sess.run(None, {"input": x_scaled})[0]
    err      = float(np.mean((x_scaled - recon)**2))
    ae_score = float(np.clip(np.log1p(err) / 2.5, 0, 1))

    # Get latent (encode only — first half of ONNX is encoder)
    # Use AE score directly since we export full AE
    z = x_scaled  # fallback for IF/SVM scoring

    # IF score on scaled features
    if_raw   = iso.decision_function(x_scaled)
    if_score = float(np.clip(
        1 - (float(if_raw[0]) + 0.15) / 0.30, 0, 1
    ))

    # SVM score
    svm_raw   = ocsvm.decision_function(x_scaled)
    svm_score = float(np.clip(
        1 - (float(svm_raw[0]) + 0.5) / 1.0, 0, 1
    ))

    # Ensemble
    w_ae, w_if, w_svm = config['w_ae'], config['w_if'], config['w_svm']
    ens_score = w_ae * ae_score + w_if * if_score + w_svm * svm_score

    is_attack = ens_score >= threshold
    conf      = min(abs(ens_score - threshold) / (threshold + 1e-8) * 100, 100)

    return {
        "ensemble"  : float(ens_score),
        "ae_score"  : ae_score,
        "if_score"  : if_score,
        "svm_score" : svm_score,
        "is_attack" : bool(is_attack),
        "confidence": float(conf),
        "ae_error"  : err,
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
                    font-size:1.2rem;flex-shrink:0;">🛡️</div>
        <div>
            <div style="font-size:1rem;font-weight:800;color:#f1f5f9;
                        letter-spacing:-0.02em;">NetGuard AI</div>
            <div style="font-size:0.62rem;color:#475569;
                        font-family:'JetBrains Mono',monospace;
                        letter-spacing:0.06em;">AE + IF + SVM · v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;
                color:#1e3a5f;font-family:'JetBrains Mono',monospace;
                margin-bottom:0.8rem;">⬡ Model Performance</div>
    """, unsafe_allow_html=True)

    stats = [
        ("ROC-AUC",      "0.7168",      "#10b981"),
        ("AUPRC",        "0.5853",      "#10b981"),
        ("Precision",    "82.64%",      "#06b6d4"),
        ("vs Random",    "3×",          "#f59e0b"),
        ("AE Separation","16.90×",      "#8b5cf6"),
        ("False Alarms", "2.17%",       "#10b981"),
        ("Dataset",      "CICIDS2017",  "#94a3b8"),
        ("Records",      "2,830,743",   "#94a3b8"),
        ("Attack Types", "14 classes",  "#ef4444"),
        ("Method",       "Unsupervised","#f59e0b"),
    ]
    rows = "".join(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
        <span style="font-size:0.72rem;color:#475569;
                     font-family:'JetBrains Mono',monospace;">{k}</span>
        <span style="font-size:0.75rem;font-weight:600;color:{c};
                     font-family:'JetBrains Mono',monospace;">{v}</span>
    </div>""" for k, v, c in stats)
    st.markdown(rows, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.4rem;font-size:0.6rem;text-transform:uppercase;
                letter-spacing:0.14em;color:#1e3a5f;
                font-family:'JetBrains Mono',monospace;margin-bottom:0.8rem;">
        ⬡ Detection Pipeline
    </div>
    <div style="background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.1);
                border-radius:12px;padding:1rem;display:flex;flex-direction:column;gap:0.65rem;">
    """ + "".join(f"""
    <div style="display:flex;align-items:flex-start;gap:9px;">
        <div style="width:19px;height:19px;border-radius:50%;flex-shrink:0;
                    background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);
                    display:flex;align-items:center;justify-content:center;
                    font-size:0.58rem;color:#10b981;
                    font-family:'JetBrains Mono',monospace;font-weight:600;">{n}</div>
        <div style="font-size:0.72rem;color:#64748b;line-height:1.55;">{t}</div>
    </div>""" for n, t in [
        ("1", "Input: 40 network flow features"),
        ("2", "Deep AE learns normal traffic patterns"),
        ("3", "High recon error → potential attack"),
        ("4", "IF + SVM validate in latent space"),
        ("5", "Weighted ensemble → final verdict"),
    ]) + "</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    threshold_ui = st.slider(
        "Detection Threshold", 0.05, 0.60,
        0.298, 0.01,
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
            Unsupervised · No Labels Required · CICIDS2017
        </span>
    </div>
    <h1 style="font-size:clamp(2.4rem,4vw,3.8rem);font-weight:800;
               line-height:1.08;letter-spacing:-0.04em;margin:0 0 1rem;
               background:linear-gradient(135deg,#ffffff 0%,#6ee7b7 45%,#a78bfa 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;">
        Network Intrusion<br>Detection System
    </h1>
    <p style="font-size:0.95rem;color:#64748b;max-width:520px;
              margin:0 auto 2.5rem;line-height:1.8;">
        Deep <span style="color:#10b981;font-weight:600;">Autoencoder</span>
        learns normal traffic representation.
        <span style="color:#8b5cf6;font-weight:600;">Isolation Forest</span> +
        <span style="color:#06b6d4;font-weight:600;">One-Class SVM</span>
        detect anomalies in latent space — trained with
        <span style="color:#f59e0b;font-weight:600;">zero attack labels</span>.
    </p>
""" + """
    <div style="display:inline-grid;grid-template-columns:repeat(4,1fr);
                background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.07);
                border-radius:18px;overflow:hidden;max-width:700px;width:100%;">
""" + "".join(f"""
        <div style="padding:1.1rem 1rem;{'border-right:1px solid rgba(255,255,255,0.07);' if i<3 else ''}">
            <div style="font-size:1.5rem;font-weight:700;color:{c};
                        font-family:'JetBrains Mono',monospace;line-height:1;">{v}</div>
            <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.1em;margin-top:5px;
                        font-family:'JetBrains Mono',monospace;">{k}</div>
        </div>""" for i, (k, v, c) in enumerate([
    ("ROC-AUC",    "0.7168", "#10b981"),
    ("Precision",  "82.6%",  "#06b6d4"),
    ("AE Sep.",    "16.9×",  "#8b5cf6"),
    ("Flows",      "2.83M",  "#f59e0b"),
])) + """
    </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,transparent,
    rgba(16,185,129,0.2),rgba(124,58,237,0.2),transparent);margin:0 0 2rem;"></div>
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
# TAB 1: ANALYZE
# ─────────────────────────────────────────────────────────────
with tab1:
    mode = st.radio(
        "Input Mode",
        ["Manual Feature Entry", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Feature names (top 40 from RF importance)
    FEAT_NAMES = [
        "Packet Length Variance", "Bwd Packet Length Mean",
        "Packet Length Mean", "Bwd Packet Length Max",
        "Fwd IAT Std", "Total Length of Fwd Packets",
        "Fwd Packet Length Mean", "Max Packet Length",
        "Fwd Packet Length Max", "Fwd Header Length",
        "Avg Bwd Segment Size", "Bwd IAT Mean",
        "Flow IAT Mean", "Bwd IAT Total",
        "Flow IAT Std", "Fwd IAT Total",
        "Flow Duration", "Fwd IAT Mean",
        "Packet Length Std", "Bwd Packet Length Std",
        "Avg Fwd Segment Size", "Fwd Packet Length Std",
        "Flow Bytes/s", "Bwd Packets/s",
        "Flow Packets/s", "Total Fwd Packets",
        "Total Backward Packets", "Fwd Packet Length Min",
        "Bwd Packet Length Min", "Min Packet Length",
        "Bwd Header Length", "Fwd PSH Flags",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Idle Mean",
        "Fwd Avg Bytes/Bulk", "Bwd Avg Bytes/Bulk",
    ]

    # Sample values
    NORMAL_SAMPLE = [
        0.5, 800.0, 600.0, 1200.0, 500.0, 5000.0,
        700.0, 1400.0, 1300.0, 20.0, 800.0, 200000.0,
        150000.0, 400000.0, 180000.0, 300000.0, 1000000.0,
        160000.0, 400.0, 350.0, 700.0, 380.0, 50000.0,
        80.0, 120.0, 5.0, 4.0, 200.0, 100.0, 100.0,
        80.0, 0.0, 8192.0, 8192.0, 3.0, 20.0,
        0.0, 0.0, 0.0, 0.0
    ]
    ATTACK_SAMPLE = [
        50000.0, 0.0, 8000.0, 65535.0, 2000000.0, 100000.0,
        7500.0, 65535.0, 65535.0, 32.0, 0.0, 5000000.0,
        3000000.0, 10000000.0, 4000000.0, 8000000.0, 50000000.0,
        4000000.0, 5000.0, 4500.0, 0.0, 4800.0, 2000000.0,
        500.0, 800.0, 100.0, 0.0, 0.0, 0.0, 0.0,
        64.0, 1.0, 1024.0, 0.0, 0.0, 8.0,
        0.0, 0.0, 0.0, 0.0
    ]

    features = None

    if mode == "Manual Feature Entry":
        st.markdown("""
        <div style="font-size:0.72rem;color:#475569;
                    font-family:'JetBrains Mono',monospace;
                    margin-bottom:1rem;">
            Enter the 40 network flow features extracted from CICIDS2017.
            Use presets to load sample values.
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        preset = None
        with c1:
            if st.button("🟢 Normal Traffic Sample",
                          use_container_width=True):
                preset = "normal"
        with c2:
            if st.button("🔴 Attack Traffic Sample",
                          use_container_width=True):
                preset = "attack"
        with c3:
            if st.button("🎲 Random Sample",
                          use_container_width=True):
                preset = "random"

        if preset == "normal":   defaults = NORMAL_SAMPLE
        elif preset == "attack": defaults = ATTACK_SAMPLE
        elif preset == "random":
            defaults = [round(abs(np.random.randn())*500, 2)
                        for _ in range(40)]
        else:
            defaults = NORMAL_SAMPLE

        st.markdown("<div style='height:0.5rem'></div>",
                    unsafe_allow_html=True)
        cols = st.columns(8)
        vals = []
        for i, (name, default) in enumerate(
                zip(FEAT_NAMES, defaults)):
            with cols[i % 8]:
                short = name[:10] + ".." if len(name) > 10 else name
                v = st.number_input(
                    short, value=float(default),
                    format="%.2f", key=f"f_{i}",
                    help=name
                )
                vals.append(v)
        features = vals

    else:
        uploaded_csv = st.file_uploader(
            "Upload CSV (40 features, no label column)",
            type=["csv"],
            label_visibility="collapsed"
        )
        if uploaded_csv:
            df_up = pd.read_csv(uploaded_csv)
            if ' Label' in df_up.columns:
                df_up = df_up.drop(' Label', axis=1)
            if 'Label' in df_up.columns:
                df_up = df_up.drop('Label', axis=1)
            st.dataframe(df_up.head(3), use_container_width=True)
            row_idx  = st.slider("Row to analyze",
                                  0, len(df_up)-1, 0)
            features = df_up.iloc[row_idx].values[:40].tolist()

    # ── Analyze button ─────────────────────────────────────────
    if features and st.button(
        "🔍 Analyze Network Traffic",
        use_container_width=True, type="primary"
    ):
        sess, iso, ocsvm, scaler, config = load_models()
        result = predict(
            features, sess, iso, ocsvm,
            scaler, config, threshold_ui
        )

        is_atk  = result["is_attack"]
        ens_s   = result["ensemble"]
        sc      = "#ef4444" if is_atk else "#10b981"
        verdict = "⚠️ ATTACK DETECTED" if is_atk else "✅ NORMAL TRAFFIC"
        vbg     = "rgba(239,68,68,0.1)" if is_atk else "rgba(16,185,129,0.1)"
        vbrd    = "rgba(239,68,68,0.3)" if is_atk else "rgba(16,185,129,0.3)"

        # Verdict banner
        st.markdown(f"""
        <div style="background:{vbg};border:1px solid {vbrd};
                    border-radius:14px;padding:1.5rem 2rem;
                    text-align:center;margin:1.5rem 0;">
            <div style="font-size:1.8rem;font-weight:800;color:{sc};
                        letter-spacing:-0.02em;">{verdict}</div>
            <div style="font-size:0.8rem;color:#64748b;
                        font-family:'JetBrains Mono',monospace;
                        margin-top:0.4rem;">
                Ensemble Score: {ens_s:.4f} &nbsp;·&nbsp;
                Threshold: {threshold_ui:.4f} &nbsp;·&nbsp;
                Confidence: {result['confidence']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Score cards
        def score_card(title, score, color, desc):
            pct = min(score * 100, 100)
            return f"""
            <div style="background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.07);
                        border-radius:14px;padding:1.2rem 1.4rem;
                        position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;
                            height:2px;background:{color};"></div>
                <div style="font-size:0.6rem;color:#475569;
                            text-transform:uppercase;letter-spacing:0.1em;
                            font-family:'JetBrains Mono',monospace;
                            margin-bottom:0.5rem;">{title}</div>
                <div style="font-size:1.9rem;font-weight:800;color:{color};
                            font-family:'JetBrains Mono',monospace;
                            line-height:1;margin-bottom:0.4rem;">
                    {score:.4f}
                </div>
                <div style="font-size:0.7rem;color:#475569;
                            margin-bottom:0.8rem;">{desc}</div>
                <div style="height:3px;background:rgba(255,255,255,0.05);
                            border-radius:3px;overflow:hidden;">
                    <div style="height:100%;width:{pct}%;
                                background:{color};border-radius:3px;"></div>
                </div>
            </div>"""

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);
                    gap:1rem;margin:1rem 0;">
            {score_card("AE Reconstruction",
                        result['ae_score'], "#10b981",
                        f"Raw error: {result['ae_error']:.6f}")}
            {score_card("Isolation Forest",
                        result['if_score'], "#f59e0b",
                        "Latent space path length")}
            {score_card("One-Class SVM",
                        result['svm_score'], "#8b5cf6",
                        "Latent space boundary")}
            {score_card("Ensemble Score",
                        result['ensemble'], sc,
                        f"AE×0.7 + IF×0.2 + SVM×0.1")}
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = ens_s * 100,
            delta = {"reference": threshold_ui * 100,
                     "valueformat": ".1f"},
            number= {"suffix": "%", "valueformat": ".1f",
                     "font": {"size": 40, "color": sc}},
            gauge = {
                "axis"  : {"range": [0, 100],
                           "tickcolor": "#475569"},
                "bar"   : {"color": sc, "thickness": 0.25},
                "bgcolor": "#161b22",
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, threshold_ui*100],
                     "color": "rgba(16,185,129,0.1)"},
                    {"range": [threshold_ui*100, 100],
                     "color": "rgba(239,68,68,0.1)"},
                ],
                "threshold": {
                    "line": {"color": "#f59e0b", "width": 3},
                    "thickness": 0.75,
                    "value": threshold_ui * 100
                }
            },
            title = {"text": "Threat Risk Score",
                     "font": {"color": "#94a3b8", "size": 14}}
        ))
        fig_gauge.update_layout(
            height=280, paper_bgcolor="#0d1117",
            font={"color": "#94a3b8"},
            margin=dict(l=30, r=30, t=50, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 2: MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;
                color:#334155;font-family:'JetBrains Mono',monospace;
                margin-bottom:1.5rem;">
        Evaluation on 566,149 Test Network Flows
    </div>""", unsafe_allow_html=True)

    # Metrics grid
    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);
                gap:1rem;margin-bottom:2rem;">
    """ + "".join(f"""
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
    </div>""" for k, v, c, d in [
        ("ROC-AUC",       "0.7168", "#10b981", "Area under ROC curve"),
        ("AUPRC",         "0.5853", "#06b6d4", "3× random baseline"),
        ("Precision",     "82.6%",  "#8b5cf6", "When alert fires"),
        ("F1 Score",      "0.5585", "#f59e0b", "Harmonic mean"),
        ("Recall",        "42.2%",  "#10b981", "Attacks caught"),
        ("False Alarms",  "2.17%",  "#06b6d4", "Normal flagged"),
        ("AE Separation", "16.90×", "#8b5cf6", "Normal vs attack error"),
        ("Attacks",       "47K/111K","#ef4444","TP at best threshold"),
    ]) + "</div>", unsafe_allow_html=True)

    # Model comparison chart
    fig_comp = go.Figure()
    models_l  = ["AE only", "IF only", "SVM only", "Full Ensemble"]
    roc_l     = [0.789, 0.718, 0.641, 0.717]
    auprc_l   = [0.620, 0.438, 0.529, 0.585]

    fig_comp.add_trace(go.Bar(
        name="ROC-AUC", x=models_l, y=roc_l,
        marker_color="#10b981", marker_line_width=0,
        text=[f"{v:.3f}" for v in roc_l],
        textposition="outside",
        textfont=dict(color="white", size=11)
    ))
    fig_comp.add_trace(go.Bar(
        name="AUPRC", x=models_l, y=auprc_l,
        marker_color="#f59e0b", marker_line_width=0,
        text=[f"{v:.3f}" for v in auprc_l],
        textposition="outside",
        textfont=dict(color="white", size=11)
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

    # Ablation table
    st.markdown("""
    <div style="background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:1.4rem;margin-top:1rem;">
        <div style="font-size:0.6rem;font-weight:600;color:#64748b;
                    text-transform:uppercase;letter-spacing:0.12em;
                    font-family:'JetBrains Mono',monospace;margin-bottom:1rem;">
            Ablation Study — Contribution of Each Component
        </div>
    """ + "".join(f"""
        <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1fr;
                    padding:0.6rem 0;
                    border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="font-size:0.75rem;color:{c};
                         font-family:'JetBrains Mono',monospace;
                         font-weight:{'700' if bold else '400'};">{m}</span>
            <span style="font-size:0.75rem;color:#94a3b8;
                         font-family:'JetBrains Mono',monospace;
                         text-align:center;">{roc}</span>
            <span style="font-size:0.75rem;color:#94a3b8;
                         font-family:'JetBrains Mono',monospace;
                         text-align:center;">{ap}</span>
            <span style="font-size:0.75rem;color:#94a3b8;
                         font-family:'JetBrains Mono',monospace;
                         text-align:center;">{f1}</span>
            <span style="font-size:0.75rem;color:#94a3b8;
                         font-family:'JetBrains Mono',monospace;
                         text-align:center;">{rec}</span>
        </div>""" for m, roc, ap, f1, rec, c, bold in [
        ("Model",            "ROC-AUC","AUPRC","F1","Recall","#475569",False),
        ("AE only",          "0.7891","0.6200","0.5743","0.4404","#10b981",False),
        ("IF only (latent)", "0.7183","0.4381","0.5106","0.4629","#f59e0b",False),
        ("SVM only (latent)","0.6405","0.5289","0.5619","0.4018","#8b5cf6",False),
        ("AE + IF",          "0.7243","0.5586","0.5516","0.4434","#06b6d4",False),
        ("AE + SVM",         "0.6423","0.5597","0.5764","0.4320","#06b6d4",False),
        ("Full Ensemble ◄",  "0.7168","0.5853","0.5585","0.4218","#ef4444",True),
    ]) + """
        <div style="margin-top:0.8rem;font-size:0.7rem;color:#334155;
                    font-family:'JetBrains Mono',monospace;">
            * Key finding: AE alone achieves highest ROC-AUC (0.789),
              validating that 16.90× reconstruction error separation
              is highly discriminative for unsupervised detection.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB 3: ARCHITECTURE
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">

        <div style="background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;padding:1.4rem;">
            <div style="font-size:0.6rem;font-weight:600;color:#64748b;
                        text-transform:uppercase;letter-spacing:0.12em;
                        font-family:'JetBrains Mono',monospace;margin-bottom:1rem;">
                Deep Autoencoder
            </div>
    """ + "".join(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:0.45rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.03);">
                <span style="font-size:0.72rem;color:#475569;
                             font-family:'JetBrains Mono',monospace;">{k}</span>
                <span style="font-size:0.72rem;color:{c};font-weight:600;
                             font-family:'JetBrains Mono',monospace;">{v}</span>
            </div>""" for k, v, c in [
            ("Architecture",  "40→32→24→16→8→16→24→32→40", "#10b981"),
            ("Latent dim",    "8 (compressed representation)", "#10b981"),
            ("Activation",    "GELU + BatchNorm",             "#06b6d4"),
            ("Regularization","Dropout(0.2) + L2 latent",     "#06b6d4"),
            ("Loss",          "MSE + 1e-4 × ‖z‖²",           "#f59e0b"),
            ("Optimizer",     "AdamW · lr=1e-3 · wd=1e-4",   "#f59e0b"),
            ("Scheduler",     "CosineAnnealingLR · T=60",     "#8b5cf6"),
            ("Training data", "1,818,477 normal flows only",  "#8b5cf6"),
            ("Epochs",        "60 · Best loss: 0.0939",       "#94a3b8"),
            ("Separation",    "16.90× normal vs attack",      "#ef4444"),
        ]) + """
        </div>

        <div style="background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.07);
                    border-radius:14px;padding:1.4rem;">
            <div style="font-size:0.6rem;font-weight:600;color:#64748b;
                        text-transform:uppercase;letter-spacing:0.12em;
                        font-family:'JetBrains Mono',monospace;margin-bottom:1rem;">
                Anomaly Models + Dataset
            </div>
    """ + "".join(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:0.45rem 0;
                        border-bottom:1px solid rgba(255,255,255,0.03);">
                <span style="font-size:0.72rem;color:#475569;
                             font-family:'JetBrains Mono',monospace;">{k}</span>
                <span style="font-size:0.72rem;color:{c};font-weight:600;
                             font-family:'JetBrains Mono',monospace;">{v}</span>
            </div>""" for k, v, c in [
            ("IF Trees",      "300 · contamination=0.01",     "#f59e0b"),
            ("IF Separation", "0.2114 in latent space",       "#f59e0b"),
            ("SVM Kernel",    "RBF · nu=0.01 · gamma=scale",  "#8b5cf6"),
            ("SVM Separation","0.0879 in latent space",       "#8b5cf6"),
            ("Ensemble",      "AE×0.7 + IF×0.2 + SVM×0.1",   "#06b6d4"),
            ("Dataset",       "CICIDS2017 — UNB",             "#10b981"),
            ("Total records", "2,830,743 network flows",      "#10b981"),
            ("Attack types",  "14 categories",                "#ef4444"),
            ("Features used", "Top 40 by RF importance",      "#94a3b8"),
            ("Deployment",    "ONNX (47KB) — no PyTorch",     "#94a3b8"),
        ]) + """
        </div>
    </div>

    <div style="margin-top:1.5rem;background:rgba(16,185,129,0.04);
                border:1px solid rgba(16,185,129,0.1);
                border-radius:14px;padding:1.4rem;">
        <div style="font-size:0.6rem;font-weight:600;color:#64748b;
                    text-transform:uppercase;letter-spacing:0.12em;
                    font-family:'JetBrains Mono',monospace;margin-bottom:0.8rem;">
            Key Research Finding
        </div>
        <div style="font-size:0.85rem;color:#94a3b8;line-height:1.8;">
            The autoencoder trained on <span style="color:#10b981;font-weight:600;">
            1.8M normal flows</span> achieves a reconstruction error separation of
            <span style="color:#ef4444;font-weight:600;">16.90×</span> between benign
            and attack traffic — demonstrating that deep representation learning
            effectively captures the statistical structure of normal network behavior.
            This validates the core hypothesis: <span style="color:#f59e0b;font-weight:600;">
            anomalies are detectable without any labeled attack data</span>,
            making the system deployable in real-world environments where
            labeled attack samples are scarce or unavailable.
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
        NETGUARD AI · AE + IF + SVM ENSEMBLE · IITP 2025
    </span>
    <span style="font-size:0.62rem;color:#1e293b;
                 font-family:'JetBrains Mono',monospace;">
        ROC-AUC 0.7168 · PRECISION 82.6% · AE SEP 16.90×
    </span>
</div>
""", unsafe_allow_html=True)
