import os
import gdown
import joblib
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from autoencoder import NetworkAutoencoder


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #030712;
    color: white;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(16,185,129,0.12), transparent 30%),
        radial-gradient(circle at bottom right, rgba(99,102,241,0.12), transparent 30%),
        #030712;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

section[data-testid="stSidebar"] {
    background: #071019;
    border-right: 1px solid rgba(255,255,255,0.06);
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 1.2rem;
    border-radius: 18px;
}

.main-title {
    font-size: 4rem;
    font-weight: 900;
    line-height: 1.0;
    background: linear-gradient(
        90deg,
        #ffffff,
        #10b981,
        #6366f1
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #94a3b8;
    font-size: 1rem;
    line-height: 1.7;
}

.attack-box {
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 18px;
    padding: 1.5rem;
}

.normal-box {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 18px;
    padding: 1.5rem;
}

.stButton>button {
    width: 100%;
    border-radius: 14px;
    height: 3.2rem;
    font-weight: 700;
    background: linear-gradient(
        90deg,
        #10b981,
        #059669
    );
    border: none;
    color: white;
}

.stButton>button:hover {
    background: linear-gradient(
        90deg,
        #059669,
        #047857
    );
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL LINKS
# =========================================================

FILES = {

    "autoencoder.pth":
    "1jO-lfstkeUehY051pSbkF6x7jbdlzdHc",

    "isolation_forest.pkl":
    "1OGFhLXB68-zpLCCiS2I19HY-zg8kC7og",

    "ocsvm.pkl":
    "1e7OFrSKaTQw6p98muXKNTGJGoltmEFtX",

    "scaler_ae.pkl":
    "1jznFfgiv0fJ1S0tFsllFpMXKfQSYpbiK",

    "config.pkl":
    "1ZSBujLvO1lWAzdSMgzX_BV6JWKtoYNsV"
}

# =========================================================
# DOWNLOAD + LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    os.makedirs("models", exist_ok=True)

    for filename, file_id in FILES.items():

        filepath = f"models/{filename}"

        if not os.path.exists(filepath):

            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                filepath,
                quiet=False
            )

    config = joblib.load("models/config.pkl")

    scaler = joblib.load(
        "models/scaler_ae.pkl"
    )

    isolation_forest = joblib.load(
        "models/isolation_forest.pkl"
    )

    ocsvm = joblib.load(
        "models/ocsvm.pkl"
    )

    model = NetworkAutoencoder(
        input_dim=config["input_dim"],
        latent_dim=config["latent_dim"]
    )

    model.load_state_dict(
        torch.load(
            "models/autoencoder.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return (
        scaler,
        isolation_forest,
        ocsvm,
        config,
        model
    )


# =========================================================
# LOAD MODELS
# =========================================================

with st.spinner("Loading NetGuard AI models..."):

    scaler, isolation_forest, ocsvm, config, model = load_models()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🛡️ NetGuard AI")

    st.markdown("""
    Advanced Hybrid Intrusion Detection System
    """)

    st.divider()

    st.subheader("📊 Model Metrics")

    st.metric("ROC-AUC", "0.7168")
    st.metric("AUPRC", "0.5853")
    st.metric("Precision", "82.64%")
    st.metric("False Alarm Rate", "2.17%")
    st.metric("Latent Space", "8-D")
    st.metric("Dataset", "CICIDS2017")

    st.divider()

    st.subheader("🧠 Architecture")

    st.markdown("""
    - Deep Autoencoder
    - Isolation Forest
    - One-Class SVM
    - Latent-Space Detection
    - Unsupervised Learning
    """)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="main-title">
NetGuard AI
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Hybrid deep-learning intrusion detection system using latent-space anomaly analysis.
The autoencoder learns normal traffic representations while Isolation Forest and
One-Class SVM detect abnormal network behavior in compressed feature space.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# PERFORMANCE CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
        <h3>ROC-AUC</h3>
        <h1 style="color:#10b981;">0.7168</h1>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
        <h3>Precision</h3>
        <h1 style="color:#06b6d4;">82.64%</h1>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
        <h3>False Alarms</h3>
        <h1 style="color:#f59e0b;">2.17%</h1>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
        <h3>Flows</h3>
        <h1 style="color:#8b5cf6;">2.83M</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# INPUT
# =========================================================

st.header("🔍 Analyze Network Traffic")

st.markdown("""
Enter the 40 numerical network-flow features.
""")

features = []

cols = st.columns(4)

for i in range(40):

    with cols[i % 4]:

        value = st.number_input(
            f"Feature {i+1}",
            value=0.0,
            format="%.4f"
        )

        features.append(value)

# =========================================================
# PREDICTION
# =========================================================

if st.button("🚀 Analyze Traffic"):

    x = np.array(features).reshape(1, -1)

    # Scale
    x_scaled = scaler.transform(x)

    x_tensor = torch.tensor(
        x_scaled,
        dtype=torch.float32
    )

    # Autoencoder inference
    with torch.no_grad():

        reconstruction, latent = model(x_tensor)

    reconstruction = reconstruction.numpy()

    latent_np = latent.numpy()

    # Reconstruction error
    ae_error = np.mean(
        (x_scaled - reconstruction) ** 2
    )

    ae_score = np.clip(
        np.log1p(ae_error) / 2.5,
        0,
        1
    )

    # Isolation Forest
    if_raw = isolation_forest.decision_function(
        latent_np
    )[0]

    if_score = np.clip(
        1 - ((if_raw + 0.15) / 0.30),
        0,
        1
    )

    # OCSVM
    svm_raw = ocsvm.decision_function(
        latent_np
    )[0]

    svm_score = np.clip(
        1 - ((svm_raw + 0.5) / 1.0),
        0,
        1
    )

    # Ensemble
    ensemble_score = (
        config["w_ae"] * ae_score
        + config["w_if"] * if_score
        + config["w_svm"] * svm_score
    )

    threshold = config["threshold"]

    attack = ensemble_score >= threshold

    # =====================================================
    # RESULT BOX
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    if attack:

        st.markdown(f"""
        <div class="attack-box">
            <h1 style="color:#ef4444;">
            ⚠️ ATTACK DETECTED
            </h1>
            <p>
            Suspicious anomalous network behavior detected.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="normal-box">
            <h1 style="color:#10b981;">
            ✅ NORMAL TRAFFIC
            </h1>
            <p>
            Network traffic appears benign.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # SCORE METRICS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "AE Score",
        f"{ae_score:.4f}"
    )

    c2.metric(
        "IF Score",
        f"{if_score:.4f}"
    )

    c3.metric(
        "SVM Score",
        f"{svm_score:.4f}"
    )

    c4.metric(
        "Ensemble",
        f"{ensemble_score:.4f}"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # GAUGE CHART
    # =====================================================

    fig = go.Figure(go.Indicator(

        mode="gauge+number",

        value=float(ensemble_score),

        title={
            "text": "Threat Probability"
        },

        gauge={

            "axis": {
                "range": [0, 1]
            },

            "bar": {
                "color":
                "#ef4444" if attack else "#10b981"
            },

            "steps": [

                {
                    "range": [0, threshold],
                    "color": "rgba(16,185,129,0.25)"
                },

                {
                    "range": [threshold, 1],
                    "color": "rgba(239,68,68,0.25)"
                }

            ],

            "threshold": {

                "line": {
                    "color": "white",
                    "width": 4
                },

                "value": threshold
            }
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================================
    # SCORE BREAKDOWN
    # =====================================================

    st.subheader("📈 Ensemble Breakdown")

    breakdown = pd.DataFrame({

        "Model": [
            "Autoencoder",
            "Isolation Forest",
            "One-Class SVM"
        ],

        "Score": [
            ae_score,
            if_score,
            svm_score
        ]
    })

    st.bar_chart(
        breakdown.set_index("Model")
    )

    # =====================================================
    # TECHNICAL DETAILS
    # =====================================================

    with st.expander("🧠 Technical Details"):

        st.markdown(f"""
        ### Inference Summary

        - **Threshold:** `{threshold:.4f}`
        - **AE Reconstruction Error:** `{ae_error:.6f}`
        - **Latent Dimension:** `{latent_np.shape[1]}`
        - **Ensemble Weights**
            - AE = `{config['w_ae']}`
            - IF = `{config['w_if']}`
            - SVM = `{config['w_svm']}`

        ### Architecture

        ```text
        Input Features (40)
                ↓
        Deep Autoencoder
                ↓
        Latent Space (8D)
                ↓
        Isolation Forest
        One-Class SVM
                ↓
        Weighted Ensemble
                ↓
        Final Intrusion Score
        ```

        ### Dataset

        - CICIDS2017
        - 2.83 million flows
        - 14 attack categories
        - Fully unsupervised anomaly detection
        """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center;color:#64748b;font-size:0.9rem;">
Built by Ankush Jha · IIT Patna<br>
Deep Learning · Anomaly Detection · Cybersecurity AI
</div>
""", unsafe_allow_html=True)