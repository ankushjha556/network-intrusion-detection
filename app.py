import os
import gdown
import joblib
import torch
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.autoencoder import NetworkAutoencoder


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NetGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🛡️ NetGuard AI")
st.subheader(
    "Hybrid Network Intrusion Detection System"
)

st.markdown("""
Deep Autoencoder + Isolation Forest + One-Class SVM  
Unsupervised anomaly detection trained on CICIDS2017
""")

# =========================================================
# GOOGLE DRIVE FILES
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
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    os.makedirs("models", exist_ok=True)

    for filename, file_id in FILES.items():

        filepath = f"models/{filename}"

        if not os.path.exists(filepath):

            url = (
                f"https://drive.google.com/uc?id={file_id}"
            )

            gdown.download(
                url,
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
# MODEL LOADING
# =========================================================

with st.spinner("Loading models..."):

    scaler, isolation_forest, ocsvm, config, model = load_models()

st.success("Models loaded successfully!")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Model Performance")

st.sidebar.metric(
    "ROC-AUC",
    "0.7168"
)

st.sidebar.metric(
    "AUPRC",
    "0.5853"
)

st.sidebar.metric(
    "Precision",
    "82.64%"
)

st.sidebar.metric(
    "False Alarm Rate",
    "2.17%"
)

# =========================================================
# INPUT
# =========================================================

st.header("Input Network Features")

st.markdown("""
Enter the 40 network-flow features.
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

if st.button(
    "Analyze Traffic",
    use_container_width=True
):

    x = np.array(features).reshape(1, -1)

    # Scaling
    x_scaled = scaler.transform(x)

    x_tensor = torch.tensor(
        x_scaled,
        dtype=torch.float32
    )

    # AE inference
    with torch.no_grad():

        reconstruction, latent = model(x_tensor)

    reconstruction = reconstruction.numpy()

    # Reconstruction Error
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
        x_scaled
    )[0]

    if_score = np.clip(
        1 - ((if_raw + 0.15) / 0.30),
        0,
        1
    )

    # OCSVM
    svm_raw = ocsvm.decision_function(
        x_scaled
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
    # RESULTS
    # =====================================================

    st.header("Detection Results")

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
        "Ensemble Score",
        f"{ensemble_score:.4f}"
    )

    st.markdown("---")

    if attack:

        st.error(
            "⚠️ ATTACK DETECTED"
        )

    else:

        st.success(
            "✅ NORMAL TRAFFIC"
        )

    st.write(
        f"Detection Threshold: {threshold:.4f}"
    )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    fig = go.Figure(go.Indicator(

        mode="gauge+number",

        value=float(ensemble_score),

        title={
            "text": "Threat Score"
        },

        gauge={

            "axis": {
                "range": [0, 1]
            },

            "threshold": {
                "line": {
                    "color": "red",
                    "width": 4
                },
                "value": threshold
            }
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )