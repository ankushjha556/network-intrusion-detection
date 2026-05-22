import streamlit as st
import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib
import gdown
import os
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NetGuard AI — Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CLEAN CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #020617 !important;
    color: #f8fafc !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    padding: 1.2rem;
    border-radius: 18px;
}

.hero {
    text-align: center;
    padding: 2rem 0 1rem 0;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(90deg,#10b981,#06b6d4,#8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    max-width: 750px;
    margin: auto;
    line-height: 1.8;
}

.sidebar-title {
    font-size: 1.1rem;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# GOOGLE DRIVE FILES
# ============================================================
FILES = {
    "models/autoencoder.onnx": "11lN5JzCotEfej2WEdU3A-YKu53tXOxVL",
    "models/isolation_forest.pkl": "1OGFhLXB68-zpLCCiS2I19HY-zg8kC7og",
    "models/ocsvm.pkl": "1e7OFrSKaTQw6p98muXKNTGJGoltmEFtX",
    "models/scaler_ae.pkl": "1jznFfgiv0fJ1S0tFsllFpMXKfQSYpbiK",
    "models/config.pkl": "1ZSBujLvO1lWAzdSMgzX_BV6JWKtoYNsV",
}

# ============================================================
# DOWNLOAD FILES
# ============================================================
@st.cache_resource
def download_models():

    os.makedirs("models", exist_ok=True)

    for path, file_id in FILES.items():

        if not os.path.exists(path):

            url = f"https://drive.google.com/uc?id={file_id}"

            with st.spinner(f"Downloading {os.path.basename(path)}..."):

                gdown.download(
                    url,
                    path,
                    quiet=False
                )

download_models()

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():

    scaler = joblib.load("models/scaler_ae.pkl")

    isolation_forest = joblib.load(
        "models/isolation_forest.pkl"
    )

    ocsvm = joblib.load(
        "models/ocsvm.pkl"
    )

    config = joblib.load(
        "models/config.pkl"
    )

    session = ort.InferenceSession(
        "models/autoencoder.onnx",
        providers=["CPUExecutionProvider"]
    )

    return scaler, isolation_forest, ocsvm, config, session

scaler, isolation_forest, ocsvm, config, ort_session = load_models()

# ============================================================
# MODEL PREDICTION
# ============================================================
def predict_network_traffic(features):

    x = np.array(features).reshape(1, -1).astype(np.float32)

    # Scaling
    x_scaled = scaler.transform(x).astype(np.float32)

    # Dynamic ONNX input name
    input_name = ort_session.get_inputs()[0].name

    # Reconstruction
    reconstructed = ort_session.run(
        None,
        {input_name: x_scaled}
    )[0]

    # AE Reconstruction Error
    reconstruction_error = np.mean(
        (x_scaled - reconstructed) ** 2
    )

    ae_score = np.clip(
        np.log1p(reconstruction_error) / 3.0,
        0,
        1
    )

    # Isolation Forest
    if_score = -isolation_forest.decision_function(
        x_scaled
    )[0]

    if_score = np.clip(
        (if_score + 0.5) / 1.5,
        0,
        1
    )

    # One-Class SVM
    svm_score = -ocsvm.decision_function(
        x_scaled
    )[0]

    svm_score = np.clip(
        (svm_score + 0.5) / 1.5,
        0,
        1
    )

    # Ensemble
    final_score = (
        config["w_ae"] * ae_score +
        config["w_if"] * if_score +
        config["w_svm"] * svm_score
    )

    threshold = config["threshold"]

    prediction = (
        "Attack"
        if final_score >= threshold
        else "Benign"
    )

    confidence = min(
        abs(final_score - threshold) /
        threshold * 100,
        100
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "ensemble_score": final_score,
        "ae_score": ae_score,
        "if_score": if_score,
        "svm_score": svm_score,
        "reconstruction_error": reconstruction_error
    }

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.markdown(
        "<div class='sidebar-title'>🛡️ NetGuard AI</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(f"""
    ### Model Metrics

    - ROC-AUC: **0.7168**
    - AUPRC: **0.5853**
    - Precision: **82.64%**
    - Recall: **42.18%**
    - False Alarm Rate: **2.17%**
    - Dataset: **CICIDS2017**
    - Method: **Unsupervised**
    """)

    st.markdown("---")

    threshold = st.slider(
        "Detection Threshold",
        0.05,
        0.60,
        float(config["threshold"]),
        0.01
    )

# ============================================================
# HERO SECTION
# ============================================================
st.markdown("""
<div class="hero">

<div class="hero-title">
Hybrid Network Intrusion Detection System
</div>

<div class="hero-sub">
Deep Autoencoder + Isolation Forest + One-Class SVM
for unsupervised cyber anomaly detection using latent
representation learning on CICIDS2017.
</div>

</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# METRICS ROW
# ============================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("ROC-AUC", "0.7168")

with c2:
    st.metric("AUPRC", "0.5853")

with c3:
    st.metric("Precision", "82.64%")

with c4:
    st.metric("AE Separation", "16.9×")

st.markdown("---")

# ============================================================
# FEATURE INPUT
# ============================================================
st.subheader("🔍 Network Traffic Analysis")

st.markdown("""
Upload a CSV containing exactly 40 preprocessed
network flow features.
""")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# ============================================================
# PROCESS FILE
# ============================================================
if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.success(
            f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns"
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        # Remove labels if present
        if "Label" in df.columns:
            df = df.drop(columns=["Label"])

        if " Label" in df.columns:
            df = df.drop(columns=[" Label"])

        if df.shape[1] != 40:

            st.error(
                f"Expected 40 features but received {df.shape[1]}"
            )

            st.stop()

        # ====================================================
        # PREDICTIONS
        # ====================================================
        results = []

        progress = st.progress(0)

        for idx, row in enumerate(df.values):

            pred = predict_network_traffic(row)

            results.append(pred)

            progress.progress(
                (idx + 1) / len(df)
            )

        progress.empty()

        result_df = pd.DataFrame(results)

        combined_df = pd.concat(
            [df, result_df],
            axis=1
        )

        # ====================================================
        # SUMMARY
        # ====================================================
        attack_count = (
            combined_df["prediction"] == "Attack"
        ).sum()

        benign_count = (
            combined_df["prediction"] == "Benign"
        ).sum()

        cc1, cc2, cc3 = st.columns(3)

        with cc1:
            st.metric(
                "Detected Attacks",
                attack_count
            )

        with cc2:
            st.metric(
                "Benign Traffic",
                benign_count
            )

        with cc3:
            st.metric(
                "Attack Rate",
                f"{100 * attack_count / len(combined_df):.2f}%"
            )

        st.markdown("---")

        # ====================================================
        # SCORE DISTRIBUTION
        # ====================================================
        st.subheader("📈 Ensemble Score Distribution")

        fig = px.histogram(
            combined_df,
            x="ensemble_score",
            color="prediction",
            nbins=50,
            template="plotly_dark"
        )

        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="yellow"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ====================================================
        # TOP ANOMALIES
        # ====================================================
        st.subheader("🚨 Highest Risk Samples")

        top_df = combined_df.sort_values(
            by="ensemble_score",
            ascending=False
        ).head(20)

        st.dataframe(
            top_df,
            use_container_width=True
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================
        csv = combined_df.to_csv(index=False)

        st.download_button(
            "⬇️ Download Results",
            csv,
            "intrusion_detection_results.csv",
            "text/csv"
        )

    except Exception as e:

        st.error(f"Error: {str(e)}")

# ============================================================
# RESEARCH FINDINGS
# ============================================================
st.markdown("---")

st.subheader("🔬 Research Findings")

c1, c2 = st.columns(2)

with c1:

    st.info("""
    Autoencoder reconstruction error produced
    stronger anomaly separation than expected.
    Ensemble detectors improved robustness
    but occasionally diluted latent-space
    discriminative structure.
    """)

with c2:

    st.warning("""
    Increasing recall beyond 90% caused
    a major increase in false positives,
    demonstrating the operational tradeoff
    between sensitivity and alert reliability.
    """)

# ============================================================
# ABLATION STUDY
# ============================================================
st.markdown("---")

st.subheader("📊 Ablation Study")

ablation_df = pd.DataFrame({
    "Model": [
        "AE only",
        "IF only",
        "OCSVM only",
        "AE + IF",
        "AE + OCSVM",
        "IF + OCSVM",
        "Full Ensemble"
    ],
    "ROC-AUC": [
        0.7891,
        0.7183,
        0.6405,
        0.7243,
        0.6423,
        0.7093,
        0.7125
    ],
    "AUPRC": [
        0.6196,
        0.4381,
        0.5289,
        0.5586,
        0.5597,
        0.5113,
        0.5675
    ],
    "F1": [
        0.5743,
        0.5106,
        0.5619,
        0.5516,
        0.5764,
        0.5270,
        0.5512
    ]
})

st.dataframe(
    ablation_df,
    use_container_width=True
)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.markdown("""
### ⚠️ Disclaimer

This project is designed for research,
educational, and experimental purposes only.

It is not intended for deployment in
real-world critical cybersecurity infrastructure.

---

**Author:** Ankush Jha  
BS — IIT Patna
""")