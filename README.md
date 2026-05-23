# Hybrid Network Intrusion Detection using Autoencoders and Ensemble Anomaly Detection

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://network-intrusion-detection-7dmy9tjcgmhsxqxjhlwrtd.streamlit.app/)

This project implements a network intrusion detection pipeline using deep representation learning and unsupervised anomaly detection. A PyTorch autoencoder learns compressed latent representations of network traffic, and anomaly scores are generated from reconstruction error and classical anomaly detection models operating in the learned latent space.

The goal is to study whether latent-space representations can improve intrusion detection on high-dimensional and imbalanced network traffic data, where attack samples are often sparse compared with benign traffic.

## Live Demo

Streamlit app: https://network-intrusion-detection-7dmy9tjcgmhsxqxjhlwrtd.streamlit.app/

The deployed app provides an interactive interface for inspecting anomaly detection outputs, model evaluation plots, threshold behavior, and latent-space analysis.

## Project Overview

Traditional intrusion detection systems can struggle with high-dimensional traffic features, class imbalance, noisy distributions, and changing attack behavior. This project approaches intrusion detection as an anomaly detection problem rather than a direct supervised multi-class classification task.

The pipeline first trains a deep autoencoder to reconstruct network traffic features. The reconstruction error is used as one anomaly signal, while the learned 8-dimensional latent embeddings are passed to additional anomaly detectors. The final system evaluates individual detectors and weighted ensembles using imbalance-aware metrics.

## Pipeline

```text
Network Traffic Features
        |
        v
Feature Scaling and Clipping
        |
        v
Deep Autoencoder
        |
        +--> Reconstruction Error
        |
        v
8-Dimensional Latent Space
        |
        +--> Isolation Forest
        +--> One-Class SVM
        +--> Local Outlier Factor Experiments
        |
        v
Score Normalization
        |
        v
Weighted Ensemble Scoring
        |
        v
Threshold Optimization
        |
        v
Intrusion Prediction
```

## Dataset

The project uses the CICIDS2017 network intrusion detection dataset.

| Attribute | Details |
| --- | --- |
| Dataset | CICIDS2017 |
| Records | Approximately 2.83 million |
| Features used | 40 |
| Traffic types | Benign and attack traffic |
| Attack categories | DDoS, DoS, brute force, botnet, port scan, infiltration, and web attacks |
| Learning setup | Unsupervised anomaly detection |

The dataset was loaded from saved NumPy arrays. Large training arrays were memory-mapped during loading to reduce RAM usage.

## Preprocessing

The preprocessing stage includes:

- loading `X_train`, `X_test`, and `y_test`
- fitting `StandardScaler` on a 200,000-sample subset of the training data
- transforming train and test features
- clipping extreme scaled values to the range `[-10, 10]`
- saving scaled arrays and the fitted scaler for reuse
- freeing unused arrays during the workflow to reduce memory pressure

## Autoencoder

The autoencoder was implemented from scratch in PyTorch. It compresses 40 input features into an 8-dimensional latent representation.

The network uses:

- fully connected encoder and decoder layers
- BatchNorm1d
- GELU activations
- Dropout regularization
- AdamW optimizer
- MSE reconstruction loss
- latent-space regularization
- CosineAnnealingLR scheduler

Training configuration:

| Item | Value |
| --- | --- |
| Input dimension | 40 |
| Latent dimension | 8 |
| Epochs | 60 |
| Batch size | 4096 |
| Loss | MSE + latent regularization |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Device | CUDA when available, otherwise CPU |

The best autoencoder checkpoint was saved based on validation loss.

## Anomaly Detection Models

After training the autoencoder, latent representations and reconstruction errors were extracted for train and test sets. Several anomaly detection methods were evaluated:

| Model | Role |
| --- | --- |
| Autoencoder reconstruction error | Measures how poorly a sample is reconstructed |
| Isolation Forest | Detects sparse or isolated latent-space regions |
| One-Class SVM | Learns a boundary around normal latent behavior |
| Local Outlier Factor | Evaluated as an additional local-density detector |

The reconstruction error showed strong separation between benign and attack traffic, with attack samples producing substantially higher reconstruction error on average.

## Ensemble Strategy

The project evaluates weighted ensemble scoring across multiple anomaly signals. Scores are normalized before fusion, and thresholds are selected using precision-recall analysis.

Several score calibration approaches were tested:

- min-max normalized reconstruction error
- percentile-clipped reconstruction error
- log-normalized reconstruction error
- corrected LOF score direction
- grid search over ensemble weights
- threshold selection for best F1-score
- high-recall operating point analysis

Final experiments showed that the autoencoder reconstruction error was the strongest individual anomaly signal. Ensemble components improved robustness in some settings, but weaker detectors could also dilute the strongest signal.

## Final Model

The selected final configuration uses an autoencoder-dominant ensemble:

| Component | Weight |
| --- | ---: |
| Autoencoder reconstruction score | 0.70 |
| Isolation Forest score | 0.20 |
| One-Class SVM score | 0.10 |

## Results

| Metric | Score |
| --- | ---: |
| ROC-AUC | 0.7891 |
| PR-AUC | 0.6196 |
| F1-score | 0.5743 |
| Precision | 82.6% |
| False Positive Rate | 2.17% |

The results show that the learned latent representation provides useful anomaly separation, especially through autoencoder reconstruction error. The evaluation also highlights a practical intrusion detection tradeoff: increasing recall can catch more attacks but also raises the false alarm rate.

## Evaluation and Analysis

The project includes:

- ROC curve analysis
- precision-recall curve analysis
- confusion matrix evaluation
- score distribution plots
- reconstruction error analysis
- model comparison plots
- ablation studies
- high-recall threshold analysis
- t-SNE visualization of latent embeddings

The t-SNE analysis samples benign and attack traffic from the learned 8-dimensional latent space and projects it into 2D. The visualization helps inspect whether attack traffic forms separable regions in the learned representation.

## Deployment

The project includes deployment preparation for Streamlit:

- trained autoencoder checkpoint
- Isolation Forest model
- One-Class SVM model
- fitted scaler
- configuration file with threshold, weights, dimensions, and metrics
- ONNX export of the autoencoder for portable inference
- Streamlit app for interactive inspection

The app is deployed on Streamlit Community Cloud.

## Tech Stack

- Python
- PyTorch
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- ONNX
- Google Colab
- Google Drive

## Repository Structure

```text
.
├── app/
│   └── streamlit_app.py
├── data/
├── models/
├── notebooks/
├── src/
│   ├── autoencoder.py
│   ├── preprocessing.py
│   ├── anomaly_models.py
│   ├── ensemble.py
│   ├── evaluation.py
│   └── visualization.py
├── requirements.txt
└── README.md
```

Some dashboard and notebook inspection features may be extended further in future versions.

## Key Learning Outcomes

This project demonstrates:

- unsupervised anomaly detection for cybersecurity
- autoencoder-based representation learning
- latent-space anomaly scoring
- ensemble calibration
- threshold optimization
- imbalanced metric evaluation
- ablation-driven model selection
- deployment of an ML workflow with Streamlit

## Important Note

This project is intended for research, education, and experimentation. It is not a replacement for production-grade intrusion detection systems or enterprise cybersecurity monitoring infrastructure.

## Author

Ankush Jha  
BS, IIT Patna  
GitHub: [@ankushjha556](https://github.com/ankushjha556)
