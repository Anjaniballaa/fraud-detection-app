# 🔍 Explainable Hybrid Fraud Detection System
### Quantum-LSTM · Classical ML · Autoencoder · SHAP · LIME · Streamlit

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/PennyLane-Quantum-6600CC?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/XGBoost-GPU-00A36C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-FFA500?style=for-the-badge"/>
</p>

<p align="center">
  <b>🚀 Live Demo →</b> <a href="https://fraud-detection-app-lhkquidwwcuwupbz2ewpqr.streamlit.app/">fraud-detection-app.streamlit.app</a>
</p>

---

## 🧠 What Makes This Project Unique

Most fraud detection projects train one model and stop. This project goes further — it builds and compares **7 completely different approaches** to fraud detection, explains every prediction using SHAP and LIME, and deploys everything as a live interactive dashboard.

The centrepiece is a **Quantum-LSTM hybrid model** — a variational quantum circuit (built with PennyLane) fused with an LSTM network (built with PyTorch) — one of the few undergraduate projects in India to implement a quantum-classical hybrid for a real financial use case.

---

## 🎯 Project Highlights

| What | Detail |
|---|---|
| **Dataset** | IEEE-CIS Fraud Detection (Kaggle) — 590,000 transactions, 400+ features |
| **Models** | 7 total: LR · DT · RF · KNN · XGBoost · Quantum-LSTM · Autoencoder |
| **Imbalance fix** | SMOTE oversampling (3.5% fraud → 50/50 balanced training set) |
| **Best AUC** | 0.999+ (XGBoost + Quantum-LSTM) |
| **Explainability** | SHAP beeswarm · waterfall · dependence · LIME per-transaction |
| **Deployment** | Streamlit Cloud — live public link, zero installation |
| **Hardware** | Trained on Google Colab T4 GPU |

---

## 🗃️ Dataset — IEEE-CIS Fraud Detection

> Real-world transaction data provided by Vesta Corporation for a Kaggle competition

- **590,540** transactions · **400+** raw features
- Only **3.5% fraud** — severely imbalanced
- Two files merged: `train_transaction.csv` + `train_identity.csv`
- Features include: transaction amount, card type, email domain, device info, time deltas, Vesta-engineered V1–V339 features

**Preprocessing pipeline:**
1. Merge transaction + identity tables on `TransactionID`
2. Drop columns with >50% missing values
3. Label encode all categorical features (ProductCD, card4, email domains, DeviceType etc.)
4. Fill remaining NaN with column median
5. StandardScaler normalization
6. SMOTE applied **only on training data** to prevent data leakage

---

## 🌐 Streamlit Dashboard

Five pages accessible from the sidebar:

| Page | What you see |
|---|---|
| 🏠 Overview | KPI cards, leaderboard, pipeline diagram, class imbalance |
| 📊 Model Comparison | ROC curves, confusion matrices, PR curves, AUC bar chart, agreement heatmap |
| 🧠 Explainability | SHAP beeswarm, waterfall, dependence, LIME — with explanations |
| 🔬 Live Prediction | Upload CSV → get predictions + probability scores + risk breakdown + download |
| 📋 Data Insights | Feature importance, fraud probability distributions, calibration curve |

**Live app:** https://fraud-detection-app-lhkquidwwcuwupbz2ewpqr.streamlit.app/

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection/app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**requirements.txt**
```
streamlit==1.32.0
pandas
numpy
scikit-learn
xgboost
joblib
shap
lime
torch
matplotlib
seaborn
pennylane
pennylane-lightning
imbalanced-learn
```

---

## 🔬 Reproduce the Results

Run the notebooks in order in Google Colab with a **T4 GPU runtime**:

```
1_data_cleaning.ipynb      → ~10 mins  (SMOTE takes longest)
2_classical_models.ipynb   → ~15 mins
3_quantum_lstm.ipynb       → ~30 mins  (quantum circuit simulation)
4_autoencoder.ipynb        → ~5 mins
5_explainability.ipynb     → ~10 mins
```

Dataset: [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)


```
