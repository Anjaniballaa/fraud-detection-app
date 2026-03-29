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

## 📁 Project Structure

```
fraud-detection/
│
├── notebooks/
│   ├── 1_data_cleaning.ipynb         # Merge, clean, encode, SMOTE
│   ├── 2_classical_models.ipynb      # LR, DT, RF, KNN, XGBoost
│   ├── 3_quantum_lstm.ipynb          # PennyLane + PyTorch hybrid
│   ├── 4_autoencoder.ipynb           # Unsupervised anomaly detection
│   └── 5_explainability.ipynb        # SHAP + LIME + all final plots
│
├── app/
│   ├── app.py                        # Streamlit dashboard
│   └── requirements.txt
│
├── models/                           # Saved .joblib and .pth files
├── plots/                            # All saved figures
└── README.md
```

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

## 🤖 Models

### 1️⃣ Logistic Regression
Baseline linear model. Uses `class_weight='balanced'` to handle imbalance. Fast to train, highly interpretable, gives a solid lower bound for comparison.

### 2️⃣ Decision Tree
Single tree with `max_depth=8`. Visualizable — you can trace exactly why a transaction was flagged. Used as a baseline for tree-based methods.

### 3️⃣ Random Forest
Ensemble of 100 decision trees. Reduces overfitting through bagging. Strong performance with built-in feature importance scores.

### 4️⃣ K-Nearest Neighbors
Instance-based learner. No training phase — classifies by proximity to known fraud transactions. Sensitive to feature scaling (hence StandardScaler applied first).

### 5️⃣ XGBoost ⭐
Gradient boosted trees with GPU acceleration (`tree_method='hist', device='cuda'`). Best performing classical model. Uses `scale_pos_weight` to handle imbalance. Tuned with 200 estimators, learning rate 0.05, max depth 6.

### 6️⃣ Quantum-LSTM Hybrid 🔬
The most novel component of this project. Architecture:

```
Input (400+ features)
    ↓
LSTM (64 hidden units, batch_first=True)
    ↓
Dropout (0.3)
    ↓
FC Bridge (64 → 4)   ← compress to quantum size
    ↓
Quantum Circuit (4 qubits, PennyLane)
    ├── Angle encoding: RY gates encode LSTM features as qubit rotations
    ├── Entanglement: CNOT gates create quantum correlations
    └── Variational layer: trainable RY gates optimized via backprop
    ↓
FC Output (4 → 1) → Sigmoid
    ↓
Fraud probability
```

```
0: ──RY(θ₀)─╭●──RY(w₀)──┤ <Z>
1: ──RY(θ₁)─╰X─╭●──RY(w₁)─┤ <Z>
2: ──RY(θ₂)────╰X─╭●──RY(w₂)─┤ <Z>
3: ──RY(θ₃)───────╰X──RY(w₃)─┤ <Z>
```

- Data reshaped into time sequences (5 timesteps) for LSTM
- Quantum weights trained end-to-end via PyTorch autograd
- Learning rate scheduler (StepLR) for stable convergence
- Best model checkpoint saved on highest validation AUC

### 7️⃣ Autoencoder (Unsupervised) 🔎
Trained **only on normal transactions** — never sees a single fraud example during training.

```
Input (400+ features)
    → Encoder: 64 → 32 → 16 → 8 (bottleneck)
    → Decoder: 8 → 16 → 32 → 64 → output
```

The key insight: the autoencoder learns to reconstruct normal transactions well. Fraud transactions have unusual patterns — the model reconstructs them poorly. High reconstruction error = likely fraud. Threshold set at 95th percentile of normal reconstruction errors.

---

## 📊 Results

| Model | Type | ROC-AUC | Notes |
|---|---|---|---|
| XGBoost | Classical | **0.999+** | Best classical, GPU accelerated |
| Quantum-LSTM | Quantum-Hybrid | **0.999** | Novel architecture |
| Random Forest | Classical | ~0.97 | Strong ensemble |
| KNN | Classical | ~0.99 | Excellent with scaling |
| Logistic Regression | Classical | ~0.98 | Fast baseline |
| Decision Tree | Classical | ~0.97 | Most interpretable |
| Autoencoder | Unsupervised | ~0.85+ | No labels needed |

> The Autoencoder's lower AUC is expected — it uses **zero fraud labels** during training, making it the most realistic model for real-world deployment where fraud patterns are unknown.

---

## 🧪 Explainability

### SHAP (SHapley Additive exPlanations)
Based on game theory. Every feature gets a Shapley value — its fair contribution to the prediction.

- **Beeswarm plot** — global view of which features drive fraud across all transactions
- **Waterfall plot** — why one specific transaction was flagged as fraud
- **Bar plot** — mean absolute SHAP values ranked by importance
- **Dependence plot** — how the top feature's value affects fraud probability

### LIME (Local Interpretable Model-agnostic Explanations)
Fits a simple interpretable model locally around one prediction. Shows which features pushed the model toward fraud for that specific transaction.

> **Key finding:** Features like `TransactionAmt`, `card1`, `addr1`, and Vesta V-series features dominate fraud prediction across all models.

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
