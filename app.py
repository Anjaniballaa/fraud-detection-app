import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from joblib import load
import json, pickle, warnings
warnings.filterwarnings('ignore')

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection — XAI Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3a5f, #2d5986);
    border-radius: 12px; padding: 18px 22px; color: white;
    text-align: center; margin-bottom: 10px;
}
.metric-card .value { font-size: 28px; font-weight: 700; color: #7ec8ff; }
.metric-card .label { font-size: 13px; color: #b0c8e0; margin-top: 4px; }
.fraud-badge { background: #c0392b; color: white; border-radius: 6px;
    padding: 2px 10px; font-size: 12px; font-weight: 600; }
.legit-badge { background: #1a7a4a; color: white; border-radius: 6px;
    padding: 2px 10px; font-size: 12px; font-weight: 600; }
.section-header { background: #1e3a5f; color: white; padding: 10px 18px;
    border-radius: 8px; font-size: 16px; font-weight: 600; margin: 16px 0 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Load assets ──────────────────────────────────────────────
@st.cache_resource
def load_everything():
    scaler = load('models/scaler.joblib')
    models = {
        'XGBoost':             load('models/xgboost.joblib'),
        'Random Forest':       load('models/random_forest.joblib'),
        'Logistic Regression': load('models/logistic_regression.joblib'),
        'Decision Tree':       load('models/decision_tree.joblib'),
        'KNN':                 load('models/knn.joblib'),
    }
    with open('models/feature_columns.json') as f:
        features = json.load(f)
    with open('models/classical_results.pkl', 'rb') as f:
        classical = pickle.load(f)
    with open('models/quantum_results.pkl', 'rb') as f:
        quantum = pickle.load(f)
    with open('models/autoencoder_results.pkl', 'rb') as f:
        ae = pickle.load(f)
    try:
        comparison = pd.read_csv('models/final_comparison.csv')
    except:
        comparison = None

    # Try sample first, fall back to None
    try:
        X_test  = np.load('models/X_test_sample.npy')
        y_test  = np.load('models/y_test_sample.npy')
    except:
        X_test  = None
        y_test  = np.load('models/y_test.npy')

    try:
        X_train = np.load('models/X_test_sample.npy')
        y_train = np.load('models/y_test_sample.npy')
    except:
        X_train = None
        y_train = np.load('models/y_train.npy')

    return scaler, models, features, classical, quantum, ae, comparison, X_test, y_test, X_train, y_train

scaler, models, features, classical, quantum, ae, comparison, X_test, y_test, X_train, y_train = load_everything()

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Fraud Detection")
    st.caption("IEEE-CIS Dataset · XAI Dashboard")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Model Comparison",
        "🧠 Explainability",
        "🔬 Live Prediction",
        "📋 Data Insights"
    ])
    st.divider()
    st.caption("Models: LR · DT · RF · KNN · XGB · Quantum-LSTM · Autoencoder")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Credit Card Fraud Detection")
    st.markdown("**IEEE-CIS Dataset** · 7-model comparison · SHAP + LIME explainability · Quantum-LSTM hybrid")

    # ── KPI cards ──
    all_aucs  = [v['auc'] for v in classical.values()] + [quantum['auc'], ae['auc']]
    all_names = list(classical.keys()) + ['Quantum-LSTM', 'Autoencoder']
    fraud_rate = y_test.mean() * 100
    # Use full test count from y_test length (118k)
    test_count = len(y_test)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(f'<div class="metric-card"><div class="value">{fraud_rate:.2f}%</div><div class="label">Fraud Rate</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="value">{test_count:,}</div><div class="label">Test Transactions</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="value">7</div><div class="label">Models Trained</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="value">{max(all_aucs):.4f}</div><div class="label">Best AUC</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="metric-card"><div class="value">{len(features)}</div><div class="label">Features</div></div>', unsafe_allow_html=True)
    c6.markdown(f'<div class="metric-card"><div class="value">SMOTE</div><div class="label">Imbalance Fix</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Leaderboard + Class distribution side by side ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">🏆 Model Leaderboard</div>', unsafe_allow_html=True)
        rows = []
        model_types = {**{k: 'Classical' for k in classical}, 'Quantum-LSTM': 'Quantum-Hybrid', 'Autoencoder': 'Unsupervised'}
        for name, auc in zip(all_names, all_aucs):
            rows.append({'Model': name, 'ROC-AUC': round(auc, 4), 'Type': model_types.get(name, '-')})
        df_leader = pd.DataFrame(rows).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
        df_leader.index = df_leader.index + 1
        df_leader['Rank'] = df_leader.index
        df_leader['ROC-AUC'] = df_leader['ROC-AUC'].apply(lambda x: f"{x:.4f}")

        def highlight_best(row):
            if row.name == 1:
                return ['background-color: #1a4a1a; color: #7eff7e'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_leader[['Rank', 'Model', 'ROC-AUC', 'Type']].style.apply(highlight_best, axis=1),
            use_container_width=True, height=280
        )

    with col_right:
        st.markdown('<div class="section-header">⚖️ Class Imbalance</div>', unsafe_allow_html=True)
        try:
            st.image('plots/class_imbalance.png', use_container_width=True)
        except:
            fig, axes = plt.subplots(1, 2, figsize=(7, 4))
            orig  = np.bincount(y_test.astype(int))
            smote = np.bincount(y_train.astype(int))
            axes[0].pie(orig,  labels=['Legit','Fraud'], autopct='%1.2f%%',
                        colors=['#4a90d9','#e74c3c'],
                        wedgeprops={'edgecolor':'white','linewidth':2})
            axes[0].set_title('Before SMOTE')
            axes[1].pie(smote, labels=['Legit','Fraud'], autopct='%1.1f%%',
                        colors=['#4a90d9','#e74c3c'],
                        wedgeprops={'edgecolor':'white','linewidth':2})
            axes[1].set_title('After SMOTE')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    # ── Pipeline diagram ──
    st.markdown('<div class="section-header">🔁 Project Pipeline</div>', unsafe_allow_html=True)
    stages = ["IEEE-CIS\nRaw Data","Data\nCleaning","SMOTE\nBalancing","5 Classical\nModels",
              "Quantum\nLSTM","Autoencoder","SHAP+LIME\nExplainability","Streamlit\nDashboard"]
    colors = ['#2c3e50','#1a5276','#117a65','#6c3483','#7d6608','#922b21','#0e6655','#1a5276']
    fig, ax = plt.subplots(figsize=(16, 2.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 2); ax.axis('off')
    fig.patch.set_facecolor('#0f1117')
    for i, (stage, col) in enumerate(zip(stages, colors)):
        x = i * 2 + 0.2
        rect = mpatches.FancyBboxPatch((x, 0.4), 1.6, 1.2, boxstyle="round,pad=0.1",
                                        facecolor=col, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.8, 1.0, stage, ha='center', va='center', fontsize=8.5,
                color='white', fontweight='bold', multialignment='center')
        if i < len(stages) - 1:
            ax.annotate('', xy=(x+1.7, 1.0), xytext=(x+1.6, 1.0),
                        arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.5))
    plt.tight_layout(pad=0)
    st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")

    tab1, tab2, tab3, tab4 = st.tabs(["ROC Curves","Confusion Matrices","Precision-Recall","AUC Bar Chart"])

    with tab1:
        st.subheader("Combined ROC Curve — All 7 Models")
        try:
            st.image('plots/final_roc_comparison.png', use_container_width=True)
        except:
            fig, ax = plt.subplots(figsize=(10, 7))
            colors_r   = ['#3498db','#2ecc71','#f39c12','#e74c3c','#9b59b6','#e67e22','#1abc9c']
            all_names_r = list(classical.keys()) + ['Quantum-LSTM','Autoencoder']
            all_fprs    = [v['fpr'] for v in classical.values()] + [quantum['fpr'], ae['fpr']]
            all_tprs    = [v['tpr'] for v in classical.values()] + [quantum['tpr'], ae['tpr']]
            all_aucs_r  = [v['auc'] for v in classical.values()] + [quantum['auc'], ae['auc']]
            for nm, fpr, tpr, auc, col in zip(all_names_r, all_fprs, all_tprs, all_aucs_r, colors_r):
                lw = 3 if nm in ['XGBoost','Quantum-LSTM'] else 1.5
                ax.plot(fpr, tpr, color=col, lw=lw, label=f'{nm} (AUC={auc:.3f})')
            ax.plot([0,1],[0,1],'--', color='gray', lw=1)
            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
            ax.legend(fontsize=10); ax.set_title('ROC Curves — All 7 Models', fontsize=14)
            ax.grid(alpha=0.3)
            st.pyplot(fig); plt.close()

    with tab2:
        st.subheader("Confusion Matrices — Classical Models")
        try:
            st.image('plots/all_confusion_matrices.png', use_container_width=True)
        except:
            st.info("Run Notebook 2 to generate confusion matrices")

    with tab3:
        st.subheader("Precision-Recall Curves")
        st.caption("PR curves are more informative than ROC for imbalanced datasets like fraud detection")
        try:
            st.image('plots/precision_recall_curves.png', use_container_width=True)
        except:
            if X_test is not None:
                fig, ax = plt.subplots(figsize=(10, 7))
                colors_pr = ['#3498db','#2ecc71','#f39c12','#e74c3c','#9b59b6']
                for (name, model), color in zip(models.items(), colors_pr):
                    probs = model.predict_proba(X_test)[:, 1]
                    prec, rec, _ = precision_recall_curve(y_test, probs)
                    ap = average_precision_score(y_test, probs)
                    ax.plot(rec, prec, color=color, lw=2, label=f'{name} (AP={ap:.3f})')
                ax.axhline(y_test.mean(), color='gray', ls='--', lw=1,
                           label=f'Baseline (fraud rate={y_test.mean():.3f})')
                ax.set_xlabel('Recall', fontsize=13); ax.set_ylabel('Precision', fontsize=13)
                ax.set_title('Precision-Recall Curves', fontsize=14)
                ax.legend(loc='upper right', fontsize=10); ax.grid(alpha=0.3)
                st.pyplot(fig); plt.close()
            else:
                st.info("PR curves saved as plot — add X_test_sample.npy to models/ for live generation")

    with tab4:
        st.subheader("AUC Score Comparison")
        try:
            st.image('plots/auc_bar_chart.png', use_container_width=True)
        except:
            all_n = list(classical.keys()) + ['Quantum-LSTM','Autoencoder']
            all_a = [v['auc'] for v in classical.values()] + [quantum['auc'], ae['auc']]
            sorted_pairs = sorted(zip(all_a, all_n))
            a_s, n_s = zip(*sorted_pairs)
            colors_b = ['#f39c12' if n in ['XGBoost','Quantum-LSTM'] else '#3498db' for n in n_s]
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(n_s, a_s, color=colors_b, height=0.6)
            for bar, auc in zip(bars, a_s):
                ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                        f'{auc:.4f}', va='center', fontsize=11, fontweight='bold')
            ax.set_xlim(0.5, 1.05); ax.set_xlabel('ROC-AUC')
            ax.set_title('Model AUC Comparison — All 7 Models', fontsize=14)
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig); plt.close()

    # ── Model agreement matrix ──
    st.markdown("---")
    st.subheader("Model Agreement Heatmap")
    st.caption("How often do pairs of models agree on the same transaction?")
    try:
        st.image('plots/model_agreement.png', use_container_width=True)
    except:
        if X_test is not None:
            preds_d   = {name: model.predict(X_test[:5000]) for name, model in models.items()}
            model_list = list(preds_d.keys())
            agreement_matrix = np.array([[
                (preds_d[m1] == preds_d[m2]).mean() for m2 in model_list
            ] for m1 in model_list])
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(agreement_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                        vmin=0.7, vmax=1.0,
                        xticklabels=model_list, yticklabels=model_list,
                        linewidths=0.5, ax=ax, annot_kws={'size':11})
            ax.set_title('Model Agreement Rate', fontsize=13)
            plt.xticks(rotation=30, ha='right'); plt.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            st.info("Add X_test_sample.npy to models/ for live model agreement chart")

    # ── Training curves ──
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quantum-LSTM Training Curve")
        try:
            st.image('plots/quantum_training_curve.png', use_container_width=True)
        except:
            st.info("Run Notebook 3")
    with col2:
        st.subheader("Autoencoder Reconstruction Error")
        try:
            st.image('plots/autoencoder_error.png', use_container_width=True)
        except:
            st.info("Run Notebook 4")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════
elif page == "🧠 Explainability":
    st.title("🧠 Explainability — SHAP & LIME")
    st.markdown("Understanding **why** the model flags a transaction as fraud is as important as the prediction itself.")

    tab1, tab2, tab3, tab4 = st.tabs(["SHAP Summary","SHAP Waterfall","SHAP Dependence","LIME"])

    with tab1:
        st.subheader("Global Feature Importance — SHAP Beeswarm")
        st.caption("Each dot = one transaction. Color = feature value (red=high, blue=low). X-axis = impact on fraud prediction.")
        try:
            st.image('plots/shap_summary.png', use_container_width=True)
        except:
            st.info("Run Notebook 5 Cell 4")

        st.subheader("Mean Absolute SHAP Values (Feature Ranking)")
        try:
            st.image('plots/shap_bar.png', use_container_width=True)
        except:
            st.info("Run Notebook 5 Cell 5")

    with tab2:
        st.subheader("SHAP Waterfall — Single Fraud Transaction")
        st.caption("Shows exactly which features pushed this transaction toward fraud. Red = pushes toward fraud. Blue = pushes toward legit.")
        try:
            st.image('plots/shap_waterfall.png', use_container_width=True)
        except:
            st.info("Run Notebook 5 Cell 6")

    with tab3:
        st.subheader("SHAP Dependence Plot — Top Feature")
        st.caption("How the top feature's value affects the fraud prediction. Color shows interaction with a second feature.")
        try:
            st.image('plots/shap_dependence.png', use_container_width=True)
        except:
            st.info("Run Notebook 5 Cell 13")

    with tab4:
        st.subheader("LIME — Local Interpretable Model Explanation")
        st.caption("LIME explains one transaction by fitting a simple model locally around it. Bars show each feature's contribution.")
        try:
            st.image('plots/lime_explanation.png', use_container_width=True)
        except:
            st.info("Run Notebook 5 Cell 7")

    # ── SHAP vs LIME ──
    st.markdown("---")
    st.markdown('<div class="section-header">📖 SHAP vs LIME — What is the difference?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**SHAP (SHapley Additive exPlanations)**
- Based on game theory (Shapley values)
- Consistent, globally comparable values
- Works well for tree-based models (XGBoost, RF)
- Shows both global and local explanations
- Exact for tree models (TreeExplainer)
        """)
    with c2:
        st.markdown("""
**LIME (Local Interpretable Model-agnostic Explanations)**
- Fits a simple model around one prediction
- Model-agnostic (works with any model)
- Shows which features matter *for one transaction*
- Faster for single-instance explanations
- Less stable than SHAP across runs
        """)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "🔬 Live Prediction":
    st.title("🔬 Live Transaction Prediction")
    st.markdown("Upload a CSV of transactions → get fraud predictions from any model, with probability scores.")

    col_settings, col_upload = st.columns([1, 2])
    with col_settings:
        model_choice = st.selectbox("Choose model", list(models.keys()))
        threshold    = st.slider("Fraud threshold", 0.1, 0.9, 0.5, 0.05,
                                 help="Transactions with fraud probability above this are flagged")
    with col_upload:
        uploaded = st.file_uploader("Upload transaction CSV", type=['csv'])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.write("**Preview** (first 5 rows):")
        st.dataframe(df_raw.head(), use_container_width=True)

        # Align columns
        df_aligned = df_raw.copy()
        for col in features:
            if col not in df_aligned.columns:
                df_aligned[col] = 0
        df_aligned = df_aligned[features]
        X_scaled = scaler.transform(df_aligned)

        chosen_model = models[model_choice]
        probs = chosen_model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= threshold).astype(int)

        results_df = pd.DataFrame({
            'Transaction #':     range(1, len(preds)+1),
            'Fraud Probability': probs.round(4),
            'Risk %':            (probs * 100).round(1),
            'Prediction':        ['🚨 FRAUD' if p == 1 else '✅ Legit' for p in preds],
        })

        fraud_count = preds.sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Transactions", len(preds))
        m2.metric("Fraud Detected", int(fraud_count),
                  delta=f"{fraud_count/len(preds)*100:.1f}% fraud rate")
        m3.metric("Threshold Used", f"{threshold:.0%}")
        m4.metric("Model", model_choice)

        def color_rows(row):
            if '🚨' in str(row['Prediction']):
                return ['background-color: rgba(192,57,43,0.3)'] * len(row)
            return ['background-color: rgba(26,122,74,0.1)'] * len(row)

        st.dataframe(
            results_df.style.apply(color_rows, axis=1)
                            .background_gradient(subset=['Fraud Probability'], cmap='RdYlGn_r'),
            use_container_width=True, height=400
        )

        st.markdown("---")
        st.subheader("Fraud Probability Distribution")
        col_hist, col_gauge = st.columns([2, 1])

        with col_hist:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(probs[preds == 0], bins=30, alpha=0.7, color='#2ecc71', label='Predicted Legit')
            ax.hist(probs[preds == 1], bins=30, alpha=0.7, color='#e74c3c', label='Predicted Fraud')
            ax.axvline(threshold, color='black', ls='--', lw=1.5, label=f'Threshold={threshold:.2f}')
            ax.set_xlabel('Fraud Probability'); ax.set_ylabel('Count')
            ax.set_title('Distribution of Predicted Fraud Probabilities')
            ax.legend(); ax.grid(alpha=0.3)
            st.pyplot(fig); plt.close()

        with col_gauge:
            st.markdown("**Risk Level Breakdown**")
            low    = (probs < 0.3).sum()
            medium = ((probs >= 0.3) & (probs < 0.7)).sum()
            high   = (probs >= 0.7).sum()
            fig, ax = plt.subplots(figsize=(4, 4))
            wedge_sizes   = [low, medium, high]
            wedge_labels  = [f'Low\n({low})', f'Medium\n({medium})', f'High\n({high})']
            wedge_colors  = ['#2ecc71', '#f39c12', '#e74c3c']
            # Avoid pie error if all zeros
            if sum(wedge_sizes) > 0:
                ax.pie(wedge_sizes, labels=wedge_labels, colors=wedge_colors,
                       autopct='%1.1f%%', startangle=90,
                       wedgeprops={'edgecolor':'white','linewidth':1.5})
            ax.set_title('Risk Tier Breakdown')
            st.pyplot(fig); plt.close()

        st.download_button("⬇️ Download Results CSV",
                           results_df.to_csv(index=False).encode(),
                           file_name='fraud_predictions.csv',
                           mime='text/csv')
    else:
        st.info("👆 Upload a CSV file above to get started. The CSV should have the same feature columns as the training data.")
        with st.expander("📋 Expected feature columns"):
            st.write(features[:30])
            st.caption(f"... and {len(features)-30} more features")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "📋 Data Insights":
    st.title("📋 Data Insights")

    tab1, tab2 = st.tabs(["Feature Statistics","Fraud Patterns"])

    with tab1:
        st.subheader("Feature Importance — XGBoost Built-in")
        xgb_model = models['XGBoost']
        if hasattr(xgb_model, 'feature_importances_'):
            importances = xgb_model.feature_importances_
            top_n   = 20
            top_idx = np.argsort(importances)[-top_n:][::-1]
            top_feats = [features[i] for i in top_idx]
            top_vals  = importances[top_idx]
            fig, ax = plt.subplots(figsize=(10, 8))
            colors_fi = ['#f39c12' if v == max(top_vals) else '#3498db' for v in top_vals]
            ax.barh(top_feats[::-1], top_vals[::-1], color=colors_fi[::-1])
            ax.set_xlabel('Feature Importance (Gain)')
            ax.set_title(f'Top {top_n} Most Important Features — XGBoost', fontsize=13)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    with tab2:
        st.subheader("Fraud Score Distribution in Test Set")
        if X_test is not None:
            xgb_model   = models['XGBoost']
            fraud_probs = xgb_model.predict_proba(X_test)[:, 1]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].hist(fraud_probs[y_test == 0], bins=60, alpha=0.7,
                         color='#3498db', label=f'Legit (n={int((y_test==0).sum()):,})')
            axes[0].hist(fraud_probs[y_test == 1], bins=60, alpha=0.7,
                         color='#e74c3c', label=f'Fraud (n={int((y_test==1).sum()):,})')
            axes[0].set_xlabel('XGBoost Fraud Probability')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Fraud Probability: True Legit vs True Fraud')
            axes[0].legend(); axes[0].grid(alpha=0.3)

            from sklearn.calibration import calibration_curve
            prob_true, prob_pred = calibration_curve(y_test, fraud_probs, n_bins=10)
            axes[1].plot([0,1],[0,1],'k--', label='Perfect calibration')
            axes[1].plot(prob_pred, prob_true, 's-', color='#e74c3c', label='XGBoost')
            axes[1].set_xlabel('Mean Predicted Probability')
            axes[1].set_ylabel('Fraction of Positives')
            axes[1].set_title('Calibration Curve — XGBoost')
            axes[1].legend(); axes[1].grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            # Show pre-saved plots instead
            st.info("Showing pre-saved analysis plots from full dataset (118k transactions)")
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.image('plots/shap_summary.png', use_container_width=True,
                             caption='SHAP Feature Importance (full dataset)')
                except:
                    st.info("SHAP summary plot not found")
            with col2:
                try:
                    st.image('plots/shap_bar.png', use_container_width=True,
                             caption='Mean SHAP values (full dataset)')
                except:
                    st.info("SHAP bar plot not found")

        st.markdown("---")
        st.subheader("Autoencoder Reconstruction Error")
        try:
            st.image('plots/autoencoder_error.png', use_container_width=True)
        except:
            st.info("Run Notebook 4")