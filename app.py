import streamlit as st
import pandas as pd

st.set_page_config(page_title="Anomaly-Aware IDS Demo", layout="centered")

st.title("🛡️ Anomaly-Aware IDS with CNN–LSTM")
st.markdown("""
A hybrid Intrusion Detection System (IDS) combining **DBSCAN anomaly detection** with 
**CNN–LSTM deep learning**.  
Enhanced with **Explainable AI (SHAP + LIME)** for transparency in cybersecurity.
""")

# --- Section 1: Results Summary ---
st.header("📊 Prediction Results (Sample)")
sample_df = pd.read_csv("assets/sample_results.csv")
st.write(sample_df.head(10))

# --- Section 2: SHAP ---
st.header("🔍 Global Explainability (SHAP)")
st.image("assets/shap_summary.png", caption="Top Features (SHAP)")

# --- Section 3: LIME ---
st.header("🎯 Local Explainability (LIME)")
st.image("assets/LIME_Explanations/lime_instance_750.png", caption="LIME Explanation for Test Instance #750")

# --- Section 4: Metrics ---
st.header("📈 Model Performance")
st.markdown("""
- Accuracy: **99.89%**  
- F1-score: **0.99912**  
- ROC-AUC: **0.99994**  
- False Positive Rate (FPR): **0.00058**  
""")

st.success("✅ Preloaded demo shows real results without rerunning heavy training.")