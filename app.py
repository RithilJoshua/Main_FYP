import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- PART 1: LOAD THE BACKEND ---
# Using the cache to load the model faster
@st.cache_resource
def load_models():
    # Load the files
    model = joblib.load('xgb_final_99acc.joblib')
    le = joblib.load('label_encoder.joblib')
    imputer = joblib.load('imputer.joblib')
    return model, le, imputer

try:
    model, le, imputer = load_models()
    st.toast("System Loaded")
except FileNotFoundError:
    st.error("Error.")
    st.stop()

# --- PART 2: FRINTEND ---
st.set_page_config(page_title="Smart CBC Diagnosis", page_icon="🩸")

st.title("🏥 AI-Powered Disease Diagnosis System")
st.markdown("Enter the patient's CBC (Complete Blood Count) values below.")

st.subheader("Patient Demographics")
col_dem1, col_dem2 = st.columns(2)
with col_dem1:
    age = st.number_input("Patient Age", min_value=1, max_value=120, value=35)
with col_dem2:
    st.write("") 
    gender = st.radio("Patient Gender", ["Male", "Female"], horizontal=True)

st.subheader("Hematology Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    hgb = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=15.0, step=0.1)
    rbc = st.number_input("RBC Count (m/mcL)", min_value=0.0, max_value=10.0, value=4.8, step=0.1)
    mcv = st.number_input("MCV (fL)", min_value=0.0, max_value=150.0, value=85.0, step=1.0)

with col2:
    wbc = st.number_input("WBC Count (cells/uL)", min_value=0.0, max_value=50000.0, value=7500.0, step=100.0)
    plt_count = st.number_input("Platelets (cells/uL)", min_value=0.0, max_value=1000000.0, value=250000.0, step=1000.0)
    mch = st.number_input("MCH (pg)", min_value=0.0, max_value=50.0, value=29.0, step=0.1)

with col3:
    hct = st.number_input("Hematocrit (%)", min_value=0.0, max_value=80.0, value=45.0, step=0.1)
    mchc = st.number_input("MCHC (g/dL)", min_value=0.0, max_value=50.0, value=33.0, step=0.1)

import plotly.graph_objects as go

# --- PART 3: MATCH ENGINE ---
def make_prediction(gender, hgb, wbc, rbc, hct, mcv, mch, mchc, plt_count):
    # 1. Feature Engineering 
    pwr = plt_count / (wbc + 1e-5)
    hpr = hgb / (plt_count + 1e-5)
    anemia_idx = hgb * rbc
    
    # SCALING THE VALS
    wbc_scaled = wbc if wbc < 100 else wbc / 1000.0 
    plt_scaled = plt_count if plt_count < 1000 else plt_count / 1000.0
    
    pwr = plt_scaled / (wbc_scaled + 1e-5)
    hpr = hgb / (plt_scaled + 1e-5)
    
    features = pd.DataFrame([[
        hgb, wbc_scaled, rbc, hct, mcv, mch, mchc, plt_scaled, 
        pwr, hpr, anemia_idx
    ]], columns=['Hemoglobin', 'WBC', 'RBC', 'Hematocrit', 'MCV', 'MCH', 'MCHC', 'Platelets', 'PWR', 'HPR', 'Anemia_Index'])

    # 3. AI Prediction & Confidence Score
    prediction_idx = model.predict(features)[0]
    initial_diagnosis = le.inverse_transform([prediction_idx])[0]
    
    # GET THE CONFIDENCE SCORE (Probability)
    probabilities = model.predict_proba(features)[0]
    confidence_score = max(probabilities) * 100 
    
    final_diagnosis = initial_diagnosis
    
    # Gender-Specific Anemia Override
    if initial_diagnosis == "Healthy":
        if (gender == "Male" and hgb < 13.5) or (gender == "Female" and hgb < 12.0):
            final_diagnosis = "Anemia"
            confidence_score = 100.0
    elif initial_diagnosis == "Anemia":
        if (gender == "Male" and hgb >= 13.5) or (gender == "Female" and hgb >= 12.0):
            final_diagnosis = "Healthy"
            confidence_score = 100.0
    
    # Infection Override
    if wbc_scaled > 12.0:
        final_diagnosis = "Infection"
        confidence_score = 100.0 # Rule-based overrides are 100% confident
        
    # Dengue Override 
    elif plt_scaled < 100 and initial_diagnosis == "Anemia":
        final_diagnosis = "Dengue"
        confidence_score = 100.0
        
    return final_diagnosis, initial_diagnosis, features, prediction_idx, confidence_score

# --- PART 4: RESULT DASHBOARD ---
if st.button("🩸 Diagnose Patient", type="primary", use_container_width=True):
    
    # Unpack all variables
    result, raw_ai, patient_features, pred_idx, confidence = make_prediction(gender, hgb, wbc, rbc, hct, mcv, mch, mchc, plt_count)
    
    # Create 3 Professional Tabs
    tab1, tab2, tab3 = st.tabs(["🩺 Clinical Verdict", "🕸️ CBC Radar Profile", "📊 AI Explainability (SHAP)"])
    
    # --- TAB 1: THE VERDICT & GAUGE CHART ---
    with tab1:
        colA, colB = st.columns([1, 1])
        
        with colA:
            st.markdown("<br>", unsafe_allow_html=True)
            if result == "Healthy":
                st.success(f"### Diagnosis: {result}")
                st.balloons()
            elif result == "Dengue":
                st.error(f"### ⚠️ Diagnosis: {result}")
                st.warning("Action: Check for fluid leakage. Monitor Platelets daily.")
            elif result == "Infection":
                st.error(f"### ⚠️ Diagnosis: {result}")
                st.warning("Action: Check temperature. Possible antibiotics required.")
            else:
                st.warning(f"### Diagnosis: {result}")
                
            if result != raw_ai:
                st.info("🚨 Clinical Safety Rules overrode the AI decision.")
                
        with colB:
            # GAUGE CHART FOR CONFIDENCE
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AI Confidence Level", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

    # --- TAB 2: CLINICAL BULLET CHARTS ---
    with tab2:
        st.markdown("### 📊 Key Biomarker Analysis")
        st.write("Patient values compared against standard physiological reference ranges.")
        
        fig_bullet = go.Figure()

        # 1. Hemoglobin Bullet (Normal: ~12-16)
        fig_bullet.add_trace(go.Indicator(
            mode = "number+gauge", value = hgb,
            domain = {'x': [0.2, 1], 'y': [0.7, 0.9]},
            title = {'text': "Hemoglobin<br>(g/dL)", 'font': {'size': 14}},
            gauge = {
                'shape': "bullet",
                'axis': {'range': [0, 20]},
                'steps': [
                    {'range': [0, 11], 'color': "#ffcccb"},   # Low (Red-ish)
                    {'range': [11, 16], 'color': "#d4edda"},  # Normal (Green-ish)
                    {'range': [16, 20], 'color': "#ffcccb"}], # High (Red-ish)
                'bar': {'color': "black", 'thickness': 0.4}
            }
        ))

        # 2. WBC Bullet (Normal: 4000 - 11000)
        fig_bullet.add_trace(go.Indicator(
            mode = "number+gauge", value = wbc,
            domain = {'x': [0.2, 1], 'y': [0.35, 0.55]},
            title = {'text': "WBC<br>(cells/uL)", 'font': {'size': 14}},
            gauge = {
                'shape': "bullet",
                'axis': {'range': [0, 20000]},
                'steps': [
                    {'range': [0, 4000], 'color': "#ffcccb"},    # Low
                    {'range': [4000, 11000], 'color': "#d4edda"},# Normal
                    {'range': [11000, 20000], 'color': "#ffcccb"}], # High (Infection zone)
                'bar': {'color': "black", 'thickness': 0.4}
            }
        ))

        # 3. Platelets Bullet (Normal: 150,000 - 450,000)
        fig_bullet.add_trace(go.Indicator(
            mode = "number+gauge", value = plt_count,
            domain = {'x': [0.2, 1], 'y': [0.0, 0.2]},
            title = {'text': "Platelets<br>(cells/uL)", 'font': {'size': 14}},
            gauge = {
                'shape': "bullet",
                'axis': {'range': [0, 600000]},
                'steps': [
                    {'range': [0, 150000], 'color': "#ffcccb"},     # Low (Dengue zone)
                    {'range': [150000, 450000], 'color': "#d4edda"},# Normal
                    {'range': [450000, 600000], 'color': "#ffcccb"}],# High
                'bar': {'color': "black", 'thickness': 0.4}
            }
        ))

        fig_bullet.update_layout(height=400, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig_bullet, use_container_width=True)

    # --- TAB 3: THE SHAP EXPLAINER ---
    with tab3:
        st.markdown(f"### Why did the AI predict **{raw_ai}**?")
        if result != raw_ai:
            st.warning(f"Note: This chart explains the raw AI prediction ({raw_ai}). The final diagnosis of {result} was enforced by the deterministic safety layer.")

        with st.spinner("Generating Biomarker Analysis..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(patient_features)
            
            fig_shap, ax = plt.subplots(figsize=(8, 5))
            patient_shap = shap_values[0, :, pred_idx]
            shap.plots.waterfall(patient_shap, show=False)
            
            st.pyplot(fig_shap)
            plt.clf()
