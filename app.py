import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime.lime_tabular
import plotly.graph_objects as go
import streamlit.components.v1 as components
import warnings
from supabase import create_client, Client

warnings.filterwarnings('ignore')

# MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Smart CBC AI", page_icon="🩸", layout="wide")

# ==========================================
# CONNECT TO CLOUD DATABASE
# ==========================================
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_connection()

# ==========================================
# SECURE DATABASE LOGIN & REGISTRATION GATE
# ==========================================
def check_password():
    if st.session_state.get("password_correct", False):
        return True

    st.markdown("<h1 style='text-align: center;'>🏥 Vertec Labs - Clinical Portal</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Create Tabs for Login and Sign Up
        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Create Account"])
        
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Secure Login", use_container_width=True)

                if submit_button:
                    try:
                        # Make sure "User_Name" matches your Supabase table perfectly!
                        response = supabase.table("User").select("*").eq("User_Name", username).eq("Password", password).execute()
                        if len(response.data) > 0:
                            st.session_state["password_correct"] = True
                            st.rerun()
                        else:
                            st.error("🚨 Authentication failed. Invalid username or password.")
                    except Exception as e:
                        st.error(f"Database connection error: {e}")
                        
        with tab_register:
            with st.form("register_form"):
                new_username = st.text_input("Choose a Username")
                new_password = st.text_input("Choose a Password", type="password")
                register_button = st.form_submit_button("Register Account", use_container_width=True)
                
                if register_button:
                    try:
                        # 1. Check if username already exists
                        check_user = supabase.table("User").select("*").eq("User_Name", new_username).execute()
                        if len(check_user.data) > 0:
                            st.error("⚠️ Username already exists. Please choose another one.")
                        elif len(new_username) < 3 or len(new_password) < 5:
                            st.warning("⚠️ Username must be 3+ chars and password 5+ chars.")
                        else:
                            # 2. Insert new user into the database
                            supabase.table("User").insert({"User_Name": new_username, "Password": new_password}).execute()
                            st.success("✅ Account created successfully! You can now log in.")
                    except Exception as e:
                        st.error(f"Error creating account: {e}")

    return False

if not check_password():
    st.stop()

# ==========================================
# MAIN APPLICATION CODE
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load('meta_ensemble_model.joblib')
    scaler = joblib.load('standard_scaler.joblib')
    features = joblib.load('elite_features_list.joblib')
    return model, scaler, features

try:
    model, scaler, elite_features = load_assets()
except FileNotFoundError:
    st.error("🚨 Missing .joblib files. Please ensure model, scaler, and features files are in the folder.")
    st.stop()

# Disease Mapping
disease_map = {0: 'Anemia', 1: 'Dengue', 2: 'Healthy', 3: 'Infection', 4: 'Kidney Disease'}
inverse_disease_map = {v: k for k, v in disease_map.items()}

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🏥 Navigation")
app_mode = st.sidebar.radio("Select Diagnostic Mode:", ["👤 Single Patient XAI", "📁 Batch Processing (CSV)"])

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status:** Online 🟢")
st.sidebar.markdown("**Engine:** Calibrated Meta-Ensemble")

if app_mode == "👤 Single Patient XAI":
    st.title("🏥 Hybrid AI-Powered CBC Diagnostics")
    st.markdown("**(Calibrated Stacked Meta-Ensemble with Multi-XAI)**")

    st.subheader("1. Patient Demographics")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        age = st.number_input("Patient Age", min_value=1, max_value=120, value=35)
    with col_d2:
        gender = st.selectbox("Patient Gender", ["Male", "Female"])
        gender_encoded = 1 if gender == "Male" else 0

    st.subheader("2. Hematology Parameters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        wbc = st.number_input("WBC (10^3/uL)", min_value=0.0, max_value=50.0, value=7.5, step=0.1)
        rbc = st.number_input("RBC (10^6/uL)", min_value=0.0, max_value=10.0, value=4.8, step=0.1)
    with col2:
        hgb = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=13.0, step=0.1)
        hct = st.number_input("Hematocrit (%)", min_value=0.0, max_value=75.0, value=40.0, step=0.1)
    with col3:
        plt_count = st.number_input("Platelets (10^3/uL)", min_value=0.0, max_value=1000.0, value=250.0, step=1.0)
        mcv = st.number_input("MCV (fL)", min_value=0.0, max_value=150.0, value=90.0, step=0.1)
    with col4:
        mch = st.number_input("MCH (pg)", min_value=0.0, max_value=50.0, value=30.0, step=0.1)
        mchc = st.number_input("MCHC (g/dL)", min_value=0.0, max_value=50.0, value=33.0, step=0.1)

    st.markdown("---")
    consent_given = st.checkbox("I confirm this data is anonymized and consent to storing it for AI research and retraining.")

    if st.button("🔬 Diagnose Patient", type="primary", use_container_width=True):
        with st.spinner("Executing Mathematical Feature Engineering & AI Inference..."):
            
            pwr = plt_count / (wbc + 1e-5)
            hpr = hgb / (plt_count + 1e-5)
            anemia_idx = hgb * rbc

            input_data = {
                'Hemoglobin': hgb, 'MCV': mcv, 'MCH': mch, 'MCHC': mchc, 
                'WBC': wbc, 'RBC': rbc, 'Hematocrit': hct, 'Platelets': plt_count,
                'Age': age, 'PWR': pwr, 'HPR': hpr, 'Anemia_Index': anemia_idx, 
                'Gender_Encoded': gender_encoded
            }
            input_df = pd.DataFrame([input_data])[elite_features]
            
            scaled_data = scaler.transform(input_df)
            scaled_df = pd.DataFrame(scaled_data, columns=elite_features)

            raw_pred_idx = model.predict(scaled_df)[0]
            raw_ai_diagnosis = disease_map[raw_pred_idx]
            probabilities = model.predict_proba(scaled_df)[0]
            max_prob = np.max(probabilities) * 100

            final_diagnosis = raw_ai_diagnosis
            final_pred_idx = raw_pred_idx

            if wbc > 12.0:
                final_diagnosis = 'Infection'
                max_prob = 100.0
            elif plt_count < 100.0:
                final_diagnosis = 'Dengue'
                max_prob = 100.0
            elif final_diagnosis == 'Anemia':
                if (gender_encoded == 1 and hgb >= 13.5) or (gender_encoded == 0 and hgb >= 12.0):
                    final_diagnosis = 'Healthy'
                    max_prob = 100.0

            final_pred_idx = inverse_disease_map[final_diagnosis]

            # --- THE FIX: SECURE CLOUD DATABASE SAVING ---
            if consent_given:
                try:
                    db_record = {
                        "age": age, "gender": gender, 
                        "wbc": wbc, "rbc": rbc, "hgb": hgb, "hct": hct, 
                        "plt": plt_count, "mcv": mcv, "mch": mch, "mchc": mchc,
                        "ai_diagnosis": final_diagnosis
                    }
                    supabase.table("Patient_Records").insert(db_record).execute()
                    st.toast("✅ Anonymized record securely saved to Vertec Labs Cloud.")
                except Exception as e:
                    st.warning(f"⚠️ Diagnosis complete, but failed to save to cloud: {e}")
            elif not consent_given:
                st.toast("⚠️ Record not saved (Consent not provided).")

            # --- DASHBOARD DISPLAY ---
            st.markdown("---")
            if final_diagnosis == 'Healthy':
                st.success(f"### Final Diagnosis: {final_diagnosis}")
            else:
                st.error(f"### Final Diagnosis: {final_diagnosis}")

            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = max_prob,
                title = {'text': "System Confidence"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}],
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("### Explainable AI & Clinical Report")
            tab1, tab2, tab3 = st.tabs(["LIME (The Rules)", "SHAP (The Tug-of-War)", "📋 Standard Clinical Report"])

            def xai_predict_proba(raw_array):
                raw_df = pd.DataFrame(raw_array, columns=elite_features)
                scaled_array = scaler.transform(raw_df)
                probs = model.predict_proba(scaled_array)
                
                for i in range(len(raw_df)):
                    wbc_val = raw_df['WBC'].iloc[i]
                    plt_val = raw_df['Platelets'].iloc[i]
                    hgb_val = raw_df['Hemoglobin'].iloc[i]
                    gender_val = raw_df['Gender_Encoded'].iloc[i]
                    current_pred = np.argmax(probs[i])
                    
                    if wbc_val > 12.0: probs[i] = [0, 0, 0, 1.0, 0] 
                    elif plt_val < 100.0: probs[i] = [0, 1.0, 0, 0, 0] 
                    elif current_pred == 0: 
                        if (gender_val == 1 and hgb_val >= 13.5) or (gender_val == 0 and hgb_val >= 12.0):
                            probs[i] = [0, 0, 1.0, 0, 0] 
                return probs

            background_scaled = np.random.normal(loc=0, scale=1, size=(100, len(elite_features)))
            background_raw = scaler.inverse_transform(background_scaled)
            background_raw = np.clip(background_raw, a_min=0, a_max=None)

            with tab1:
                st.markdown("#### LIME: The Mathematical Rules")
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=background_raw, 
                    feature_names=elite_features,
                    class_names=[disease_map[i] for i in range(5)],
                    mode='classification'
                )
                lime_exp = lime_explainer.explain_instance(
                    input_df.iloc[0].values, xai_predict_proba, num_features=5, labels=(final_pred_idx,) 
                )
                raw_html = lime_exp.as_html(labels=[final_pred_idx]) 
                white_background_html = f"""<div style="background-color: white; padding: 20px; border-radius: 8px; color: black;">{raw_html}</div>"""
                components.html(white_background_html, height=450, scrolling=True)

            with tab2:
                st.markdown("#### SHAP: The 'Tug-of-War'")
                shap_explainer = shap.KernelExplainer(xai_predict_proba, background_raw[:10])
                shap_values_raw = shap_explainer.shap_values(input_df)
                
                if isinstance(shap_values_raw, list): shap_vals = shap_values_raw[final_pred_idx][0]
                else: shap_vals = shap_values_raw[0, :, final_pred_idx]
                    
                shap_df = pd.DataFrame({'Feature': elite_features, 'SHAP Value': shap_vals, 'Patient Value': input_df.iloc[0].values})
                shap_df['Abs Impact'] = shap_df['SHAP Value'].abs()
                shap_df = shap_df.sort_values(by='Abs Impact', ascending=True)
                colors = ['#ff4b4b' if val > 0 else '#1f77b4' for val in shap_df['SHAP Value']]
                
                fig_shap = go.Figure(go.Bar(
                    x=shap_df['SHAP Value'], y=shap_df['Feature'], orientation='h', marker_color=colors,
                    text=[f"Actual Value: {val:.1f}" for val in shap_df['Patient Value']], hoverinfo='text+x', textposition='inside'
                ))
                fig_shap.update_layout(title=f"What drove the System to predict {final_diagnosis}?", xaxis_title="Impact (Longer = More Important)", yaxis_title="Blood Marker", height=500, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_shap, use_container_width=True)

            with tab3:
                st.markdown("#### Complete Blood Count (CBC) - Patient Results")
                if gender_encoded == 1: 
                    hgb_range, hgb_min, hgb_max = "13.5 - 17.5", 13.5, 17.5
                    rbc_range, rbc_min, rbc_max = "4.3 - 5.7", 4.3, 5.7
                    hct_range, hct_min, hct_max = "38.0 - 50.0", 38.0, 50.0
                else: 
                    hgb_range, hgb_min, hgb_max = "12.0 - 15.5", 12.0, 15.5
                    rbc_range, rbc_min, rbc_max = "3.9 - 5.0", 3.9, 5.0
                    hct_range, hct_min, hct_max = "35.0 - 45.0", 35.0, 45.0
                
                report_data = [
                    {"Parameter": "Hemoglobin (g/dL)", "Result": hgb, "Reference Range": hgb_range, "Status": "🔽 Low" if hgb < hgb_min else "🔼 High" if hgb > hgb_max else "✅ Normal"},
                    {"Parameter": "WBC (10^3/uL)", "Result": wbc, "Reference Range": "4.5 - 11.0", "Status": "🔽 Low" if wbc < 4.5 else "🔼 High" if wbc > 11.0 else "✅ Normal"},
                    {"Parameter": "RBC (10^6/uL)", "Result": rbc, "Reference Range": rbc_range, "Status": "🔽 Low" if rbc < rbc_min else "🔼 High" if rbc > rbc_max else "✅ Normal"},
                    {"Parameter": "Platelets (10^3/uL)", "Result": plt_count, "Reference Range": "150 - 450", "Status": "🔽 Low" if plt_count < 150 else "🔼 High" if plt_count > 450 else "✅ Normal"},
                    {"Parameter": "Hematocrit (%)", "Result": hct, "Reference Range": hct_range, "Status": "🔽 Low" if hct < hct_min else "🔼 High" if hct > hct_max else "✅ Normal"},
                    {"Parameter": "MCV (fL)", "Result": mcv, "Reference Range": "80.0 - 100.0", "Status": "🔽 Low" if mcv < 80 else "🔼 High" if mcv > 100 else "✅ Normal"},
                    {"Parameter": "MCH (pg)", "Result": mch, "Reference Range": "27.0 - 33.0", "Status": "🔽 Low" if mch < 27 else "🔼 High" if mch > 33 else "✅ Normal"},
                    {"Parameter": "MCHC (g/dL)", "Result": mchc, "Reference Range": "32.0 - 36.0", "Status": "🔽 Low" if mchc < 32 else "🔼 High" if mchc > 36 else "✅ Normal"}
                ]
                st.table(pd.DataFrame(report_data).set_index("Parameter"))

elif app_mode == "📁 Batch Processing (CSV)":
    st.title("📁 Laboratory Batch Processing")
    st.markdown("Upload a CSV file containing multiple patient records. The AI will engineer the features, apply clinical overrides, and generate a mass diagnostic report.")
    st.info("💡 **CSV Format Required:** Your file must contain the following columns:\n`Age`, `Gender` (Male/Female), `Hemoglobin`, `WBC`, `RBC`, `Platelets`, `Hematocrit`, `MCV`, `MCH`, `MCHC`")

    uploaded_file = st.file_uploader("Upload Patient CBC Data (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(batch_df)} patient records.")
            
            with st.spinner("Running AI Engine and Clinical Overrides..."):
                process_df = batch_df.copy()
                
                if 'Gender' in process_df.columns:
                    process_df['Gender_Encoded'] = process_df['Gender'].apply(lambda x: 1 if str(x).strip().lower() == 'male' else 0)
                else:
                    process_df['Gender_Encoded'] = 0 
                
                process_df['PWR'] = process_df['Platelets'] / (process_df['WBC'] + 1e-5)
                process_df['HPR'] = process_df['Hemoglobin'] / (process_df['Platelets'] + 1e-5)
                process_df['Anemia_Index'] = process_df['Hemoglobin'] * process_df['RBC']

                ai_input_df = process_df[elite_features]
                scaled_batch = scaler.transform(ai_input_df)
                scaled_batch_df = pd.DataFrame(scaled_batch, columns=elite_features)

                raw_predictions = model.predict(scaled_batch_df)
                
                final_diagnoses = []
                override_flags = []
                final_pred_indices = []

                for i in range(len(process_df)):
                    raw_diagnosis = disease_map[raw_predictions[i]]
                    wbc_val = process_df['WBC'].iloc[i]
                    plt_val = process_df['Platelets'].iloc[i]
                    hgb_val = process_df['Hemoglobin'].iloc[i]
                    gender_val = process_df['Gender_Encoded'].iloc[i]
                    
                    final_diag = raw_diagnosis
                    flag = "None"

                    if wbc_val > 12.0:
                        final_diag = 'Infection'
                        flag = "⚠️ High WBC"
                    elif plt_val < 100.0:
                        final_diag = 'Dengue'
                        flag = "⚠️ Low Platelets"
                    elif final_diag == 'Anemia':
                        if (gender_val == 1 and hgb_val >= 13.5) or (gender_val == 0 and hgb_val >= 12.0):
                            final_diag = 'Healthy'
                            flag = "✅ Normal HGB"
                            
                    final_diagnoses.append(final_diag)
                    override_flags.append(flag)
                    final_pred_indices.append(inverse_disease_map[final_diag])

                batch_df['AI Final Diagnosis'] = final_diagnoses
                batch_df['Safety Override Trigger'] = override_flags
                process_df['Final_Pred_Idx'] = final_pred_indices 

            st.markdown("### 📊 Mass Diagnostic Results")
            st.dataframe(batch_df, use_container_width=True)

            col_d1, col_d2 = st.columns([1, 3])
            with col_d1:
                csv_export = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Diagnostic Report",
                    data=csv_export,
                    file_name='AI_Mass_Diagnostics_Report.csv',
                    mime='text/csv',
                    type='primary'
                )

            # ==========================================
            # BULK CLOUD SAVE 
            # ==========================================
            st.markdown("---")
            st.markdown("### 💾 Cloud Database Sync")
            batch_consent = st.checkbox("I confirm these records are anonymized and consent to storing them for AI research.", key="batch_consent")
            
            if st.button("☁️ Save Batch to Vertec Labs Cloud", type="primary"):
                if batch_consent:
                    with st.spinner("Uploading batch to database..."):
                        try:
                            # Package all rows into a single list
                            records_to_insert = []
                            for i in range(len(process_df)):
                                records_to_insert.append({
                                    "age": float(process_df['Age'].iloc[i]), 
                                    "gender": str(process_df['Gender'].iloc[i]) if 'Gender' in process_df.columns else "Unknown", 
                                    "wbc": float(process_df['WBC'].iloc[i]), 
                                    "rbc": float(process_df['RBC'].iloc[i]), 
                                    "hgb": float(process_df['Hemoglobin'].iloc[i]), 
                                    "hct": float(process_df['Hematocrit'].iloc[i]), 
                                    "plt": float(process_df['Platelets'].iloc[i]), 
                                    "mcv": float(process_df['MCV'].iloc[i]), 
                                    "mch": float(process_df['MCH'].iloc[i]), 
                                    "mchc": float(process_df['MCHC'].iloc[i]),
                                    "ai_diagnosis": str(batch_df['AI Final Diagnosis'].iloc[i])
                                })
                            
                            # Notice I capitalized "Patient_Records" here to match your other code perfectly!
                            supabase.table("Patient_Records").insert(records_to_insert).execute()
                            st.success(f"✅ Successfully synced {len(records_to_insert)} anonymized patient records to the cloud database!")
                        except Exception as e:
                            st.error(f"⚠️ Failed to save batch to cloud: {e}")
                else:
                    st.warning("⚠️ Please check the consent box to synchronize data.")

            # ==========================================
            # DEEP DIVE INSPECTOR
            # ==========================================
            st.markdown("---")
            st.markdown("### 🔍 Deep Dive: Patient Inspector")
            selected_row = st.selectbox(
                "Select Patient Record:",
                options=range(len(batch_df)),
                format_func=lambda x: f"Row {x+1}  |  Predicted: {batch_df['AI Final Diagnosis'].iloc[x]}  |  Age: {batch_df['Age'].iloc[x]}"
            )

            if st.button("Generate XAI For Selected Patient"):
                st.markdown(f"#### Analyzing Patient from Row {selected_row + 1}")
                
                pat_raw_data = process_df.iloc[selected_row]
                pat_input_df = pd.DataFrame([pat_raw_data])[elite_features]
                pat_final_diag = batch_df['AI Final Diagnosis'].iloc[selected_row]
                pat_final_idx = pat_raw_data['Final_Pred_Idx']
                
                wbc = pat_raw_data['WBC']
                rbc = pat_raw_data['RBC']
                hgb = pat_raw_data['Hemoglobin']
                hct = pat_raw_data['Hematocrit']
                plt_count = pat_raw_data['Platelets']
                mcv = pat_raw_data['MCV']
                mch = pat_raw_data['MCH']
                mchc = pat_raw_data['MCHC']
                gender_encoded = pat_raw_data['Gender_Encoded']

                def batch_xai_predict_proba(raw_array):
                    raw_df_xai = pd.DataFrame(raw_array, columns=elite_features)
                    scaled_array = scaler.transform(raw_df_xai)
                    probs = model.predict_proba(scaled_array)
                    
                    for i in range(len(raw_df_xai)):
                        w_val = raw_df_xai['WBC'].iloc[i]
                        p_val = raw_df_xai['Platelets'].iloc[i]
                        h_val = raw_df_xai['Hemoglobin'].iloc[i]
                        g_val = raw_df_xai['Gender_Encoded'].iloc[i]
                        c_pred = np.argmax(probs[i])
                        
                        if w_val > 12.0: probs[i] = [0, 0, 0, 1.0, 0] 
                        elif p_val < 100.0: probs[i] = [0, 1.0, 0, 0, 0] 
                        elif c_pred == 0: 
                            if (g_val == 1 and h_val >= 13.5) or (g_val == 0 and h_val >= 12.0):
                                probs[i] = [0, 0, 1.0, 0, 0] 
                    return probs

                background_scaled = np.random.normal(loc=0, scale=1, size=(100, len(elite_features)))
                background_raw = scaler.inverse_transform(background_scaled)
                background_raw = np.clip(background_raw, a_min=0, a_max=None)

                tab1, tab2, tab3 = st.tabs(["LIME (The Rules)", "SHAP (The Tug-of-War)", "📋 Standard Clinical Report"])
                
                with tab1:
                    st.markdown("#### LIME: The Mathematical Rules")
                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                        training_data=background_raw, feature_names=elite_features,
                        class_names=[disease_map[i] for i in range(5)], mode='classification'
                    )
                    lime_exp = lime_explainer.explain_instance(
                        pat_input_df.iloc[0].values, batch_xai_predict_proba, num_features=5, labels=(pat_final_idx,) 
                    )
                    raw_html = lime_exp.as_html(labels=[pat_final_idx]) 
                    white_background_html = f"""<div style="background-color: white; padding: 20px; border-radius: 8px; color: black;">{raw_html}</div>"""
                    components.html(white_background_html, height=450, scrolling=True)

                with tab2:
                    st.markdown("#### SHAP: The 'Tug-of-War'")
                    shap_explainer = shap.KernelExplainer(batch_xai_predict_proba, background_raw[:10])
                    shap_values_raw = shap_explainer.shap_values(pat_input_df)
                    
                    if isinstance(shap_values_raw, list): shap_vals = shap_values_raw[pat_final_idx][0]
                    else: shap_vals = shap_values_raw[0, :, pat_final_idx]
                        
                    shap_df = pd.DataFrame({'Feature': elite_features, 'SHAP Value': shap_vals, 'Patient Value': pat_input_df.iloc[0].values})
                    shap_df['Abs Impact'] = shap_df['SHAP Value'].abs()
                    shap_df = shap_df.sort_values(by='Abs Impact', ascending=True)
                    colors = ['#ff4b4b' if val > 0 else '#1f77b4' for val in shap_df['SHAP Value']]
                    
                    fig_shap = go.Figure(go.Bar(
                        x=shap_df['SHAP Value'], y=shap_df['Feature'], orientation='h', marker_color=colors,
                        text=[f"Actual Value: {val:.1f}" for val in shap_df['Patient Value']], hoverinfo='text+x', textposition='inside'
                    ))
                    fig_shap.update_layout(title=f"What drove the System to predict {pat_final_diag}?", xaxis_title="Impact", yaxis_title="Blood Marker", height=500, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_shap, use_container_width=True)

                with tab3:
                    st.markdown("#### Complete Blood Count (CBC) - Patient Results")
                    if gender_encoded == 1: 
                        hgb_range, hgb_min, hgb_max = "13.5 - 17.5", 13.5, 17.5
                        rbc_range, rbc_min, rbc_max = "4.3 - 5.7", 4.3, 5.7
                        hct_range, hct_min, hct_max = "38.0 - 50.0", 38.0, 50.0
                    else: 
                        hgb_range, hgb_min, hgb_max = "12.0 - 15.5", 12.0, 15.5
                        rbc_range, rbc_min, rbc_max = "3.9 - 5.0", 3.9, 5.0
                        hct_range, hct_min, hct_max = "35.0 - 45.0", 35.0, 45.0
                    
                    report_data = [
                        {"Parameter": "Hemoglobin (g/dL)", "Result": hgb, "Reference Range": hgb_range, "Status": "🔽 Low" if hgb < hgb_min else "🔼 High" if hgb > hgb_max else "✅ Normal"},
                        {"Parameter": "WBC (10^3/uL)", "Result": wbc, "Reference Range": "4.5 - 11.0", "Status": "🔽 Low" if wbc < 4.5 else "🔼 High" if wbc > 11.0 else "✅ Normal"},
                        {"Parameter": "RBC (10^6/uL)", "Result": rbc, "Reference Range": rbc_range, "Status": "🔽 Low" if rbc < rbc_min else "🔼 High" if rbc > rbc_max else "✅ Normal"},
                        {"Parameter": "Platelets (10^3/uL)", "Result": plt_count, "Reference Range": "150 - 450", "Status": "🔽 Low" if plt_count < 150 else "🔼 High" if plt_count > 450 else "✅ Normal"},
                        {"Parameter": "Hematocrit (%)", "Result": hct, "Reference Range": hct_range, "Status": "🔽 Low" if hct < hct_min else "🔼 High" if hct > hct_max else "✅ Normal"},
                        {"Parameter": "MCV (fL)", "Result": mcv, "Reference Range": "80.0 - 100.0", "Status": "🔽 Low" if mcv < 80 else "🔼 High" if mcv > 100 else "✅ Normal"},
                        {"Parameter": "MCH (pg)", "Result": mch, "Reference Range": "27.0 - 33.0", "Status": "🔽 Low" if mch < 27 else "🔼 High" if mch > 33 else "✅ Normal"},
                        {"Parameter": "MCHC (g/dL)", "Result": mchc, "Reference Range": "32.0 - 36.0", "Status": "🔽 Low" if mchc < 32 else "🔼 High" if mchc > 36 else "✅ Normal"}
                    ]
                    st.table(pd.DataFrame(report_data).set_index("Parameter"))

        except Exception as e:
            st.error(f"🚨 Error processing file: Make sure your CSV has the exact column names required. Details: {e}")