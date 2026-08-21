import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import tabpfn_client  # Ensure this package is explicitly imported

# Bypass terminal configurations entirely by explicitly injecting the token
if "TABPFN_TOKEN" in st.secrets:
    try:
        # Use the built-in direct setter method to completely skip 
        # file-system configuration folder creation hooks.
        tabpfn_client.set_access_token(st.secrets["TABPFN_TOKEN"])
        
        # Enforce an explicit connection state in global memory
        os.environ["TABPFN_TOKEN"] = st.secrets["TABPFN_TOKEN"]
    except Exception as auth_err:
        st.error(f"Authentication token registration failure: {auth_err}")
else:
    st.error("❌ TABPFN_TOKEN missing from Streamlit App Secrets Manager Configuration Window.")

class ThresholdOptimizedTabPFN(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, threshold=0.5):
        self.base_model = base_model
        self.threshold = threshold
        self.classes_ = np.array([0, 1])
    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)
    def predict(self, X):
        probas = self.predict_proba(X)[:, 1]
        return (probas >= self.threshold).astype(int)

st.set_page_config(page_title="EHR AKI Risk Evaluator", layout="wide")
st.title("🫘Predict AKI Web Application")
st.write("Clinical decision support pipeline leveraging an optimized TabPFN classification engine.")
st.write("This tool is designed for general diagnostics. It is up to the clinicians' judgement to determine if AKI is present.")


@st.cache_resource
def load_model():
    return joblib.load("optimized_tabpfn_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model file. Error: {e}")
    st.stop()

# ---------------------------------------------------------------------
# VIEW PATH A: BULK DATA UPLOAD & BATCH PROCESSING (Main Page Area)
# ---------------------------------------------------------------------
st.header("📋 Option A: Batch Patient Screening")
uploaded_file = st.file_uploader("Upload a patient dataset CSV file matching your 33-column EHR schema:", type=["csv"])

if uploaded_file is not None:
    try:
        bulk_df = pd.read_csv(uploaded_file)
        
        # Verify required columns exist (ignoring target 'AKI' if absent)
        expected_features = [
            "Age", "Patient Ethnicity Code", "Patient Gender", "Weight", "Height (m)", "BMI",
            "Procedure_code", "Contrast type_code", "Volume Vial", "Patient Type 1inpt 2OutptDaysx 3Emer",
            "Elevated Cr or CKD before scan 0N1Y", "DM code, no DM-0, DM-1", "HTN code, no HTN-0, HTN-1",
            "Cardiovascular disease (PVD or IHD or CVA): Absent-0, Present-1", "History of Ca",
            "Procal_code (Grp 1-4: <0.5, 0.5-<2, 2-<10, >=10, nd9)", "HCT low=2, normal=1, high= 3, nd-9 (normal 38-52%)",
            "Hypoalbuminemia No-0, Yes-1, nd-9", "Glucose_code: <4_2, 4-11_1, >11_3, nd9", "High Lactate- No: 0, Yes:1",
            "High Uric Acid_1", "High Thyroxine >14.4_1", "Acidbase pH (<7.35=1, Normal2, >7.45=3, nd9)",
            "Hb notdone9, 11.5-17=1, <8=2, 8-<11.5=3, >17=4)", "WBC notdone9, 4-10=1, <4.0=2, >10=3",
            "ProBNPmt300_1, nd9", "Proteinuria <=0.5=1, >0.5=2, nd9", "CRP <=10=1, >10=2, nd9",
            "ACEi/ARB", "Diuretics", "NSAIDS", "SGLT2i", "Calculated baseline creatinine"
        ]
        
        # Strip potential target column if present in the upload
        eval_df = bulk_df[expected_features].copy()
        
        # Generate model calculations
        probabilities = model.predict_proba(eval_df)[:, 1]
        alerts = model.predict(eval_df)
        
        # Append outputs to visualization dataframe
        bulk_df["Calculated Risk Probability"] = probabilities.round(4)
        bulk_df["Clinical Alert Triggered"] = ["⚠️ HIGH RISK" if a == 1 else "✅ Low/Normal" for a in alerts]
        
        st.success("Dataset successfully screened!")
        st.dataframe(bulk_df[["Calculated Risk Probability", "Clinical Alert Triggered"] + expected_features[:5]])
        
        # Export Option
        csv_output = bulk_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Annotated Risk Report CSV",
            data=csv_output,
            file_name="aki_risk_predictions_report.csv",
            mime="text/csv"
        )
    except Exception as err:
        st.error(f"Error parsing data file: {err}. Verify all 33 column headers exactly match your specification.")

# ---------------------------------------------------------------------
# VIEW PATH B: BEDSIDE TESTING SYSTEM (Sidebar Entry Form)
# ---------------------------------------------------------------------
st.sidebar.header("🔬 Option B: Bedside Patient Input")
with st.sidebar.form("individual_patient_form"):
    
    # Tier 1: Demographics and Physical Indexes
    st.markdown("**Demographics & Vitals**")
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=65)
    ethnicity = st.number_input("Patient Ethnicity Code", min_value=0, max_value=20, value=1)
    gender = st.selectbox("Patient Gender", options=[1, 2], format_func=lambda x: "Male (1)" if x == 1 else "Female (2)")
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=75.0)
    height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.75)
    bmi = st.number_input("BMI Index", min_value=10.0, max_value=60.0, value=24.5)
    
    # Tier 2: Contrast/Procedure Metrics
    st.markdown("--- \n**Procedural Exposure Metrics**")
    p_code = st.number_input("Procedure Code ID", value=101)
    c_code = st.number_input("Contrast Type Code ID", value=1)
    v_vial = st.number_input("Volume Vial Administered", value=100)
    p_type = st.selectbox("Patient Admissions Type", options=[1, 2, 3], format_func=lambda x: {1:"Inpatient (1)", 2:"Outpatient (2)", 3:"Emergency (3)"}[x])
    
    # Tier 3: Pre-existing Pathologies & Target History Indicators
    st.markdown("--- \n**Pre-existing Comorbidities**")
    ckd = st.selectbox("Elevated Cr or CKD History", options=[0, 1])
    dm = st.selectbox("Diabetes Mellitus (DM) Status", options=[0, 1])
    htn = st.selectbox("Hypertension (HTN) Status", options=[0, 1])
    cvd = st.selectbox("Cardiovascular Disease (PVD/IHD/CVA)", options=[0, 1])
    ca_hist = st.selectbox("History of Malignancy (Ca)", options=[0, 1])
    
    # Tier 4: Stratified Biomarkers and Lab Panels
    st.markdown("--- \n**EHR Sourced Laboratory Panels**")
    procal = st.selectbox("Procalcitonin Bin Class (1-4, nd9)", options=[1, 2, 3, 4, 9])
    hct = st.selectbox("Hematocrit Tier (Normal=1, Low=2, High=3, nd9)", options=[1, 2, 3, 9])
    albumin = st.selectbox("Hypoalbuminemia Status (No=0, Yes=1, nd9)", options=[0, 1, 9])
    glucose = st.selectbox("Glucose Categorization Range (<4=2, 4-11=1, >11=3, nd9)", options=[1, 2, 3, 9])
    lactate = st.selectbox("High Lactate Serum (No=0, Yes=1)", options=[0, 1])
    uric = st.selectbox("High Uric Acid Serum Indicator", options=[0, 1])
    thyroxine = st.selectbox("High Thyroxine >14.4 Indicator", options=[0, 1])
    ph_tier = st.selectbox("Acid-Base pH Status (<7.35=1, Normal=2, >7.45=3, nd9)", options=[2, 1, 3, 9])
    hb_tier = st.selectbox("Hemoglobin Bin (11.5-17=1, <8=2, 8-11.5=3, >17=4, nd9)", options=[1, 2, 3, 4, 9])
    wbc_tier = st.selectbox("WBC Code Range (4-10=1, <4=2, >10=3, nd9)", options=[1, 2, 3, 9])
    probnp = st.selectbox("ProBNP > 300 Alert Flag (nd9)", options=[0, 1, 9])
    proteinuria = st.selectbox("Proteinuria Status (<=0.5=1, >0.5=2, nd9)", options=[1, 2, 9])
    crp_tier = st.selectbox("CRP Threshold Tier (<=10=1, >10=2, nd9)", options=[1, 2, 9])
    base_cr = st.number_input("Calculated Baseline Creatinine Value", value=0.90)

    # Tier 5: Current Nephrotoxic/High Risk Pharmacotherapy Exposures
    st.markdown("--- \n**Active Medication Regimen Profile**")
    ace_arb = st.selectbox("ACEi/ARB Exposure Status", options=[0, 1])
    diuretics = st.selectbox("Diuretics Exposure Status", options=[0, 1])
    nsaids = st.selectbox("NSAIDS Exposure Status", options=[0, 1])
    sglt2i = st.selectbox("SGLT2i Exposure Status", options=[0, 1])
    
    submit_single = st.form_submit_button("Run Bedside Evaluation")

# 6. Single Patient Calculation Logic Execution
if submit_single:
    # Build single vector following exact feature matrix sequence
    single_patient_data = pd.DataFrame([{
        "Age": age, "Patient Ethnicity Code": ethnicity, "Patient Gender": gender, "Weight": weight, "Height (m)": height, "BMI": bmi,
        "Procedure_code": p_code, "Contrast type_code": c_code, "Volume Vial": v_vial, "Patient Type 1inpt 2OutptDaysx 3Emer": p_type,
        "Elevated Cr or CKD before scan 0N1Y": ckd, "DM code, no DM-0, DM-1": dm, "HTN code, no HTN-0, HTN-1": htn,
        "Cardiovascular disease (PVD or IHD or CVA): Absent-0, Present-1": cvd, "History of Ca": ca_hist,
        "Procal_code (Grp 1-4: <0.5, 0.5-<2, 2-<10, >=10, nd9)": procal, "HCT low=2, normal=1, high= 3, nd-9 (normal 38-52%)": hct,
        "Hypoalbuminemia No-0, Yes-1, nd-9": albumin, "Glucose_code: <4_2, 4-11_1, >11_3, nd9": glucose, "High Lactate- No: 0, Yes:1": lactate,
        "High Uric Acid_1": uric, "High Thyroxine >14.4_1": thyroxine, "Acidbase pH (<7.35=1, Normal2, >7.45=3, nd9)": ph_tier,
        "Hb notdone9, 11.5-17=1, <8=2, 8-<11.5=3, >17=4)": hb_tier, "WBC notdone9, 4-10=1, <4.0=2, >10=3": wbc_tier,
        "ProBNPmt300_1, nd9": probnp, "Proteinuria <=0.5=1, >0.5=2, nd9": proteinuria, "CRP <=10=1, >10=2, nd9": crp_tier,
        "ACEi/ARB": ace_arb, "Diuretics": diuretics, "NSAIDS": nsaids, "SGLT2i": sglt2i, "Calculated baseline creatinine": base_cr
    }])
    
    # Calculate probabilities via loaded pkl artifact
    single_prob = model.predict_proba(single_patient_data)[0, 1]
    single_alert = model.predict(single_patient_data)[0]
    
    st.markdown("---")
    st.subheader("🔬 Single Patient Risk Evaluation Metrics")
    
    c1, c2 = st.columns(2)
    c1.metric(label="Calculated Post-Contrast AKI Risk Probability", value=f"{single_prob:.2%}")
    
    if single_alert == 1:
        c2.error("🚨 CRITICAL WARNING: PATIENT CLASSIFIED AS HIGH RISK")
        st.warning("**Recommended Clinical Precautions:** Ensure robust peri-procedural hydration, minimize contrast volume delivery where feasible, and schedule automated serum creatinine labs 48 hours post-exposure.")
    else:
        c2.success("✅ Patient Classified as Low/Normal Risk")
        st.info("**Recommended Clinical Precautions:** Standard post-procedural monitoring and follow-up labs as per institutional protocol.")
