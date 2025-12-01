import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import random
import time
from streamlit_option_menu import option_menu
from datetime import datetime
from fpdf import FPDF

# ---------- CONFIG (FIRST STREAMLIT CALL) ----------
st.set_page_config(page_title="Smart Health Monitor", layout="wide")

# ---------- GLOBAL UI THEME ----------
def apply_custom_css():
    custom_css = """
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #111827, #1f2937);
        color: #f9fafb;
    }
    /* Main content width */
    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1f2933;
    }
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #ef4444;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.35rem 1.1rem;
        font-weight: 600;
    }
    div.stButton > button:first-child:hover {
        background-color: #f97373;
    }
    /* Inputs */
    textarea, input, select {
        border-radius: 6px !important;
    }
    /* Subheaders */
    h2, h3, h4 {
        color: #e5e7eb;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

apply_custom_css()

# ---------- NLP SYMPTOM KEYWORDS (ENGLISH + HINGLISH) ----------
SYMPTOM_KEYWORDS = {
    # HEART
    "chest pain": ["heart"],
    "pressure in chest": ["heart"],
    "shortness of breath": ["heart"],
    "breathless": ["heart"],
    "breathlessness": ["heart"],
    "palpitation": ["heart"],
    "dil me dard": ["heart"],
    "seene me dard": ["heart"],
    "saans phool": ["heart"],
    "saans phool rahi": ["heart"],

    # FATIGUE / THAKAVAT
    "tired": ["diabetes", "kidney", "heart"],
    "fatigue": ["diabetes", "kidney", "heart"],
    "weakness": ["diabetes", "kidney"],
    "thakavat": ["diabetes", "kidney", "heart"],
    "thakawat": ["diabetes", "kidney", "heart"],
    "bahut thak": ["diabetes", "kidney", "heart"],

    # DIABETES – URINATION / PYAAS
    "frequent urination": ["diabetes", "kidney"],
    "pee a lot": ["diabetes"],
    "urinate a lot": ["diabetes"],
    "bar bar bathroom": ["diabetes", "kidney"],
    "bar bar susu": ["diabetes", "kidney"],

    "thirsty": ["diabetes"],
    "increased thirst": ["diabetes"],
    "dry mouth": ["diabetes"],
    "pyaas lag": ["diabetes"],
    "bahut pyaas": ["diabetes"],
    "zyada pyaas": ["diabetes"],

    # KIDNEY – SWELLING / ITCH
    "swelling in legs": ["kidney", "heart"],
    "ankle swelling": ["kidney", "heart"],
    "puffy eyes": ["kidney"],
    "pair sujan": ["kidney", "heart"],
    "pair me sujan": ["kidney", "heart"],
    "legs me sujan": ["kidney", "heart"],

    "itchy skin": ["kidney", "liver"],
    "khujli": ["kidney", "liver"],

    # LIVER – JAUNDICE / NAUSEA
    "yellow skin": ["liver"],
    "yellow eyes": ["liver"],
    "jaundice": ["liver"],
    "pili aankh": ["liver"],
    "pili skin": ["liver"],

    "abdominal pain": ["liver"],
    "stomach pain": ["liver"],
    "pet dard": ["liver"],
    "nausea": ["liver", "kidney"],
    "vomiting": ["liver", "kidney"],
    "ulti": ["liver", "kidney"]
}

# ---------- PATHS ----------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
HISTORY_CSV = os.path.join(PROJECT_ROOT, "history.csv")

# ---------- HELPERS ----------
def load_model_and_scaler(name):
    mpath = os.path.join(MODELS_DIR, f"{name}_model.pkl")
    spath = os.path.join(MODELS_DIR, f"{name}_scaler.pkl")
    try:
        if os.path.exists(mpath) and os.path.exists(spath):
            model = joblib.load(mpath)
            scaler = joblib.load(spath)
            return model, scaler
    except Exception as e:
        st.error(f"Error loading {name} model/scaler: {e}")
    return None, None

def save_history(row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(HISTORY_CSV):
        df.to_csv(HISTORY_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(HISTORY_CSV, index=False)

def create_pdf_report(patient_name, details, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Health Report", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Name: {patient_name}", ln=True)
    pdf.ln(4)
    for k, v in details.items():
        safe_text = str(f"{k}: {v}")[:200]
        pdf.multi_cell(150, 8, safe_text)
        pdf.ln(2)
    pdf.output(filename)
    return filename

def compute_health_insights(age, gender, bmi, systolic, diastolic, heart_rate):
    insights = []

    if bmi < 18.5:
        bmi_cat = "Underweight"
        insights.append("Weight is below normal range. Consider a nutrition-rich diet.")
    elif bmi < 25:
        bmi_cat = "Normal"
        insights.append("Weight is in normal range. Maintain a balanced lifestyle.")
    elif bmi < 30:
        bmi_cat = "Overweight"
        insights.append("Weight is above normal. Regular exercise and diet control are recommended.")
    else:
        bmi_cat = "Obese"
        insights.append("High BMI. Risk for heart disease and diabetes increases. Consider medical advice.")

    if systolic < 90 or diastolic < 60:
        bp_cat = "Low"
        insights.append("Blood pressure is low. Dizziness or fatigue may occur.")
    elif systolic < 120 and diastolic < 80:
        bp_cat = "Normal"
        insights.append("Blood pressure is in the healthy range.")
    elif systolic < 140 or diastolic < 90:
        bp_cat = "Pre-hypertensive"
        insights.append("Blood pressure is slightly high. Monitor regularly and reduce salt intake.")
    else:
        bp_cat = "High"
        insights.append("Blood pressure is high. Strongly recommended to consult a doctor.")

    if heart_rate < 60:
        hr_cat = "Low"
        insights.append("Heart rate is low (bradycardia). If not an athlete, consider medical checkup.")
    elif heart_rate <= 100:
        hr_cat = "Normal"
        insights.append("Heart rate is in normal resting range.")
    else:
        hr_cat = "High"
        insights.append("Heart rate is high (tachycardia). Could be due to stress, anxiety, or other issues.")

    risk_score = 0
    if bmi_cat in ["Overweight", "Obese"]:
        risk_score += 25
    if bp_cat in ["Pre-hypertensive", "High"]:
        risk_score += 35
    if hr_cat in ["Low", "High"]:
        risk_score += 20
    if age >= 45:
        risk_score += 20

    risk_score = min(100, risk_score)

    return {
        "bmi_category": bmi_cat,
        "bp_category": bp_cat,
        "hr_category": hr_cat,
        "risk_score": risk_score,
        "recommendations": insights
    }

def disease_detector():
    st.markdown("### Symptom-based Disease Detector (Rule-based)")
    symptoms = {
        "Chest pain": ["Heart"],
        "Shortness of breath": ["Heart"],
        "Fatigue": ["Heart", "Diabetes", "Kidney"],
        "Frequent urination": ["Diabetes"],
        "Increased thirst": ["Diabetes"],
        "Swelling in legs": ["Kidney", "Heart"],
        "Itchy skin": ["Kidney"],
        "Yellow skin": ["Liver"],
        "Abdominal pain": ["Liver"],
        "Nausea": ["Liver"],
        "High BP": ["Heart", "Kidney"],
        "High cholesterol": ["Heart"]
    }
    selected_symptoms = st.multiselect("Select your symptoms:", list(symptoms.keys()))
    if st.button("Analyze Symptoms"):
        if not selected_symptoms:
            st.info("No symptoms selected.")
            return
        disease_scores = {"Heart": 0, "Diabetes": 0, "Kidney": 0, "Liver": 0}
        for s in selected_symptoms:
            for d in symptoms[s]:
                disease_scores[d] += 1
        st.markdown("#### Symptom-based risk")
        for d, s in sorted(disease_scores.items(), key=lambda x: x[1], reverse=True):
            if s > 0:
                st.write(f"- {d}: score {s}")
        high_risk = [d for d, s in disease_scores.items() if s >= 2]
        if high_risk:
            st.warning("High risk indications for: " + ", ".join(high_risk))
        else:
            st.success("No strong indication from selected symptoms.")
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disease": "symptom_detector",
            "input": str(selected_symptoms),
            "prediction": 1 if high_risk else 0,
            "probability": np.mean(list(disease_scores.values()))
        }
        save_history(row)

# ---------- SIDEBAR ----------
with st.sidebar:
    choice = option_menu(
        "Smart Health System",
        ["Home", "Monitoring", "Symptom Description Analyzer", "Predictor", "History", "About"],
        icons=["house", "activity", "chat-dots", "heart", "clock-history", "info-circle"],
        default_index=0
    )

# ---------- HOME ----------
if choice == "Home":
    st.title("Smart Health Monitoring & Early Disease Prediction System")
    st.markdown(
        "<h4 style='color:#d1d5db;'>AI/ML-powered virtual health assistant with monitoring, NLP and multi-disease prediction.</h4>",
        unsafe_allow_html=True
    )
    st.markdown("""
This system combines:
- AI Health Insights (rule-based scoring from vitals)
- Symptom Description Analyzer (NLP on English + Hinglish text)
- ML Disease Prediction for Heart, Diabetes, Kidney, Liver
- PDF report generation and prediction history logging

It is an academic AI/ML project and not a medical diagnosis tool.
""")

    if os.path.exists(HISTORY_CSV):
        df_home = pd.read_csv(HISTORY_CSV)
        if not df_home.empty and "disease" in df_home.columns:
            st.subheader("Overall predictions by disease")
            st.bar_chart(df_home["disease"].value_counts())

# ---------- MONITORING / AI INSIGHTS ----------
elif choice == "Monitoring":
    st.title("AI Health Insights & Recommendations")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 10, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height_cm = st.number_input("Height (cm)", 120, 220, 170)
        weight_kg = st.number_input("Weight (kg)", 30, 200, 65)
    with col2:
        systolic = st.number_input("Systolic BP (upper, mm Hg)", 80, 200, 120)
        diastolic = st.number_input("Diastolic BP (lower, mm Hg)", 50, 130, 80)
        heart_rate = st.number_input("Resting Heart Rate (bpm)", 40, 200, 75)

    if st.button("Analyze Health Insights"):
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2) if height_m > 0 else 0.0
        st.markdown("### Calculated Parameter")
        st.write(f"BMI: {bmi:.2f}")
        result = compute_health_insights(age, gender, bmi, systolic, diastolic, heart_rate)
        c3, c4 = st.columns(2)
        with c3:
            st.metric("BMI Category", result["bmi_category"])
            st.metric("BP Category", result["bp_category"])
        with c4:
            st.metric("Heart Rate Category", result["hr_category"])
            st.metric("Overall Risk Score (0–100)", result["risk_score"])
        st.markdown("### Recommendations")
        for rec in result["recommendations"]:
            st.write("- " + rec)
        st.info("Note: This rule-based engine is for awareness only, not for clinical decisions.")

# ---------- SYMPTOM DESCRIPTION ANALYZER ----------
elif choice == "Symptom Description Analyzer":
    st.title("Symptom Description Analyzer")
    st.markdown("#### Enter your symptom description below")
    st.markdown(
        "Type your symptoms in natural language (English or simple Hinglish). "
        "The system uses simple keyword-based NLP to suggest which disease modules "
        "you should check first (Heart, Diabetes, Kidney, Liver)."
    )
    user_text = st.text_area(
        "Describe your symptoms (English / simple Hinglish):",
        placeholder="Example: I feel very tired and thirsty."
    )
    if st.button("Analyze Symptoms (NLP)"):
        if not user_text.strip():
            st.warning("Please type something about your symptoms.")
        else:
            text = user_text.lower()
            scores = {"heart": 0, "diabetes": 0, "kidney": 0, "liver": 0}
            for phrase, diseases in SYMPTOM_KEYWORDS.items():
                if phrase in text:
                    for d in diseases:
                        scores[d] += 1
            st.markdown("### NLP-based suggestions")
            if all(v == 0 for v in scores.values()):
                st.info("No strong keyword matches found. You can still use the ML Predictor tab directly.")
            else:
                st.write("Scores (higher means more matching symptoms):")
                for d, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    if s > 0:
                        st.write(f"- {d.capitalize()}: {s}")
                best_disease = max(scores, key=scores.get)
                if scores[best_disease] > 0:
                    st.success(
                        f"Based on your text, **{best_disease.capitalize()}** module is recommended first. "
                        "Go to the Predictor tab and run that disease predictor."
                    )
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "disease": "nlp_symptom_analyzer",
                "input": user_text,
                "prediction": 0,
                "probability": max(scores.values()) if scores else "NA"
            }
            save_history(row)

# ---------- PREDICTOR ----------
elif choice == "Predictor":
    st.title("Disease Prediction Suite")
    st.markdown(
        "<p style='color:#d1d5db;'>First analyze symptoms, then run ML models to estimate disease risk.</p>",
        unsafe_allow_html=True
    )
    st.markdown("#### Step 1: Symptom-based check")
    disease_detector()
    st.markdown("---")
    st.markdown("#### Step 2: ML model prediction")
    st.markdown("### ML Disease Predictor")

    disease = st.selectbox("Choose disease", ["Heart", "Diabetes", "Kidney", "Liver"])
    model, scaler = load_model_and_scaler(disease.lower())

    if model is None:
        st.warning(
            f"Model for {disease} not found in `{MODELS_DIR}`. "
            "Train or place the model and scaler files first."
        )
    else:
        st.success(f"{disease} model loaded successfully.")
        st.write("Enter patient details below:")

        X = None
        if disease == "Heart":
            age = st.number_input("Age", 20, 100, 50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
            chol = st.number_input("Cholesterol", 100, 500, 200)
            thalach = st.number_input("Max heart rate achieved", 50, 220, 150)
            exang = st.selectbox("Exercise induced angina", [0, 1])
            X = np.array([[age, 1 if sex == "Male" else 0, trestbps, chol, thalach, exang]])
        elif disease == "Diabetes":
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose", 0, 300, 120)
            bp = st.number_input("Blood Pressure", 0, 200, 70)
            skin = st.number_input("Skin Thickness", 0, 100, 20)
            insulin = st.number_input("Insulin", 0, 900, 80)
            bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
            X = np.array([[pregnancies, glucose, bp, skin, insulin, bmi]])
        elif disease == "Kidney":
            bp = st.number_input("Blood Pressure", 40, 200, 80)
            sg = st.number_input("Specific Gravity (e.g., 1.01 -> 1.03)", 1.00, 1.10, 1.02, format="%.2f")
            al = st.number_input("Albumin (0-5)", 0, 5, 0)
            su = st.number_input("Sugar (0-5)", 0, 5, 0)
            bgr = st.number_input("Blood Glucose Random", 0, 500, 100)
            X = np.array([[bp, sg, al, su, bgr]])
        else:  # Liver
            age = st.number_input("Age", 10, 100, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tb = st.number_input("Total Bilirubin", 0.0, 20.0, 1.0)
            db = st.number_input("Direct Bilirubin", 0.0, 10.0, 0.2)
            alkph = st.number_input("Alkaline Phosphatase", 0, 1000, 200)
            X = np.array([[age, 1 if gender == "Male" else 0, tb, db, alkph]])

        if st.button("Predict Disease") and X is not None:
            try:
                Xs = scaler.transform(X)
                pred = int(model.predict(Xs)[0])
                prob = None
                if hasattr(model, "predict_proba"):
                    try:
                        prob = float(model.predict_proba(Xs)[0][1])
                    except Exception:
                        prob = None

                st.markdown("### Prediction Result")
                if pred == 1:
                    st.error("Disease predicted. Please consult a doctor.")
                else:
                    st.success("No disease predicted.")

                if prob is not None:
                    st.info(f"Risk probability: {prob:.1%}")

                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "disease": disease.lower(),
                    "input": str(X.tolist()),
                    "prediction": pred,
                    "probability": prob if prob else "NA"
                }
                save_history(row)

                pdf_name = os.path.join(
                    PROJECT_ROOT,
                    f"report_{disease.lower()}_{int(datetime.now().timestamp())}.pdf"
                )
                details = {
                    "Disease": disease,
                    "ML Prediction": "Disease Detected" if pred == 1 else "No Disease",
                    "ML Risk %": f"{prob:.1%}" if prob else "NA",
                    "Patient Data": str(X.tolist())
                }
                create_pdf_report("Patient", details, pdf_name)
                with open(pdf_name, "rb") as f:
                    st.download_button(
                        label="Download ML Health Report",
                        data=f,
                        file_name=f"ml_report_{disease.lower()}.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------- HISTORY ----------
elif choice == "History":
    st.title("Prediction History")

    if os.path.exists(HISTORY_CSV):
        df = pd.read_csv(HISTORY_CSV)
        if df.empty:
            st.info("History file exists but is empty.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                disease_filter = st.selectbox(
                    "Filter by disease",
                    ["All"] + sorted(df["disease"].unique())
                )
            with c2:
                only_positive = st.checkbox("Only cases where disease predicted")
            with c3:
                max_rows = st.number_input("Rows to view", 10, 500, 50)

            filtered_df = df.copy()
            if disease_filter != "All":
                filtered_df = filtered_df[filtered_df["disease"] == disease_filter]
            if only_positive:
                filtered_df = filtered_df[filtered_df["prediction"] == 1]

            filtered_df = filtered_df.sort_values(by="timestamp", ascending=False).head(max_rows)
            st.dataframe(filtered_df)

            st.download_button(
                "Download Filtered History CSV",
                filtered_df.to_csv(index=False).encode("utf-8"),
                "health_history_filtered.csv"
            )

            if not filtered_df.empty:
                st.markdown("### Visualization")
                st.subheader("Predictions per disease")
                st.bar_chart(filtered_df["disease"].value_counts())

                st.subheader("Disease vs No Disease")
                outcome = filtered_df["prediction"].map({0: "No disease", 1: "Disease"})
                st.bar_chart(outcome.value_counts())

                if "probability" in filtered_df.columns:
                    try:
                        prob_df = filtered_df[filtered_df["probability"] != "NA"].copy()
                        prob_df["timestamp"] = pd.to_datetime(prob_df["timestamp"], errors="coerce")
                        prob_df["probability"] = prob_df["probability"].astype(float)
                        prob_df = prob_df.dropna(subset=["timestamp"])
                        prob_df = prob_df.sort_values("timestamp")
                        if not prob_df.empty:
                            st.subheader("Risk probability trend over time")
                            st.line_chart(prob_df.set_index("timestamp")["probability"])
                    except Exception:
                        st.info("Could not plot probability trend (parsing issue).")
    else:
        st.info("No predictions yet. Make some predictions first!")

# ---------- ABOUT ----------
elif choice == "About":
    st.title("About Smart Health System")
    st.markdown("""
## Smart Health Monitoring & Early Disease Prediction

Major features:
- AI Health Insights (rule-based risk scoring using vitals)
- Symptom Description Analyzer (NLP on English + Hinglish text)
- ML-based disease prediction for Heart, Diabetes, Kidney, Liver
- PDF health report generation
- Prediction history logging and analytics

Technologies:
- Python, Streamlit
- Scikit-learn, NumPy, Pandas, Joblib
- FPDF for PDF reports

Developed as a 7th semester AI/ML major project for academic purposes only.
""")

