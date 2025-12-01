import streamlit as st
import pandas as pd
import numpy as np
import os, joblib
from streamlit_option_menu import option_menu
from src.simulator import generate_stream, generate_vitals
from src.report_generator import HealthReport
from src.utils import load_model_and_scaler

st.set_page_config(layout='wide', page_title='Smart Health Monitor')

MODELS_DIR = 'models'
HISTORY_CSV = 'history/history.csv'
os.makedirs('history', exist_ok=True)

with st.sidebar:
    choice = option_menu('Smart Health System', ['Home','Monitoring','Predictor','History','About'], icons=['house','activity','heartbeat','clock-history','info-circle'])

if choice == 'Home':
    st.title('Smart Health Monitoring & Early Disease Prediction (Software-only)')
    st.markdown('''**What this does:** Multi-disease prediction (Heart, Diabetes, Kidney, Liver), virtual monitoring simulator (no hardware), SHAP explainability and report generation.''')

elif choice == 'Monitoring':
    st.title('Virtual Smart Monitoring Dashboard (Simulated)')
    col1, col2 = st.columns([3,1])
    with col2:
        base_hr = st.slider('Base heart rate', 60, 95, 75)
        n = st.number_input('Points for quick demo', 10, 500, 80)
        if st.button('Generate quick stream'):
            df = generate_stream(n, base_hr)
            st.dataframe(df.tail(10))
            st.line_chart(df.set_index('timestamp')[['heart_rate']])
    with col1:
        st.subheader('Live simulated vitals (press Start)')
        placeholder = st.empty()
        start = st.button('Start Live Simulation')
        if start:
            for i in range(40):
                vit = generate_vitals(base_hr)
                placeholder.metric('Heart Rate', vit['heart_rate'])
                st.write(vit)

elif choice == 'Predictor':
    st.title('Disease Prediction Suite')
    disease = st.selectbox('Choose disease', ['Heart','Diabetes','Kidney','Liver'])
    model, scaler = load_model_and_scaler(disease.lower())
    if model is None:
        st.warning(f'Model for {disease} not found in {MODELS_DIR}. If you trained models, place {disease.lower()}_model.pkl and {disease.lower()}_scaler.pkl there.')
    else:
        st.success(f'{disease} model loaded.')
        st.write('Enter patient information:')
        if disease == 'Heart':
            age = st.number_input('Age', 20, 100, 50)
            sex = st.selectbox('Sex', ['Male','Female'])
            trestbps = st.number_input('Resting BP (mm Hg)', 80, 200, 120)
            chol = st.number_input('Cholesterol', 100, 500, 200)
            thalach = st.number_input('Max heart rate achieved', 50, 220, 150)
            exang = st.selectbox('Exercise induced angina', [0,1])
            X = np.array([[age, 1 if sex=='Male' else 0, trestbps, chol, thalach, exang]])
        elif disease == 'Diabetes':
            pregnancies = st.number_input('Pregnancies', 0, 20, 1)
            glucose = st.number_input('Glucose', 0, 300, 120)
            bp = st.number_input('Blood Pressure', 0, 200, 70)
            skin = st.number_input('Skin Thickness', 0, 100, 20)
            insulin = st.number_input('Insulin', 0, 900, 80)
            bmi = st.number_input('BMI', 10.0, 70.0, 25.0)
            X = np.array([[pregnancies, glucose, bp, skin, insulin, bmi]])
        elif disease == 'Kidney':
            bp = st.number_input('Blood Pressure', 40, 200, 80)
            sg = st.number_input('Specific Gravity (e.g., 1.01->1.03)', 1.00, 1.10, 1.02)
            al = st.number_input('Albumin (0-5)', 0, 5, 0)
            su = st.number_input('Sugar (0-5)', 0, 5, 0)
            bgr = st.number_input('Blood Glucose Random', 0, 500, 100)
            X = np.array([[bp, sg, al, su, bgr]])
        else:
            age = st.number_input('Age', 10, 100, 45)
            gender = st.selectbox('Gender', ['Male','Female'])
            tb = st.number_input('Total Bilirubin', 0.0, 20.0, 1.0)
            db = st.number_input('Direct Bilirubin', 0.0, 10.0, 0.2)
            alkph = st.number_input('Alkaline Phosphatase', 0, 1000, 200)
            X = np.array([[age, 1 if gender=='Male' else 0, tb, db, alkph]])

        if st.button('Predict'):
            try:
                Xs = scaler.transform(X)
            except Exception as e:
                st.error('Error transforming input. Make sure the scaler matches expected feature count.\\n' + str(e))
                Xs = None
            if Xs is not None:
                pred = model.predict(Xs)[0]
                prob = None
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(Xs)[0][1]
                st.write('**Prediction:**', 'Disease' if int(pred)==1 else 'No Disease')
                if prob is not None:
                    st.write('**Risk probability:**', round(float(prob),3))
                row = dict(disease=disease.lower(), input=str(X.tolist()), pred=int(pred), prob=float(prob) if prob is not None else 'NA')
                dfh = pd.DataFrame([row])
                if os.path.exists(HISTORY_CSV):
                    dfh.to_csv(HISTORY_CSV, mode='a', header=False, index=False)
                else:
                    dfh.to_csv(HISTORY_CSV, index=False)
                st.success('Saved prediction to history.')
                report = HealthReport('Patient', {'Disease': disease, 'Prediction': 'Disease' if int(pred)==1 else 'No Disease', 'Probability': prob})
                fname = f"history/{disease}_report.pdf"
                report.create_pdf(fname)
                with open(fname, 'rb') as f:
                    st.download_button('Download PDF report', f, file_name=os.path.basename(fname))
                try:
                    import shap
                    explainer = shap.Explainer(model, Xs)
                    shap_vals = explainer(Xs)
                    st.write('SHAP explanation:')
                    st.pyplot(shap.plots.bar(shap_vals, show=False))
                except Exception as e:
                    st.info('SHAP explanation not available: ' + str(e))

elif choice == 'History':
    st.title('Prediction History')
    if os.path.exists(HISTORY_CSV):
        df = pd.read_csv(HISTORY_CSV)
        st.dataframe(df.tail(200))
        st.download_button('Download history CSV', df.to_csv(index=False).encode('utf-8'), file_name='history.csv')
    else:
        st.info('No history saved yet.')

elif choice == 'About':
    st.title('About Project')
    st.markdown('Smart Health Monitoring & Early Disease Prediction — Software-only final year project. Trained heart model: train locally and save to models/heart_model.pkl. Pretrained models for diabetes/kidney/liver are included.')

