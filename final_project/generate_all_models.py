import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Create models directory
os.makedirs('models', exist_ok=True)
print("📁 Created 'models/' folder")

# ========== HEART DISEASE MODEL (6 features) ==========
print("\n❤️ Generating Heart Disease models...")
heart_data = pd.DataFrame({
    'age': np.random.normal(55, 10, 1000),
    'sex': np.random.choice([0, 1], 1000),
    'trestbps': np.random.normal(130, 17, 1000),
    'chol': np.random.normal(246, 51, 1000),
    'thalach': np.random.normal(150, 22, 1000),
    'exang': np.random.choice([0, 1], 1000, p=[0.68, 0.32])
})
heart_data['target'] = np.random.choice([0, 1], 1000, p=[0.55, 0.45])
X_heart = heart_data[['age', 'sex', 'trestbps', 'chol', 'thalach', 'exang']]
scaler_heart = StandardScaler().fit_transform(X_heart)
model_heart = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler_heart, heart_data['target'])
joblib.dump(model_heart, 'models/heart_model.pkl')
joblib.dump(StandardScaler().fit(X_heart), 'models/heart_scaler.pkl')
print("✅ heart_model.pkl & heart_scaler.pkl SAVED")

# ========== DIABETES MODEL (6 features) ==========
print("\n🍬 Generating Diabetes models...")
diabetes_data = pd.DataFrame({
    'pregnancies': np.random.poisson(3, 1000),
    'glucose': np.random.normal(120, 30, 1000),
    'bp': np.random.normal(72, 12, 1000),
    'skin': np.random.normal(20, 15, 1000),
    'insulin': np.random.exponential(80, 1000),
    'bmi': np.random.normal(32, 7, 1000)
})
diabetes_data['target'] = np.random.choice([0, 1], 1000, p=[0.65, 0.35])
X_diabetes = diabetes_data.drop('target', axis=1)
scaler_diabetes = StandardScaler().fit_transform(X_diabetes)
model_diabetes = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler_diabetes, diabetes_data['target'])
joblib.dump(model_diabetes, 'models/diabetes_model.pkl')
joblib.dump(StandardScaler().fit(X_diabetes), 'models/diabetes_scaler.pkl')
print("✅ diabetes_model.pkl & diabetes_scaler.pkl SAVED")

# ========== KIDNEY DISEASE MODEL (5 features) ==========
print("\n🫘 Generating Kidney Disease models...")
kidney_data = pd.DataFrame({
    'bp': np.random.normal(76, 13, 1000),
    'sg': np.random.normal(1.02, 0.005, 1000),
    'al': np.random.poisson(1, 1000),
    'su': np.random.choice([0,1,2,3,4,5], 1000, p=[0.7,0.15,0.08,0.04,0.02,0.01]),
    'bgr': np.random.normal(140, 70, 1000)
})
kidney_data['target'] = np.random.choice([0, 1], 1000, p=[0.75, 0.25])
X_kidney = kidney_data.drop('target', axis=1)
scaler_kidney = StandardScaler().fit_transform(X_kidney)
model_kidney = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler_kidney, kidney_data['target'])
joblib.dump(model_kidney, 'models/kidney_model.pkl')
joblib.dump(StandardScaler().fit(X_kidney), 'models/kidney_scaler.pkl')
print("✅ kidney_model.pkl & kidney_scaler.pkl SAVED")

# ========== LIVER DISEASE MODEL (5 features) ==========
print("\n🧠 Generating Liver Disease models...")
liver_data = pd.DataFrame({
    'age': np.random.normal(45, 12, 1000),
    'gender': np.random.choice([0, 1], 1000),
    'tb': np.random.exponential(1.5, 1000),
    'db': np.random.exponential(0.3, 1000),
    'alkph': np.random.normal(291, 126, 1000)
})
liver_data['target'] = np.random.choice([0, 1], 1000, p=[0.71, 0.29])
X_liver = liver_data.drop('target', axis=1)
scaler_liver = StandardScaler().fit_transform(X_liver)
model_liver = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler_liver, liver_data['target'])
joblib.dump(model_liver, 'models/liver_model.pkl')
joblib.dump(StandardScaler().fit(X_liver), 'models/liver_scaler.pkl')
print("✅ liver_model.pkl & liver_scaler.pkl SAVED")

print("\n🎉 ALL 8 MODELS GENERATED SUCCESSFULLY!")
print("📂 Files created in 'models/' folder:")
print("- heart_model.pkl, heart_scaler.pkl")
print("- diabetes_model.pkl, diabetes_scaler.pkl") 
print("- kidney_model.pkl, kidney_scaler.pkl")
print("- liver_model.pkl, liver_scaler.pkl")
print("\n🚀 Now run: streamlit run app.py")
print("✅ Your Smart Health System is READY!")
