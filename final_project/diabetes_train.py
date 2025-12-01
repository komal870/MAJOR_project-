# diabetes_train.py - High Accuracy Diabetes Model Training
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score

# Common diabetes datasets: Pima Indians or similar
# Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
DATA_PATH = "diabetes.csv"  # Place diabetes.csv in same folder
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading Diabetes Dataset...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print("Columns:", df.columns.tolist())
except FileNotFoundError:
    print("❌ diabetes.csv not found!")
    print("Download from: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
    print("Save as 'diabetes.csv' and retry.")
    exit()

# Standard Pima Indians Diabetes columns (adjust if different)
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
target_col = 'Outcome'  # 1=Diabetes, 0=No Diabetes

# Handle zeros as missing values (common in diabetes data)
for col in ['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']:
    if col in df.columns:
        df[col].replace(0, np.nan, inplace=True)
        df[col].fillna(df[col].median(), inplace=True)

X = df[feature_cols]
y = df[target_col]

print(f"Class distribution:\n{y.value_counts()}")

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE for class imbalance
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train_bal)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train_bal, cv=5)
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Save for app.py/appp.py
joblib.dump(model, os.path.join(MODEL_DIR, "diabetesmodel.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "diabetesscaler.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "diabetes_features.pkl"))

print("\n✅ Diabetes model trained and saved!")
print("Files: diabetesmodel.pkl, diabetesscaler.pkl, diabetes_features.pkl")
print("Expected accuracy: 82-85% (standard for Pima dataset)")

