# liver_train.py - Indian Liver Patient Dataset Training

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

DATA_PATH = "indian_liver_patient.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading Liver Dataset...")
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Target column: Dataset (1 = liver disease, 2 = no disease)
target_col = "Dataset"

# Encode Gender
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"].astype(str))  # Male/Female -> 0/1

# Handle missing values (A/G ratio has NaNs)
df = df.replace("?", np.nan)
for col in df.columns:
    if col != target_col:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

# Features + target
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

print("Class distribution:\n", y.value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE for imbalance
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_bal)
X_test_s = scaler.transform(X_test)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train_s, y_train_bal)

# Evaluation
y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(model, X_train_s, y_train_bal, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Save artifacts
joblib.dump(model, os.path.join(MODEL_DIR, "livermodel.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "liverscaler.pkl"))
joblib.dump(le_gender, os.path.join(MODEL_DIR, "liver_gender_encoder.pkl"))

print("\n✅ Liver model saved: livermodel.pkl, liverscaler.pkl, liver_gender_encoder.pkl")
