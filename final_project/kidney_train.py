import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DATA_CSV = r"C:\Users\Asus\OneDrive\Documents\final_project\Chronic_Kidney_Disease.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_CSV)
print("Columns:", df.columns.tolist())

# Clean and convert necessary columns
for col in ["bp", "sg", "al", "su", "bgr"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=["bp", "sg", "al", "su", "bgr"], inplace=True)

# ------------------ FEATURES from app.py ------------------
X = df[["bp", "sg", "al", "su", "bgr"]]

# ------------------ TARGET ------------------
# CKD datasets use "classification" or "class"
# Convert "ckd" → 1 and "notckd" → 0
df["classification"] = df["classification"].map({"ckd": 1, "notckd": 0})
y = df["classification"]

# ------------------ SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------ SCALE ------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------ MODEL ------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# ------------------ ACCURACY ------------------
pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)
print("KIDNEY Model Accuracy:", acc)

# ------------------ SAVE ------------------
joblib.dump(model, os.path.join(MODELS_DIR, "kidney_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "kidney_scaler.pkl"))

print("Saved kidney_model.pkl and kidney_scaler.pkl")
