import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DATA_CSV = r"C:\Users\Asus\OneDrive\Documents\final_project\heart_disease_uci.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_CSV)

print(df.columns)

# Convert Male/Female to 1/0
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})

# Convert exang maybe yes/no? → 1/0
if df["exang"].dtype == object:
    df["exang"] = df["exang"].map({"Yes": 1, "No": 0})

# FEATURES
X = df[["age", "sex", "trestbps", "chol", "thalch", "exang"]]

# TARGET column
y = df["num"]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SCALE
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print("Heart model accuracy:", acc)

# SAVE
joblib.dump(model, os.path.join(MODELS_DIR, "heart_model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "heart_scaler.pkl"))

print("Saved heart_model.pkl and heart_scaler.pkl")
