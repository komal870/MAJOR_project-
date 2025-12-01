import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib, os

def load_csv(path):
    return pd.read_csv(path)

def save_model(model, scaler, name, models_dir='models'):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, f"{name}_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, f"{name}_scaler.pkl"))

def load_model_and_scaler(name, models_dir='models'):
    mpath = os.path.join(models_dir, f"{name}_model.pkl")
    spath = os.path.join(models_dir, f"{name}_scaler.pkl")
    if os.path.exists(mpath) and os.path.exists(spath):
        model = joblib.load(mpath)
        scaler = joblib.load(spath)
        return model, scaler
    return None, None
PY
