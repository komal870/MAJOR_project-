import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import save_model

def train(data_path, name, target_col):
    if not os.path.exists(data_path):
        print(f"❌ Put your dataset at {data_path}")
        return
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        print(f"❌ Target column {target_col} not found in dataset")
        return
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\\n', classification_report(y_test, y_pred))
    save_model(model, scaler, name, 'models')
    print(f'Saved {name}_model.pkl and {name}_scaler.pkl to models/')

if __name__ == '__main__':
    pass