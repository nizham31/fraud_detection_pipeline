import pandas as pd
import mlflow
import mlflow.sklearn
import os
import shutil
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

# 1. SETUP MLFLOW 
load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("CreditCard_Fraud_Detection_Production")

# logging hanya pakai autoLog
mlflow.sklearn.autolog(log_models=True)

def load_data():
    print("Loading processed data...")
    train_df = pd.read_csv('data_clean/train_data.csv')
    test_df = pd.read_csv('data_clean/test_data.csv')

    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']

    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    print("Training model with Autologging...")
    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42,
        "class_weight": "balanced"
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def export_model_locally(model):
    export_path = "exported_model"
    if os.path.exists(export_path):
        shutil.rmtree(export_path) 
    
    print(f"Menyimpan model ke folder lokal: {export_path}...")
    mlflow.sklearn.save_model(sk_model=model, path=export_path)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model = train_model(X_train, y_train)
    export_model_locally(model)
    print("Proses Training dan Ekspor Lokal Selesai.")
