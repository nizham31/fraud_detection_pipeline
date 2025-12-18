import pandas as pd
import mlflow
import mlflow.sklearn
import os
import shutil
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
tracking_user = os.getenv("MLFLOW_TRACKING_USERNAME")
tracking_pass = os.getenv("MLFLOW_TRACKING_PASSWORD")

if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

if tracking_user and tracking_pass:
    os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_pass

mlflow.set_experiment("CreditCard_Fraud_Detection_Production")

# autolog
mlflow.sklearn.autolog(log_models=True)

def load_data():
    train_df = pd.read_csv("data_clean/train_data.csv")
    test_df = pd.read_csv("data_clean/test_data.csv")

    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]
    X_test  = test_df.drop("Class", axis=1)
    y_test  = test_df["Class"]
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42,
        "class_weight": "balanced",
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def export_model_locally(model, export_path="exported_model"):
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    mlflow.sklearn.save_model(sk_model=model, path=export_path)
    return export_path

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    with mlflow.start_run(run_name="rf-production"):
        model = train_model(X_train, y_train)

        export_path = export_model_locally(model)

        mlflow.log_artifacts(export_path, artifact_path="exported_model")

        print("Training + upload artifacts selesai.")
