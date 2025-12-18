import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. SETUP MLFLOW 
load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    print("WARNING: MLFLOW_TRACKING_URI tidak ditemukan. Cek .env atau Secrets.")

# Set MLflow
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("CreditCard_Fraud_Detection_Production")

mlflow.sklearn.autolog(log_models=True)

# 2. LOAD DATA
def load_data():
    print("Loading processed data...")
    train_df = pd.read_csv('data_clean/train_data.csv')
    test_df = pd.read_csv('data_clean/test_data.csv')

    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']

    return X_train, y_train, X_test, y_test

# 3. TRAIN MODEL 
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

    return model, params

# 4. EVALUATE & LOG CUSTOM ARTIFACTS
def evaluate_and_log(model, X_test, y_test):
    print("Logging custom artifacts to MLflow...")

    with mlflow.start_run(run_name="Manual_Evaluation_Artifacts", nested=True):
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        print(f"Metrics - F1: {f1:.4f}, Recall: {rec:.4f}")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="CreditCard_Fraud_Model" 
        )

        # Artifact Confusion Matrix
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Artifact Feature Importance
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10,6))
            feat_importances = pd.Series(model.feature_importances_, index=X_test.columns)
            feat_importances.nlargest(10).plot(kind='barh')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()

        print("Autologging complete & Artifacts uploaded to MLflow/DagsHub")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model, params = train_model(X_train, y_train)
    evaluate_and_log(model, X_test, y_test)
