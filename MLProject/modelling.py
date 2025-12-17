import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI tidak ditemukan di file .env")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("CreditCard_Fraud_Detection_Experiment")

# LOAD DATAA
def load_data():
    print("Loading processed data...")
    train_df = pd.read_csv('data_clean/train_data.csv')
    test_df = pd.read_csv('data_clean/test_data.csv')

    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']

    return X_train, y_train, X_test, y_test


# TRAIN MODEL

def train_model(X_train, y_train):
    print("Training model without hyperparameter tuning.....")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    best_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2
    }

    return model, best_params


# EVALUATE & LOG
def evaluate_and_log(model, best_params, X_test, y_test):
    print("Logging to MLflow...")

    with mlflow.start_run():

        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("tuning_method", "none")
        mlflow.log_param("environment", "DagsHub MLflow Tracking")

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        print(f"Metrics - F1: {f1:.4f}, Recall: {rec:.4f}")

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Artifact 1: Confusion Matrix
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Artifact 2: Feature Importance
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10,6))
            feat_importances = pd.Series(model.feature_importances_, index=X_test.columns)
            feat_importances.nlargest(10).plot(kind='barh')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()

        print("Artifacts uploaded to MLflow/DagsHub")

# MAIN
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    model, best_params = train_model(X_train, y_train)
    evaluate_and_log(model, best_params, X_test, y_test)