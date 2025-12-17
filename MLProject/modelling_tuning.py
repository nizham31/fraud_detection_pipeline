import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

load_dotenv() 

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI tidak ditemukan di file .env")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(" ")

def load_data():
    print(" Loading processed data...")
    train_df = pd.read_csv('../preprocessing/data_clean/train_data.csv')
    test_df = pd.read_csv('../preprocessing/data_clean/test_data.csv')
    
    X_train = train_df.drop('Class', axis=1)
    y_train = train_df['Class']
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    return X_train, y_train, X_test, y_test

def train_with_tuning(X_train, y_train):
    print("Set Hyperparameter Tuning...")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    param_dist = {
        'n_estimators': [30, 100],
        'max_depth': [None, 10, 50],
        'min_samples_split': [2, 5]
    }
    
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=3,
        cv=3,
        scoring='f1',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    print(f"Best Params: {random_search.best_params_}")
    return random_search.best_estimator_, random_search.best_params_

def evaluate_and_log(model, best_params, X_test, y_test):
    print("Logging to MLflow...")
    
    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("environment", "DagsHub MLflow Tracking")
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        print(f"Metrics - F1: {f1:.4f}, Recall: {rec:.4f}")

        mlflow.sklearn.log_model(model, "model")
        
        # Artifacts
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        
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

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    best_model, best_params = train_with_tuning(X_train, y_train)
    evaluate_and_log(best_model, best_params, X_test, y_test)