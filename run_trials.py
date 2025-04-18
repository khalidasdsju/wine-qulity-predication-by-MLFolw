import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-quality-trials")

# Load the data
def load_data():
    data_path = Path("artifacts/data_ingestion/winequality-red.csv")
    df = pd.read_csv(data_path, sep=";")
    return df

# Preprocess the data
def preprocess_data(df):
    # Load the preprocessor
    preprocessor_path = Path("artifacts/data_transformation/preprocessor.joblib")
    preprocessor = joblib.load(preprocessor_path)
    
    # Split into features and target
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Apply preprocessing
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, y_train, y_test, X.columns

# Evaluate the model
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# Run a trial with specific hyperparameters
def run_trial(alpha, l1_ratio, X_train, X_test, y_train, y_test, feature_names):
    with mlflow.start_run():
        # Train the model
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        rmse, mae, r2 = eval_metrics(y_test, y_pred)
        
        # Log parameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log feature importances
        feature_importance = pd.DataFrame(
            list(zip(feature_names, model.coef_)),
            columns=["Feature", "Importance"]
        )
        feature_importance = feature_importance.sort_values("Importance", ascending=False)
        mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        logger.info(f"Trial with alpha={alpha}, l1_ratio={l1_ratio} - RMSE: {rmse}, MAE: {mae}, R2: {r2}")
        
        return rmse, mae, r2, model

def main():
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Define hyperparameter grid
    alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
    l1_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Run trials
    logger.info("Starting trials...")
    results = []
    
    for i in range(10):
        # Select hyperparameters for this trial
        alpha = alphas[i]
        l1_ratio = l1_ratios[i]
        
        # Run the trial
        logger.info(f"Running trial {i+1}/10 with alpha={alpha}, l1_ratio={l1_ratio}")
        rmse, mae, r2, model = run_trial(
            alpha, l1_ratio, X_train, X_test, y_train, y_test, feature_names
        )
        
        # Store results
        results.append({
            "trial": i+1,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
    
    # Find the best model
    best_trial = min(results, key=lambda x: x["rmse"])
    logger.info(f"Best trial: {best_trial}")
    
    # Save results to a file
    with open("trial_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info("All trials completed. Results saved to trial_results.json")

if __name__ == "__main__":
    main()
