import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-quality-best-model")

def load_data():
    data_path = Path("artifacts/data_ingestion/winequality-red.csv")
    df = pd.read_csv(data_path, sep=";")
    return df

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

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main():
    # Load the best hyperparameters from the trials
    with open("trial_results.json", "r") as f:
        results = json.load(f)
    
    # Find the best trial (lowest RMSE)
    best_trial = min(results, key=lambda x: x["rmse"])
    logger.info(f"Best trial: {best_trial}")
    
    # Extract the best hyperparameters
    best_alpha = best_trial["alpha"]
    best_l1_ratio = best_trial["l1_ratio"]
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = load_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Train the model with the best hyperparameters
    logger.info(f"Training model with best hyperparameters: alpha={best_alpha}, l1_ratio={best_l1_ratio}")
    with mlflow.start_run():
        # Train the model
        model = ElasticNet(
            alpha=best_alpha,
            l1_ratio=best_l1_ratio,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        rmse, mae, r2 = eval_metrics(y_test, y_pred)
        
        # Log parameters
        mlflow.log_param("alpha", best_alpha)
        mlflow.log_param("l1_ratio", best_l1_ratio)
        
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
        
        # Register the model
        mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "WineQualityBestModel")
        
        # Save the model to the artifacts directory
        model_path = Path("artifacts/model_trainer/best_model.joblib")
        joblib.dump(model, model_path)
        
        logger.info(f"Best model saved at {model_path}")
        logger.info(f"Model metrics - RMSE: {rmse}, MAE: {mae}, R2: {r2}")
        
        # Update the app.py file to use the best model
        update_app_model_path(model_path)

def update_app_model_path(model_path):
    """Update the app.py file to use the best model"""
    app_path = Path("app.py")
    with open(app_path, "r") as f:
        app_content = f.read()
    
    # Replace the model path
    updated_content = app_content.replace(
        'model_path = Path("artifacts/model_trainer/model.joblib")',
        f'model_path = Path("{model_path}")'
    )
    
    with open(app_path, "w") as f:
        f.write(updated_content)
    
    logger.info(f"Updated app.py to use the best model at {model_path}")

if __name__ == "__main__":
    main()
