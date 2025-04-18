import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import json
from mlProject import logger
from mlProject.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        try:
            # Read the test data
            test_data = pd.read_csv(self.config.test_data_path, sep=";")
            
            # Split into features and target
            X_test = test_data.drop(self.config.target_column, axis=1)
            y_test = test_data[self.config.target_column]
            
            # Load the model
            model = joblib.load(self.config.model_path)
            
            # Set MLflow registry URI
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            with mlflow.start_run():
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Evaluate the model
                rmse, mae, r2 = self.eval_metrics(y_test, y_pred)
                
                # Log metrics
                mlflow.log_metric("test_rmse", rmse)
                mlflow.log_metric("test_mae", mae)
                mlflow.log_metric("test_r2", r2)
                
                # Log the model if tracking URI is not a local file path
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticNetModel")
                else:
                    mlflow.sklearn.log_model(model, "model")
                    
                # Save metrics to a file
                scores = {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
                
                with open(self.config.metric_file_name, "w") as f:
                    json.dump(scores, f, indent=4)
                    
                logger.info(f"Model evaluation completed. Metrics saved at {self.config.metric_file_name}")
                
                return scores
                
        except Exception as e:
            logger.exception(e)
            raise e
