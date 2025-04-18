import os
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.sklearn
from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def train(self):
        try:
            # Read the data
            df = pd.read_csv(self.config.train_data_path, sep=";")
            
            # Split into features and target
            X = df.drop(self.config.target_column, axis=1)
            y = df[self.config.target_column]
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Start MLflow run
            mlflow.set_registry_uri("sqlite:///mlflow.db")
            tracking_url_type_store = os.path.join(self.config.root_dir, "mlruns")
            mlflow.set_tracking_uri(tracking_url_type_store)
            
            with mlflow.start_run():
                # Train the model
                model = ElasticNet(
                    alpha=self.config.alpha,
                    l1_ratio=self.config.l1_ratio,
                    random_state=self.config.random_state
                )
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Evaluate the model
                rmse, mae, r2 = self.eval_metrics(y_test, y_pred)
                
                # Log parameters and metrics
                mlflow.log_param("alpha", self.config.alpha)
                mlflow.log_param("l1_ratio", self.config.l1_ratio)
                
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                
                # Log the model
                mlflow.sklearn.log_model(model, "model")
                
                # Save the model locally
                model_path = os.path.join(self.config.root_dir, self.config.model_name)
                joblib.dump(model, model_path)
                
                logger.info(f"Model saved at {model_path}")
                logger.info(f"Model metrics - RMSE: {rmse}, MAE: {mae}, R2: {r2}")
                
                return model_path, rmse, mae, r2
                
        except Exception as e:
            logger.exception(e)
            raise e
