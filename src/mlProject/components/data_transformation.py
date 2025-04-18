import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self):
        try:
            # Define the preprocessing steps
            preprocessor = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            logger.exception(e)
            raise e
            
    def initiate_data_transformation(self):
        try:
            # Read the data
            df = pd.read_csv(self.config.data_path, sep=";")
            
            # Split into features and target
            X = df.drop('quality', axis=1)
            y = df['quality']
            
            # Create preprocessor object
            preprocessor = self.get_data_transformer_object()
            
            # Fit and transform the data
            X_transformed = preprocessor.fit_transform(X)
            
            # Convert transformed arrays back to DataFrames
            transformed_df = pd.DataFrame(X_transformed, columns=X.columns)
            
            # Add target column back
            transformed_df['quality'] = y
            
            # Save the preprocessor object
            joblib.dump(preprocessor, self.config.preprocessor_path)
            logger.info(f"Preprocessor saved at {self.config.preprocessor_path}")
            
            return transformed_df, self.config.preprocessor_path
            
        except Exception as e:
            logger.exception(e)
            raise e
