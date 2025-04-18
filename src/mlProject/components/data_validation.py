import os
import pandas as pd
from mlProject import logger
from mlProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True
            data = pd.read_csv(self.config.data_path, sep=";")
            all_cols = list(data.columns)
            all_schema = self.config.all_schema.columns

            for col in all_cols:
                if col not in all_schema.keys():
                    validation_status = False
                    logger.info(f"Column {col} not in schema")
                    
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
                
            logger.info(f"Validation status: {validation_status}")
            return validation_status
        
        except Exception as e:
            logger.exception(e)
            raise e
