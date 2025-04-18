# Wine Quality Prediction with MLflow

This project demonstrates an end-to-end machine learning pipeline for predicting wine quality using MLflow for experiment tracking and model management.

## Project Overview

The goal of this project is to predict the quality of wine based on various physicochemical properties. The project includes:

- Data ingestion from a public dataset
- Data validation to ensure schema correctness
- Data transformation using preprocessing techniques
- Model training with hyperparameter optimization
- Model evaluation and tracking using MLflow
- Model deployment via a Flask web application

## Project Structure

```
├── artifacts/                  # Generated artifacts from the pipeline
│   ├── data_ingestion/         # Raw and processed data
│   ├── data_validation/        # Validation results
│   ├── data_transformation/    # Preprocessor objects
│   ├── model_trainer/          # Trained models and MLflow runs
│   └── model_evaluation/       # Model evaluation metrics
├── config/                     # Configuration files
│   └── config.yaml             # Main configuration
├── logs/                       # Application logs
├── mlruns/                     # MLflow experiment tracking data
├── src/                        # Source code
│   └── mlProject/              # Main package
│       ├── components/         # Pipeline components
│       ├── config/             # Configuration management
│       ├── constants/          # Constants and paths
│       ├── entity/             # Data classes
│       ├── pipeline/           # Pipeline stages
│       └── utils/              # Utility functions
├── templates/                  # Flask templates
├── visualizations/             # Performance visualizations
├── app.py                      # Flask web application
├── main.py                     # Main pipeline execution
├── params.yaml                 # Model hyperparameters
├── run_trials.py               # Script for hyperparameter trials
├── schema.yaml                 # Data schema definition
├── setup.py                    # Package setup
└── update_best_model.py        # Script to update with best model
```

## Hyperparameter Optimization Results

We conducted 10 trials with different hyperparameter combinations for the ElasticNet model. The best performing model had:

- Alpha: 0.01
- L1 Ratio: 0.1
- RMSE: 0.625
- MAE: 0.504
- R²: 0.402

The performance of different hyperparameter combinations can be visualized in the `visualizations/` directory.

## Running the Project

1. Install dependencies:
   ```
   pip install -e .
   ```

2. Run the full pipeline:
   ```
   python main.py
   ```

3. Run hyperparameter trials:
   ```
   python run_trials.py
   ```

4. Update with the best model:
   ```
   python update_best_model.py
   ```

5. Start the web application:
   ```
   python app.py
   ```

6. View MLflow UI:
   ```
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
   ```

## Web Application

The web application allows users to input wine characteristics and get a quality prediction. Access it at http://127.0.0.1:5000 when running the app.

## MLflow Tracking

MLflow is used to track experiments, metrics, and models. The MLflow UI can be accessed at http://127.0.0.1:5001 when running the MLflow server.

## Model Registry

The best model is registered in the MLflow Model Registry as "WineQualityBestModel" for easy deployment and versioning.

## Future Improvements

- Implement more advanced feature engineering
- Try different model architectures
- Add cross-validation for more robust evaluation
- Implement CI/CD pipeline for model deployment
- Add user authentication to the web application