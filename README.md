<div align="center">

# 🍷 Wine Quality Prediction with MLflow

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.2.2-blue.svg)](https://mlflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0.1-red.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An end-to-end machine learning pipeline for predicting wine quality using MLflow for experiment tracking and model management.**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Results](#results) • [Contributing](#contributing) • [License](#license)

<!-- Image will be added later -->
<!-- <img src="path-to-image" alt="Wine Quality Prediction" width="600px"> -->

</div>

## 📋 Features

- **Modular Pipeline Architecture**: Organized into reusable components for easy maintenance and extension
- **Automated Data Validation**: Schema validation ensures data consistency
- **Hyperparameter Optimization**: Systematic trials to find optimal model parameters
- **MLflow Integration**: Comprehensive experiment tracking and model versioning
- **Interactive Web Interface**: User-friendly Flask application for making predictions
- **Visualization Tools**: Performance metrics visualization across different hyperparameter combinations

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/khalidasdsju/wine-qulity-predication-by-MLFolw.git
   cd wine-qulity-predication-by-MLFolw
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

## 🚀 Usage

### Running the Complete Pipeline

Execute the full ML pipeline from data ingestion to model evaluation:

```bash
python main.py
```

This will:
- Download and validate the wine quality dataset
- Preprocess the data
- Train an ElasticNet model
- Evaluate the model performance
- Log all metrics and artifacts to MLflow

### Hyperparameter Optimization

Run multiple trials with different hyperparameters:

```bash
python run_trials.py
```

Update the model with the best hyperparameters:

```bash
python update_best_model.py
```

### Web Application

Start the Flask web application for making predictions:

```bash
python app.py
```

Access the application at http://127.0.0.1:5000

### MLflow UI

View experiment tracking and model registry:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Access the MLflow UI at http://127.0.0.1:5001

## 🏗️ Architecture

### Project Structure

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

### Pipeline Components

1. **Data Ingestion**: Downloads and prepares the wine quality dataset
2. **Data Validation**: Validates the schema of the dataset
3. **Data Transformation**: Preprocesses the data using StandardScaler
4. **Model Training**: Trains an ElasticNet model with MLflow tracking
5. **Model Evaluation**: Evaluates the model and logs metrics to MLflow

## 📊 Results

### Hyperparameter Optimization

We conducted 10 trials with different hyperparameter combinations for the ElasticNet model. The best performing model had:

| Parameter | Value |
|-----------|-------|
| Alpha     | 0.01  |
| L1 Ratio  | 0.1   |
| RMSE      | 0.625 |
| MAE       | 0.504 |
| R²        | 0.402 |

The performance across different hyperparameter combinations is visualized in the `visualizations/` directory.

### Key Findings

- Lower alpha values (around 0.01) produced the best results
- Higher alpha values (> 0.7) resulted in poor model performance
- The model achieves moderate predictive power for wine quality

## 🤝 Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) for the Wine Quality dataset
- [MLflow](https://mlflow.org/) for experiment tracking and model management
- [scikit-learn](https://scikit-learn.org/) for machine learning tools

## 📞 Contact

Khalid - [khalid.asds.ju@gmail.com](mailto:khalid.asds.ju@gmail.com)

Project Link: [https://github.com/khalidasdsju/wine-qulity-predication-by-MLFolw](https://github.com/khalidasdsju/wine-qulity-predication-by-MLFolw)