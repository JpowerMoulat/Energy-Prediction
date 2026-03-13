# Wind Power Forecasting Project

This project allows youto  build and evaluate machine learning models to forecast wind power generation using weather data. It includes data pipeline setup, model training, and experiment tracking using MLFlow.

## Project Structure

- `Assignment_2_Template_MLFlow_Project.py`: Main script for MLFlow project execution. You should edit this file for submission.
- `Assignment_2-Template.ipynb`: Jupyter notebook with detailed code explanations. Just to play around, idally not intended for submission, as it is not compatible with MLFlow projects.
- `python_env.yaml`: Python environment configuration
- `requirements.txt`: Project dependencies
- `MLproject`: MLFlow project configuration

## Setup

1. Create a virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
   pip install -r requirements2.txt
```

## Project Overview

This project demonstrates how to:

1. Retrieve wind and power generation data from the given .csv files
2. Process and analyze time series data using pandas
3. Build ML pipelines with scikit-learn
4. Track experiments using MLFlow
5. Deploy models for production use

## Data Sources

The project uses two main data sources:
- Wind forecast data (speed and direction)
- Power generation data from Orkney

## Model Training

The project includes:
- Time series cross-validation
- Feature engineering pipeline
- Model evaluation metrics (MAE, MSE, R²)
- Experiment tracking with MLFlow

## MLFlow Integration

MLFlow is used for:
- Experiment tracking
- Model versioning
- Parameter logging
- Metric visualization
- Model deployment

## Usage

1. Start the MLFlow server:
~~~ bash
git clone https://github.itu.dk/jonie/Windmill-Energy-Prediction.git
cd Windmill-Energy-Prediction

py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements2.txt
mlflow server
~~~

2. View experiments at http://localhost:5000. Then Run the main script in a new terminal. Replace brackets with your specific ML model:
```bash
.venv\Scripts\activate
mlflow run . --experiment-name XGBoost
mlflow models serve -m mlartifacts/[experiment]/[run]/artifacts/model --env-manager local
```

3. Close old terminal and create new terminal. Replace Time, Speed and Direction with given variables.
~~~bash
curl -X POST http://127.0.0.1:5000/invocations -H "Content-Type: application/json" -d "{\"dataframe_records\":[{\"Time\":\"2025-01-10T15:00:00Z\",\"Speed\":7.5,\"Direction\":\"NW\"}]}"
~~~


## Requirements

See `requirements.txt` and `requirements2.txt` for full list of dependencies. Key requirements:
- Python 3.11+
- MLFlow
- scikit-learn
- pandas
- InfluxDB client
