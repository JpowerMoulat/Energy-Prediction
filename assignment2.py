print("Hello 1")
# You absolutely need these
import mlflow

# You will probably need these
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# This are for example purposes. You may discard them if you don't use them.
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
from mlflow.models import infer_signature
from urllib.parse import urlparse

# Read the data into DataFrames
power_df = pd.read_csv('data/power.csv', parse_dates=["time"], index_col="time")
wind_df = pd.read_csv('data/weather.csv', parse_dates=["time"], index_col="time")

# Add timeindex as a column 
power_df['Time'] = power_df.index

# Join power and forecast on power
df = power_df.join(wind_df, how="left")
df['Speed'] = ( df['Speed'].interpolate(method='time') )
df = df.dropna(subset = ['Speed'])

def convert_timestamp (df):
    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])

    # Convert timestamp to usable integers
    df['Hour'] = df["Time"].dt.hour
    df['Weekday'] = df["Time"].dt.dayofweek
    df['Month'] = df["Time"].dt.month
    return df

# Convert Direction in string to numeric sin and cos
def convert_direction(df): 
    direction_to_degrees = { 'N':   0.0, 'NNE': 22.5, 'NE':  45.0, 'ENE': 67.5, 'E':   90.0, 'ESE': 112.5, 'SE':  135.0, 'SSE': 157.5, 'S':   180.0, 'SSW': 202.5, 'SW':  225.0, 'WSW': 247.5, 'W':   270.0, 'WNW': 292.5, 'NW':  315.0, 'NNW': 337.5 }

    # Create a new column with the mapped degrees
    df['Degrees'] = df['Direction'].map(direction_to_degrees)
    df['Radians'] = df['Direction'].map(direction_to_degrees) * np.pi / 180

    # Define sin and cos columns for continous values of circular direction
    df['Sin'] = np.sin(df['Radians'])
    df['Cos'] = np.cos(df['Radians'])

    # Ensure Time is set as index for new incoming data
    df = df.set_index(pd.to_datetime(df["Time"]))

    # Interpolate between rows in the same manner as 'Speed'
    df[['Sin', 'Cos']] = ( df[['Sin', 'Cos']].
                        interpolate(method='time') 
                        .ffill()
                        .bfill() )
    return df

# Apply transformations for Timestamp and Direction
time_transformer = FunctionTransformer(convert_timestamp)
direction_transformer = FunctionTransformer(convert_direction)

# Only keep selected features
FEATURES = ["Speed", "Hour", "Weekday", "Month", "Sin", "Cos"]
scaler = ColumnTransformer([ ("scaler", MinMaxScaler(), FEATURES) ])

# Construct pipeline
pipeline_base = Pipeline([
    # Transformations
    ("time_features", time_transformer), # Explode timestamp to numeric features
    ("direction_features", direction_transformer), # Transform direction to sin and cos values
    ("scaler", scaler), # Normalize data from 0 to 1
])

# Define raw training features
RAW_FEATURES = ["Speed", "Time", "Direction"]
X = df[RAW_FEATURES] 
y = df["Total"]

# Split using a timeseries-sensitive method 5 times
tscv = TimeSeriesSplit(n_splits=3)

 # Start MLFlow
#mlflow.sklearn.autolog() # This is to help us track scikit learn metrics.
mlflow.set_tracking_uri("http://127.0.0.1:5000") # We set the MLFlow UI to display in our local host.
#mlflow.sklearn.autolog() # This is to help us track scikit learn metrics.

# Define function for gridsearch prediction
def gridsearch (model, parameter_grid, name):
    
    # Define pipeline:
    pipeline_grid = Pipeline(
        pipeline_base.steps + [ ("regressor", model) ]
        )
    
    # Define grid search
    grid_search = GridSearchCV (
    estimator = pipeline_grid,
    param_grid = parameter_grid,
    cv = tscv,
    scoring = "neg_mean_absolute_error", 
    n_jobs = 1
    )

    # Initiate grid search and log best run
    #mlflow.set_experiment(name)
        
    mdl = grid_search.fit(X,y)
    mlflow.log_metric("Average MAE", -mdl.best_score_)
    mlflow.sklearn.log_model(
        sk_model = mdl.best_estimator_, 
        artifact_path = "model",
        )
       
# Perform gridstyle search on XGBOost to find hyperparameter

# Define name and model
model = XGBRegressor()
name = "XGBoost"

# Define paramter grid
param_grid = {
    "regressor__n_estimators" : [20, 100, 200],
    "regressor__learning_rate" : [0.05, 0.1, 0.5],
    "regressor__max_depth" : [2, 5, 10],
}

gridsearch(model, param_grid, name)
print('Hello 2')