import pandas as pd
import polars as pl
import numpy as np
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from statistics import mean
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

leap_raw_df = pl.read_csv('s3://capstone-project-ucsd-mle-bootcamp/datasets/train.csv')

input_columns = leap_raw_df.columns[1:557]
output_columns = leap_raw_df.columns[557:]
X = leap_raw_df.select(pl.col(input_columns))
Y = leap_raw_df.select(pl.col(output_columns))

def custom_log_transform(column):
    """Applies a log1p transformation to a given column with an adjustment for negative values."""
    min_value = column.min()
    transformed_column = np.log1p(column - min_value + 1)  # log1p is log(1 + x) which is stable for small x
    return transformed_column

def log_transformation(data):
    """Applies custom log transformation to each column in the data."""
    data_log_transformed = np.array(data, dtype="float64")
    for i in range(data.shape[1]):
        data_log_transformed[:, i] = custom_log_transform(data[:, i])
    return data_log_transformed

def preprocess_data(X, Y, transform=True, log=False):
    """Preprocesses the data by scaling and optionally applying log transformation."""
    X_np = X.to_numpy()
    Y_np = Y.to_numpy()

    if transform:
        # Standardize features
        X_scaler = StandardScaler().fit(X_np)
        Y_scaler = StandardScaler().fit(Y_np)

        X_transformed = X_scaler.transform(X_np)
        Y_transformed = Y_scaler.transform(Y_np)

        if log:
            X_transformed = log_transformation(X_transformed)
            Y_transformed = log_transformation(Y_transformed)
        
        # Clean up
        del X, Y, X_np, Y_np

    else:
        # Split data into training and validation sets without transformation
        #X_train, X_val, Y_train, Y_val = train_test_split(X_np, Y_np, test_size=0.2, random_state=42)
        
        # Clean up
        del X, Y, X_np, Y_np

    return X_transformed, Y_transformed

# Example usage
# Assuming X and Y are defined and are Polars DataFrames
X_process, Y_process = preprocess_data(X, Y, transform=True, log=False)
X_train, X_val, Y_train, Y_val = train_test_split(X_process, Y_process, test_size=0.2, random_state=42)