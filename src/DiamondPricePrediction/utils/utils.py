# =====================================================================
# UTILS FILE FOR DIAMOND PRICE PREDICTION
# Contains helper functions for:
# 1. Saving Python objects (models, preprocessors)
# 2. Loading saved objects
# 3. Evaluating ML models with test data
# Integrated with MLflow for experiment tracking
# =====================================================================

# ---------------------------------------------
# IMPORTING LIBRARIES
# ---------------------------------------------
import os   # for file path handling (create/read directories)
import sys  # for system-specific error handling
import pickle  # for saving/loading Python objects (like trained models)
import numpy as np  # for numerical operations on arrays
import pandas as pd  # for handling datasets in tabular form

# custom logging and exception handling (project-specific)
from src.DiamondPricePrediction.logger import logging   # logs important events/errors
from src.DiamondPricePrediction.exception import customexception   # raises structured errors

# ML model evaluation metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  

# MLflow for experiment tracking & model registry
import mlflow  


# ---------------------------------------------
# MLFLOW TRACKING URI SETUP
# ---------------------------------------------
# ✅ Check environment variable "MLFLOW_TRACKING_URI"
# ✅ If not set, fall back to local folder "F:/plswork/mlruns"
# This helps in keeping experiment tracking flexible (local/server/cloud)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:///F:/plswork/mlruns"))


# ---------------------------------------------
# FUNCTION: SAVE OBJECT
# ---------------------------------------------
def save_object(file_path, obj):
    # Saves any Python object (model, scaler, encoder) to a file using pickle
    # Future use: load model directly without retraining again
    try:
        dir_path = os.path.dirname(file_path)  # extract directory path
        os.makedirs(dir_path, exist_ok=True)   # create folder if it doesn’t exist

        with open(file_path, "wb") as file_obj:  # open file in write-binary mode
            pickle.dump(obj, file_obj)  # save object using pickle

    except Exception as e:
        raise customexception(e, sys)  # raise custom exception if saving fails


# ---------------------------------------------
# FUNCTION: EVALUATE MULTIPLE MODELS
# ---------------------------------------------
def evaluate_model(X_train, y_train, X_test, y_test, models):
    # Trains and evaluates multiple ML models, returns R2 scores for each
    # Input: dictionary of models {name: model}
    # Output: dictionary {model_name: R2_score}
    try:
        report = {}  # dictionary to store evaluation results

        for i in range(len(models)):  # iterate over all models
            model = list(models.values())[i]  # pick model object
            model_name = list(models.keys())[i]  # get model name

            model.fit(X_train, y_train)  # train the model
            y_test_pred = model.predict(X_test)  # predict on test data
            test_model_score = r2_score(y_test, y_test_pred)  # calculate R2 score

            report[model_name] = test_model_score  # save model’s score in dictionary

        return report  # return dictionary of results

    except Exception as e:
        logging.info('Exception occurred during model evaluation')  # log error
        raise customexception(e, sys)  # raise custom error


# ---------------------------------------------
# FUNCTION: LOAD OBJECT
# ---------------------------------------------
def load_object(file_path):
    # Loads any previously saved Python object (model, scaler, encoder) from pickle file
    # Future use: reuse trained model without retraining
    try:
        with open(file_path, 'rb') as file_obj:  # open file in read-binary mode
            return pickle.load(file_obj)  # load object back into memory

    except Exception as e:
        logging.info('Exception occurred in load_object function')  # log error
        raise customexception(e, sys)  # raise structured error
