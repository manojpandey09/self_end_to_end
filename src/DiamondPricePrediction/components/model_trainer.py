#class ModelTrainer:
    #def __init__(self):
        #pass

    #def initiate_model_training(self):
       # pass



    #coming from model training file comment above 


import pandas as pd
import numpy as np
import os
import sys
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customexception
from dataclasses import dataclass
from ..utils.utils import save_object, evaluate_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import mlflow
import mlflow.sklearn


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple regression models, evaluates them, selects the best one, 
        logs metrics in MLflow, and saves the trained model to a pickle file.
        """
        try:
            logging.info('Splitting features and target from train and test data')

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models to train
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'ElasticNet': ElasticNet()
            }
            
            logging.info('Evaluating models...')
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            logging.info(f'Model Report: {model_report}')
            print("\nModel Report:", model_report)

            # Get best model details
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f'Best Model: {best_model_name} | R2 Score: {best_model_score}')
            print(f"\nBest Model Found: {best_model_name} with R2 Score: {best_model_score}")

            # Train the best model
            best_model.fit(X_train, y_train)

            # MLflow logging starts here
            with mlflow.start_run():
                mlflow.log_param("best_model", best_model_name)

                # Log metrics for all models: R2, MAE, RMSE
                for model_name, score in model_report.items():
                    # Calculate MAE and RMSE for each model
                    y_pred = models[model_name].fit(X_train, y_train).predict(X_test)
                    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
                    mae = np.mean(np.abs(y_test - y_pred))
                    r2 = score

                    # Log metrics with model name as prefix
                    mlflow.log_metric(f"{model_name}_r2", r2)
                    mlflow.log_metric(f"{model_name}_rmse", rmse)
                    mlflow.log_metric(f"{model_name}_mae", mae)

                # Log best model separately as well
                y_pred_best = best_model.predict(X_test)
                mlflow.log_metric("best_model_r2", best_model_score)
                mlflow.log_metric("best_model_rmse", np.sqrt(np.mean((y_test - y_pred_best)**2)))
                mlflow.log_metric("best_model_mae", np.mean(np.abs(y_test - y_pred_best)))

                # Save model with MLflow and provide input_example to avoid warning
                mlflow.sklearn.log_model(
                    best_model,
                    "model",
                    input_example=X_train[:5]
                )

            # Save the best model as pickle file also
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully.")

        except Exception as e:
            logging.info('Exception occurred in initiate_model_trainer')
            raise customexception(e, sys)

#make a utilis file inside utils and code it lets go to utils file now