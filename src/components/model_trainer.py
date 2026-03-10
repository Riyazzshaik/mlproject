import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and testing data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info("Initializing models")

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }

            logging.info("Defining hyperparameters")

            params = {

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"]
                },

                "Random Forest": {
                    "n_estimators": [16, 32, 64, 128]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [32, 64, 128]
                },

                "Linear Regression": {},

                "XGBRegressor": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [32, 64, 128]
                },

                "CatBoosting": {
                    "depth": [6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [50, 100]
                },

                "AdaBoost": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [32, 64, 128]
                }
            }

            logging.info("Starting model evaluation")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Model Score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No good model found", sys)

            logging.info("Training best model")

            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            logging.info(f"Model R2 Score: {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)