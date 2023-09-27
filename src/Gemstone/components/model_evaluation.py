import sys
from typing import List
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Gemstone.config.exception import CustomException
from Gemstone.config.logger import logging


class Evaluation:
    """
    Evaluation class for model performance evaluation using sklearn metrics.
    """

    def __init__(self):
        """Initialize the Evaluation class."""
        pass

    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error (MSE).
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            return mean_squared_error(y_true, y_pred)
        except Exception as e:
            logging.error("Error in mean_squared_error method:", e)
            raise CustomException(e, sys) from e

    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error (MAE).
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mae: float
        """
        try:
            return mean_absolute_error(y_true, y_pred)
        except Exception as e:
            logging.error("Error in mean_absolute_error method:", e)
            raise CustomException(e, sys) from e

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R-squared (R2) score.
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        """
        try:
            return r2_score(y_true, y_pred)
        except Exception as e:
            logging.error("Error in r2_score method:", e)
            raise CustomException(e, sys) from e

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            rmse: float
        """
        try:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        except Exception as e:
            logging.error("Error in root_mean_squared_error method:", e)
            raise CustomException(e, sys) from e

    def get_metrics_scores(self, y_true, y_pred) -> dict:
        """
        Calculate evaluation metrics and return them as a dictionary.
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            metrics: dict
        """
        try:
            mae = self.mean_absolute_error(y_true, y_pred)
            mse = self.mean_squared_error(y_true, y_pred)
            rmse = self.root_mean_squared_error(y_true, y_pred)
            r2_score_val = self.r2_score(y_true, y_pred)
            metrics = {
                "Mean Absolute Error": mae,
                "Mean Squared Error": mse,
                "Root Mean Squared Error": rmse,
                "R-squared (R2) Score": r2_score_val,
            }
            return metrics
        except Exception as e:
            logging.error("Error in get_metrics_scores method:", e)
            raise CustomException(e, sys) from e

    def evaluate_single_model(self, X_train, X_test, y_train, y_test, model) -> dict:
        """
        Evaluate a single machine learning model.
        Args:
            X_train: np.ndarray
            X_test: np.ndarray
            y_train: np.ndarray
            y_test: np.ndarray
            model: sklearn model
        Returns:
            evaluation_metrics: dict
        """
        try:
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_metrics = self.get_metrics_scores(y_train, y_train_pred)
            test_metrics = self.get_metrics_scores(y_test, y_test_pred)

            evaluation_metrics = {
                "Train Metrics": train_metrics,
                "Test Metrics": test_metrics,
            }

            logging.info("-" * 20)
            logging.info("Train Evaluation Scores:")
            logging.info(train_metrics)
            logging.info("-" * 20)
            logging.info("Test Evaluation Scores:")
            logging.info(test_metrics)
            return evaluation_metrics
        except Exception as e:
            logging.error("Error in evaluate_single_model method:", e)
            raise CustomException(e, sys) from e

    def initiate_multi_model_evaluation(self, X_train, X_test, y_train, y_test, models) -> dict:
        """
        Evaluate multiple machine learning models and return their R2 scores.
        Args:
            X_train: np.ndarray
            X_test: np.ndarray
            y_train: np.ndarray
            y_test: np.ndarray
            models: dict of sklearn models
        Returns:
            model_scores: dict
        """
        try:
            model_scores = {}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                r2_score_val = self.r2_score(y_test, y_test_pred)
                model_scores[model_name] = r2_score_val
                logging.info(f"Model Name: {model_name}")
                logging.info("-" * 20)
                logging.info(f"R-squared (R2) Score: {r2_score_val}")
            return model_scores
        except Exception as e:
            logging.error("Error in initiate_multi_model_evaluation method:", e)
            raise CustomException(e, sys) from e
