import logging

import pandas as pd
from code.training.src.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from sklearn.base import RegressorMixin


def train_model(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        config: dict,
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        config: ModelNameConfig
    Returns:
        model: RegressorMixin
    """
    try:

        model_factory = ModelFactory()
        model = model_factory.createModel(config['model_name'])
        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if config['fine_tune']:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
        return trained_model
    except Exception as e:
        logging.error(e)
        raise e


class ModelFactory:
    """
    Factory class to create the model based on the config.
    """

    @staticmethod
    def createModel(model_name: str) -> RegressorMixin:
        if model_name == "lightgbm":
            model = LightGBMModel()
        elif model_name == "randomforest":
            model = RandomForestModel()
        elif model_name == "xgboost":
            model = XGBoostModel()
        elif model_name == "linear_regression":
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")
        return model
