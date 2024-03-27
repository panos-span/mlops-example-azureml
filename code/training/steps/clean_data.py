import logging
from typing import Tuple

import pandas as pd
from code.training.src.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
    DataScalerStrategy,
    CategoryEncoderStrategy
)
from typing_extensions import Annotated
import json


def clean_data(
        data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:

        # Read the config file
        with open('../config.json') as f:
            config = json.load(f)

        # Preprocess the data
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        # Scale the data
        scale_strategy = DataScalerStrategy()
        data_cleaning = DataCleaning(preprocessed_data, scale_strategy)
        scaled_data = data_cleaning.handle_data(scaler_type=config['scaler_type'])

        # Encode the data
        encode_strategy = CategoryEncoderStrategy()
        data_cleaning = DataCleaning(scaled_data, encode_strategy)
        encoded_data = data_cleaning.handle_data()

        # Divide the data
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(encoded_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e
