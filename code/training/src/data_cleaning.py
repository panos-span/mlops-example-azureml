import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        try:

            # Remove rows with missing values
            data = data.dropna()

            # Remove rows where a row contains a value less than 0
            data = data[(data >= 0).all(1)]

            # Remove outliers from the data using the IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)

            IQR = Q3 - Q1

            data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)).any(axis=1))]

            return data
        except Exception as e:
            logging.error(e)
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame, **kwargs) -> Tuple[
        Annotated[pd.DataFrame, "X_train"],
        Annotated[pd.DataFrame, "X_test"],
        Annotated[pd.Series, "y_train"],
        Annotated[pd.Series, "y_test"],
    ]:
        """
        Divides the data into train and test data.
        """
        try:
            X = data.drop("target", axis=1)
            y = data["target"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.data = data
        self.strategy = strategy

    def handle_data(self, **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle the data using the strategy.
        """
        try:
            return self.strategy.handle_data(self.data, **kwargs)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e


class DataScalerStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Scales the data using MinMaxScaler.
        """
        try:
            scalerFactory = ScalerFactory()
            scaler = scalerFactory.get_scaler(kwargs["scaler_type"])
            # Get the columns to be scaled
            columns = [col for col in data.columns if col not in ["target", "sex"]]
            # Scale the data
            data[columns] = scaler.fit_transform(data[columns])
            return data
        except Exception as e:
            logging.error(e)
            raise e


class CategoryEncoderStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Encodes the categorical data.
        """
        try:
            data = pd.get_dummies(data, columns=data['sex'])
            return data
        except Exception as e:
            logging.error(e)
            raise e


class ScalerFactory:
    def get_scaler(self, scaler_type: str):
        if scaler_type == "StandardScaler":
            return StandardScaler()
        elif scaler_type == "MinMaxScaler":
            return MinMaxScaler()
        elif scaler_type == "RobustScaler":
            return RobustScaler()


class TokenizeStrategy(DataStrategy):
    """
    Strategy to tokenize data.
    """

    def handle_data(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Tokenize the data.
        Args:
            data: the data to be tokenized
        Returns:
            pd.Series: the tokenized data
        """
        try:
            # Tokenize the review_comment_message column with Hugging Face's tokenizers
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
            data["review_comment_message"] = data["review_comment_message"].apply(
                lambda x: tokenizer.encode(x, add_special_tokens=True)
            )
            return data["review_comment_message"]
        except Exception as e:
            logging.error(f"Error in tokenizing data: {e}")
            raise e
