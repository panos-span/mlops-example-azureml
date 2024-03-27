import logging

import pandas as pd


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self) -> None:
        """Initialize the data ingestion class."""
        pass

    def get_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df


def ingest_data(path: str = "../data/diabetes.csv") -> pd.DataFrame:
    """
    Args:
        path: Path to the data file.
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data(path)
        return df
    except Exception as e:
        logging.error(e)
        raise e
