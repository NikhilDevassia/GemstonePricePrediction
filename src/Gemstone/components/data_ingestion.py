import os
import sys
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

from Gemstone.config.exception import CustomException
from Gemstone.config.logger import logging

@dataclass
class DataIngestionConfig:
    """
    Data ingestion Config class that returns file paths.
    """
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngester:
    """
    Data ingestion class that ingests data from the source and returns a DataFrame.
    """

    def __init__(self):
        """ Initialize the data ingestion class. """
        self.ingestion_config = DataIngestionConfig()

    def get_data(self) -> pd.DataFrame:
        """
        Get the dataset containing gemstone prices.

        Returns:
            pd.DataFrame: The dataset containing gemstone prices.

        Raises:
            CustomException: If there is an error reading the dataset.
        """
        try:
            logging.info("Collecting dataset")
            data_file_path = Path(__file__).parent / "data" / "gemstone_price.csv"
            return pd.read_csv(data_file_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_ingest_data(self) -> pd.DataFrame:
        """
        Ingests data by fetching it, saving it to a file, and returning it as a DataFrame.

        Returns:
            df (pd.DataFrame): The ingested data as a DataFrame.
        """
        try:
            # Fetch the data
            df = self.get_data()

            # Create the directory for the raw data file if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the data to a CSV file
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            return df
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys) from e

