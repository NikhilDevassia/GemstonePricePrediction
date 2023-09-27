import os
import pickle
import sys

from Gemstone.config.exception import CustomException
from Gemstone.config.logger import logging


def load_object(file_path):
    """
    Load an object from a specific file path.

    Args:
        file_path (str): The path to the file containing the object.

    Returns:
        The loaded object.

    Raises:
        CustomException: If there is an error while loading the object.
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys) from e


def save_object(file_path, obj):
    """
    Save an object to a specific file path.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj: The object to be saved.

    Raises:
        CustomException: If there is an error while saving the object.

    Returns:
        None
    """
    try:
        # Create the directory path if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Save the object to the file
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        # Log the successful save
        logging.info(f"Object Saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e