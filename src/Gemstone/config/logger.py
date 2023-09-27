import logging
import os
import sys
from datetime import datetime

# Define the logging format
logging_format = "[%(asctime)s: %(lineno)d : %(name)s : %(levelname)s : %(module)s : %(message)s]"

# Create a unique log file name based on the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the log directory path
log_dir = os.path.join(os.getcwd(), 'logs', LOG_FILE)

# Create the full log file path
log_file_path = os.path.join(log_dir, LOG_FILE)

# Create the log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,

    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
