import sys


import sys

def error_message_detail(error, error_details):
    """
    Generate an error message with detailed information.
    
    Args:
        error: The error that occurred.
        error_details: Details about the error.
        
    Returns:
        The error message with detailed information.
    """
    # Check if error_details is empty
    if error_details == "":
        return f"Error message {str(error)}"
    
    # Get information about the error
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Generate the error message
    return "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )


class CustomException(Exception):
    def __init__(self, error_message: str, error_details: sys):
        """
        Initialize the class with an error message and error details.

        Args:
            error_message (str): The error message.
            error_details (sys): The error details.

        Returns:
            None
        """
        super().__init__(error_message)  # Call the parent's __init__ method
        self.error_message = error_message_detail(
            error=error_message, error_details=error_details)

    def __str__(self) -> str:
        """
        Returns a string representation of the object.
        """
        return self.error_message
