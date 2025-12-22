import sys
import traceback
from src.logger.logging import get_logger

logger = get_logger()


def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        f"Error occurred in file [{file_name}] "
        f"at line [{line_number}] "
        f"with message [{str(error)}]"
    )

    return error_message


class ZomatoException(Exception):
    """
    Base exception class for the entire project.
    All custom exceptions should inherit from this.
    """

    def __init__(self, error: Exception, error_detail: sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

        # Log full traceback
        logger.error(self.error_message)
        logger.error("Traceback:\n" + traceback.format_exc())

    def __str__(self):
        return self.error_message


class DatabaseError(ZomatoException):
    """Raised for MongoDB / data ingestion related errors"""
    pass


class ModelTrainingError(ZomatoException):
    """Raised for model training related errors"""
    pass
