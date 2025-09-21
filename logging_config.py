import logging
import sys
from logging.handlers import RotatingFileHandler

# Define log format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s"

def setup_logging():
    """
    Set up logging for the application.
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create handlers
    # Console handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # File handler (for errors)
    file_handler = RotatingFileHandler(
        "app.log", 
        maxBytes=1024 * 1024 * 5,  # 5 MB
        backupCount=5
    )
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    
    # Add handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize and configure logging
logger = setup_logging()

# Log unhandled exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Log unhandled exceptions.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception
