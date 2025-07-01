import logging
import os
import datetime

def setup_logger(name, log_file=None, level=logging.DEBUG):
    """
    Set up a logger with consistent formatting and handlers.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file. If None, uses name_timestamp.log
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate default log filename with timestamp if not provided
    if log_file is None:
        log_file = f"logs/{name}.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers to prevent duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger