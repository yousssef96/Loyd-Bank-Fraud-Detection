import sys
from loguru import logger
from datetime import datetime
from pathlib import Path


LOG_FILE = Path("logs") /f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}.log"

# This clears any default handlers (like the default console output) 
# so you can start with a clean slate and apply your own custom configuration.
logger.remove()

# This tells the logger to send all messages at the INFO level (and above) to the console (stderr).
logger.add(sys.stderr, level="INFO")

logger.add(
    LOG_FILE,
    rotation="10 MB", # Automatically closes the current file and starts a new one when it hits 10 MB.
    retention="7 days", # Automatically deletes log files older than one week to save disk space.
    compression="zip",
    level="INFO", # Only logs at the INFO level or higher are saved to the file.
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} - {message}"
)