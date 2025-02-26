import logging
from colorama import Fore, Style, init

# Initialize Colorama (for Windows compatibility)
init(autoreset=True)

class SimpleColorFormatter(logging.Formatter):
    """
    Custom logging formatter to display only the message with colors based on log level.
    """
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # Get the color based on log level
        color = self.COLORS.get(record.levelno, "")
        reset = Style.RESET_ALL  # Reset color after the message

        # Only display the log message with color
        return f"{color}{record.msg}{reset}"

# Set up the logger with the custom formatter
def setup_logger():
    logger = logging.getLogger("custom_logger")

    # Check if handlers already exist to prevent duplicates
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all logs

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Add the custom formatter to the handler
        console_handler.setFormatter(SimpleColorFormatter())

        # Add the handler to the logger
        logger.addHandler(console_handler)
    
    return logger