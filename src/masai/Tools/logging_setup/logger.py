import logging
import os
import sys

# Windows Console API for guaranteed colors
if os.name == 'nt':
    try:
        import ctypes

        # Windows console color constants
        FOREGROUND_BLUE = 0x01
        FOREGROUND_GREEN = 0x02
        FOREGROUND_RED = 0x04
        FOREGROUND_YELLOW = 0x06
        FOREGROUND_INTENSITY = 0x08
        FOREGROUND_WHITE = 0x07

        STD_OUTPUT_HANDLE = -11
        kernel32 = ctypes.windll.kernel32

        def set_console_color(color):
            """Set Windows console color"""
            try:
                handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
                kernel32.SetConsoleTextAttribute(handle, color)
                return True
            except:
                return False

        def reset_console_color():
            """Reset to default color"""
            set_console_color(FOREGROUND_WHITE)

        WINDOWS_COLORS_AVAILABLE = True

    except:
        WINDOWS_COLORS_AVAILABLE = False
else:
    WINDOWS_COLORS_AVAILABLE = False

# Fallback to colorama for non-Windows
if not WINDOWS_COLORS_AVAILABLE:
    try:
        from colorama import Fore, Style, init
        init(autoreset=True, convert=True)
        COLORAMA_AVAILABLE = True
    except ImportError:
        COLORAMA_AVAILABLE = False
else:
    COLORAMA_AVAILABLE = False

class WindowsColorHandler(logging.StreamHandler):
    """
    Custom handler that uses Windows Console API for guaranteed colors.
    """

    def emit(self, record):
        try:
            message = self.format(record)

            if WINDOWS_COLORS_AVAILABLE:
                # Set color based on log level using Windows API
                if record.levelno == logging.DEBUG:
                    color = FOREGROUND_BLUE | FOREGROUND_INTENSITY
                elif record.levelno == logging.INFO:
                    color = FOREGROUND_GREEN | FOREGROUND_INTENSITY
                elif record.levelno == logging.WARNING:
                    color = FOREGROUND_YELLOW | FOREGROUND_INTENSITY
                elif record.levelno == logging.ERROR:
                    color = FOREGROUND_RED | FOREGROUND_INTENSITY
                elif record.levelno == logging.CRITICAL:
                    color = FOREGROUND_RED | FOREGROUND_INTENSITY
                else:
                    color = FOREGROUND_WHITE

                # Apply color, write message, reset color
                set_console_color(color)
                self.stream.write(message + '\n')
                self.stream.flush()
                reset_console_color()

            elif COLORAMA_AVAILABLE:
                # Fallback to colorama
                if record.levelno == logging.DEBUG:
                    colored_message = Fore.BLUE + message + Style.RESET_ALL
                elif record.levelno == logging.INFO:
                    colored_message = Fore.GREEN + Style.BRIGHT + message + Style.RESET_ALL
                elif record.levelno == logging.WARNING:
                    colored_message = Fore.YELLOW + message + Style.RESET_ALL
                elif record.levelno == logging.ERROR:
                    colored_message = Fore.RED + message + Style.RESET_ALL
                elif record.levelno == logging.CRITICAL:
                    colored_message = Fore.RED + Style.BRIGHT + message + Style.RESET_ALL
                else:
                    colored_message = message

                self.stream.write(colored_message + '\n')
                self.stream.flush()
            else:
                # No colors available
                self.stream.write(message + '\n')
                self.stream.flush()

        except Exception:
            self.handleError(record)

class SimpleFormatter(logging.Formatter):
    """Simple formatter for the Windows color handler."""

    def format(self, record):
        return super().format(record)

# Singleton logger instance
_shared_logger = None

def setup_logger():
    """
    Returns the same logger instance across the entire project.
    Ensures no duplicates and consistent logging behavior.
    """
    global _shared_logger

    if _shared_logger is None:
        _shared_logger = logging.getLogger("mas_logger")

        # Check if handlers already exist to prevent duplicates
        if not _shared_logger.hasHandlers():
            _shared_logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all logs

            # Use the Windows color handler
            console_handler = WindowsColorHandler()
            console_handler.setLevel(logging.DEBUG)

            # Add the simple formatter to the handler
            console_handler.setFormatter(SimpleFormatter())

            # Add the handler to the logger
            _shared_logger.addHandler(console_handler)

    return _shared_logger