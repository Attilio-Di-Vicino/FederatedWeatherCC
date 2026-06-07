import logging
from colorama import Fore, Style, init

# Define colors for each log level
COLORS = {
    'DEBUG': Fore.GREEN + Style.BRIGHT,
    'WARNING': Fore.YELLOW + Style.BRIGHT,
    'ERROR': Fore.RED + Style.BRIGHT,
    'CRITICAL': Fore.MAGENTA + Style.BRIGHT,
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLORS.get(record.levelname, "")
        log_message = super().format(record)
        return f"{log_color}{log_message}{Style.RESET_ALL}"

class Logger:
    def __init__(self, level=logging.INFO):
        init(autoreset=True)
        self.logger = logging.getLogger("ColorLogger")
        self.logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter("[%(levelname)s] %(message)s"))
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    def set_level(self, level=logging.INFO):
        self.logger.setLevel(level)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)