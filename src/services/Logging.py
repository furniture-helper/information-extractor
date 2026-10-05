import logging


COLOURS = {
    logging.DEBUG:    "\033[36m",   # Cyan
    logging.INFO:     "\033[32m",   # Green
    logging.WARNING:  "\033[33m",   # Yellow
    logging.ERROR:    "\033[31m",   # Red
    logging.CRITICAL: "\033[35m",   # Magenta
}
RESET = "\033[0m"


class _ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = COLOURS.get(record.levelno, RESET)
        message = super().format(record)
        return f"{colour}{message}{RESET}"


class LoggingService:

    @staticmethod
    def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
        logger = logging.getLogger(name)

        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                _ColouredFormatter(
                    fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)

        logger.setLevel(level)
        return logger


