"""Secure structured logging."""
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class SecureJSONFormatter(logging.Formatter):
    """JSON formatter that redacts sensitive data."""

    SENSITIVE_KEYS = {
        'api_key', 'token', 'password', 'secret', 'auth',
        'apikey', 'api_token', 'bearer', 'credentials'
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with sensitive data redacted."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add extra fields (redact sensitive)
        if hasattr(record, 'extra_data'):
            for key, value in record.extra_data.items():
                if any(sensitive in key.lower() for sensitive in self.SENSITIVE_KEYS):
                    log_data[key] = '***REDACTED***'
                else:
                    log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logger(
    name: str = 'adapt-ai',
    log_file: Path = Path('./logs/adapt-ai.log'),
    level: str = 'INFO'
) -> logging.Logger:
    """Setup secure logger with JSON formatting.

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create log directory
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # File handler with JSON formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(SecureJSONFormatter())
    logger.addHandler(file_handler)

    # Console handler (simpler format for readability)
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    return logger


# Global logger instance
logger = setup_logger()
