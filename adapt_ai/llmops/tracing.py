"""LangSmith tracing activation.

Call setup_tracing() once at application startup. It reads settings and sets
the LangChain environment variables that langsmith SDK picks up automatically.
If LANGSMITH_TRACING is False or no API key is configured, this is a no-op.
"""
from __future__ import annotations
import logging
import os

logger = logging.getLogger(__name__)


def setup_tracing() -> None:
    """Activate LangSmith tracing when configured via settings."""
    from adapt_ai.config import settings  # local import avoids circular deps

    if not settings.langsmith_tracing:
        return

    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is not set — tracing disabled")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
