"""Secure settings management using Pydantic."""
from pydantic_settings import BaseSettings
from pydantic import SecretStr, Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys (SecretStr prevents logging)
    anthropic_api_key: SecretStr = Field(..., alias='ANTHROPIC_API_KEY')
    langsmith_api_key: Optional[SecretStr] = Field(None, alias='LANGSMITH_API_KEY')
    langsmith_project: Optional[str] = Field('adapt-ai-poc', alias='LANGSMITH_PROJECT')

    # Model Configuration
    model_name: str = Field('claude-haiku-4-5-20251001', alias='MODEL_NAME')
    max_tokens: int = Field(4000, alias='MAX_TOKENS_PER_REQUEST')
    temperature: float = Field(0.7, alias='MODEL_TEMPERATURE')

    # Rate Limiting
    max_requests_per_minute: int = Field(60, alias='MAX_REQUESTS_PER_MINUTE')
    cost_alert_threshold: float = Field(10.0, alias='COST_ALERT_THRESHOLD')

    # Database Paths
    vector_db_path: Path = Field(Path('./data/chroma_db'), alias='VECTOR_DB_PATH')
    session_db_path: Path = Field(Path('./data/sessions.db'), alias='SESSION_DB_PATH')
    metrics_db_path: Path = Field(Path('./data/metrics.db'), alias='METRICS_DB_PATH')

    # Logging
    log_level: str = Field('INFO', alias='LOG_LEVEL')
    log_file: Path = Field(Path('./logs/adapt-ai.log'), alias='LOG_FILE')

    # Embedding Configuration (using sentence-transformers instead of OpenAI)
    embedding_model: str = Field('all-MiniLM-L6-v2', alias='EMBEDDING_MODEL')

    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'case_sensitive': False,
        'extra': 'ignore'
    }


# Global settings instance
settings = Settings()
