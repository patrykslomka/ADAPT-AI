"""Centralised settings - reads from .env."""
from typing import Optional
from pathlib import Path
from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: SecretStr = Field(..., alias="ANTHROPIC_API_KEY")
    model_name: str = Field("claude-haiku-4-5-20251001", alias="MODEL_NAME")
    max_tokens: int = Field(2048, alias="MAX_TOKENS_PER_REQUEST")
    temperature: float = Field(0.3, alias="MODEL_TEMPERATURE")

    # LLM provider selection (anthropic | openai_compatible)
    llm_provider: str = Field("anthropic", alias="LLM_PROVIDER")
    llm_base_url: str = Field("http://localhost:11434/v1", alias="LLM_BASE_URL")
    llm_api_key: Optional[str] = Field(None, alias="LLM_API_KEY")

    # Redis (session management)
    redis_url: str = Field("redis://localhost:6379", alias="REDIS_URL")
    redis_fallback_memory: bool = Field(True, alias="REDIS_FALLBACK_MEMORY")

    # PostgreSQL (domain data)
    postgres_url: str = Field(
        "postgresql+asyncpg://adapt:adapt@localhost:5432/adapt_ai",
        alias="POSTGRES_URL",
    )
    postgres_fallback_json: bool = Field(True, alias="POSTGRES_FALLBACK_JSON")

    # ChromaDB (vector store - reuses existing seeded collection)
    chroma_persist_dir: str = Field("./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field("clinical_knowledge", alias="CHROMA_COLLECTION")

    # Domain data paths
    data_dir: Path = Field(Path("./data"), alias="DATA_DIR")
    regulations_dir: Path = Field(
        Path("./adapt_ai/domain/regulations"), alias="REGULATIONS_DIR"
    )
    profiles_dir: Path = Field(
        Path("./adapt_ai/domain/profiles"), alias="PROFILES_DIR"
    )

    # RAT configuration
    rat_max_steps: int = Field(3, alias="RAT_MAX_STEPS")
    rag_n_results: int = Field(5, alias="RAG_N_RESULTS")

    # Neo4j (domain ontology)
    neo4j_uri: str = Field("bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field("neo4j", alias="NEO4J_USER")
    neo4j_password: Optional[SecretStr] = Field(None, alias="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", alias="NEO4J_DATABASE")

    # LangSmith (optional tracing)
    langsmith_api_key: Optional[SecretStr] = Field(None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field("adapt-ai", alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(False, alias="LANGSMITH_TRACING")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


settings = Settings()
