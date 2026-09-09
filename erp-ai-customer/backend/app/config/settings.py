"""Application settings loaded from environment variables and the project .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

AppEnvironment = Literal["local", "development", "testing", "staging", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["text", "json"]
EmbeddingProviderName = Literal["sentence_transformers"]
EmbeddingDevice = Literal["auto", "cpu", "cuda"]


class Settings(BaseSettings):
    """Validated runtime configuration for the ERP customer service API."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "ERP AI Customer Service"
    app_version: str = "0.1.0"
    app_env: AppEnvironment = "local"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "text"
    knowledge_upload_max_mb: int = 10
    embedding_provider: EmbeddingProviderName = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: EmbeddingDevice = "auto"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True

    @property
    def knowledge_upload_max_bytes(self) -> int:
        """Return the upload limit in bytes."""
        return self.knowledge_upload_max_mb * 1024 * 1024

    @field_validator("knowledge_upload_max_mb")
    @classmethod
    def validate_upload_limit(cls, value: int) -> int:
        if not 1 <= value <= 100:
            raise ValueError("KNOWLEDGE_UPLOAD_MAX_MB must be between 1 and 100")
        return value

    @field_validator("embedding_provider", "embedding_device", mode="before")
    @classmethod
    def normalize_embedding_options(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("EMBEDDING_MODEL must not be blank")
        return normalized

    @field_validator("embedding_batch_size")
    @classmethod
    def validate_embedding_batch_size(cls, value: int) -> int:
        if not 1 <= value <= 512:
            raise ValueError("EMBEDDING_BATCH_SIZE must be between 1 and 512")
        return value

    @field_validator("api_v1_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        """Normalize the API prefix and reject the root path."""
        normalized = "/" + value.strip().strip("/")
        if normalized == "/":
            raise ValueError("API_V1_PREFIX must not be the root path")
        return normalized

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("log_format", mode="before")
    @classmethod
    def normalize_log_format(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""
    return Settings()
