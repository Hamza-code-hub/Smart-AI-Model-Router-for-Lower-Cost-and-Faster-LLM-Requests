from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_version: str = "0.1.0"

    database_url: str = "postgresql+asyncpg://gateway:gateway@localhost:5432/gateway"
    database_pool_size: int = 10

    redis_url: str = "redis://localhost:6379"

    config_dir: Path = Field(default=Path("config"))

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    admin_api_key: str = "admin-secret-change-me"

    # Phase 2.2 — local Ollama. Default targets host Ollama from inside the
    # gateway container via Docker Desktop's host.docker.internal alias.
    ollama_base_url: str = "http://host.docker.internal:11434"
    # Bumped vs OpenAI/Anthropic because local Llama on CPU can be slow on
    # first call (model load). Tier-3 classifier reads keep max_tokens small.
    ollama_timeout_seconds: float = 120.0

    log_level: str = "INFO"
    log_json: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_yaml(filename: str) -> dict[str, Any]:
    settings = get_settings()
    path = settings.config_dir / filename
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}
