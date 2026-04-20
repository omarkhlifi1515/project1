from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "ENISO Enterprise Assistant API"
    api_prefix: str = "/api"
    environment: str = "dev"
    debug: bool = False

    secret_key: str = Field(default="change-this-in-production", alias="SECRET_KEY")
    access_token_expire_minutes: int = 60 * 8
    algorithm: str = "HS256"

    database_url: str = "sqlite:///./app.db"

    processed_data_dir: str = str(Path("processed_data").resolve())
    chroma_dir: str = str(Path("chroma_multi").resolve())
    gold_chroma_dir: str = str(Path("chroma_gold_multi").resolve())
    eniso_raw_data_dir: str = str(
        Path("EnisoData1-20260418T142956Z-3-001/EnisoData1").resolve()
    )

    default_model_preset: str = Field(default="ollama-qwen2.5-3b", alias="DEFAULT_MODEL_PRESET")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:3b", alias="OLLAMA_MODEL")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
