from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openrouter_api_key: str = Field(alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="mistralai/mistral-7b-instruct:free", alias="OPENROUTER_MODEL"
    )
    apify_token: str = Field(alias="APIFY_TOKEN")
    apify_youtube_actor_id: str = Field(default="streamers/youtube-scraper", alias="APIFY_YOUTUBE_ACTOR_ID")
    youtube_query: str = Field(
        default="indian stock market trading analysis", alias="YOUTUBE_QUERY"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    vector_dim: int = Field(default=384, alias="VECTOR_DIM")
    top_k_retrieval: int = Field(default=8, alias="TOP_K_RETRIEVAL")
    chat_cache_ttl_seconds: int = Field(default=21600, alias="CHAT_CACHE_TTL_SECONDS")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")


settings = Settings()
