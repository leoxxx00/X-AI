from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str
    REDIS_URL: str
    WORKER_POLL_SECONDS: int = 3
    OUTPUT_DIR: str = "/tmp/outputs"

    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


settings = Settings()