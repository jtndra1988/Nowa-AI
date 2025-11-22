import os
from pathlib import Path
from functools import cached_property  # ✅ You forgot this import
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# --- Load .env from project root ---
ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
print(f"✅ Loaded .env from: {ENV_PATH}")


class Settings(BaseSettings):
    @cached_property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """Return SQLAlchemy-style database URI."""
        if hasattr(self, "DATABASE_URL") and self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    # --- Redis ---
    REDIS_HOST: str 
    REDIS_PORT: int
    # --- General ---
    ENVIRONMENT: str = Field(default="development")
    SECRET_KEY: str

    # --- Database ---
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    DATABASE_URL: str | None = None  # ✅ make optional for flexibility

    # --- Redis ---
    REDIS_HOST: str
    REDIS_PORT: int

    # --- Bybit API ---
    BYBIT_TESTNET_API_KEY: str
    BYBIT_TESTNET_API_SECRET: str
    USE_BINANCE_FOR_DATA: bool = True
    BINANCE_API_KEY:str
    BINANCE_API_SECRET:str

    # --- Trading ---
    TRADING_MODE: str

    # --- News & Sentiment APIs ---
    NEWSAPI_API_KEY: str
    SANTIMENT_API_KEY: str
    LUNARCRUSH_API_KEY: str
    CRYPTOPANIC_API_KEY: str
    COINMARKETCAP_API_KEY: str
    TELEGRAM_BOT_TOKEN:str
    TELEGRAM_CHAT_ID:str
    GEMINI_API_KEY: str
    PROMETHEUS_ENABLED: bool = False
    PROMETHEUS_PORT: int = 9090
    class Config:
        env_file = ENV_PATH
        env_file_encoding = "utf-8"

    @property
    def CELERY_BROKER_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    @property
    def CELERY_RESULT_BACKEND(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"
    
    @property
    def celery_broker_url(self) -> str:
        """Alias for lowercase access if your app.py uses settings.celery_broker_url"""
        return self.CELERY_BROKER_URL
# ✅ Instantiate settings at import
settings = Settings()
