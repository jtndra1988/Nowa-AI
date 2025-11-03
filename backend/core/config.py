import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# --- Load .env from project root ---
ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
print(f"✅ Loaded .env from: {ENV_PATH}")

class Settings(BaseSettings):
    # General
    ENVIRONMENT: str = Field(default="development")
    SECRET_KEY: str
   
    # Database
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    DATABASE_URL: str

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int

    # Bybit API
    BYBIT_TESTNET_API_KEY: str
    BYBIT_TESTNET_API_SECRET: str
    USE_BINANCE_FOR_DATA: bool = True
    # Trading
    TRADING_MODE: str
    
    # News & Sentiment APIs
    NEWSAPI_API_KEY: str
    
    # --- FIX: Add the new API keys here ---
    SANTIMENT_API_KEY: str
    LUNARCRUSH_API_KEY: str
    CRYPTOPANIC_API_KEY: str
    COINMARKETCAP_API_KEY: str
    # ------------------------------------

    class Config:
        env_file = ENV_PATH
        env_file_encoding = "utf-8"

# Instantiate settings
settings = Settings()