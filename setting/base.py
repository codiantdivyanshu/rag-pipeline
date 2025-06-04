from pydantic import BaseSettings

class AppSettings(BaseSettings):
    groq_api_key: str
    env: str = "dev"

    class Config:
        env_file = ".env"
