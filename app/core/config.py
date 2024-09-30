from  pydantic_settings import BaseSettings

class Settings(BaseSettings):
    file_path: str = "transaction_data/creditcard.csv"
    model_cache_expiry: int = 300 #cache expiry time in seconds

    class Config:
        env_file = ".env"

settings = Settings()