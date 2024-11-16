from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import NonNegativeInt
from typing import Literal

class Settings(BaseSettings):
    scheme: Literal["http", 'https'] = 'http'
    host: str = "127.0.0.1"
    port: NonNegativeInt = 7063
    
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.web"),
        env_prefix="BABAISWEB_",
        env_nested_delimiter="__",
        env_file_encoding="utf-8",
    )
    
settings = Settings()
print(f'{settings=}')