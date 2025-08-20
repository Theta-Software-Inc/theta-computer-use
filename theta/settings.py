from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="allow"
    )
    
    api_key: Optional[str] = Field(
        default=None, 
        alias="THETA_API_KEY", 
        validation_alias="THETA_API_KEY"
    )
    
    base_url: str = Field(
        default="https://cub.trythetasoftware.com",
        alias="THETA_BASE_URL",
        validation_alias="THETA_BASE_URL",
        description="Base URL for the Theta Computer Use Environments API"
    )

    openai_api_key: Optional[str] = Field(
        default=None,
        alias="OPENAI_API_KEY",
        validation_alias="OPENAI_API_KEY"
    )


settings = Settings()