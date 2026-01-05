"""Configuration settings for the menu service."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "Menu & Restaurant Browse Service"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Database Settings
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/swifteats_menu"
    
    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 1  # Different DB from order-service
    REDIS_SOCKET_TIMEOUT: Optional[int] = None  # None for blocking operations
    
    # Cache Settings
    CACHE_RESTAURANT_TTL: int = 3600  # 1 hour (fallback, but event-based invalidation primary)
    CACHE_MENU_TTL: int = 1800  # 30 minutes (fallback)
    CACHE_STATUS_TTL: int = 30  # 30 seconds (fallback, but event-based invalidation primary)
    CACHE_PREFIX: str = "menu_service"
    
    # Performance Settings
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    ENABLE_RESPONSE_COMPRESSION: bool = True
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

