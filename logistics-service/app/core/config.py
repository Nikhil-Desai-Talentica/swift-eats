"""Configuration settings for the logistics service."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "Real-Time Logistics & Analytics Service"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Database Settings (TimescaleDB)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/swifteats_logistics"
    
    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 2  # Different DB from other services
    REDIS_SOCKET_TIMEOUT: Optional[int] = None
    
    # Redis Streams Settings
    STREAM_NAME: str = "driver-locations"
    CONSUMER_GROUP: str = "location-processors"
    CONSUMER_NAME: str = "processor-1"  # Unique per worker instance
    STREAM_BATCH_SIZE: int = 100  # Read batch size
    STREAM_BLOCK_TIME: int = 1000  # Block time in milliseconds
    
    # Processing Settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5  # seconds
    PENDING_MESSAGE_TIMEOUT: int = 60000  # milliseconds (1 minute)
    DEAD_LETTER_STREAM: str = "driver-locations-dlq"
    
    # Current Location Storage (Redis)
    LOCATION_TTL: int = 300  # 5 minutes TTL for current locations
    GEO_KEY: str = "drivers:geo"  # Redis GEO key
    
    # Pub/Sub Settings
    PUBSUB_CHANNEL_PREFIX: str = "driver:location"
    
    # Performance Settings
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    WORKER_CONCURRENCY: int = 10  # Concurrent processing tasks
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

