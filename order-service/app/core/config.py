"""Configuration settings for the order service."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_TITLE: str = "Order Service"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Database Settings
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/swifteats"
    
    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_QUEUE_NAME: str = "payment_queue"
    
    # Payment Gateway Settings (Mocked)
    PAYMENT_GATEWAY_URL: str = "http://mock-payment-gateway:8080"
    PAYMENT_TIMEOUT: int = 5  # seconds
    PAYMENT_RETRY_ATTEMPTS: int = 3
    PAYMENT_RETRY_DELAY: int = 1  # seconds (base delay)
    
    # Circuit Breaker Settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 60  # seconds
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: type = Exception
    
    # Payment Mock Settings
    PAYMENT_MOCK_SUCCESS_RATE: float = 0.95  # 95% success rate
    PAYMENT_MOCK_LATENCY_MIN: int = 50  # milliseconds
    PAYMENT_MOCK_LATENCY_MAX: int = 200  # milliseconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

