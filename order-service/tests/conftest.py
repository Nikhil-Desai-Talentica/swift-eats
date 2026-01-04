"""Pytest configuration and fixtures."""
import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from httpx import AsyncClient
from unittest.mock import Mock, patch

from app.core.database import Base, get_db
from app.main import app
from app.core.config import settings

# Test database URL (in-memory SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session_maker = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client."""
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock_redis = Mock()
    mock_redis.ping.return_value = True
    mock_redis.lpush.return_value = 1
    mock_redis.brpop.return_value = None  # Default: no data
    mock_redis.decode_responses = True
    return mock_redis


@pytest.fixture
def sample_order_data():
    """Sample order data for testing."""
    return {
        "customer_id": "customer_123",
        "items": [
            {
                "item_id": "item_001",
                "item_name": "Burger",
                "quantity": 2,
                "price": 12.99
            },
            {
                "item_id": "item_002",
                "item_name": "Fries",
                "quantity": 1,
                "price": 4.99
            }
        ]
    }

