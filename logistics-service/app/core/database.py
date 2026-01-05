"""Database connection and session management for TimescaleDB."""
import logging
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import settings

logger = logging.getLogger(__name__)

# Create async engine with connection pooling
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base = declarative_base()


async def get_db() -> AsyncSession:
    """Dependency for getting database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables and TimescaleDB hypertable."""
    from sqlalchemy import text
    
    async with engine.begin() as conn:
        # Create tables
        await conn.run_sync(Base.metadata.create_all)
        
        # Enable TimescaleDB extension if available
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
        except Exception as e:
            logger.warning(f"TimescaleDB extension not available: {e}")
        
        # Create TimescaleDB hypertable if it doesn't exist
        # Note: This requires TimescaleDB extension to be installed
        try:
            await conn.execute(
                text("""
                    SELECT create_hypertable('driver_locations', 'time', 
                        if_not_exists => TRUE);
                """)
            )
            logger.info("TimescaleDB hypertable created")
        except Exception as e:
            # If hypertable already exists or TimescaleDB not installed, continue
            logger.info(f"Hypertable creation skipped: {e}")

