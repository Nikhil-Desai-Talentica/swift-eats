"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.core.database import init_db
from app.core.redis_streams import stream_processor
from app.api.routes import locations

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Real-time logistics and analytics service for driver location tracking"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(locations.router, prefix=settings.API_PREFIX)


@app.on_event("startup")
async def startup_event():
    """Initialize database and Redis on startup."""
    logger.info("Starting up logistics service...")
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
    
    # Ensure Redis consumer group exists
    try:
        await stream_processor.ensure_consumer_group()
        logger.info("Redis consumer group initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis consumer group: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.core.redis_streams import stream_processor
    
    redis_available = False
    pending_count = 0
    
    try:
        redis_client = await stream_processor.get_client()
        await redis_client.ping()
        redis_available = True
        pending_count = await stream_processor.get_pending_count()
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
    
    return {
        "status": "healthy",
        "service": "logistics-service",
        "redis": "available" if redis_available else "unavailable",
        "pending_messages": pending_count
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Real-Time Logistics & Analytics Service",
        "version": settings.API_VERSION,
        "docs": "/docs"
    }

