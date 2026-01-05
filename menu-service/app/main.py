"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging

from app.core.config import settings
from app.core.database import init_db
from app.api.routes import restaurants

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
    description="High-performance menu and restaurant browse service"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip compression middleware for performance
if settings.ENABLE_RESPONSE_COMPRESSION:
    app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(restaurants.router, prefix=settings.API_PREFIX)


@app.on_event("startup")
async def startup_event():
    """Initialize database and cache on startup."""
    logger.info("Starting up menu service...")
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
    
    # Check cache availability
    from app.core.cache import cache_service
    if cache_service.is_available():
        logger.info("Redis cache is available")
    else:
        logger.warning("Redis cache is not available - performance may be degraded")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.core.cache import cache_service
    
    cache_status = "available" if cache_service.is_available() else "unavailable"
    
    return {
        "status": "healthy",
        "service": "menu-service",
        "cache": cache_status
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Menu & Restaurant Browse Service",
        "version": settings.API_VERSION,
        "docs": "/docs"
    }

