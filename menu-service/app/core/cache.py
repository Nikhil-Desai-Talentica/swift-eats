"""Cache service with event-based invalidation."""
import json
import logging
import asyncio
from typing import Optional, Any, Dict, List
import redis
from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Service for managing Redis cache with event-based invalidation."""
    
    def __init__(self):
        """Initialize Redis connection."""
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                socket_keepalive=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _get_key(self, key_type: str, identifier: str) -> str:
        """Generate cache key."""
        return f"{settings.CACHE_PREFIX}:{key_type}:{identifier}"
    
    async def get(self, key_type: str, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get cached data.
        
        Args:
            key_type: Type of cache key (restaurant, menu, status)
            identifier: Identifier (restaurant_id, etc.)
            
        Returns:
            Cached data or None
        """
        if not self.redis_client:
            return None
        
        try:
            key = self._get_key(key_type, identifier)
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting cache key {key_type}:{identifier}: {e}")
            return None
    
    async def set(
        self,
        key_type: str,
        identifier: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set cached data.
        
        Args:
            key_type: Type of cache key
            identifier: Identifier
            data: Data to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            key = self._get_key(key_type, identifier)
            
            # Use default TTL if not specified
            if ttl is None:
                ttl_map = {
                    "restaurant": settings.CACHE_RESTAURANT_TTL,
                    "menu": settings.CACHE_MENU_TTL,
                    "status": settings.CACHE_STATUS_TTL,
                }
                ttl = ttl_map.get(key_type, 3600)
            
            serialized = json.dumps(data)
            self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key_type}:{identifier}: {e}")
            return False
    
    async def delete(self, key_type: str, identifier: str) -> bool:
        """
        Delete cached data (invalidate).
        
        Args:
            key_type: Type of cache key
            identifier: Identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            key = self._get_key(key_type, identifier)
            deleted = self.redis_client.delete(key)
            logger.info(f"Invalidated cache: {key_type}:{identifier}")
            return deleted > 0
        except Exception as e:
            logger.error(f"Error deleting cache key {key_type}:{identifier}: {e}")
            return False
    
    async def invalidate_restaurant(self, restaurant_id: str) -> bool:
        """
        Invalidate all cache entries for a restaurant.
        
        Args:
            restaurant_id: Restaurant identifier
            
        Returns:
            True if successful
        """
        # Invalidate restaurant, menu, and status
        results = await asyncio.gather(
            self.delete("restaurant", restaurant_id),
            self.delete("menu", restaurant_id),
            self.delete("status", restaurant_id),
            return_exceptions=True
        )
        
        success = all(r is True for r in results if not isinstance(r, Exception))
        logger.info(f"Invalidated all cache for restaurant {restaurant_id}")
        return success
    
    async def invalidate_menu(self, restaurant_id: str) -> bool:
        """
        Invalidate menu cache for a restaurant.
        
        Args:
            restaurant_id: Restaurant identifier
            
        Returns:
            True if successful
        """
        return await self.delete("menu", restaurant_id)
    
    async def invalidate_status(self, restaurant_id: str) -> bool:
        """
        Invalidate status cache for a restaurant.
        
        Args:
            restaurant_id: Restaurant identifier
            
        Returns:
            True if successful
        """
        return await self.delete("status", restaurant_id)
    
    async def get_multi(
        self,
        key_type: str,
        identifiers: List[str]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get multiple cache entries at once.
        
        Args:
            key_type: Type of cache key
            identifiers: List of identifiers
            
        Returns:
            Dictionary mapping identifier to cached data (or None)
        """
        if not self.redis_client or not identifiers:
            return {id: None for id in identifiers}
        
        try:
            keys = [self._get_key(key_type, id) for id in identifiers]
            values = self.redis_client.mget(keys)
            
            result = {}
            for identifier, value in zip(identifiers, values):
                if value:
                    try:
                        result[identifier] = json.loads(value)
                    except json.JSONDecodeError:
                        result[identifier] = None
                else:
                    result[identifier] = None
            
            return result
        except Exception as e:
            logger.error(f"Error getting multiple cache keys: {e}")
            return {id: None for id in identifiers}
    
    def is_available(self) -> bool:
        """Check if Redis cache is available."""
        if not self.redis_client:
            return False
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False


# Global cache service instance
cache_service = CacheService()

