"""Location service for business logic."""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.models.location import DriverLocation
from app.core.config import settings

logger = logging.getLogger(__name__)


class LocationService:
    """Service for location operations."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize with optional Redis client."""
        self.redis_client = redis_client
    
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self.redis_client:
            from app.core.redis_streams import stream_processor
            self.redis_client = await stream_processor.get_client()
        return self.redis_client
    
    async def update_current_location(
        self,
        driver_id: int,
        latitude: float,
        longitude: float,
        speed: Optional[float] = None,
        heading: Optional[float] = None,
        accuracy: Optional[float] = None,
        order_id: Optional[str] = None
    ) -> bool:
        """
        Update current location in Redis.
        
        Args:
            driver_id: Driver identifier
            latitude: Latitude
            longitude: Longitude
            speed: Speed in km/h
            heading: Heading in degrees
            accuracy: Accuracy in meters
            order_id: Associated order ID
            
        Returns:
            True if successful
        """
        client = await self.get_redis_client()
        
        try:
            # Store current location
            location_key = f"driver:location:{driver_id}"
            location_data = {
                "driver_id": str(driver_id),
                "latitude": str(latitude),
                "longitude": str(longitude),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            if speed is not None:
                location_data["speed"] = str(speed)
            if heading is not None:
                location_data["heading"] = str(heading)
            if accuracy is not None:
                location_data["accuracy"] = str(accuracy)
            if order_id:
                location_data["order_id"] = order_id
            
            # Set with TTL (using JSON for proper serialization)
            import json
            await client.setex(
                location_key,
                settings.LOCATION_TTL,
                json.dumps(location_data)
            )
            
            # Update GEO index for geospatial queries
            await client.geoadd(
                settings.GEO_KEY,
                (longitude, latitude, driver_id)
            )
            
            logger.debug(f"Updated current location for driver {driver_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating current location for driver {driver_id}: {e}")
            return False
    
    async def get_current_location(self, driver_id: int) -> Optional[Dict[str, Any]]:
        """
        Get current location from Redis.
        
        Args:
            driver_id: Driver identifier
            
        Returns:
            Location data or None
        """
        client = await self.get_redis_client()
        
        try:
            location_key = f"driver:location:{driver_id}"
            data = await client.get(location_key)
            
            if data:
                # Parse location data (JSON)
                import json
                try:
                    location_data = json.loads(data)
                    return location_data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse location data for driver {driver_id}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current location for driver {driver_id}: {e}")
            return None
    
    async def store_historical_location(
        self,
        db: AsyncSession,
        driver_id: int,
        latitude: float,
        longitude: float,
        speed: Optional[float] = None,
        heading: Optional[float] = None,
        accuracy: Optional[float] = None,
        order_id: Optional[str] = None
    ) -> DriverLocation:
        """
        Store location in TimescaleDB for historical analysis.
        
        Args:
            db: Database session
            driver_id: Driver identifier
            latitude: Latitude
            longitude: Longitude
            speed: Speed in km/h
            heading: Heading in degrees
            accuracy: Accuracy in meters
            order_id: Associated order ID
            
        Returns:
            Created location record
        """
        location = DriverLocation(
            driver_id=driver_id,
            latitude=latitude,
            longitude=longitude,
            speed=speed,
            heading=heading,
            accuracy=accuracy,
            order_id=order_id,
            time=datetime.utcnow()
        )
        
        db.add(location)
        await db.commit()
        await db.refresh(location)
        
        logger.debug(f"Stored historical location for driver {driver_id}")
        return location
    
    async def get_location_history(
        self,
        db: AsyncSession,
        driver_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[DriverLocation]:
        """
        Get location history for a driver.
        
        Args:
            db: Database session
            driver_id: Driver identifier
            start_time: Start time (default: 1 hour ago)
            end_time: End time (default: now)
            limit: Maximum number of records
            
        Returns:
            List of location records
        """
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=1)
        
        query = select(DriverLocation).where(
            and_(
                DriverLocation.driver_id == driver_id,
                DriverLocation.time >= start_time,
                DriverLocation.time <= end_time
            )
        ).order_by(DriverLocation.time.desc()).limit(limit)
        
        result = await db.execute(query)
        locations = result.scalars().all()
        
        return list(locations)
    
    async def publish_location_update(
        self,
        driver_id: int,
        latitude: float,
        longitude: float,
        speed: Optional[float] = None,
        heading: Optional[float] = None,
        order_id: Optional[str] = None
    ) -> bool:
        """
        Publish location update to Redis Pub/Sub for real-time distribution.
        
        Args:
            driver_id: Driver identifier
            latitude: Latitude
            longitude: Longitude
            speed: Speed in km/h
            heading: Heading in degrees
            order_id: Associated order ID
            
        Returns:
            True if successful
        """
        client = await self.get_redis_client()
        
        try:
            # Publish to driver-specific channel
            channel = f"{settings.PUBSUB_CHANNEL_PREFIX}:{driver_id}"
            
            # Also publish to order-specific channel if order_id provided
            message = {
                "driver_id": driver_id,
                "latitude": latitude,
                "longitude": longitude,
                "speed": speed,
                "heading": heading,
                "timestamp": datetime.utcnow().isoformat(),
                "order_id": order_id
            }
            
            import json
            await client.publish(channel, json.dumps(message))
            
            if order_id:
                order_channel = f"{settings.PUBSUB_CHANNEL_PREFIX}:order:{order_id}"
                await client.publish(order_channel, json.dumps(message))
            
            logger.debug(f"Published location update for driver {driver_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing location update for driver {driver_id}: {e}")
            return False
    
    async def get_nearby_drivers(
        self,
        latitude: float,
        longitude: float,
        radius: float = 1000,  # meters
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get nearby drivers using Redis GEO commands.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius: Search radius in meters
            limit: Maximum number of drivers
            
        Returns:
            List of nearby drivers with distance
        """
        client = await self.get_redis_client()
        
        try:
            # Use GEORADIUS to find nearby drivers
            results = await client.georadius(
                name=settings.GEO_KEY,
                longitude=longitude,
                latitude=latitude,
                radius=radius,
                unit="m",
                withdist=True,  # Include distance
                withcoord=True,  # Include coordinates
                count=limit,
                sort="ASC"  # Closest first
            )
            
            drivers = []
            for result in results:
                driver_id, distance, (lng, lat) = result
                drivers.append({
                    "driver_id": int(driver_id),
                    "latitude": lat,
                    "longitude": lng,
                    "distance": distance
                })
            
            return drivers
            
        except Exception as e:
            logger.error(f"Error getting nearby drivers: {e}")
            return []


# Global location service instance
location_service = LocationService()

