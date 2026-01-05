"""Location API routes."""
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from datetime import datetime, timedelta
import json
import logging

from app.core.database import get_db
from app.core.redis_streams import stream_processor
from app.schemas.location import (
    LocationUpdateRequest, LocationResponse, LocationHistoryResponse,
    NearbyDriversResponse, WebSocketMessage, WebSocketLocationUpdate
)
from app.services.location_service import location_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/drivers", tags=["locations"])


@router.post("/{driver_id}/location", status_code=status.HTTP_202_ACCEPTED)
async def update_location(
    driver_id: int,
    location_data: LocationUpdateRequest
):
    """
    Update driver location (Ingestion endpoint).
    
    Accepts location update and adds to Redis Streams queue for reliable processing.
    Returns immediately with 202 Accepted.
    """
    try:
        # Add message to Redis Streams
        message_id = await stream_processor.add_location_message(
            driver_id=str(driver_id),
            latitude=location_data.latitude,
            longitude=location_data.longitude,
            speed=location_data.speed,
            heading=location_data.heading,
            accuracy=location_data.accuracy,
            order_id=location_data.order_id
        )
        
        logger.info(f"Queued location update for driver {driver_id}: {message_id}")
        
        return {
            "status": "accepted",
            "message_id": message_id,
            "driver_id": driver_id
        }
        
    except Exception as e:
        logger.error(f"Error queuing location update for driver {driver_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue location update: {str(e)}"
        )


@router.get("/{driver_id}/location", response_model=LocationResponse)
async def get_current_location(driver_id: int):
    """
    Get current driver location.
    
    Returns the most recent location from Redis cache.
    """
    location = await location_service.get_current_location(driver_id)
    
    if not location:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location not found for driver {driver_id}"
        )
    
    return LocationResponse(
        driver_id=driver_id,
        latitude=float(location.get("latitude", 0)),
        longitude=float(location.get("longitude", 0)),
        speed=float(location.get("speed")) if location.get("speed") else None,
        heading=float(location.get("heading")) if location.get("heading") else None,
        accuracy=float(location.get("accuracy")) if location.get("accuracy") else None,
        order_id=location.get("order_id"),
        timestamp=datetime.fromisoformat(location.get("timestamp", datetime.utcnow().isoformat()))
    )


@router.get("/{driver_id}/route", response_model=LocationHistoryResponse)
async def get_route_history(
    driver_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    db: AsyncSession = Depends(get_db)
):
    """
    Get driver route history.
    
    Returns historical location data from TimescaleDB.
    """
    locations = await location_service.get_location_history(
        db=db,
        driver_id=driver_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )
    
    if not locations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No location history found for driver {driver_id}"
        )
    
    location_responses = [
        LocationResponse(
            driver_id=loc.driver_id,
            latitude=loc.latitude,
            longitude=loc.longitude,
            speed=loc.speed,
            heading=loc.heading,
            accuracy=loc.accuracy,
            order_id=loc.order_id,
            timestamp=loc.time
        )
        for loc in locations
    ]
    
    start = locations[-1].time if locations else datetime.utcnow()
    end = locations[0].time if locations else datetime.utcnow()
    
    return LocationHistoryResponse(
        driver_id=driver_id,
        locations=location_responses,
        start_time=start,
        end_time=end
    )


@router.get("/nearby", response_model=NearbyDriversResponse)
async def get_nearby_drivers(
    lat: float,
    lng: float,
    radius: float = 1000,  # meters
    limit: int = 50
):
    """
    Get nearby drivers using geospatial query.
    
    Uses Redis GEO commands for fast location-based search.
    """
    drivers = await location_service.get_nearby_drivers(
        latitude=lat,
        longitude=lng,
        radius=radius,
        limit=limit
    )
    
    return NearbyDriversResponse(
        drivers=drivers,
        center_latitude=lat,
        center_longitude=lng,
        radius=radius
    )


@router.get("/orders/{order_id}/driver-location", response_model=LocationResponse)
async def get_driver_location_by_order(order_id: str):
    """
    Get driver location by order ID.
    
    Finds the driver associated with an order and returns their current location.
    """
    # This is a simplified implementation
    # In production, you'd query order service or maintain order-driver mapping
    
    # For now, search through recent locations
    # In production, maintain order_id -> driver_id mapping in Redis
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Order-based location lookup not yet implemented"
    )


@router.websocket("/orders/{order_id}/driver-location")
async def websocket_driver_location(websocket: WebSocket, order_id: str):
    """
    WebSocket endpoint for real-time driver location updates.
    
    Customers connect to receive live location updates for their order's driver.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for order {order_id}")
    
    # Subscribe to Redis Pub/Sub channel for this order
    from app.core.config import settings
    import redis.asyncio as redis
    
    redis_client = await stream_processor.get_client()
    pubsub = redis_client.pubsub()
    
    # Subscribe to order-specific channel
    order_channel = f"{settings.PUBSUB_CHANNEL_PREFIX}:order:{order_id}"
    await pubsub.subscribe(order_channel)
    
    try:
        # Send initial message
        await websocket.send_json({
            "type": "connected",
            "order_id": order_id,
            "message": "Subscribed to driver location updates"
        })
        
        # Listen for location updates
        while True:
            # Check for WebSocket messages (for unsubscribe, etc.)
            try:
                # Non-blocking check for WebSocket messages
                data = await websocket.receive_text(timeout=0.1)
                message = json.loads(data)
                
                if message.get("action") == "unsubscribe":
                    break
                    
            except:
                # No WebSocket message, continue
                pass
            
            # Check for Redis Pub/Sub messages
            try:
                message = await pubsub.get_message(timeout=0.1)
                if message and message["type"] == "message":
                    location_data = json.loads(message["data"])
                    
                    # Send to WebSocket client
                    await websocket.send_json({
                        "type": "location_update",
                        "data": location_data
                    })
            except:
                # No Pub/Sub message, continue
                pass
            
            # Small sleep to prevent busy loop
            import asyncio
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for order {order_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection for order {order_id}: {e}")
    finally:
        await pubsub.unsubscribe(order_channel)
        await pubsub.close()
        await websocket.close()

