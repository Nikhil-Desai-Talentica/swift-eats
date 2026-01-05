"""Pydantic schemas for location API."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class LocationUpdateRequest(BaseModel):
    """Request schema for driver location update."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    speed: Optional[float] = Field(None, ge=0, description="Speed in km/h")
    heading: Optional[float] = Field(None, ge=0, le=360, description="Heading in degrees")
    accuracy: Optional[float] = Field(None, ge=0, description="Accuracy in meters")
    order_id: Optional[str] = Field(None, description="Associated order ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "speed": 45.5,
                "heading": 180,
                "accuracy": 10.0,
                "order_id": "order_123"
            }
        }


class LocationResponse(BaseModel):
    """Response schema for current location."""
    driver_id: int
    latitude: float
    longitude: float
    speed: Optional[float]
    heading: Optional[float]
    accuracy: Optional[float]
    order_id: Optional[str]
    timestamp: datetime


class LocationHistoryResponse(BaseModel):
    """Response schema for location history."""
    driver_id: int
    locations: list[LocationResponse]
    start_time: datetime
    end_time: datetime


class NearbyDriversResponse(BaseModel):
    """Response schema for nearby drivers."""
    drivers: list[dict]
    center_latitude: float
    center_longitude: float
    radius: float  # meters


class WebSocketMessage(BaseModel):
    """WebSocket message schema."""
    action: str = Field(..., description="Action: subscribe, unsubscribe")
    order_id: Optional[str] = Field(None, description="Order ID to track")


class WebSocketLocationUpdate(BaseModel):
    """WebSocket location update message."""
    driver_id: int
    latitude: float
    longitude: float
    speed: Optional[float]
    heading: Optional[float]
    timestamp: datetime
    order_id: Optional[str]

