"""Pydantic schemas for restaurant and menu API."""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from app.models.restaurant import RestaurantStatus, CuisineType


# Restaurant Schemas
class RestaurantBase(BaseModel):
    """Base restaurant schema."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    image_url: Optional[str] = None
    address: str
    city: str
    state: Optional[str] = None
    zip_code: Optional[str] = None
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    cuisine_type: CuisineType
    max_capacity: Optional[int] = Field(None, gt=0)


class RestaurantResponse(BaseModel):
    """Restaurant response schema."""
    id: int
    name: str
    description: Optional[str]
    image_url: Optional[str]
    address: str
    city: str
    state: Optional[str]
    zip_code: Optional[str]
    latitude: float
    longitude: float
    status: RestaurantStatus
    cuisine_type: CuisineType
    rating: float
    review_count: int
    max_capacity: Optional[int]
    current_capacity: int
    operating_hours: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RestaurantListResponse(BaseModel):
    """Restaurant list response schema."""
    restaurants: List[RestaurantResponse]
    total: int
    page: int
    page_size: int


class RestaurantStatusResponse(BaseModel):
    """Restaurant status response schema."""
    restaurant_id: int
    status: RestaurantStatus
    current_capacity: int
    max_capacity: Optional[int]
    updated_at: datetime


# Menu Schemas
class MenuItemResponse(BaseModel):
    """Menu item response schema."""
    id: int
    name: str
    description: Optional[str]
    price: float
    image_url: Optional[str]
    available: bool
    in_stock: bool
    dietary_info: Optional[str]  # JSON string
    preparation_time: Optional[int]
    
    class Config:
        from_attributes = True


class MenuCategoryResponse(BaseModel):
    """Menu category response schema."""
    id: int
    name: str
    description: Optional[str]
    display_order: int
    items: List[MenuItemResponse]
    
    class Config:
        from_attributes = True


class MenuResponse(BaseModel):
    """Menu response schema."""
    restaurant_id: int
    categories: List[MenuCategoryResponse]
    updated_at: datetime
    
    class Config:
        from_attributes = True


class MenuWithStatusResponse(BaseModel):
    """Combined menu and status response."""
    restaurant: RestaurantResponse
    menu: MenuResponse
    status: RestaurantStatusResponse


# Operator Update Schemas
class UpdateRestaurantStatusRequest(BaseModel):
    """Request to update restaurant status."""
    status: RestaurantStatus
    current_capacity: Optional[int] = Field(None, ge=0)


class UpdateMenuItemAvailabilityRequest(BaseModel):
    """Request to update menu item availability."""
    item_id: int
    available: bool
    in_stock: bool


class UpdateMenuItemsAvailabilityRequest(BaseModel):
    """Request to update multiple menu items availability."""
    items: List[UpdateMenuItemAvailabilityRequest] = Field(..., min_items=1)


class UpdateRestaurantStatusResponse(BaseModel):
    """Response for status update."""
    success: bool
    message: str
    restaurant_id: int
    status: RestaurantStatus


class UpdateMenuItemAvailabilityResponse(BaseModel):
    """Response for menu item availability update."""
    success: bool
    message: str
    restaurant_id: int
    item_id: int
    available: bool
    in_stock: bool

