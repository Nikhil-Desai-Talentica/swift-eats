"""Restaurant API routes."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import logging

from app.core.database import get_db
from app.schemas.restaurant import (
    RestaurantResponse, RestaurantListResponse, RestaurantStatusResponse,
    MenuResponse, MenuWithStatusResponse, UpdateRestaurantStatusRequest,
    UpdateRestaurantStatusResponse, UpdateMenuItemAvailabilityRequest,
    UpdateMenuItemsAvailabilityRequest, UpdateMenuItemAvailabilityResponse
)
from app.services.restaurant_service import RestaurantService, MenuService
from app.models.restaurant import RestaurantStatus, CuisineType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/restaurants", tags=["restaurants"])


@router.get("", response_model=RestaurantListResponse)
async def list_restaurants(
    cuisine_type: Optional[CuisineType] = Query(None, description="Filter by cuisine type"),
    status: Optional[RestaurantStatus] = Query(None, description="Filter by status"),
    min_rating: Optional[float] = Query(None, ge=0, le=5, description="Minimum rating"),
    city: Optional[str] = Query(None, description="Filter by city"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db)
):
    """
    List restaurants with optional filters.
    
    Supports filtering by cuisine type, status, minimum rating, and city.
    Results are paginated and ordered by rating.
    """
    restaurants, total = await RestaurantService.list_restaurants(
        db=db,
        cuisine_type=cuisine_type,
        status=status,
        min_rating=min_rating,
        city=city,
        page=page,
        page_size=page_size
    )
    
    return RestaurantListResponse(
        restaurants=restaurants,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{restaurant_id}", response_model=RestaurantResponse)
async def get_restaurant(
    restaurant_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get restaurant details by ID.
    
    Uses caching for optimal performance.
    """
    restaurant = await RestaurantService.get_restaurant(db, restaurant_id)
    
    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant {restaurant_id} not found"
        )
    
    return restaurant


@router.get("/{restaurant_id}/menu", response_model=MenuResponse)
async def get_menu(
    restaurant_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get restaurant menu.
    
    Returns full menu with categories and items.
    Uses caching for optimal performance.
    """
    # Verify restaurant exists
    restaurant = await RestaurantService.get_restaurant(db, restaurant_id)
    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant {restaurant_id} not found"
        )
    
    menu = await MenuService.get_menu(db, restaurant_id)
    
    if not menu:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Menu not found for restaurant {restaurant_id}"
        )
    
    return menu


@router.get("/{restaurant_id}/status", response_model=RestaurantStatusResponse)
async def get_status(
    restaurant_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get restaurant current status.
    
    Returns current status, capacity, and last update time.
    Uses caching with short TTL for frequently changing data.
    """
    status_data = await RestaurantService.get_restaurant_status(db, restaurant_id)
    
    if not status_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant {restaurant_id} not found"
        )
    
    return RestaurantStatusResponse(**status_data)


@router.get("/{restaurant_id}/menu-with-status", response_model=MenuWithStatusResponse)
async def get_menu_with_status(
    restaurant_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get restaurant menu and status in a single request.
    
    Optimized endpoint that fetches both menu and status in parallel.
    Uses caching for optimal performance.
    """
    # Fetch restaurant, menu, and status
    restaurant = await RestaurantService.get_restaurant(db, restaurant_id)
    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant {restaurant_id} not found"
        )
    
    menu = await MenuService.get_menu(db, restaurant_id)
    if not menu:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Menu not found for restaurant {restaurant_id}"
        )
    
    status_data = await RestaurantService.get_restaurant_status(db, restaurant_id)
    if not status_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Status not found for restaurant {restaurant_id}"
        )
    
    return MenuWithStatusResponse(
        restaurant=restaurant,
        menu=menu,
        status=RestaurantStatusResponse(**status_data)
    )


# Operator Update Endpoints
@router.put(
    "/{restaurant_id}/status",
    response_model=UpdateRestaurantStatusResponse,
    status_code=status.HTTP_200_OK
)
async def update_restaurant_status(
    restaurant_id: int,
    request: UpdateRestaurantStatusRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Update restaurant status (Operator endpoint).
    
    Restaurant operators can update their status (open/closed/busy/maintenance)
    and current capacity. This triggers cache invalidation.
    """
    restaurant = await RestaurantService.update_restaurant_status(
        db=db,
        restaurant_id=restaurant_id,
        status=request.status,
        current_capacity=request.current_capacity
    )
    
    if not restaurant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Restaurant {restaurant_id} not found"
        )
    
    return UpdateRestaurantStatusResponse(
        success=True,
        message="Restaurant status updated successfully",
        restaurant_id=restaurant_id,
        status=restaurant.status
    )


@router.put(
    "/{restaurant_id}/menu/items/{item_id}/availability",
    response_model=UpdateMenuItemAvailabilityResponse,
    status_code=status.HTTP_200_OK
)
async def update_menu_item_availability(
    restaurant_id: int,
    item_id: int,
    request: UpdateMenuItemAvailabilityRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Update menu item availability (Operator endpoint).
    
    Restaurant operators can mark items as available/unavailable or in/out of stock.
    This triggers cache invalidation for the menu.
    """
    item = await MenuService.update_item_availability(
        db=db,
        restaurant_id=restaurant_id,
        item_id=item_id,
        available=request.available,
        in_stock=request.in_stock
    )
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Menu item {item_id} not found for restaurant {restaurant_id}"
        )
    
    return UpdateMenuItemAvailabilityResponse(
        success=True,
        message="Menu item availability updated successfully",
        restaurant_id=restaurant_id,
        item_id=item_id,
        available=item.available,
        in_stock=item.in_stock
    )


@router.put(
    "/{restaurant_id}/menu/items/availability",
    response_model=dict,
    status_code=status.HTTP_200_OK
)
async def update_menu_items_availability(
    restaurant_id: int,
    request: UpdateMenuItemsAvailabilityRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Update multiple menu items availability (Operator endpoint).
    
    Restaurant operators can update availability for multiple items at once.
    Useful when dishes run out of stock or become available.
    This triggers cache invalidation for the menu.
    """
    updates = [
        {
            "item_id": item.item_id,
            "available": item.available,
            "in_stock": item.in_stock
        }
        for item in request.items
    ]
    
    updated_items = await MenuService.update_items_availability(
        db=db,
        restaurant_id=restaurant_id,
        updates=updates
    )
    
    if not updated_items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No menu items found for restaurant {restaurant_id}"
        )
    
    return {
        "success": True,
        "message": f"Updated {len(updated_items)} menu items",
        "restaurant_id": restaurant_id,
        "updated_items": [
            {
                "item_id": item.id,
                "available": item.available,
                "in_stock": item.in_stock
            }
            for item in updated_items
        ]
    }

