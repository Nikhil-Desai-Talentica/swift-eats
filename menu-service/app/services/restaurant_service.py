"""Restaurant service for business logic."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload, joinedload
from typing import List, Optional, Tuple
from app.models.restaurant import (
    Restaurant, Menu, MenuCategory, MenuItem,
    RestaurantStatus, CuisineType
)
from app.core.cache import cache_service
import logging

logger = logging.getLogger(__name__)


class RestaurantService:
    """Service for restaurant operations."""
    
    @staticmethod
    async def get_restaurant(
        db: AsyncSession,
        restaurant_id: int,
        use_cache: bool = True
    ) -> Optional[Restaurant]:
        """
        Get restaurant by ID with caching.
        
        Args:
            db: Database session
            restaurant_id: Restaurant identifier
            use_cache: Whether to use cache
            
        Returns:
            Restaurant if found, None otherwise
        """
        # Try cache first
        if use_cache:
            cached = await cache_service.get("restaurant", str(restaurant_id))
            if cached:
                logger.debug(f"Cache hit for restaurant {restaurant_id}")
                pass  # For now, fall through to DB query
        
        # Query database
        result = await db.execute(
            select(Restaurant).where(Restaurant.id == restaurant_id)
        )
        restaurant = result.scalar_one_or_none()
        
        # Cache the result
        if restaurant and use_cache:
            await RestaurantService._cache_restaurant(restaurant)
        
        return restaurant
    
    @staticmethod
    async def _cache_restaurant(restaurant: Restaurant):
        """Cache restaurant data."""
        data = {
            "id": restaurant.id,
            "name": restaurant.name,
            "description": restaurant.description,
            "image_url": restaurant.image_url,
            "address": restaurant.address,
            "city": restaurant.city,
            "state": restaurant.state,
            "zip_code": restaurant.zip_code,
            "latitude": restaurant.latitude,
            "longitude": restaurant.longitude,
            "status": restaurant.status.value,
            "cuisine_type": restaurant.cuisine_type.value,
            "rating": restaurant.rating,
            "review_count": restaurant.review_count,
            "max_capacity": restaurant.max_capacity,
            "current_capacity": restaurant.current_capacity,
            "operating_hours": restaurant.operating_hours,
        }
        await cache_service.set("restaurant", str(restaurant.id), data)
    
    @staticmethod
    async def list_restaurants(
        db: AsyncSession,
        cuisine_type: Optional[CuisineType] = None,
        status: Optional[RestaurantStatus] = None,
        min_rating: Optional[float] = None,
        city: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[Restaurant], int]:
        """
        List restaurants with filters and pagination.
        
        Args:
            db: Database session
            cuisine_type: Filter by cuisine type
            status: Filter by status
            min_rating: Minimum rating
            city: Filter by city
            page: Page number
            page_size: Items per page
            
        Returns:
            Tuple of (restaurants list, total count)
        """
        # Build query
        query = select(Restaurant)
        count_query = select(func.count()).select_from(Restaurant)
        
        conditions = []
        if cuisine_type:
            conditions.append(Restaurant.cuisine_type == cuisine_type)
        if status:
            conditions.append(Restaurant.status == status)
        if min_rating is not None:
            conditions.append(Restaurant.rating >= min_rating)
        if city:
            conditions.append(Restaurant.city.ilike(f"%{city}%"))
        
        if conditions:
            query = query.where(and_(*conditions))
            count_query = count_query.where(and_(*conditions))
        
        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination and ordering
        query = query.order_by(Restaurant.rating.desc(), Restaurant.review_count.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        restaurants = result.scalars().all()
        
        return list(restaurants), total
    
    @staticmethod
    async def get_restaurant_status(
        db: AsyncSession,
        restaurant_id: int,
        use_cache: bool = True
    ) -> Optional[dict]:
        """
        Get restaurant status with caching.
        
        Args:
            db: Database session
            restaurant_id: Restaurant identifier
            use_cache: Whether to use cache
            
        Returns:
            Status dict or None
        """
        # Try cache first
        if use_cache:
            cached = await cache_service.get("status", str(restaurant_id))
            if cached:
                logger.debug(f"Cache hit for status {restaurant_id}")
                return cached
        
        # Query database
        restaurant = await RestaurantService.get_restaurant(db, restaurant_id, use_cache=False)
        if not restaurant:
            return None
        
        status_data = {
            "restaurant_id": restaurant.id,
            "status": restaurant.status.value,
            "current_capacity": restaurant.current_capacity,
            "max_capacity": restaurant.max_capacity,
            "updated_at": restaurant.updated_at.isoformat() if restaurant.updated_at else None,
        }
        
        # Cache the result
        if use_cache:
            await cache_service.set("status", str(restaurant_id), status_data)
        
        return status_data
    
    @staticmethod
    async def update_restaurant_status(
        db: AsyncSession,
        restaurant_id: int,
        status: RestaurantStatus,
        current_capacity: Optional[int] = None
    ) -> Optional[Restaurant]:
        """
        Update restaurant status (operator endpoint).
        
        Args:
            db: Database session
            restaurant_id: Restaurant identifier
            status: New status
            current_capacity: New capacity (optional)
            
        Returns:
            Updated restaurant or None
        """
        restaurant = await RestaurantService.get_restaurant(db, restaurant_id, use_cache=False)
        if not restaurant:
            return None
        
        restaurant.status = status
        if current_capacity is not None:
            restaurant.current_capacity = current_capacity
        
        await db.commit()
        await db.refresh(restaurant)
        
        # Invalidate cache
        await cache_service.invalidate_restaurant(str(restaurant_id))
        
        logger.info(f"Updated restaurant {restaurant_id} status to {status}")
        return restaurant


class MenuService:
    """Service for menu operations."""
    
    @staticmethod
    async def get_menu(
        db: AsyncSession,
        restaurant_id: int,
        use_cache: bool = True
    ) -> Optional[Menu]:
        """
        Get menu for a restaurant with caching.
        
        Args:
            db: Database session
            restaurant_id: Restaurant identifier
            use_cache: Whether to use cache
            
        Returns:
            Menu if found, None otherwise
        """
        # Try cache first
        if use_cache:
            cached = await cache_service.get("menu", str(restaurant_id))
            if cached:
                logger.debug(f"Cache hit for menu {restaurant_id}")
                # In production, reconstruct from cache
                pass
        
        # Query database with eager loading
        result = await db.execute(
            select(Menu)
            .where(Menu.restaurant_id == restaurant_id)
            .options(
                selectinload(Menu.categories).selectinload(MenuCategory.items)
            )
        )
        menu = result.scalar_one_or_none()
        
        # Cache the result
        if menu and use_cache:
            await MenuService._cache_menu(menu)
        
        return menu
    
    @staticmethod
    async def _cache_menu(menu: Menu):
        """Cache menu data."""
        categories_data = []
        for category in menu.categories:
            items_data = []
            for item in category.items:
                items_data.append({
                    "id": item.id,
                    "name": item.name,
                    "description": item.description,
                    "price": item.price,
                    "image_url": item.image_url,
                    "available": item.available,
                    "in_stock": item.in_stock,
                    "dietary_info": item.dietary_info,
                    "preparation_time": item.preparation_time,
                })
            categories_data.append({
                "id": category.id,
                "name": category.name,
                "description": category.description,
                "display_order": category.display_order,
                "items": items_data,
            })
        
        data = {
            "restaurant_id": menu.restaurant_id,
            "categories": categories_data,
            "updated_at": menu.updated_at.isoformat() if menu.updated_at else None,
        }
        await cache_service.set("menu", str(menu.restaurant_id), data)
    
    @staticmethod
    async def update_item_availability(
        db: AsyncSession,
        restaurant_id: int,
        item_id: int,
        available: bool,
        in_stock: bool
    ) -> Optional[MenuItem]:
        """
        Update menu item availability (operator endpoint).
        
        Args:
            db: Database session
            restaurant_id: Restaurant identifier
            item_id: Menu item identifier
            available: Availability flag
            in_stock: Stock flag
            
        Returns:
            Updated menu item or None
        """
        # Verify restaurant exists
        restaurant = await RestaurantService.get_restaurant(db, restaurant_id, use_cache=False)
        if not restaurant:
            return None
        
        # Get menu item
        result = await db.execute(
            select(MenuItem)
            .join(MenuCategory)
            .join(Menu)
            .where(
                and_(
                    Menu.restaurant_id == restaurant_id,
                    MenuItem.id == item_id
                )
            )
        )
        item = result.scalar_one_or_none()
        
        if not item:
            return None
        
        # Update item
        item.available = available
        item.in_stock = in_stock
        
        await db.commit()
        await db.refresh(item)
        
        # Invalidate menu cache
        await cache_service.invalidate_menu(str(restaurant_id))
        
        logger.info(f"Updated menu item {item_id} availability for restaurant {restaurant_id}")
        return item
    
    @staticmethod
    async def update_items_availability(
        db: AsyncSession,
        restaurant_id: int,
        updates: List[dict]
    ) -> List[MenuItem]:
        """
        Update multiple menu items availability (operator endpoint).
        
        Args:
            db: Database session
            restaurant_id: Restaurant identifier
            updates: List of updates with item_id, available, in_stock
            
        Returns:
            List of updated menu items
        """
        # Verify restaurant exists
        restaurant = await RestaurantService.get_restaurant(db, restaurant_id, use_cache=False)
        if not restaurant:
            return []
        
        item_ids = [update["item_id"] for update in updates]
        
        # Get menu items
        result = await db.execute(
            select(MenuItem)
            .join(MenuCategory)
            .join(Menu)
            .where(
                and_(
                    Menu.restaurant_id == restaurant_id,
                    MenuItem.id.in_(item_ids)
                )
            )
        )
        items = result.scalars().all()
        
        # Create update map
        update_map = {update["item_id"]: update for update in updates}
        
        # Update items
        updated_items = []
        for item in items:
            if item.id in update_map:
                update = update_map[item.id]
                item.available = update["available"]
                item.in_stock = update["in_stock"]
                updated_items.append(item)
        
        await db.commit()
        
        # Refresh all updated items
        for item in updated_items:
            await db.refresh(item)
        
        # Invalidate menu cache
        await cache_service.invalidate_menu(str(restaurant_id))
        
        logger.info(f"Updated {len(updated_items)} menu items for restaurant {restaurant_id}")
        return updated_items

