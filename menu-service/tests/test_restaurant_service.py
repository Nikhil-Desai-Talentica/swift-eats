import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.restaurant_service import RestaurantService, MenuService
from app.models.restaurant import (
    Restaurant,
    Menu,
    MenuCategory,
    MenuItem,
    RestaurantStatus,
    CuisineType,
)


@pytest.mark.asyncio
async def test_get_restaurant(db_session: AsyncSession):
    r = Restaurant(
        name="R1",
        description=None,
        image_url=None,
        address="Addr",
        city="City",
        state=None,
        zip_code=None,
        latitude=0.0,
        longitude=0.0,
        status=RestaurantStatus.OPEN,
        cuisine_type=CuisineType.OTHER,
    )
    db_session.add(r)
    await db_session.commit()
    await db_session.refresh(r)

    got = await RestaurantService.get_restaurant(db_session, r.id)
    assert got is not None
    assert got.id == r.id


@pytest.mark.asyncio
async def test_list_restaurants_filters(db_session: AsyncSession):
    for i in range(3):
        db_session.add(
            Restaurant(
                name=f"R{i}",
                description=None,
                image_url=None,
                address="Addr",
                city="City" if i < 2 else "Other",
                state=None,
                zip_code=None,
                latitude=0.0,
                longitude=0.0,
                status=RestaurantStatus.OPEN if i < 2 else RestaurantStatus.CLOSED,
                cuisine_type=CuisineType.OTHER,
                rating=4.5 if i == 0 else 3.0,
            )
        )
    await db_session.commit()

    rs, total = await RestaurantService.list_restaurants(
        db_session, status=RestaurantStatus.OPEN, city="City"
    )
    assert total == 2
    assert all(r.status == RestaurantStatus.OPEN for r in rs)


@pytest.mark.asyncio
async def test_update_items_availability(db_session: AsyncSession):
    r = Restaurant(
        name="R",
        description=None,
        image_url=None,
        address="Addr",
        city="City",
        state=None,
        zip_code=None,
        latitude=0.0,
        longitude=0.0,
        status=RestaurantStatus.OPEN,
        cuisine_type=CuisineType.OTHER,
    )
    db_session.add(r)
    await db_session.flush()

    menu = Menu(restaurant_id=r.id)
    db_session.add(menu)
    await db_session.flush()

    cat = MenuCategory(menu_id=menu.id, name="Mains", description=None, display_order=0)
    db_session.add(cat)
    await db_session.flush()

    item = MenuItem(
        category_id=cat.id,
        name="Burger",
        description="",
        price=9.99,
        available=True,
        in_stock=True,
    )
    db_session.add(item)
    await db_session.commit()
    await db_session.refresh(item)

    updates = [{"item_id": item.id, "available": False, "in_stock": False}]
    updated_items = await MenuService.update_items_availability(db_session, r.id, updates)
    assert len(updated_items) == 1
    assert updated_items[0].available is False
    assert updated_items[0].in_stock is False