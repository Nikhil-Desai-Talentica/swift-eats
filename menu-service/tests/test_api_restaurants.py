import pytest
from httpx import AsyncClient

from app.models.restaurant import (
    Restaurant,
    Menu,
    MenuCategory,
    MenuItem,
    RestaurantStatus,
    CuisineType,
)


@pytest.mark.asyncio
async def test_list_restaurants_empty(client: AsyncClient):
    resp = await client.get("/api/v1/restaurants")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["restaurants"] == []


@pytest.mark.asyncio
async def test_get_restaurant_not_found(client: AsyncClient):
    resp = await client.get("/api/v1/restaurants/999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_restaurant_menu_status_combined_flow(client: AsyncClient, db_session):
    # Seed DB
    r = Restaurant(
        name="Test R",
        description="",
        image_url=None,
        address="Addr",
        city="City",
        state=None,
        zip_code=None,
        latitude=1.0,
        longitude=2.0,
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

    # GET /restaurants/{id}
    res = await client.get(f"/api/v1/restaurants/{r.id}")
    assert res.status_code == 200
    assert res.json()["name"] == "Test R"

    # GET /restaurants/{id}/menu
    res = await client.get(f"/api/v1/restaurants/{r.id}/menu")
    assert res.status_code == 200
    data = res.json()
    assert data["restaurant_id"] == r.id
    assert len(data["categories"]) == 1
    assert len(data["categories"][0]["items"]) == 1

    # GET /restaurants/{id}/status
    res = await client.get(f"/api/v1/restaurants/{r.id}/status")
    assert res.status_code == 200
    assert res.json()["status"] == RestaurantStatus.OPEN.value

    # GET /restaurants/{id}/menu-with-status
    res = await client.get(f"/api/v1/restaurants/{r.id}/menu-with-status")
    assert res.status_code == 200
    comb = res.json()
    assert comb["restaurant"]["id"] == r.id
    assert comb["menu"]["restaurant_id"] == r.id
    assert comb["status"]["restaurant_id"] == r.id