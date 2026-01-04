"""Tests for order API endpoints."""
import pytest
import json
from unittest.mock import patch, Mock
from httpx import AsyncClient

from app.models.order import OrderStatus


@pytest.mark.asyncio
async def test_create_order_success(client: AsyncClient, sample_order_data, mock_redis):
    """Test successful order creation."""
    with patch("app.api.routes.orders.get_or_create_redis_client", return_value=mock_redis):
        response = await client.post("/api/v1/orders", json=sample_order_data)
        
        assert response.status_code == 202
        data = response.json()
        assert data["customer_id"] == sample_order_data["customer_id"]
        assert data["status"] == OrderStatus.PAYMENT_PROCESSING.value
        assert len(data["items"]) == 2
        assert data["total_amount"] == (2 * 12.99) + (1 * 4.99)
        assert "id" in data
        assert "created_at" in data


@pytest.mark.asyncio
async def test_create_order_invalid_data(client: AsyncClient):
    """Test order creation with invalid data."""
    invalid_data = {
        "customer_id": "",  # Empty customer ID
        "items": []
    }
    
    response = await client.post("/api/v1/orders", json=invalid_data)
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_create_order_empty_items(client: AsyncClient):
    """Test order creation with empty items list."""
    invalid_data = {
        "customer_id": "customer_123",
        "items": []
    }
    
    response = await client.post("/api/v1/orders", json=invalid_data)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_order_redis_unavailable(client: AsyncClient, sample_order_data):
    """Test order creation when Redis is unavailable."""
    with patch("app.api.routes.orders.get_or_create_redis_client", return_value=None):
        response = await client.post("/api/v1/orders", json=sample_order_data)
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_order_success(client: AsyncClient, sample_order_data, mock_redis, db_session):
    """Test getting an order by ID."""
    from app.services.order_service import OrderService
    from app.schemas.order import OrderCreate
    
    # Create order first
    order_data = OrderCreate(**sample_order_data)
    order = await OrderService.create_order(db_session, order_data)
    
    with patch("app.api.routes.orders.get_or_create_redis_client", return_value=mock_redis):
        response = await client.get(f"/api/v1/orders/{order.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == order.id
        assert data["customer_id"] == sample_order_data["customer_id"]


@pytest.mark.asyncio
async def test_get_order_not_found(client: AsyncClient):
    """Test getting a non-existent order."""
    response = await client.get("/api/v1/orders/99999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_orders(client: AsyncClient, sample_order_data, mock_redis, db_session):
    """Test listing orders."""
    from app.services.order_service import OrderService
    from app.schemas.order import OrderCreate
    
    # Create a few orders
    for i in range(3):
        order_data = OrderCreate(**sample_order_data)
        order_data.customer_id = f"customer_{i}"
        await OrderService.create_order(db_session, order_data)
    
    with patch("app.api.routes.orders.get_or_create_redis_client", return_value=mock_redis):
        response = await client.get("/api/v1/orders?page=1&page_size=10")
        
        assert response.status_code == 200
        data = response.json()
        assert "orders" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert len(data["orders"]) == 3


@pytest.mark.asyncio
async def test_list_orders_with_filters(client: AsyncClient, sample_order_data, mock_redis, db_session):
    """Test listing orders with customer_id filter."""
    from app.services.order_service import OrderService
    from app.schemas.order import OrderCreate
    
    # Create orders for different customers
    order_data = OrderCreate(**sample_order_data)
    order_data.customer_id = "customer_filter"
    await OrderService.create_order(db_session, order_data)
    
    order_data.customer_id = "other_customer"
    await OrderService.create_order(db_session, order_data)
    
    with patch("app.api.routes.orders.get_or_create_redis_client", return_value=mock_redis):
        response = await client.get("/api/v1/orders?customer_id=customer_filter")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["orders"][0]["customer_id"] == "customer_filter"


@pytest.mark.asyncio
async def test_list_orders_pagination(client: AsyncClient, sample_order_data, mock_redis, db_session):
    """Test order list pagination."""
    from app.services.order_service import OrderService
    from app.schemas.order import OrderCreate
    
    # Create 5 orders
    for i in range(5):
        order_data = OrderCreate(**sample_order_data)
        order_data.customer_id = f"customer_{i}"
        await OrderService.create_order(db_session, order_data)
    
    with patch("app.api.routes.orders.get_or_create_redis_client", return_value=mock_redis):
        # First page
        response = await client.get("/api/v1/orders?page=1&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["orders"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2
        
        # Second page
        response = await client.get("/api/v1/orders?page=2&page_size=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["orders"]) == 2


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test root endpoint."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data

