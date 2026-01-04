"""Tests for order service business logic."""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.order_service import OrderService
from app.schemas.order import OrderCreate
from app.models.order import OrderStatus, PaymentStatus


@pytest.mark.asyncio
async def test_create_order(db_session: AsyncSession):
    """Test order creation."""
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[
            {
                "item_id": "item_001",
                "item_name": "Burger",
                "quantity": 2,
                "price": 12.99
            }
        ]
    )
    
    order = await OrderService.create_order(db_session, order_data)
    
    assert order.id is not None
    assert order.customer_id == "customer_123"
    assert order.status == OrderStatus.PENDING
    assert order.total_amount == 2 * 12.99
    assert len(order.items) == 1
    assert order.items[0].item_id == "item_001"
    assert order.items[0].quantity == 2


@pytest.mark.asyncio
async def test_create_order_multiple_items(db_session: AsyncSession):
    """Test order creation with multiple items."""
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[
            {
                "item_id": "item_001",
                "item_name": "Burger",
                "quantity": 2,
                "price": 12.99
            },
            {
                "item_id": "item_002",
                "item_name": "Fries",
                "quantity": 1,
                "price": 4.99
            }
        ]
    )
    
    order = await OrderService.create_order(db_session, order_data)
    
    assert order.total_amount == (2 * 12.99) + (1 * 4.99)
    assert len(order.items) == 2


@pytest.mark.asyncio
async def test_get_order(db_session: AsyncSession):
    """Test getting an order by ID."""
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
    )
    
    created_order = await OrderService.create_order(db_session, order_data)
    retrieved_order = await OrderService.get_order(db_session, created_order.id)
    
    assert retrieved_order is not None
    assert retrieved_order.id == created_order.id
    assert retrieved_order.customer_id == "customer_123"
    assert len(retrieved_order.items) == 1


@pytest.mark.asyncio
async def test_get_order_not_found(db_session: AsyncSession):
    """Test getting a non-existent order."""
    order = await OrderService.get_order(db_session, 99999)
    assert order is None


@pytest.mark.asyncio
async def test_update_order_status(db_session: AsyncSession):
    """Test updating order status."""
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
    )
    
    order = await OrderService.create_order(db_session, order_data)
    assert order.status == OrderStatus.PENDING
    
    updated_order = await OrderService.update_order_status(
        db_session, order.id, OrderStatus.CONFIRMED
    )
    
    assert updated_order.status == OrderStatus.CONFIRMED


@pytest.mark.asyncio
async def test_get_orders_with_filters(db_session: AsyncSession):
    """Test getting orders with filters."""
    # Create orders for different customers
    for i in range(3):
        order_data = OrderCreate(
            customer_id=f"customer_{i % 2}",  # customer_0 and customer_1
            items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
        )
        await OrderService.create_order(db_session, order_data)
    
    # Filter by customer_id
    orders, total = await OrderService.get_orders(
        db_session, customer_id="customer_0"
    )
    assert total == 2
    assert all(order.customer_id == "customer_0" for order in orders)
    
    # Filter by status
    orders, total = await OrderService.get_orders(
        db_session, status=OrderStatus.PENDING
    )
    assert total == 3


@pytest.mark.asyncio
async def test_get_orders_pagination(db_session: AsyncSession):
    """Test order pagination."""
    # Create 5 orders
    for i in range(5):
        order_data = OrderCreate(
            customer_id=f"customer_{i}",
            items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
        )
        await OrderService.create_order(db_session, order_data)
    
    # First page
    orders, total = await OrderService.get_orders(db_session, page=1, page_size=2)
    assert len(orders) == 2
    assert total == 5
    
    # Second page
    orders, total = await OrderService.get_orders(db_session, page=2, page_size=2)
    assert len(orders) == 2
    assert total == 5


@pytest.mark.asyncio
async def test_create_payment_attempt(db_session: AsyncSession):
    """Test creating a payment attempt."""
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
    )
    
    order = await OrderService.create_order(db_session, order_data)
    payment_attempt = await OrderService.create_payment_attempt(
        db_session, order.id, order.total_amount, attempt_number=1
    )
    
    assert payment_attempt.id is not None
    assert payment_attempt.order_id == order.id
    assert payment_attempt.amount == order.total_amount
    assert payment_attempt.attempt_number == 1
    assert payment_attempt.status == PaymentStatus.PROCESSING


@pytest.mark.asyncio
async def test_update_payment_attempt(db_session: AsyncSession):
    """Test updating payment attempt status."""
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
    )
    
    order = await OrderService.create_order(db_session, order_data)
    payment_attempt = await OrderService.create_payment_attempt(
        db_session, order.id, order.total_amount, attempt_number=1
    )
    
    updated_attempt = await OrderService.update_payment_attempt(
        db_session, payment_attempt.id, PaymentStatus.SUCCESS
    )
    
    assert updated_attempt.status == PaymentStatus.SUCCESS


@pytest.mark.asyncio
async def test_update_payment_attempt_with_error(db_session: AsyncSession):
    """Test updating payment attempt with error message."""
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
    )
    
    order = await OrderService.create_order(db_session, order_data)
    payment_attempt = await OrderService.create_payment_attempt(
        db_session, order.id, order.total_amount, attempt_number=1
    )
    
    error_message = "Payment failed: Insufficient funds"
    updated_attempt = await OrderService.update_payment_attempt(
        db_session, payment_attempt.id, PaymentStatus.FAILED, error_message
    )
    
    assert updated_attempt.status == PaymentStatus.FAILED
    assert updated_attempt.error_message == error_message

