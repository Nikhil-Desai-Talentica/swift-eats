"""Tests for database models."""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.order import Order, OrderItem, PaymentAttempt, OrderStatus, PaymentStatus
from app.services.order_service import OrderService
from app.schemas.order import OrderCreate


@pytest.mark.asyncio
async def test_order_model(db_session: AsyncSession):
    """Test Order model creation and relationships."""
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[
            {"item_id": "item_001", "item_name": "Burger", "quantity": 2, "price": 12.99}
        ]
    )
    
    order = await OrderService.create_order(db_session, order_data)
    
    assert isinstance(order, Order)
    assert order.customer_id == "customer_123"
    assert order.total_amount == 2 * 12.99
    assert order.status == OrderStatus.PENDING
    assert len(order.items) == 1
    assert order.items[0].item_id == "item_001"


@pytest.mark.asyncio
async def test_order_item_model(db_session: AsyncSession):
    """Test OrderItem model."""
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[
            {"item_id": "item_001", "item_name": "Burger", "quantity": 2, "price": 12.99}
        ]
    )
    
    order = await OrderService.create_order(db_session, order_data)
    
    item = order.items[0]
    assert isinstance(item, OrderItem)
    assert item.order_id == order.id
    assert item.item_id == "item_001"
    assert item.item_name == "Burger"
    assert item.quantity == 2
    assert item.price == 12.99


@pytest.mark.asyncio
async def test_payment_attempt_model(db_session: AsyncSession):
    """Test PaymentAttempt model."""
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
    )
    
    order = await OrderService.create_order(db_session, order_data)
    payment_attempt = await OrderService.create_payment_attempt(
        db_session, order.id, order.total_amount, attempt_number=1
    )
    
    assert isinstance(payment_attempt, PaymentAttempt)
    assert payment_attempt.order_id == order.id
    assert payment_attempt.amount == order.total_amount
    assert payment_attempt.attempt_number == 1
    assert payment_attempt.status == PaymentStatus.PROCESSING
    
    # Test relationship
    assert payment_attempt.order.id == order.id


@pytest.mark.asyncio
async def test_order_status_enum():
    """Test OrderStatus enum values."""
    assert OrderStatus.PENDING.value == "pending"
    assert OrderStatus.PAYMENT_PROCESSING.value == "payment_processing"
    assert OrderStatus.CONFIRMED.value == "confirmed"
    assert OrderStatus.PAYMENT_FAILED.value == "payment_failed"
    assert OrderStatus.CANCELLED.value == "cancelled"


@pytest.mark.asyncio
async def test_payment_status_enum():
    """Test PaymentStatus enum values."""
    assert PaymentStatus.PENDING.value == "pending"
    assert PaymentStatus.PROCESSING.value == "processing"
    assert PaymentStatus.SUCCESS.value == "success"
    assert PaymentStatus.FAILED.value == "failed"


@pytest.mark.asyncio
async def test_order_cascade_delete(db_session: AsyncSession):
    """Test that order items are deleted when order is deleted."""
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[
            {"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}
        ]
    )
    
    order = await OrderService.create_order(db_session, order_data)
    order_id = order.id
    item_id = order.items[0].id
    
    # Delete order
    await db_session.delete(order)
    await db_session.commit()
    
    # Verify item is also deleted
    from sqlalchemy import select
    result = await db_session.execute(
        select(OrderItem).where(OrderItem.id == item_id)
    )
    item = result.scalar_one_or_none()
    assert item is None

