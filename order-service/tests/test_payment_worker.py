"""Tests for payment worker."""
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.workers.payment_worker import (
    process_payment_with_retry,
    process_order_from_queue,
    get_redis_client
)
from app.models.order import OrderStatus, PaymentStatus
from app.services.order_service import OrderService
from app.schemas.order import OrderCreate


@pytest.mark.asyncio
async def test_process_payment_with_retry_success(db_session: AsyncSession):
    """Test successful payment processing with retry."""
    # Create an order
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
    )
    order = await OrderService.create_order(db_session, order_data)
    await OrderService.update_order_status(db_session, order.id, OrderStatus.PAYMENT_PROCESSING)
    
    # Mock successful payment
    with patch('app.workers.payment_worker.PaymentService.process_payment_with_circuit_breaker') as mock_payment:
        mock_payment.return_value = {
            "success": True,
            "transaction_id": "txn_123",
            "amount": order.total_amount,
            "status": "completed"
        }
        
        result = await process_payment_with_retry(
            db_session, order.id, order.total_amount, order.customer_id, max_attempts=3
        )
        
        assert result is True
        mock_payment.assert_called_once()
        
        # Verify order status updated
        updated_order = await OrderService.get_order(db_session, order.id)
        assert updated_order.status == OrderStatus.CONFIRMED


@pytest.mark.asyncio
async def test_process_payment_with_retry_failure(db_session: AsyncSession):
    """Test payment processing with retry that eventually fails."""
    # Create an order
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
    )
    order = await OrderService.create_order(db_session, order_data)
    await OrderService.update_order_status(db_session, order.id, OrderStatus.PAYMENT_PROCESSING)
    
    # Mock failed payment
    with patch('app.workers.payment_worker.PaymentService.process_payment_with_circuit_breaker') as mock_payment:
        mock_payment.side_effect = Exception("Payment failed")
        
        result = await process_payment_with_retry(
            db_session, order.id, order.total_amount, order.customer_id, max_attempts=2
        )
        
        assert result is False
        assert mock_payment.call_count == 2  # Should retry
        
        # Verify order status updated to failed
        updated_order = await OrderService.get_order(db_session, order.id)
        assert updated_order.status == OrderStatus.PAYMENT_FAILED


@pytest.mark.asyncio
async def test_process_payment_with_retry_success_after_failure(db_session: AsyncSession):
    """Test payment processing that succeeds after initial failure."""
    # Create an order
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
    )
    order = await OrderService.create_order(db_session, order_data)
    await OrderService.update_order_status(db_session, order.id, OrderStatus.PAYMENT_PROCESSING)
    
    # Mock payment that fails first time, succeeds second time
    with patch('app.workers.payment_worker.PaymentService.process_payment_with_circuit_breaker') as mock_payment:
        mock_payment.side_effect = [
            Exception("Payment failed"),
            {
                "success": True,
                "transaction_id": "txn_123",
                "amount": order.total_amount,
                "status": "completed"
            }
        ]
        
        result = await process_payment_with_retry(
            db_session, order.id, order.total_amount, order.customer_id, max_attempts=3
        )
        
        assert result is True
        assert mock_payment.call_count == 2
        
        # Verify order status updated to confirmed
        updated_order = await OrderService.get_order(db_session, order.id)
        assert updated_order.status == OrderStatus.CONFIRMED


@pytest.mark.asyncio
async def test_process_order_from_queue_success(db_session: AsyncSession):
    """Test processing order from queue successfully."""
    # Create an order
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
    )
    order = await OrderService.create_order(db_session, order_data)
    await OrderService.update_order_status(db_session, order.id, OrderStatus.PAYMENT_PROCESSING)
    await db_session.commit()
    
    queue_data = {
        "order_id": order.id,
        "customer_id": order.customer_id,
        "amount": float(order.total_amount)
    }
    
    # Mock the database session context manager
    from unittest.mock import MagicMock, AsyncMock
    
    mock_session_context = MagicMock()
    mock_session_context.__aenter__ = AsyncMock(return_value=db_session)
    mock_session_context.__aexit__ = AsyncMock(return_value=None)
    
    with patch('app.workers.payment_worker.AsyncSessionLocal', return_value=mock_session_context), \
         patch('app.workers.payment_worker.process_payment_with_retry', return_value=True) as mock_retry:
        
        await process_order_from_queue(queue_data)
        
        # Verify payment was attempted
        assert mock_retry.called
        call_args = mock_retry.call_args
        assert call_args[0][1] == order.id
        assert call_args[0][2] == order.total_amount


@pytest.mark.asyncio
async def test_process_order_from_queue_order_not_found():
    """Test processing order from queue when order doesn't exist."""
    queue_data = {
        "order_id": 99999,
        "customer_id": "customer_123",
        "amount": 12.99
    }
    
    # Should not raise exception, just log error
    await process_order_from_queue(queue_data)


@pytest.mark.asyncio
async def test_process_order_from_queue_wrong_status(db_session: AsyncSession):
    """Test processing order from queue when order is in wrong status."""
    # Create an order with CONFIRMED status
    order_data = OrderCreate(
        customer_id="customer_123",
        items=[{"item_id": "item_001", "item_name": "Burger", "quantity": 1, "price": 12.99}]
    )
    order = await OrderService.create_order(db_session, order_data)
    await OrderService.update_order_status(db_session, order.id, OrderStatus.CONFIRMED)
    await db_session.commit()
    
    queue_data = {
        "order_id": order.id,
        "customer_id": order.customer_id,
        "amount": float(order.total_amount)
    }
    
    # Mock the database session context manager
    from unittest.mock import MagicMock, AsyncMock
    
    mock_session_context = MagicMock()
    mock_session_context.__aenter__ = AsyncMock(return_value=db_session)
    mock_session_context.__aexit__ = AsyncMock(return_value=None)
    
    with patch('app.workers.payment_worker.AsyncSessionLocal', return_value=mock_session_context), \
         patch('app.workers.payment_worker.process_payment_with_retry') as mock_retry:
        
        await process_order_from_queue(queue_data)
        # Should skip processing when order is not in PAYMENT_PROCESSING status
        mock_retry.assert_not_called()


def test_get_redis_client():
    """Test Redis client creation."""
    with patch('app.workers.payment_worker.settings') as mock_settings:
        mock_settings.REDIS_HOST = "localhost"
        mock_settings.REDIS_PORT = 6379
        mock_settings.REDIS_DB = 0
        
        client = get_redis_client()
        assert client is not None
        assert client.connection_pool.connection_kwargs['host'] == "localhost"
        assert client.connection_pool.connection_kwargs['port'] == 6379

