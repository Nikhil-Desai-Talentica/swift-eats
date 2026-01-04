"""Tests for payment service."""
import pytest
from unittest.mock import patch, AsyncMock
from app.services.payment_service import PaymentService
from app.core.circuit_breaker import payment_circuit_breaker, CircuitState


@pytest.mark.asyncio
async def test_process_payment_success():
    """Test successful payment processing."""
    with patch('random.random', return_value=0.5), \
         patch('random.randint', return_value=100), \
         patch('asyncio.sleep', new_callable=AsyncMock):
        
        result = await PaymentService.process_payment(
            order_id=1,
            amount=25.99,
            customer_id="customer_123"
        )
        
        assert result["success"] is True
        assert "transaction_id" in result
        assert result["amount"] == 25.99
        assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_process_payment_failure():
    """Test failed payment processing."""
    with patch('random.random', return_value=0.99), \
         patch('random.randint', return_value=100), \
         patch('asyncio.sleep', new_callable=AsyncMock):
        
        with pytest.raises(Exception) as exc_info:
            await PaymentService.process_payment(
                order_id=1,
                amount=25.99,
                customer_id="customer_123"
            )
        
        assert "Payment failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_process_payment_with_circuit_breaker_success():
    """Test payment processing with circuit breaker (success)."""
    # Reset circuit breaker
    payment_circuit_breaker.reset()
    
    with patch('random.random', return_value=0.5), \
         patch('random.randint', return_value=100), \
         patch('asyncio.sleep', new_callable=AsyncMock):
        
        result = await PaymentService.process_payment_with_circuit_breaker(
            order_id=1,
            amount=25.99,
            customer_id="customer_123"
        )
        
        assert result["success"] is True
        assert payment_circuit_breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_process_payment_circuit_breaker_opens():
    """Test circuit breaker opening after failures."""
    # Reset circuit breaker
    payment_circuit_breaker.reset()
    
    # Simulate multiple failures
    with patch('random.random', return_value=0.99), \
         patch('random.randint', return_value=100), \
         patch('asyncio.sleep', new_callable=AsyncMock):
        
        # Cause failures to open circuit
        for _ in range(6):
            try:
                await PaymentService.process_payment_with_circuit_breaker(
                    order_id=1,
                    amount=25.99,
                    customer_id="customer_123"
                )
            except Exception:
                pass
        
        # Circuit should be open now
        assert payment_circuit_breaker.state == CircuitState.OPEN
        
        # Next call should fail immediately with circuit breaker error
        with pytest.raises(Exception) as exc_info:
            await PaymentService.process_payment_with_circuit_breaker(
                order_id=1,
                amount=25.99,
                customer_id="customer_123"
            )
        
        assert "Circuit breaker is OPEN" in str(exc_info.value)


@pytest.mark.asyncio
async def test_payment_latency_simulation():
    """Test that payment processing simulates latency."""
    import time
    
    with patch('random.random', return_value=0.5), \
         patch('random.randint', return_value=150):  # 150ms latency
        
        start_time = time.time()
        await PaymentService.process_payment(
            order_id=1,
            amount=25.99,
            customer_id="customer_123"
        )
        elapsed = time.time() - start_time
        
        # Should take at least 0.1 seconds (100ms) due to simulated latency
        assert elapsed >= 0.1


@pytest.mark.asyncio
async def test_payment_success_rate_configurable():
    """Test that payment success rate is configurable."""
    # Test with high success rate (should mostly succeed)
    with patch('app.services.payment_service.settings.PAYMENT_MOCK_SUCCESS_RATE', 0.99), \
         patch('random.random', return_value=0.5), \
         patch('random.randint', return_value=100), \
         patch('asyncio.sleep', new_callable=AsyncMock):
        
        result = await PaymentService.process_payment(
            order_id=1,
            amount=25.99,
            customer_id="customer_123"
        )
        assert result["success"] is True
    
    # Test with low success rate (should mostly fail)
    with patch('app.services.payment_service.settings.PAYMENT_MOCK_SUCCESS_RATE', 0.01), \
         patch('random.random', return_value=0.5), \
         patch('random.randint', return_value=100), \
         patch('asyncio.sleep', new_callable=AsyncMock):
        
        with pytest.raises(Exception):
            await PaymentService.process_payment(
                order_id=1,
                amount=25.99,
                customer_id="customer_123"
            )

