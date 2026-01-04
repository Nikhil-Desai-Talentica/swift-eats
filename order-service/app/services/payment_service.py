"""Payment service with mocked payment gateway."""
import asyncio
import random
import logging
from typing import Dict, Any
from app.core.config import settings
from app.core.circuit_breaker import payment_circuit_breaker

logger = logging.getLogger(__name__)


class PaymentService:
    """Service for processing payments through mocked gateway."""
    
    @staticmethod
    async def process_payment(order_id: int, amount: float, customer_id: str) -> Dict[str, Any]:
        """
        Process payment through mocked payment gateway.
        
        Args:
            order_id: Order identifier
            amount: Payment amount
            customer_id: Customer identifier
            
        Returns:
            Dict with payment result
            
        Raises:
            Exception: If payment fails or circuit breaker is open
        """
        # Simulate network latency
        latency = random.randint(
            settings.PAYMENT_MOCK_LATENCY_MIN,
            settings.PAYMENT_MOCK_LATENCY_MAX
        )
        await asyncio.sleep(latency / 1000.0)  # Convert to seconds
        
        # Simulate payment success/failure based on configured rate
        success = random.random() < settings.PAYMENT_MOCK_SUCCESS_RATE
        
        if success:
            logger.info(f"Payment successful for order {order_id}: ${amount:.2f}")
            return {
                "success": True,
                "transaction_id": f"txn_{order_id}_{random.randint(100000, 999999)}",
                "amount": amount,
                "status": "completed"
            }
        else:
            error_message = random.choice([
                "Insufficient funds",
                "Card declined",
                "Payment gateway timeout",
                "Invalid payment method"
            ])
            logger.warning(f"Payment failed for order {order_id}: {error_message}")
            raise Exception(f"Payment failed: {error_message}")
    
    @staticmethod
    async def process_payment_with_circuit_breaker(
        order_id: int,
        amount: float,
        customer_id: str
    ) -> Dict[str, Any]:
        """
        Process payment with circuit breaker protection.
        
        Args:
            order_id: Order identifier
            amount: Payment amount
            customer_id: Customer identifier
            
        Returns:
            Dict with payment result
            
        Raises:
            Exception: If payment fails or circuit breaker is open
        """
        try:
            return await payment_circuit_breaker.call_async(
                PaymentService.process_payment,
                order_id,
                amount,
                customer_id
            )
        except Exception as e:
            # Check if it's a circuit breaker error
            if "Circuit breaker is OPEN" in str(e):
                logger.error(f"Circuit breaker OPEN - payment service unavailable for order {order_id}")
            raise e

