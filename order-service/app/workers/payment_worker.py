"""Payment processing worker."""
import asyncio
import json
import redis
import logging
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.services.payment_service import PaymentService
from app.services.order_service import OrderService
from app.models.order import OrderStatus, PaymentStatus

logger = logging.getLogger(__name__)

def get_redis_client():
    """Get Redis client with proper timeout settings for blocking operations."""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=None,  # None allows blocking operations like brpop to work properly
        socket_keepalive=True,
        health_check_interval=30
    )

# Redis connection
try:
    redis_client = get_redis_client()
    # Test connection
    redis_client.ping()
    logger.info("Connected to Redis")
except (redis.ConnectionError, redis.TimeoutError) as e:
    logger.error(f"Failed to connect to Redis: {e}")
    raise


async def process_payment_with_retry(
    db: AsyncSession,
    order_id: int,
    amount: float,
    customer_id: str,
    max_attempts: int = None
) -> bool:
    """
    Process payment with retry logic.
    
    Args:
        db: Database session
        order_id: Order identifier
        amount: Payment amount
        customer_id: Customer identifier
        max_attempts: Maximum retry attempts
        
    Returns:
        True if payment successful, False otherwise
    """
    max_attempts = max_attempts or settings.PAYMENT_RETRY_ATTEMPTS
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Create payment attempt record
            payment_attempt = await OrderService.create_payment_attempt(
                db, order_id, amount, attempt
            )
            
            # Process payment with circuit breaker
            result = await PaymentService.process_payment_with_circuit_breaker(
                order_id, amount, customer_id
            )
            
            # Update payment attempt as successful
            await OrderService.update_payment_attempt(
                db, payment_attempt.id, PaymentStatus.SUCCESS
            )
            
            # Update order status
            await OrderService.update_order_status(db, order_id, OrderStatus.CONFIRMED)
            
            logger.info(f"Payment successful for order {order_id} on attempt {attempt}")
            return True
            
        except Exception as e:
            error_message = str(e)
            logger.warning(
                f"Payment attempt {attempt} failed for order {order_id}: {error_message}"
            )
            
            # Update payment attempt as failed
            if 'payment_attempt' in locals():
                await OrderService.update_payment_attempt(
                    db, payment_attempt.id, PaymentStatus.FAILED, error_message
                )
            
            # If not the last attempt, wait before retrying
            if attempt < max_attempts:
                # Exponential backoff
                delay = settings.PAYMENT_RETRY_DELAY * (2 ** (attempt - 1))
                logger.info(f"Retrying payment for order {order_id} in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                # All attempts failed
                await OrderService.update_order_status(
                    db, order_id, OrderStatus.PAYMENT_FAILED
                )
                logger.error(f"All payment attempts failed for order {order_id}")
                return False
    
    return False


async def process_order_from_queue(order_data: Dict[str, Any]):
    """
    Process a single order from the queue.
    
    Args:
        order_data: Order data from queue
    """
    order_id = order_data["order_id"]
    amount = order_data["amount"]
    customer_id = order_data["customer_id"]
    
    async with AsyncSessionLocal() as db:
        try:
            # Verify order exists and is in correct state
            order = await OrderService.get_order(db, order_id)
            if not order:
                logger.error(f"Order {order_id} not found in database")
                return
            
            if order.status != OrderStatus.PAYMENT_PROCESSING:
                logger.warning(
                    f"Order {order_id} is in {order.status} state, skipping payment processing"
                )
                return
            
            # Process payment
            success = await process_payment_with_retry(
                db, order_id, amount, customer_id
            )
            
            if success:
                logger.info(f"Successfully processed payment for order {order_id}")
            else:
                logger.error(f"Failed to process payment for order {order_id}")
                
        except Exception as e:
            logger.error(f"Error processing order {order_id}: {str(e)}", exc_info=True)


async def worker_loop():
    """Main worker loop to process orders from queue."""
    logger.info("Payment worker started")
    global redis_client
    
    while True:
        try:
            # Blocking pop from queue (waits up to 5 seconds)
            # brpop returns None if timeout, or (queue_name, data) if data available
            queue_data = redis_client.brpop(
                settings.REDIS_QUEUE_NAME,
                timeout=5
            )
            
            if queue_data:
                # queue_data is a tuple: (queue_name, data)
                _, data = queue_data
                try:
                    order_data = json.loads(data)
                    logger.info(f"Processing order {order_data['order_id']} from queue")
                    await process_order_from_queue(order_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse queue data: {e}, data: {data}")
                except KeyError as e:
                    logger.error(f"Missing required field in order data: {e}")
            else:
                # No data in queue (timeout), continue loop
                # This is normal behavior, not an error
                await asyncio.sleep(0.1)
                
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error(f"Redis connection error: {e}, attempting to reconnect...")
            try:
                # Attempt to reconnect
                redis_client = get_redis_client()
                redis_client.ping()
                logger.info("Reconnected to Redis")
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect to Redis: {reconnect_error}")
                await asyncio.sleep(5)  # Wait before retrying
        except KeyboardInterrupt:
            logger.info("Worker interrupted, shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in worker loop: {str(e)}", exc_info=True)
            await asyncio.sleep(1)  # Brief pause before retrying


def main():
    """Entry point for payment worker."""
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("Worker stopped")


if __name__ == "__main__":
    main()

