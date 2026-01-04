"""Order API routes."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import redis
import json
import logging

from app.core.database import get_db
from app.core.config import settings
from app.schemas.order import OrderCreate, OrderResponse, OrderListResponse
from app.services.order_service import OrderService
from app.models.order import OrderStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orders", tags=["orders"])

def get_redis_client():
    """Get Redis client for API operations."""
    return redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,  # Short timeout for non-blocking operations
        socket_keepalive=True,
        health_check_interval=30
    )

def get_or_create_redis_client():
    """Get existing Redis client or create a new one."""
    global redis_client
    if redis_client is None:
        try:
            redis_client = get_redis_client()
            redis_client.ping()
            logger.info("API: Connected to Redis")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error(f"API: Failed to connect to Redis: {e}")
            return None
    return redis_client

# Redis connection for queue
redis_client = None
try:
    redis_client = get_redis_client()
    redis_client.ping()
    logger.info("API: Connected to Redis")
except (redis.ConnectionError, redis.TimeoutError) as e:
    logger.error(f"API: Failed to connect to Redis: {e}")
    redis_client = None


@router.post("", response_model=OrderResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_order(
    order_data: OrderCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new order.
    
    The order is accepted immediately and queued for payment processing.
    Returns 202 Accepted with the order details.
    """
    try:
        # Create order in database
        order = await OrderService.create_order(db, order_data)
        
        # Enqueue order for payment processing
        queue_data = {
            "order_id": order.id,
            "customer_id": order.customer_id,
            "amount": float(order.total_amount)
        }
        
        # Get or create Redis client
        client = get_or_create_redis_client()
        if client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Queue service unavailable"
            )
        
        try:
            client.lpush(settings.REDIS_QUEUE_NAME, json.dumps(queue_data))
            logger.info(f"Order {order.id} queued for payment processing")
        except (redis.RedisError, redis.ConnectionError, redis.TimeoutError) as e:
            logger.error(f"Failed to enqueue order {order.id}: {e}")
            # Try to reconnect and retry once
            try:
                client = get_or_create_redis_client()
                if client:
                    client.lpush(settings.REDIS_QUEUE_NAME, json.dumps(queue_data))
                    logger.info(f"Order {order.id} queued after reconnection")
                else:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Failed to queue order for processing"
                    )
            except HTTPException:
                raise
            except Exception as retry_error:
                logger.error(f"Failed to enqueue order {order.id} after retry: {retry_error}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to queue order for processing"
                )
        
        # Update order status to payment_processing
        await OrderService.update_order_status(db, order.id, OrderStatus.PAYMENT_PROCESSING)
        
        # Refresh order to get updated status
        order = await OrderService.get_order(db, order.id)
        
        return order
        
    except Exception as e:
        logger.error(f"Error creating order: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create order: {str(e)}"
        )


@router.get("/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get order by ID.
    """
    order = await OrderService.get_order(db, order_id)
    
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    return order


@router.get("", response_model=OrderListResponse)
async def list_orders(
    customer_id: Optional[str] = Query(None, description="Filter by customer ID"),
    status: Optional[OrderStatus] = Query(None, description="Filter by order status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db)
):
    """
    List orders with pagination and optional filters.
    """
    orders, total = await OrderService.get_orders(
        db=db,
        customer_id=customer_id,
        status=status,
        page=page,
        page_size=page_size
    )
    
    return OrderListResponse(
        orders=orders,
        total=total,
        page=page,
        page_size=page_size
    )

