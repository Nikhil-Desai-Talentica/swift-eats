"""Order service for business logic."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from typing import List, Optional
from app.models.order import Order, OrderItem, PaymentAttempt, OrderStatus, PaymentStatus
from app.schemas.order import OrderCreate
import logging

logger = logging.getLogger(__name__)


class OrderService:
    """Service for order operations."""
    
    @staticmethod
    async def create_order(db: AsyncSession, order_data: OrderCreate) -> Order:
        """
        Create a new order.
        
        Args:
            db: Database session
            order_data: Order creation data
            
        Returns:
            Created order
        """
        # Calculate total amount
        total_amount = sum(item.price * item.quantity for item in order_data.items)
        
        # Create order
        order = Order(
            customer_id=order_data.customer_id,
            total_amount=total_amount,
            status=OrderStatus.PENDING
        )
        db.add(order)
        await db.flush()  # Get order ID
        
        # Create order items
        for item_data in order_data.items:
            order_item = OrderItem(
                order_id=order.id,
                item_id=item_data.item_id,
                item_name=item_data.item_name,
                quantity=item_data.quantity,
                price=item_data.price
            )
            db.add(order_item)
        
        await db.commit()
        await db.refresh(order)
        
        logger.info(f"Created order {order.id} for customer {order.customer_id}")
        return order
    
    @staticmethod
    async def get_order(db: AsyncSession, order_id: int) -> Optional[Order]:
        """
        Get order by ID.
        
        Args:
            db: Database session
            order_id: Order identifier
            
        Returns:
            Order if found, None otherwise
        """
        result = await db.execute(
            select(Order)
            .options(selectinload(Order.items), selectinload(Order.payment_attempts))
            .where(Order.id == order_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_orders(
        db: AsyncSession,
        customer_id: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[Order], int]:
        """
        Get orders with pagination and filters.
        
        Args:
            db: Database session
            customer_id: Filter by customer ID
            status: Filter by order status
            page: Page number (1-indexed)
            page_size: Number of items per page
            
        Returns:
            Tuple of (orders list, total count)
        """
        # Build query
        query = select(Order).options(
            selectinload(Order.items),
            selectinload(Order.payment_attempts)
        )
        
        if customer_id:
            query = query.where(Order.customer_id == customer_id)
        if status:
            query = query.where(Order.status == status)
        
        # Get total count
        count_query = select(func.count()).select_from(Order)
        if customer_id:
            count_query = count_query.where(Order.customer_id == customer_id)
        if status:
            count_query = count_query.where(Order.status == status)
        
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        query = query.order_by(Order.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        orders = result.scalars().all()
        
        return list(orders), total
    
    @staticmethod
    async def update_order_status(
        db: AsyncSession,
        order_id: int,
        status: OrderStatus
    ) -> Optional[Order]:
        """
        Update order status.
        
        Args:
            db: Database session
            order_id: Order identifier
            status: New status
            
        Returns:
            Updated order if found, None otherwise
        """
        order = await OrderService.get_order(db, order_id)
        if not order:
            return None
        
        order.status = status
        await db.commit()
        await db.refresh(order)
        
        logger.info(f"Updated order {order_id} status to {status}")
        return order
    
    @staticmethod
    async def create_payment_attempt(
        db: AsyncSession,
        order_id: int,
        amount: float,
        attempt_number: int = 1
    ) -> PaymentAttempt:
        """
        Create a payment attempt record.
        
        Args:
            db: Database session
            order_id: Order identifier
            amount: Payment amount
            attempt_number: Attempt number
            
        Returns:
            Created payment attempt
        """
        payment_attempt = PaymentAttempt(
            order_id=order_id,
            attempt_number=attempt_number,
            status=PaymentStatus.PROCESSING,
            amount=amount
        )
        db.add(payment_attempt)
        await db.commit()
        await db.refresh(payment_attempt)
        
        return payment_attempt
    
    @staticmethod
    async def update_payment_attempt(
        db: AsyncSession,
        payment_attempt_id: int,
        status: PaymentStatus,
        error_message: Optional[str] = None
    ) -> Optional[PaymentAttempt]:
        """
        Update payment attempt status.
        
        Args:
            db: Database session
            payment_attempt_id: Payment attempt identifier
            status: New status
            error_message: Error message if failed
            
        Returns:
            Updated payment attempt if found, None otherwise
        """
        result = await db.execute(
            select(PaymentAttempt).where(PaymentAttempt.id == payment_attempt_id)
        )
        payment_attempt = result.scalar_one_or_none()
        
        if not payment_attempt:
            return None
        
        payment_attempt.status = status
        payment_attempt.error_message = error_message
        await db.commit()
        await db.refresh(payment_attempt)
        
        return payment_attempt

