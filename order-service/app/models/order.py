"""Order database models."""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum as SQLEnum, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum
from app.core.database import Base


class OrderStatus(str, enum.Enum):
    """Order status enumeration."""
    PENDING = "pending"
    PAYMENT_PROCESSING = "payment_processing"
    CONFIRMED = "confirmed"
    PAYMENT_FAILED = "payment_failed"
    CANCELLED = "cancelled"


class PaymentStatus(str, enum.Enum):
    """Payment status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


class Order(Base):
    """Order model."""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(100), nullable=False, index=True)
    total_amount = Column(Float, nullable=False)
    status = Column(SQLEnum(OrderStatus), default=OrderStatus.PENDING, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")
    payment_attempts = relationship("PaymentAttempt", back_populates="order", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Order(id={self.id}, customer_id={self.customer_id}, status={self.status})>"


class OrderItem(Base):
    """Order item model."""
    __tablename__ = "order_items"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False, index=True)
    item_id = Column(String(100), nullable=False)
    item_name = Column(String(255), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    
    # Relationships
    order = relationship("Order", back_populates="items")
    
    def __repr__(self):
        return f"<OrderItem(id={self.id}, order_id={self.order_id}, item_id={self.item_id})>"


class PaymentAttempt(Base):
    """Payment attempt model for tracking payment processing."""
    __tablename__ = "payment_attempts"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False, index=True)
    attempt_number = Column(Integer, nullable=False, default=1)
    status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.PENDING, nullable=False)
    amount = Column(Float, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    order = relationship("Order", back_populates="payment_attempts")
    
    def __repr__(self):
        return f"<PaymentAttempt(id={self.id}, order_id={self.order_id}, status={self.status})>"

