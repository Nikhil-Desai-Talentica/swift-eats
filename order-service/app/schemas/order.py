"""Pydantic schemas for order API."""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from app.models.order import OrderStatus, PaymentStatus


class OrderItemCreate(BaseModel):
    """Schema for creating an order item."""
    item_id: str = Field(..., description="Item identifier")
    item_name: str = Field(..., description="Item name")
    quantity: int = Field(..., gt=0, description="Quantity of items")
    price: float = Field(..., gt=0, description="Price per item")


class OrderCreate(BaseModel):
    """Schema for creating an order."""
    customer_id: str = Field(..., description="Customer identifier")
    items: List[OrderItemCreate] = Field(..., min_items=1, description="List of order items")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "customer_123",
                "items": [
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
            }
        }


class OrderItemResponse(BaseModel):
    """Schema for order item response."""
    id: int
    item_id: str
    item_name: str
    quantity: int
    price: float
    
    class Config:
        from_attributes = True


class PaymentAttemptResponse(BaseModel):
    """Schema for payment attempt response."""
    id: int
    attempt_number: int
    status: PaymentStatus
    amount: float
    error_message: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class OrderResponse(BaseModel):
    """Schema for order response."""
    id: int
    customer_id: str
    total_amount: float
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    items: List[OrderItemResponse]
    payment_attempts: List[PaymentAttemptResponse] = []
    
    class Config:
        from_attributes = True


class OrderListResponse(BaseModel):
    """Schema for paginated order list response."""
    orders: List[OrderResponse]
    total: int
    page: int
    page_size: int

