"""Restaurant and menu database models."""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Enum as SQLEnum, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum
from app.core.database import Base


class RestaurantStatus(str, enum.Enum):
    """Restaurant status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    BUSY = "busy"  # Open but very busy
    MAINTENANCE = "maintenance"


class CuisineType(str, enum.Enum):
    """Cuisine type enumeration."""
    ITALIAN = "italian"
    CHINESE = "chinese"
    INDIAN = "indian"
    MEXICAN = "mexican"
    JAPANESE = "japanese"
    AMERICAN = "american"
    THAI = "thai"
    MEDITERRANEAN = "mediterranean"
    OTHER = "other"


class Restaurant(Base):
    """Restaurant model."""
    __tablename__ = "restaurants"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    image_url = Column(String(500), nullable=True)
    
    # Location
    address = Column(String(500), nullable=False)
    city = Column(String(100), nullable=False, index=True)
    state = Column(String(100), nullable=True)
    zip_code = Column(String(20), nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # Status and metadata
    status = Column(SQLEnum(RestaurantStatus), default=RestaurantStatus.OPEN, nullable=False, index=True)
    cuisine_type = Column(SQLEnum(CuisineType), nullable=False, index=True)
    rating = Column(Float, default=0.0, nullable=False, index=True)
    review_count = Column(Integer, default=0, nullable=False)
    
    # Capacity
    max_capacity = Column(Integer, nullable=True)
    current_capacity = Column(Integer, default=0, nullable=False)
    
    # Operating hours (stored as JSON string for simplicity)
    operating_hours = Column(Text, nullable=True)  # JSON format
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    menu = relationship("Menu", back_populates="restaurant", uselist=False, cascade="all, delete-orphan")
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_restaurant_cuisine_status', 'cuisine_type', 'status'),
        Index('idx_restaurant_location', 'latitude', 'longitude'),
        Index('idx_restaurant_rating', 'rating', 'review_count'),
    )
    
    def __repr__(self):
        return f"<Restaurant(id={self.id}, name={self.name}, status={self.status})>"


class Menu(Base):
    """Menu model (one per restaurant)."""
    __tablename__ = "menus"
    
    id = Column(Integer, primary_key=True, index=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False, unique=True, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    restaurant = relationship("Restaurant", back_populates="menu")
    categories = relationship("MenuCategory", back_populates="menu", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Menu(id={self.id}, restaurant_id={self.restaurant_id})>"


class MenuCategory(Base):
    """Menu category model."""
    __tablename__ = "menu_categories"
    
    id = Column(Integer, primary_key=True, index=True)
    menu_id = Column(Integer, ForeignKey("menus.id"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    display_order = Column(Integer, default=0, nullable=False)
    
    # Relationships
    menu = relationship("Menu", back_populates="categories")
    items = relationship("MenuItem", back_populates="category", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<MenuCategory(id={self.id}, name={self.name}, menu_id={self.menu_id})>"


class MenuItem(Base):
    """Menu item model."""
    __tablename__ = "menu_items"
    
    id = Column(Integer, primary_key=True, index=True)
    category_id = Column(Integer, ForeignKey("menu_categories.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    price = Column(Float, nullable=False, index=True)
    image_url = Column(String(500), nullable=True)
    
    # Availability
    available = Column(Boolean, default=True, nullable=False, index=True)
    in_stock = Column(Boolean, default=True, nullable=False, index=True)
    
    # Dietary information (stored as JSON string)
    dietary_info = Column(Text, nullable=True)  # JSON: ["vegan", "gluten-free", etc.]
    
    # Metadata
    preparation_time = Column(Integer, nullable=True)  # minutes
    display_order = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    category = relationship("MenuCategory", back_populates="items")
    
    # Indexes
    __table_args__ = (
        Index('idx_menu_item_availability', 'available', 'in_stock'),
    )
    
    def __repr__(self):
        return f"<MenuItem(id={self.id}, name={self.name}, available={self.available})>"

