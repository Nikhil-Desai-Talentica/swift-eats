"""Driver location database models."""
from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from sqlalchemy.sql import func
from datetime import datetime
from app.core.database import Base


class DriverLocation(Base):
    """Driver location model for TimescaleDB."""
    __tablename__ = "driver_locations"
    
    time = Column(DateTime(timezone=True), primary_key=True, server_default=func.now(), nullable=False)
    driver_id = Column(Integer, primary_key=True, nullable=False, index=True)
    order_id = Column(String(100), nullable=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    speed = Column(Float, nullable=True)  # km/h
    heading = Column(Float, nullable=True)  # degrees (0-360)
    accuracy = Column(Float, nullable=True)  # meters
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_driver_locations_driver_time', 'driver_id', 'time'),
        Index('idx_driver_locations_order_time', 'order_id', 'time'),
    )
    
    def __repr__(self):
        return f"<DriverLocation(driver_id={self.driver_id}, time={self.time}, lat={self.latitude}, lng={self.longitude})>"

