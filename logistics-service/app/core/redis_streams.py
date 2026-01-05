"""Reliable Redis Streams queue service."""
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import redis.asyncio as redis
from app.core.config import settings

logger = logging.getLogger(__name__)


class ReliableStreamProcessor:
    """Reliable Redis Streams processor with temporary queue pattern."""
    
    def __init__(self):
        """Initialize Redis connection."""
        self.redis_client: Optional[redis.Redis] = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis (synchronous for initialization)."""
        try:
            # Note: We'll use async Redis client
            # For now, create connection pool
            self.redis_pool = redis.ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
                socket_keepalive=True,
                health_check_interval=30
            )
            logger.info("Redis connection pool created")
        except Exception as e:
            logger.error(f"Failed to create Redis connection pool: {e}")
            self.redis_pool = None
    
    async def get_client(self) -> redis.Redis:
        """Get async Redis client."""
        if not self.redis_pool:
            raise ConnectionError("Redis connection pool not available")
        return redis.Redis(connection_pool=self.redis_pool)
    
    async def ensure_consumer_group(self):
        """Ensure consumer group exists."""
        client = await self.get_client()
        try:
            await client.xgroup_create(
                name=settings.STREAM_NAME,
                groupname=settings.CONSUMER_GROUP,
                id="0",  # Start from beginning
                mkstream=True  # Create stream if it doesn't exist
            )
            logger.info(f"Created consumer group: {settings.CONSUMER_GROUP}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group already exists: {settings.CONSUMER_GROUP}")
            else:
                raise
    
    async def add_location_message(
        self,
        driver_id: str,
        latitude: float,
        longitude: float,
        **kwargs
    ) -> str:
        """
        Add location update message to stream.
        
        Args:
            driver_id: Driver identifier
            latitude: Latitude
            longitude: Longitude
            **kwargs: Additional fields (speed, heading, order_id, etc.)
            
        Returns:
            Message ID
        """
        client = await self.get_client()
        
        message_data = {
            "driver_id": driver_id,
            "latitude": str(latitude),
            "longitude": str(longitude),
            **{k: str(v) for k, v in kwargs.items()}
        }
        
        message_id = await client.xadd(
            name=settings.STREAM_NAME,
            fields=message_data
        )
        
        logger.debug(f"Added location message for driver {driver_id}: {message_id}")
        return message_id
    
    async def read_messages(
        self,
        count: Optional[int] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Read messages from stream using consumer group.
        Messages are automatically moved to pending/processing state.
        
        Args:
            count: Number of messages to read (default: STREAM_BATCH_SIZE)
            
        Returns:
            List of (message_id, message_data) tuples
        """
        client = await self.get_client()
        count = count or settings.STREAM_BATCH_SIZE
        
        try:
            # Read messages from stream
            # Messages are automatically claimed by this consumer
            messages = await client.xreadgroup(
                groupname=settings.CONSUMER_GROUP,
                consumername=settings.CONSUMER_NAME,
                streams={settings.STREAM_NAME: ">"},  # ">" means new messages
                count=count,
                block=settings.STREAM_BLOCK_TIME
            )
            
            if not messages:
                return []
            
            # Parse messages
            result = []
            stream_name, stream_messages = messages[0]
            
            for message_id, message_data in stream_messages:
                # Parse message data
                parsed_data = {}
                for key, value in message_data.items():
                    # Try to parse numeric values
                    try:
                        if '.' in value:
                            parsed_data[key] = float(value)
                        else:
                            parsed_data[key] = int(value)
                    except ValueError:
                        parsed_data[key] = value
                
                result.append((message_id, parsed_data))
            
            logger.debug(f"Read {len(result)} messages from stream")
            return result
            
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error reading messages: {e}")
            return []
    
    async def acknowledge_message(self, message_id: str) -> bool:
        """
        Acknowledge successful processing of a message.
        This removes it from the pending/processing queue.
        
        Args:
            message_id: Message ID to acknowledge
            
        Returns:
            True if successful
        """
        client = await self.get_client()
        
        try:
            await client.xack(
                name=settings.STREAM_NAME,
                groupname=settings.CONSUMER_GROUP,
                *[message_id]
            )
            logger.debug(f"Acknowledged message: {message_id}")
            return True
        except Exception as e:
            logger.error(f"Error acknowledging message {message_id}: {e}")
            return False
    
    async def acknowledge_messages(self, message_ids: List[str]) -> int:
        """
        Acknowledge multiple messages.
        
        Args:
            message_ids: List of message IDs
            
        Returns:
            Number of successfully acknowledged messages
        """
        if not message_ids:
            return 0
        
        client = await self.get_client()
        
        try:
            acknowledged = await client.xack(
                name=settings.STREAM_NAME,
                groupname=settings.CONSUMER_GROUP,
                *message_ids
            )
            logger.debug(f"Acknowledged {acknowledged} messages")
            return acknowledged
        except Exception as e:
            logger.error(f"Error acknowledging messages: {e}")
            return 0
    
    async def claim_pending_messages(
        self,
        min_idle_time: Optional[int] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Claim pending messages that have been idle too long.
        Useful for recovering from failed processing.
        
        Args:
            min_idle_time: Minimum idle time in milliseconds
            
        Returns:
            List of (message_id, message_data) tuples
        """
        client = await self.get_client()
        min_idle_time = min_idle_time or settings.PENDING_MESSAGE_TIMEOUT
        
        try:
            # Get pending messages
            pending = await client.xpending_range(
                name=settings.STREAM_NAME,
                groupname=settings.CONSUMER_GROUP,
                min="-",
                max="+",
                count=settings.STREAM_BATCH_SIZE
            )
            
            if not pending:
                return []
            
            # Claim messages that have been idle too long
            idle_message_ids = [
                msg["message_id"] for msg in pending
                if msg["time_since_delivered"] > min_idle_time
            ]
            
            if not idle_message_ids:
                return []
            
            # Claim these messages
            claimed = await client.xclaim(
                name=settings.STREAM_NAME,
                groupname=settings.CONSUMER_GROUP,
                consumername=settings.CONSUMER_NAME,
                min_idle_time=min_idle_time,
                message_ids=idle_message_ids
            )
            
            # Parse claimed messages
            result = []
            for message_id, message_data in claimed:
                parsed_data = {}
                for key, value in message_data.items():
                    try:
                        if '.' in value:
                            parsed_data[key] = float(value)
                        else:
                            parsed_data[key] = int(value)
                    except ValueError:
                        parsed_data[key] = value
                
                result.append((message_id, parsed_data))
            
            logger.info(f"Claimed {len(result)} pending messages")
            return result
            
        except Exception as e:
            logger.error(f"Error claiming pending messages: {e}")
            return []
    
    async def move_to_dead_letter_queue(
        self,
        message_id: str,
        message_data: Dict[str, Any],
        error: str
    ) -> bool:
        """
        Move a failed message to dead letter queue.
        
        Args:
            message_id: Original message ID
            message_data: Message data
            error: Error message
            
        Returns:
            True if successful
        """
        client = await self.get_client()
        
        try:
            # Add error information
            dlq_data = {
                **message_data,
                "original_message_id": message_id,
                "error": error,
                "failed_at": str(asyncio.get_event_loop().time())
            }
            
            # Add to dead letter queue
            await client.xadd(
                name=settings.DEAD_LETTER_STREAM,
                fields={k: str(v) for k, v in dlq_data.items()}
            )
            
            # Acknowledge original message (remove from processing queue)
            await self.acknowledge_message(message_id)
            
            logger.warning(f"Moved message {message_id} to DLQ: {error}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving message to DLQ: {e}")
            return False
    
    async def get_pending_count(self) -> int:
        """Get count of pending messages."""
        client = await self.get_client()
        
        try:
            info = await client.xpending(
                name=settings.STREAM_NAME,
                groupname=settings.CONSUMER_GROUP
            )
            return info.get("pending", 0)
        except Exception as e:
            logger.error(f"Error getting pending count: {e}")
            return 0
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


# Global stream processor instance
stream_processor = ReliableStreamProcessor()

