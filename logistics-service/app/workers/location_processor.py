"""Location processing worker with reliable queue pattern."""
import asyncio
import logging
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.redis_streams import stream_processor
from app.services.location_service import location_service

logger = logging.getLogger(__name__)


async def process_location_message(
    message_id: str,
    message_data: Dict[str, Any],
    db: AsyncSession
) -> bool:
    """
    Process a single location message.
    
    Args:
        message_id: Message ID from stream
        message_data: Parsed message data
        db: Database session
        
    Returns:
        True if processing successful, False otherwise
    """
    driver_id = int(message_data.get("driver_id"))
    latitude = float(message_data.get("latitude"))
    longitude = float(message_data.get("longitude"))
    speed = message_data.get("speed")
    heading = message_data.get("heading")
    accuracy = message_data.get("accuracy")
    order_id = message_data.get("order_id")
    
    try:
        # Step 1: Update current location in Redis
        success = await location_service.update_current_location(
            driver_id=driver_id,
            latitude=latitude,
            longitude=longitude,
            speed=speed,
            heading=heading,
            accuracy=accuracy,
            order_id=order_id
        )
        
        if not success:
            logger.warning(f"Failed to update current location for driver {driver_id}")
            return False
        
        # Step 2: Store historical location in TimescaleDB
        await location_service.store_historical_location(
            db=db,
            driver_id=driver_id,
            latitude=latitude,
            longitude=longitude,
            speed=speed,
            heading=heading,
            accuracy=accuracy,
            order_id=order_id
        )
        
        # Step 3: Publish to Redis Pub/Sub for real-time updates
        await location_service.publish_location_update(
            driver_id=driver_id,
            latitude=latitude,
            longitude=longitude,
            speed=speed,
            heading=heading,
            order_id=order_id
        )
        
        logger.debug(f"Successfully processed location message {message_id} for driver {driver_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing location message {message_id}: {e}", exc_info=True)
        return False


async def process_messages_batch(
    messages: list[tuple[str, Dict[str, Any]]],
    db: AsyncSession
) -> tuple[list[str], list[tuple[str, Dict[str, Any]]]]:
    """
    Process a batch of messages.
    
    Args:
        messages: List of (message_id, message_data) tuples
        db: Database session
        
    Returns:
        Tuple of (successful_message_ids, failed_messages)
    """
    successful = []
    failed = []
    
    for message_id, message_data in messages:
        success = await process_location_message(message_id, message_data, db)
        
        if success:
            successful.append(message_id)
        else:
            failed.append((message_id, message_data))
    
    # Acknowledge successful messages (removes from processing queue)
    if successful:
        await stream_processor.acknowledge_messages(successful)
        logger.info(f"Acknowledged {len(successful)} messages")
    
    return successful, failed


async def worker_loop():
    """Main worker loop for processing location messages."""
    logger.info("Location processing worker started")
    
    # Ensure consumer group exists
    await stream_processor.ensure_consumer_group()
    
    # Track retry counts for failed messages
    retry_counts: Dict[str, int] = {}
    
    while True:
        try:
            # Read messages from stream
            # Messages automatically move to pending/processing state
            messages = await stream_processor.read_messages()
            
            if messages:
                logger.info(f"Processing {len(messages)} location messages")
                
                # Process messages in batches
                async with AsyncSessionLocal() as db:
                    successful_ids, failed_messages = await process_messages_batch(messages, db)
                    
                    logger.info(f"Processed batch: {len(successful_ids)} successful, {len(failed_messages)} failed")
                    
                    # Handle failed messages with retry logic
                    for message_id, message_data in failed_messages:
                        retry_counts[message_id] = retry_counts.get(message_id, 0) + 1
                        
                        if retry_counts[message_id] >= settings.MAX_RETRIES:
                            # Move to dead letter queue
                            await stream_processor.move_to_dead_letter_queue(
                                message_id=message_id,
                                message_data=message_data,
                                error=f"Failed after {settings.MAX_RETRIES} retries"
                            )
                            retry_counts.pop(message_id, None)
                        else:
                            # Will retry on next claim
                            logger.info(f"Message {message_id} will be retried (attempt {retry_counts[message_id]})")
            
            # Claim pending messages that have been idle too long
            pending_messages = await stream_processor.claim_pending_messages()
            if pending_messages:
                logger.info(f"Claimed {len(pending_messages)} pending messages")
                async with AsyncSessionLocal() as db:
                    successful_ids, failed_messages = await process_messages_batch(pending_messages, db)
                    logger.info(f"Processed pending batch: {len(successful_ids)} successful, {len(failed_messages)} failed")
                    
                    # Handle failed pending messages
                    for message_id, message_data in failed_messages:
                        retry_counts[message_id] = retry_counts.get(message_id, 0) + 1
                        
                        if retry_counts[message_id] >= settings.MAX_RETRIES:
                            await stream_processor.move_to_dead_letter_queue(
                                message_id=message_id,
                                message_data=message_data,
                                error=f"Failed after {settings.MAX_RETRIES} retries"
                            )
                            retry_counts.pop(message_id, None)
            
            # Small sleep to prevent busy loop
            await asyncio.sleep(0.1)
            
        except KeyboardInterrupt:
            logger.info("Worker interrupted, shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in worker loop: {e}", exc_info=True)
            await asyncio.sleep(1)  # Brief pause before retrying


def main():
    """Entry point for location processing worker."""
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

