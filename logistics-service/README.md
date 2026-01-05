# Real-Time Logistics & Analytics Service

A high-performance microservice for real-time driver location tracking and analytics. Designed to handle 10,000 concurrent drivers sending GPS updates every 5 seconds (2,000 events/second peak load).

## Features

- **High Throughput**: Handles 2,000+ location updates per second
- **Reliable Processing**: Redis Streams with reliable queue pattern (messages only removed after successful processing)
- **Real-Time Updates**: WebSocket support for live driver location tracking
- **Geospatial Queries**: Redis GEO commands for fast location-based searches
- **Historical Analytics**: TimescaleDB for time-series location data
- **Fault Tolerance**: Automatic retry and dead letter queue for failed messages

## Architecture

```
Driver GPS → Ingestion API → Redis Streams → Processing Worker
                                              ├── Redis (current location + GEO)
                                              ├── TimescaleDB (historical)
                                              └── Redis Pub/Sub → WebSocket → Customer
```

## Reliable Queue Pattern

**Key Feature**: Messages are only removed from the processing queue after successful processing.

1. **Message Ingestion**: Driver sends location update → Added to Redis Streams
2. **Message Processing**: Worker reads message → Message moves to pending/processing state
3. **Processing Steps**:
   - Update current location in Redis
   - Store historical location in TimescaleDB
   - Publish to Redis Pub/Sub for real-time updates
4. **Success**: ACK message (XACK) → Removes from processing queue
5. **Failure**: Message remains in pending → Can be retried
6. **Max Retries**: After max retries → Moved to dead letter queue

This ensures **no message loss** even if processing fails.

## Technology Stack

- **FastAPI**: High-performance async API framework
- **Redis Streams**: Reliable message queue with consumer groups
- **Redis GEO**: Geospatial queries for nearby drivers
- **Redis Pub/Sub**: Real-time distribution to customers
- **TimescaleDB**: Time-series optimized PostgreSQL for historical data
- **WebSocket**: Real-time bidirectional communication

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Running with Docker Compose

1. Navigate to the logistics-service directory:
```bash
cd logistics-service
```

2. Start all services:
```bash
docker-compose up -d
```

This will start:
- TimescaleDB database (port 5434)
- Redis (port 6381)
- Logistics Service API (port 8002)
- Location Processing Worker

3. Check service health:
```bash
curl http://localhost:8002/health
```

4. Access API documentation:
- Swagger UI: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start TimescaleDB and Redis (using Docker):
```bash
docker-compose up -d postgres redis
```

4. Run the API service:
```bash
uvicorn app.main:app --reload
```

5. Run the location worker (in a separate terminal):
```bash
python -m app.workers.location_processor
```

## API Endpoints

### Driver Endpoints

#### Update Location
```bash
POST /api/v1/drivers/{driver_id}/location
Content-Type: application/json

{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "speed": 45.5,
    "heading": 180,
    "accuracy": 10.0,
    "order_id": "order_123"
}
```

**Response**: `202 Accepted` - Message queued for processing

#### Get Current Location
```bash
GET /api/v1/drivers/{driver_id}/location
```

#### Get Route History
```bash
GET /api/v1/drivers/{driver_id}/route?start_time=2024-01-01T00:00:00Z&end_time=2024-01-01T23:59:59Z
```

### Customer Endpoints

#### WebSocket Connection (Real-Time Updates)
```javascript
ws://localhost:8002/api/v1/drivers/orders/{order_id}/driver-location
```

**Messages:**
- Server → Client: `{"type": "location_update", "data": {...}}`

#### Get Nearby Drivers
```bash
GET /api/v1/drivers/nearby?lat=40.7128&lng=-74.0060&radius=1000
```

## Reliable Processing Flow

### Normal Flow
1. Driver sends location update
2. Message added to Redis Streams
3. Worker reads message (moves to pending)
4. Worker processes:
   - Updates Redis current location
   - Stores in TimescaleDB
   - Publishes to Pub/Sub
5. Worker ACKs message (removes from pending)
6. Customer receives real-time update via WebSocket

### Failure Handling
1. If processing fails, message remains in pending
2. Worker claims pending messages after timeout
3. Retries processing (up to MAX_RETRIES)
4. After max retries → Moves to dead letter queue
5. No message loss guaranteed

## Performance

- **Ingestion**: < 10ms per request (P99)
- **Current Location Read**: < 1ms (Redis)
- **WebSocket Update Latency**: < 50ms end-to-end
- **Historical Query**: < 500ms for typical queries
- **Throughput**: 2,000+ events/second

## Monitoring

- Health check endpoint: `/health`
- Pending message count in health check
- Dead letter queue for failed messages
- Logs output to stdout

## Configuration

Key settings in `.env`:

- `STREAM_NAME`: Redis Stream name (default: "driver-locations")
- `CONSUMER_GROUP`: Consumer group name
- `MAX_RETRIES`: Maximum retry attempts (default: 3)
- `PENDING_MESSAGE_TIMEOUT`: Timeout for claiming pending messages (ms)
- `LOCATION_TTL`: TTL for current locations in Redis (seconds)

## Future Enhancements

- [ ] Driver authentication and authorization
- [ ] Rate limiting per driver
- [ ] Advanced analytics queries
- [ ] Route optimization
- [ ] ETA calculations
- [ ] Metrics and monitoring (Prometheus/Grafana)

## License

Part of the Swift Eats food delivery platform.

