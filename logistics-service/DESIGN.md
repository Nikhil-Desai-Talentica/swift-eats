# Real-Time Logistics & Analytics Service - Design Document

## Requirements

- **Scale**: 10,000 concurrent drivers
- **Update Frequency**: Every 5 seconds per driver
- **Peak Load**: 2,000 events/second
- **Use Case**: Real-time driver location tracking for customers
- **Analytics**: Historical location data analysis

## Architecture Overview

```
┌─────────────┐
│   Drivers   │ (10,000 concurrent)
│  GPS Devices│
└──────┬──────┘
       │ HTTP/WebSocket (2,000 events/sec)
       ▼
┌─────────────────────────────────┐
│  Location Ingestion Service     │
│  - FastAPI endpoints            │
│  - Rate limiting                │
│  - Validation                   │
└──────┬──────────────────────────┘
       │
       ├──► Message Queue (Kafka/Redis Streams)
       │    - Buffering
       │    - Decoupling
       │
       ▼
┌─────────────────────────────────┐
│  Location Processing Service   │
│  - Consume from queue           │
│  - Update current location      │
│  - Store historical data        │
│  - Publish to subscribers       │
└──────┬──────────────────────────┘
       │
       ├──► Redis (Current Locations)
       │    - Fast reads
       │    - Geospatial queries
       │
       ├──► TimescaleDB (Historical)
       │    - Time-series optimized
       │    - Analytics queries
       │
       └──► Redis Pub/Sub
            │
            ▼
    ┌──────────────────────────┐
    │  WebSocket Service       │
    │  - Customer connections  │
    │  - Real-time updates     │
    └──────────────────────────┘
```

## Technology Stack

### Core Technologies

1. **FastAPI** - High-performance async API framework
2. **Redis** - Multiple roles:
   - Current location storage (fast reads)
   - Geospatial queries (GEO commands)
   - Pub/Sub for real-time distribution
   - Redis Streams for message queue (alternative to Kafka)
3. **TimescaleDB** - PostgreSQL extension optimized for time-series data
4. **WebSocket** - Real-time bidirectional communication

### Why These Technologies?

**Redis:**
- **Current Locations**: Sub-millisecond reads
- **Geospatial**: Built-in GEO commands for location queries
- **Pub/Sub**: Lightweight real-time distribution
- **Streams**: Message queue without additional infrastructure
- **Already in use**: Consistent with other services

**TimescaleDB:**
- **Time-series optimized**: Perfect for GPS location data
- **PostgreSQL compatible**: Familiar SQL interface
- **Automatic partitioning**: Efficient storage and queries
- **Analytics**: Built-in functions for time-series analysis
- **Hypertables**: Automatic data retention policies

**FastAPI:**
- **Async**: Handles high concurrency
- **Performance**: Fast request processing
- **WebSocket support**: Built-in WebSocket endpoints

## Data Flow

### 1. Driver Location Update Flow (Reliable Processing)

```
Driver Device
    ↓ HTTP POST /api/v1/drivers/{driver_id}/location
Location Ingestion Service
    ↓ Validate
Redis Streams: "driver-locations" (main queue)
    ↓
Location Processing Worker
    ↓ Read message (XREADGROUP)
    ↓ Move to processing queue (temporary)
    ↓
    ├──► Update Redis (current location)
    ├──► Store in TimescaleDB (historical)
    └──► Publish to Redis Pub/Sub (real-time)
            ↓
        If all successful:
            ↓ ACK message (XACK) - removes from processing queue
        If failure:
            ↓ Retry or move to dead letter queue
            ↓
        WebSocket Service
            ↓
        Connected Customers
```

**Reliable Processing Pattern:**
1. Worker reads message from stream using consumer group
2. Message automatically moves to pending/processing state
3. Worker processes: updates Redis, stores in DB, publishes to Pub/Sub
4. On success: ACK message (XACK) - removes from processing queue
5. On failure: Message remains in pending, can be retried
6. Dead letter queue: Failed messages after max retries

### 2. Customer Location Query Flow

```
Customer App
    ↓ WebSocket Connection
WebSocket Service
    ↓ Subscribe to driver location updates
Redis Pub/Sub Channel: "driver:{driver_id}"
    ↓
Real-time location updates pushed to customer
```

### 3. Historical Query Flow

```
Analytics Service / API
    ↓ Query historical data
TimescaleDB
    ↓ Time-series optimized queries
Return aggregated location data
```

## Data Models

### Current Location (Redis)
```
Key: driver:location:{driver_id}
Value: {
    "driver_id": "12345",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "timestamp": "2024-01-01T12:00:00Z",
    "speed": 45.5,
    "heading": 180,
    "order_id": "order_123"
}
```

### Historical Location (TimescaleDB)
```sql
CREATE TABLE driver_locations (
    time TIMESTAMPTZ NOT NULL,
    driver_id INTEGER NOT NULL,
    order_id VARCHAR(100),
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    speed DOUBLE PRECISION,
    heading DOUBLE PRECISION,
    accuracy DOUBLE PRECISION,
    PRIMARY KEY (time, driver_id)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('driver_locations', 'time');
```

### Geospatial Index (Redis)
```
GEOADD drivers:geo {longitude} {latitude} {driver_id}
```

## API Design

### Driver Endpoints

#### Update Location
```http
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

**Response**: `202 Accepted` (async processing)

#### Get Current Location
```http
GET /api/v1/drivers/{driver_id}/location
```

**Response**: Current location with timestamp

### Customer Endpoints

#### WebSocket Connection
```javascript
ws://host/api/v1/orders/{order_id}/driver-location
```

**Messages:**
- Client → Server: `{"action": "subscribe", "order_id": "order_123"}`
- Server → Client: `{"driver_id": "12345", "latitude": 40.7128, "longitude": -74.0060, "timestamp": "..."}`

#### Get Driver Location (HTTP fallback)
```http
GET /api/v1/orders/{order_id}/driver-location
```

### Analytics Endpoints

#### Get Driver Route History
```http
GET /api/v1/drivers/{driver_id}/route?start_time=...&end_time=...
```

#### Get Drivers in Area
```http
GET /api/v1/drivers/nearby?lat=40.7128&lng=-74.0060&radius=1000
```

## Performance Optimizations

### 1. Ingestion Layer
- **Async processing**: FastAPI async endpoints
- **Rate limiting**: Per-driver rate limiting
- **Batch processing**: Group updates when possible
- **Connection pooling**: Efficient database connections

### 2. Message Queue (Reliable Processing)
- **Redis Streams**: Lightweight, fast, already in infrastructure
- **Consumer groups**: Parallel processing
- **Reliable Processing Pattern**:
  - Messages read into temporary processing queue
  - Only removed after successful processing
  - Failed messages can be retried or moved to dead letter queue
  - Prevents message loss during processing failures
- **Backpressure handling**: Prevent queue overflow

### 3. Current Location Storage
- **Redis GEO**: Geospatial queries in microseconds
- **TTL**: Auto-expire stale locations (e.g., 5 minutes)
- **Memory efficient**: Only store current location

### 4. Historical Storage
- **TimescaleDB hypertables**: Automatic partitioning
- **Compression**: Compress old data
- **Retention policies**: Auto-delete old data (configurable)
- **Indexes**: Optimized for time-series queries

### 5. Real-time Distribution
- **Redis Pub/Sub**: Lightweight pub/sub
- **Selective subscriptions**: Customers only get relevant updates
- **Connection pooling**: Efficient WebSocket management
- **Message batching**: Group updates when possible

## Scalability Considerations

### Horizontal Scaling
- **Stateless services**: All services can scale horizontally
- **Load balancer**: Distribute driver connections
- **Redis cluster**: Scale Redis for high availability
- **TimescaleDB read replicas**: Scale read queries

### Vertical Scaling
- **Connection limits**: Tune OS and application limits
- **Memory**: Redis memory for current locations
- **CPU**: Processing workers can scale

### Capacity Planning

**Current Locations (Redis):**
- 10,000 drivers × ~200 bytes = ~2 MB (negligible)

**Message Queue (Redis Streams):**
- 2,000 events/sec × 1 KB = 2 MB/sec
- Buffer 1 minute = 120 MB (very manageable)

**Historical Data (TimescaleDB):**
- 2,000 events/sec × 200 bytes = 400 KB/sec
- Per day: ~35 GB
- With compression: ~10-15 GB/day
- Retention: 30-90 days (configurable)

## Monitoring & Observability

### Key Metrics
- **Ingestion rate**: Events/second
- **Processing latency**: Queue → Storage
- **WebSocket connections**: Active customer connections
- **Redis memory**: Current location storage
- **Database query performance**: Historical queries
- **Error rates**: Failed updates, connection failures

### Alerts
- Queue depth > threshold
- Processing latency > 1 second
- Redis memory > 80%
- Database connection pool exhaustion
- WebSocket connection failures

## Security Considerations

1. **Authentication**: Driver tokens, customer session validation
2. **Rate limiting**: Prevent abuse
3. **Data validation**: Validate GPS coordinates
4. **Privacy**: Only authorized customers see driver location
5. **Encryption**: TLS for all connections

## Implementation Phases

### Phase 1: Core Ingestion & Storage
- Location ingestion API
- Redis Streams integration
- Current location storage (Redis)
- Historical storage (TimescaleDB)

### Phase 2: Real-time Distribution
- WebSocket service
- Redis Pub/Sub integration
- Customer connection management

### Phase 3: Analytics & Optimization
- Historical query endpoints
- Geospatial queries
- Performance tuning
- Monitoring integration

## Alternative Considerations

### Kafka vs Redis Streams
**Redis Streams** (Chosen):
- ✅ Already in infrastructure
- ✅ Simpler setup
- ✅ Sufficient for 2K events/sec
- ✅ Lower operational overhead

**Kafka** (Alternative):
- ✅ Higher throughput (if needed)
- ✅ Better durability guarantees
- ❌ Additional infrastructure
- ❌ More complex operations

### WebSocket vs Server-Sent Events (SSE)
**WebSocket** (Chosen):
- ✅ Bidirectional communication
- ✅ Lower overhead
- ✅ Better for real-time updates

**SSE** (Alternative):
- ✅ Simpler (HTTP-based)
- ✅ Better browser compatibility
- ❌ Unidirectional only
- ❌ Higher overhead

## Expected Performance

- **Ingestion**: < 10ms per request (P99)
- **Queue Processing**: < 100ms latency
- **Current Location Read**: < 1ms (Redis)
- **WebSocket Update Latency**: < 50ms end-to-end
- **Historical Query**: < 500ms for typical queries

