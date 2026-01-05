# Menu & Restaurant Browse Service

A high-performance microservice for browsing restaurants and menus. Designed to achieve P99 response times under 200ms even under heavy user load.

## Features

- **High Performance**: P99 < 200ms with multi-layer caching
- **Event-Based Cache Invalidation**: Operators update data, cache automatically invalidates
- **Optimized Queries**: Database indexes and eager loading
- **Response Compression**: Gzip compression for faster responses
- **Operator API**: Endpoints for restaurant operators to update status and menu availability

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  Menu Service (FastAPI)         │
│  - Multi-layer caching           │
│  - Optimized queries            │
│  - Response compression          │
└──────┬──────────────────────────┘
       │
       ├──► Redis Cache (L1)
       │    - Restaurant data
       │    - Menu data
       │    - Status data
       │
       └──► PostgreSQL (L2)
            - Optimized indexes
            - Eager loading
```

## Cache Invalidation Strategy

**Event-Based Invalidation**: When restaurant operators update data:
- Restaurant status update → Invalidates restaurant, menu, and status cache
- Menu item availability update → Invalidates menu cache
- Multiple items update → Invalidates menu cache

TTL values serve as fallback safety mechanisms.

## Technology Stack

- **FastAPI**: Modern, fast web framework
- **PostgreSQL**: Reliable relational database
- **Redis**: Distributed caching layer
- **SQLAlchemy**: Async ORM
- **Pydantic**: Data validation

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Running with Docker Compose

1. Navigate to the menu-service directory:
```bash
cd menu-service
```

2. Start all services:
```bash
docker-compose up -d
```

This will start:
- PostgreSQL database (port 5433)
- Redis (port 6380)
- Menu Service API (port 8001)

3. Check service health:
```bash
curl http://localhost:8001/health
```

4. Access API documentation:
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

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

3. Start PostgreSQL and Redis (using Docker):
```bash
docker-compose up -d postgres redis
```

4. Run the API service:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

### Browse Endpoints

#### List Restaurants
```bash
GET /api/v1/restaurants?cuisine_type=italian&status=open&page=1&page_size=20
```

#### Get Restaurant
```bash
GET /api/v1/restaurants/{restaurant_id}
```

#### Get Menu
```bash
GET /api/v1/restaurants/{restaurant_id}/menu
```

#### Get Status
```bash
GET /api/v1/restaurants/{restaurant_id}/status
```

#### Get Menu with Status (Optimized)
```bash
GET /api/v1/restaurants/{restaurant_id}/menu-with-status
```

### Operator Update Endpoints

#### Update Restaurant Status
```bash
PUT /api/v1/restaurants/{restaurant_id}/status
Content-Type: application/json

{
  "status": "closed",
  "current_capacity": 0
}
```

#### Update Menu Item Availability
```bash
PUT /api/v1/restaurants/{restaurant_id}/menu/items/{item_id}/availability
Content-Type: application/json

{
  "available": false,
  "in_stock": false
}
```

#### Update Multiple Menu Items
```bash
PUT /api/v1/restaurants/{restaurant_id}/menu/items/availability
Content-Type: application/json

{
  "items": [
    {
      "item_id": 1,
      "available": false,
      "in_stock": false
    },
    {
      "item_id": 2,
      "available": true,
      "in_stock": true
    }
  ]
}
```

## Performance Optimizations

1. **Multi-Layer Caching**
   - Redis cache with event-based invalidation
   - Database query optimization
   - Strategic indexes

2. **Response Optimization**
   - Gzip compression
   - Eager loading (avoid N+1 queries)
   - Parallel data fetching

3. **Database Optimization**
   - Composite indexes for common queries
   - Connection pooling
   - Efficient queries

## Cache Invalidation Flow

1. Operator updates restaurant status or menu item availability
2. Database is updated
3. Cache service automatically invalidates relevant cache keys
4. Next request fetches fresh data from database and caches it

## Performance Targets

- **P99 Response Time**: < 200ms
- **Cache Hit Rate**: > 80% (target)
- **Cache Hit Latency**: < 10ms
- **Cache Miss Latency**: 50-150ms (DB query + cache write)

## Monitoring

- Health check endpoint: `/health`
- Cache status included in health check
- Logs output to stdout

## Future Enhancements

- [ ] Location-based search (lat/lng/radius)
- [ ] Search functionality (Elasticsearch integration)
- [ ] Real-time status updates (WebSocket)
- [ ] Metrics and monitoring (Prometheus/Grafana)
- [ ] Rate limiting
- [ ] API authentication for operator endpoints

## License

Part of the Swift Eats food delivery platform.

