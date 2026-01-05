# Menu & Restaurant Browse Service - Design Document

## Performance Requirements

- **P99 Response Time**: < 200ms
- **Load**: Heavy read traffic (browse-heavy workload)
- **Availability**: High availability for browsing experience

## Architecture

### Multi-Layer Caching Strategy

```
Client Request
    ↓
FastAPI Service
    ↓
L1: Redis Cache (Distributed)
    ├── Restaurant Metadata (TTL: 1 hour)
    ├── Menu Data (TTL: 30 minutes)
    └── Restaurant Status (TTL: 30 seconds)
    ↓
L2: PostgreSQL (Optimized)
    ├── Read Replicas
    ├── Strategic Indexes
    └── Materialized Views
    ↓
L3: In-Memory Cache (Optional)
    └── Hot Data (Top 10% restaurants)
```

## Data Model

### Restaurant
- Basic info: id, name, description, image_url
- Location: address, latitude, longitude
- Status: open/closed/busy, current_capacity
- Metadata: rating, review_count, cuisine_type
- Operating hours: schedule, timezone

### Menu
- Restaurant relationship (one-to-many)
- Categories: appetizers, mains, desserts, etc.
- Items: id, name, description, price, image_url
- Availability: in_stock, dietary_info (vegan, gluten-free, etc.)

## API Endpoints

### Core Endpoints
1. `GET /api/v1/restaurants` - List restaurants (paginated, filtered)
2. `GET /api/v1/restaurants/{id}` - Restaurant details
3. `GET /api/v1/restaurants/{id}/menu` - Full menu
4. `GET /api/v1/restaurants/{id}/status` - Current status
5. `GET /api/v1/restaurants/{id}/menu-with-status` - Combined (optimized)

### Query Parameters
- `page`, `page_size` - Pagination
- `cuisine_type` - Filter by cuisine
- `status` - Filter by open/closed
- `lat`, `lng`, `radius` - Location-based search
- `min_rating` - Filter by rating
- `fields` - Sparse fieldsets (select specific fields)

## Performance Optimizations

### 1. Caching Strategy
- **Cache-Aside Pattern**: Check cache first, fallback to DB
- **TTL Strategy**:
  - Restaurant metadata: 1 hour (changes infrequently)
  - Menu data: 30 minutes (changes occasionally)
  - Status: 30 seconds (changes frequently)
- **Cache Keys**: `restaurant:{id}`, `menu:{restaurant_id}`, `status:{restaurant_id}`

### 2. Database Optimization
- **Indexes**:
  - Primary: restaurant.id
  - Composite: (cuisine_type, status, rating)
  - Spatial: (latitude, longitude) for location queries
  - Foreign key: menu.restaurant_id
- **Query Optimization**:
  - Eager loading (avoid N+1 queries)
  - Select only needed columns
  - Use materialized views for complex aggregations
- **Connection Pooling**: 20-50 connections per instance

### 3. Response Optimization
- **Compression**: Gzip middleware
- **Pagination**: Cursor-based for large datasets
- **Field Selection**: Allow clients to request only needed fields
- **Parallel Fetching**: Fetch menu and status in parallel when needed

### 4. Redis Optimization
- **Pipelining**: Batch multiple cache operations
- **Connection Pooling**: Reuse connections
- **Serialization**: Use msgpack or JSON (msgpack faster)

## Cache Invalidation Strategy

### Primary: Event-Based Invalidation

**Restaurant operators notify the system of changes**, triggering automatic cache invalidation:

1. **Restaurant Status Update** (facility shutting down, opening, etc.)
   - Operator calls: `PUT /api/v1/restaurants/{id}/status`
   - Cache invalidation: `restaurant:{id}`, `menu:{id}`, `status:{id}`
   - Next request fetches fresh data from DB and caches it

2. **Menu Item Availability Update** (dish running out of stock, etc.)
   - Operator calls: `PUT /api/v1/restaurants/{id}/menu/items/{item_id}/availability`
   - Cache invalidation: `menu:{restaurant_id}`
   - Next request fetches fresh menu from DB and caches it

3. **Bulk Menu Items Update** (multiple dishes)
   - Operator calls: `PUT /api/v1/restaurants/{id}/menu/items/availability`
   - Cache invalidation: `menu:{restaurant_id}`
   - Efficient for updating multiple items at once

### Fallback: Time-Based (TTL)

TTL values serve as safety mechanisms:
- Restaurant metadata: 1 hour TTL (fallback)
- Menu data: 30 minutes TTL (fallback)
- Status: 30 seconds TTL (fallback, but event-based is primary)

### Benefits of Event-Based Invalidation

- **Immediate Updates**: Changes reflected immediately on next request
- **Efficient**: Only invalidates what changed
- **No Stale Data**: Operators control when cache is cleared
- **Performance**: Reduces unnecessary cache misses

## Monitoring & Metrics

### Key Metrics
- Response time (P50, P95, P99)
- Cache hit rate
- Database query time
- Redis latency
- Request rate

### Alerts
- P99 > 200ms
- Cache hit rate < 80%
- Database connection pool exhaustion
- Redis connection failures

## Scalability Considerations

### Horizontal Scaling
- Stateless API service (scale horizontally)
- Redis cluster for distributed caching
- PostgreSQL read replicas
- Load balancer for traffic distribution

### Vertical Scaling
- Database connection pool tuning
- Redis memory optimization
- Application memory for in-memory cache

## Technology Stack

- **API Framework**: FastAPI (async, high performance)
- **Database**: PostgreSQL (with read replicas)
- **Cache**: Redis (distributed caching)
- **ORM**: SQLAlchemy (async)
- **Validation**: Pydantic
- **Monitoring**: Prometheus metrics (future)

## Implementation Phases

### Phase 1: Core Service
- Database models
- Basic API endpoints
- Redis caching layer
- Database optimization

### Phase 2: Performance Tuning
- Query optimization
- Cache strategy refinement
- Response compression
- Monitoring integration

### Phase 3: Advanced Features
- Location-based search
- Advanced filtering
- Search functionality (Elasticsearch integration)
- Real-time status updates

