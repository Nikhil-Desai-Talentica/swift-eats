# Swift Eats - System Architecture

## Table of Contents

1. [Overview](#overview)
2. [Why Microservices?](#why-microservices)
3. [System Architecture](#system-architecture)
4. [Service Architectures](#service-architectures)
5. [Technology Choices](#technology-choices)
6. [Communication Patterns](#communication-patterns)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [Scalability & Deployment](#scalability--deployment)
9. [Future Considerations](#future-considerations)

---

## Overview

Swift Eats is a food delivery platform backend built as a collection of independent microservices. The system is designed to handle:

- **High-throughput order processing**: 500+ orders per minute
- **Low-latency browsing**: P99 < 200ms for menu/restaurant queries
- **Real-time logistics**: 10,000 concurrent drivers, 2,000 events/second

### Core Services

1. **Order Service**: Handles order creation, payment processing, and order lifecycle
2. **Menu Service**: High-performance restaurant and menu browsing with caching
3. **Logistics Service**: Real-time driver location tracking and analytics

---

## Why Microservices?

### 1. **Independent Scalability**

Each service has different scaling requirements:
- **Order Service**: CPU-intensive during payment processing, scales with order volume
- **Menu Service**: Memory-intensive (caching), scales with read traffic
- **Logistics Service**: Network-intensive (WebSocket connections), scales with concurrent drivers

Microservices allow us to scale each service independently based on its specific needs.

### 2. **Technology Diversity**

Different services benefit from different technologies:
- **Order Service**: Needs reliable queues and transaction support
- **Menu Service**: Needs high-performance caching and read optimization
- **Logistics Service**: Needs time-series database and real-time messaging

Microservices allow each service to use the best technology for its domain.

### 3. **Team Autonomy**

- Teams can develop, deploy, and maintain services independently
- Faster iteration cycles
- Reduced coordination overhead
- Technology choices don't affect other teams

### 4. **Fault Isolation**

- Failures in one service don't cascade to others
- Order service payment failures don't affect menu browsing
- Logistics service issues don't impact order creation
- Better system resilience

### 5. **Domain-Driven Design**

Each service owns a clear business domain:
- **Order Service**: Order management domain
- **Menu Service**: Restaurant/menu catalog domain
- **Logistics Service**: Driver tracking domain

Clear boundaries enable better code organization and maintainability.

### Trade-offs

**Challenges:**
- Increased operational complexity
- Network latency between services
- Distributed transaction management
- Service discovery and configuration

**Mitigations:**
- Async communication patterns
- Event-driven architecture
- Comprehensive monitoring
- API versioning and contracts

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Swift Eats Platform                     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Order      │    │    Menu      │    │  Logistics   │
│   Service    │    │   Service    │    │   Service    │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                    │                    │
       ├──► PostgreSQL      ├──► PostgreSQL      ├──► TimescaleDB
       ├──► Redis Queue     ├──► Redis Cache     ├──► Redis Streams
       └──► Payment Worker  └──► (Operator API)  └──► Location Worker
```

### Component Communication

```
┌─────────────┐
│   Clients   │
│ (Web/Mobile)│
└──────┬──────┘
       │
       ├─────────────────────────────────────────┐
       │                                         │
       ▼                                         ▼
┌──────────────┐                        ┌──────────────┐
│   Order      │                        │    Menu     │
│   Service    │                        │   Service   │
│              │                        │             │
│  Port: 8000  │                        │ Port: 8001  │
└──────┬───────┘                        └──────┬──────┘
       │                                        │
       │                                        │
       ▼                                        ▼
┌──────────────┐                        ┌──────────────┐
│  PostgreSQL  │                        │  PostgreSQL  │
│  Port: 5432  │                        │  Port: 5433  │
└──────────────┘                        └──────────────┘
       │                                        │
       │                                        │
       └──────────────┬─────────────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │    Redis     │
              │  Port: 6379  │
              │  Port: 6380  │
              │  Port: 6381  │
              └──────┬───────┘
                     │
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Order      │ │    Menu      │ │  Logistics   │
│   Service    │ │   Service    │ │   Service    │
│              │ │              │ │              │
│  Workers     │ │  (Caching)    │ │  Workers     │
└──────────────┘ └──────────────┘ └──────┬───────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │ TimescaleDB  │
                                   │  Port: 5434  │
                                   └──────────────┘
```

---

## Service Architectures

### 1. Order Service

**Purpose**: Handle order creation and payment processing at scale

**Architecture**:

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /api/v1/orders
       ▼
┌─────────────────────────────────┐
│   Order Service (FastAPI)       │
│  - Accept orders                │
│  - Validate order data          │
│  - Persist to DB                │
│  - Enqueue for payment          │
└──────┬──────────────────────────┘
       │
       ├──► PostgreSQL (Order DB)
       │    - Orders table
       │    - Order items table
       │    - Payment attempts table
       │
       └──► Redis Queue (Payment Queue)
              │
              ▼
       ┌──────────────────────────┐
       │  Payment Worker          │
       │  - Process payment       │
       │  - Update order status   │
       │  - Mock payment gateway  │
       │  - Circuit breaker       │
       └──────────────────────────┘
```

**Key Design Decisions**:

1. **Asynchronous Payment Processing**
   - Orders accepted immediately (202 Accepted)
   - Payment processed in background
   - Prevents payment gateway failures from blocking order acceptance

2. **Circuit Breaker Pattern**
   - Protects against payment gateway failures
   - Prevents cascading failures
   - Automatic recovery after timeout

3. **Retry Logic**
   - Exponential backoff for failed payments
   - Configurable retry attempts
   - Payment attempts tracked in database

**Technology Justification**:
- **FastAPI**: High-performance async framework, perfect for high-throughput APIs
- **PostgreSQL**: ACID compliance for financial transactions
- **Redis Queue**: Lightweight, fast message queue for payment processing
- **Circuit Breaker**: Resilience pattern for third-party service failures

---

### 2. Menu Service

**Purpose**: High-performance restaurant and menu browsing with P99 < 200ms

**Architecture**:

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ GET /api/v1/restaurants/{id}/menu
       ▼
┌─────────────────────────────────┐
│  Menu Service (FastAPI)         │
│  - Multi-layer caching           │
│  - Optimized queries            │
│  - Response compression          │
└──────┬──────────────────────────┘
       │
       ├──► Redis Cache (L1)
       │    - Restaurant data (TTL: 1h)
       │    - Menu data (TTL: 30m)
       │    - Status data (TTL: 30s)
       │
       └──► PostgreSQL (L2)
            - Optimized indexes
            - Eager loading
            - Strategic queries
```

**Key Design Decisions**:

1. **Multi-Layer Caching**
   - Redis for distributed caching
   - Database as fallback
   - Cache-aside pattern

2. **Event-Based Cache Invalidation**
   - Operators update data → Cache invalidates automatically
   - TTL as fallback safety mechanism
   - Immediate consistency for updates

3. **Database Optimization**
   - Composite indexes for common queries
   - Eager loading to avoid N+1 queries
   - Connection pooling

**Technology Justification**:
- **FastAPI**: Async framework for high concurrency
- **Redis**: Sub-millisecond cache reads, distributed caching
- **PostgreSQL**: Reliable relational database with advanced indexing
- **Gzip Compression**: Reduces response size, improves latency

---

### 3. Logistics Service

**Purpose**: Real-time driver location tracking (10K drivers, 2K events/sec)

**Architecture**:

```
┌─────────────┐
│   Drivers   │ (10,000 concurrent)
│  GPS Devices│
└──────┬──────┘
       │ POST /api/v1/drivers/{id}/location
       ▼
┌─────────────────────────────────┐
│  Location Ingestion Service     │
│  - Accept location updates      │
│  - Validate coordinates          │
│  - Add to Redis Streams          │
└──────┬──────────────────────────┘
       │
       └──► Redis Streams (Reliable Queue)
              │
              ▼
       ┌──────────────────────────┐
       │  Location Processing      │
       │  Worker                   │
       │  - Read from stream       │
       │  - Process location       │
       │  - ACK on success         │
       └──────┬─────────────────────┘
              │
              ├──► Redis (Current Locations + GEO)
              │    - Fast reads (< 1ms)
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

**Key Design Decisions**:

1. **Reliable Queue Pattern**
   - Messages only removed after successful processing
   - Automatic retry for failed messages
   - Dead letter queue for persistent failures
   - No message loss guarantee

2. **Dual Storage Strategy**
   - Redis for current locations (fast reads)
   - TimescaleDB for historical data (analytics)
   - Optimized for each use case

3. **Real-Time Distribution**
   - Redis Pub/Sub for lightweight distribution
   - WebSocket for customer connections
   - Low-latency updates (< 50ms)

**Technology Justification**:
- **Redis Streams**: Reliable message queue, already in infrastructure
- **TimescaleDB**: Time-series optimization for GPS data
- **Redis GEO**: Built-in geospatial queries
- **WebSocket**: Real-time bidirectional communication

---

## Technology Choices

### Core Technologies

#### FastAPI
**Why**: 
- High-performance async framework
- Built-in WebSocket support
- Automatic API documentation
- Type validation with Pydantic
- Excellent for high-throughput services

**Alternatives Considered**:
- Django: Too heavy, synchronous by default
- Flask: Lower performance, no async support
- Node.js/Express: Python ecosystem preferred

#### PostgreSQL
**Why**:
- ACID compliance for financial data (Order Service)
- Advanced indexing capabilities (Menu Service)
- Reliable and battle-tested
- Rich ecosystem and tooling

**Alternatives Considered**:
- MongoDB: No ACID guarantees, not ideal for financial data
- MySQL: PostgreSQL has better JSON support and advanced features

#### TimescaleDB
**Why**:
- PostgreSQL extension (familiar SQL)
- Automatic partitioning for time-series data
- Built-in compression and retention policies
- Perfect for GPS location data

**Alternatives Considered**:
- InfluxDB: Different query language, less familiar
- MongoDB Time Series: Less mature, PostgreSQL preferred

#### Redis
**Why**:
- Multi-purpose: Cache, Queue, Pub/Sub, GEO
- Sub-millisecond latency
- Built-in data structures (Streams, GEO, Pub/Sub)
- Already in infrastructure (consistency)

**Alternatives Considered**:
- Memcached: No advanced features (Streams, GEO)
- RabbitMQ: Additional infrastructure, Redis sufficient
- Kafka: Overkill for our scale, Redis Streams sufficient

### Design Patterns

#### Circuit Breaker (Order Service)
**Why**: Protects against third-party service failures
**Implementation**: Custom Python implementation
**Benefits**: Prevents cascading failures, automatic recovery

#### Cache-Aside (Menu Service)
**Why**: Simple, reliable caching pattern
**Implementation**: Redis with TTL and event-based invalidation
**Benefits**: High cache hit rates, immediate consistency

#### Reliable Queue (Logistics Service)
**Why**: Guarantee no message loss
**Implementation**: Redis Streams with consumer groups
**Benefits**: Fault tolerance, automatic retry, dead letter queue

---

## Communication Patterns

### 1. Synchronous HTTP (REST APIs)

**Used For**:
- Client → Service communication
- Service → Service communication (when needed)
- Operator API endpoints

**Example**:
```
Client → POST /api/v1/orders → Order Service
Client → GET /api/v1/restaurants → Menu Service
Driver → POST /api/v1/drivers/{id}/location → Logistics Service
```

### 2. Asynchronous Message Queue

**Used For**:
- Order Service → Payment Worker
- Logistics Service → Location Processor

**Example**:
```
Order Service → Redis Queue → Payment Worker
Logistics Service → Redis Streams → Location Processor
```

**Benefits**:
- Decoupling
- Resilience
- Scalability

### 3. Pub/Sub (Real-Time)

**Used For**:
- Logistics Service → WebSocket Service
- Real-time location updates

**Example**:
```
Location Processor → Redis Pub/Sub → WebSocket → Customer
```

**Benefits**:
- Low latency
- Efficient distribution
- Scalable

### 4. WebSocket (Bidirectional)

**Used For**:
- Customer → Logistics Service (real-time driver tracking)

**Example**:
```
Customer ←→ WebSocket ←→ Logistics Service
```

**Benefits**:
- Real-time updates
- Low overhead
- Bidirectional communication

---

## Data Flow Diagrams

### Order Creation Flow

```
┌─────────┐
│ Client  │
└────┬────┘
     │ 1. POST /orders
     ▼
┌─────────────────┐
│  Order Service  │
│  (FastAPI)      │
└────┬────────────┘
     │ 2. Validate & Persist
     ▼
┌─────────────────┐      ┌─────────────────┐
│   PostgreSQL    │      │   Redis Queue    │
│   (Orders DB)   │      │  (Payment Queue) │
└─────────────────┘      └────────┬─────────┘
                                   │
     ┌─────────────────────────────┘
     │ 3. Return 202 Accepted
     ▼
┌─────────┐
│ Client  │
└─────────┘

     │ 4. Worker processes
     ▼
┌─────────────────┐
│ Payment Worker  │
└────┬────────────┘
     │ 5. Process payment
     ▼
┌─────────────────┐      ┌─────────────────┐
│ Payment Gateway │      │   PostgreSQL    │
│    (Mocked)     │      │  (Update Order) │
└─────────────────┘      └─────────────────┘
```

### Menu Browsing Flow

```
┌─────────┐
│ Client  │
└────┬────┘
     │ 1. GET /restaurants/{id}/menu
     ▼
┌─────────────────┐
│  Menu Service   │
│  (FastAPI)      │
└────┬────────────┘
     │ 2. Check cache
     ▼
┌─────────────────┐
│  Redis Cache    │
│  (L1 Cache)     │
└────┬────────────┘
     │
     ├──► Cache Hit → Return (P99 < 10ms)
     │
     └──► Cache Miss
          │ 3. Query database
          ▼
     ┌─────────────────┐
     │   PostgreSQL    │
     │  (Menu Data)    │
     └────┬────────────┘
          │ 4. Cache result
          ▼
     ┌─────────────────┐
     │  Redis Cache    │
     └────┬────────────┘
          │ 5. Return response
          ▼
     ┌─────────┐
     │ Client  │
     └─────────┘
```

### Driver Location Update Flow

```
┌─────────┐
│ Driver  │
└────┬────┘
     │ 1. POST /drivers/{id}/location
     ▼
┌─────────────────────────┐
│ Location Ingestion      │
│ Service (FastAPI)       │
└────┬────────────────────┘
     │ 2. Add to stream
     ▼
┌─────────────────────────┐
│   Redis Streams         │
│  (Reliable Queue)       │
└────┬────────────────────┘
     │ 3. Worker reads (moves to pending)
     ▼
┌─────────────────────────┐
│ Location Processor      │
│ Worker                  │
└────┬────────────────────┘
     │
     ├──► 4a. Update Redis (current location)
     │    ┌─────────────────┐
     │    │  Redis (GEO)    │
     │    └─────────────────┘
     │
     ├──► 4b. Store historical
     │    ┌─────────────────┐
     │    │  TimescaleDB    │
     │    └─────────────────┘
     │
     └──► 4c. Publish update
          ┌─────────────────┐
          │  Redis Pub/Sub  │
          └────┬────────────┘
               │ 5. Real-time update
               ▼
          ┌─────────────────┐
          │  WebSocket      │
          │  Service        │
          └────┬────────────┘
               │ 6. Push to customer
               ▼
          ┌─────────┐
          │Customer │
          └─────────┘
     
     │ 7. ACK message (removes from pending)
     ▼
┌─────────────────────────┐
│   Redis Streams         │
│  (Message removed)      │
└─────────────────────────┘
```

### Cache Invalidation Flow (Menu Service)

```
┌──────────────┐
│   Operator   │
└──────┬───────┘
       │ PUT /restaurants/{id}/status
       ▼
┌─────────────────┐
│  Menu Service   │
│  (FastAPI)      │
└────┬────────────┘
     │ 1. Update database
     ▼
┌─────────────────┐
│   PostgreSQL    │
│  (Restaurant)   │
└────┬────────────┘
     │ 2. Invalidate cache
     ▼
┌─────────────────┐
│  Redis Cache    │
│  (Delete keys)  │
└────┬────────────┘
     │ 3. Return success
     ▼
┌──────────────┐
│   Operator   │
└──────────────┘

     │ Next request
     ▼
┌─────────┐
│ Client  │
└────┬────┘
     │ GET /restaurants/{id}
     ▼
┌─────────────────┐
│  Menu Service   │
└────┬────────────┘
     │ Cache miss → Fetch from DB
     ▼
┌─────────────────┐
│   PostgreSQL    │
└────┬────────────┘
     │ Cache fresh data
     ▼
┌─────────────────┐
│  Redis Cache    │
└─────────────────┘
```

---

## Scalability & Deployment

### Horizontal Scaling

All services are **stateless** and can scale horizontally:

```
┌─────────────┐
│ Load        │
│ Balancer    │
└──────┬──────┘
       │
   ┌───┼───┐
   │   │   │
   ▼   ▼   ▼
┌────┐┌────┐┌────┐
│ S1 ││ S2 ││ S3 │  Order Service Instances
└────┘└────┘└────┘
```

### Database Scaling

**PostgreSQL**:
- Read replicas for read-heavy services (Menu Service)
- Connection pooling per service
- Vertical scaling for write-heavy services

**TimescaleDB**:
- Read replicas for analytics queries
- Automatic partitioning (hypertables)
- Compression for old data

**Redis**:
- Redis Cluster for high availability
- Different databases per service (isolation)
- Memory optimization

### Deployment Strategy

**Containerization**:
- Docker containers for each service
- Docker Compose for local development
- Kubernetes-ready (for production)

**Service Discovery**:
- Environment variables for service URLs
- Can be replaced with service mesh (Istio, Consul)

**Monitoring**:
- Health check endpoints per service
- Logs to stdout (container-friendly)
- Ready for Prometheus/Grafana integration

---

## Future Considerations

### Service Mesh
- **Istio** or **Linkerd** for advanced traffic management
- Mutual TLS for service-to-service communication
- Advanced observability

### API Gateway
- **Kong** or **Envoy** for unified API entry point
- Rate limiting, authentication, routing
- Request/response transformation

### Event-Driven Architecture
- **Event bus** for service-to-service communication
- Decoupled services via events
- Event sourcing for audit trails

### Caching Layer
- **CDN** for static menu data
- **Varnish** or **CloudFlare** for edge caching
- Reduced latency for global users

### Database Optimization
- **Read replicas** for Menu Service
- **Sharding** if needed for scale
- **Connection pooling** optimization

### Monitoring & Observability
- **Prometheus** for metrics
- **Grafana** for visualization
- **Jaeger** or **Zipkin** for distributed tracing
- **ELK Stack** for log aggregation

### Security
- **API authentication** (JWT, OAuth)
- **Rate limiting** per service
- **Encryption** at rest and in transit
- **Secrets management** (Vault, AWS Secrets Manager)

---

## Summary

Swift Eats is built as a **microservices architecture** to achieve:

1. **Independent Scalability**: Each service scales based on its needs
2. **Technology Diversity**: Best tool for each job
3. **Fault Isolation**: Failures don't cascade
4. **Team Autonomy**: Independent development and deployment
5. **Domain Clarity**: Clear service boundaries

The architecture leverages:
- **FastAPI** for high-performance APIs
- **PostgreSQL/TimescaleDB** for reliable data storage
- **Redis** for caching, queuing, and real-time distribution
- **Async patterns** for high throughput
- **Reliable queues** for fault tolerance

This design enables the platform to handle:
- ✅ 500+ orders/minute
- ✅ P99 < 200ms for browsing
- ✅ 10,000 concurrent drivers
- ✅ 2,000+ events/second

While maintaining:
- ✅ High availability
- ✅ Fault tolerance
- ✅ Scalability
- ✅ Maintainability

