# Project Structure

## Overview

Swift Eats is organized as three independent microservices, each with its own codebase, database, and infrastructure.

```
swift-eats/
├── ARCHITECTURE.md              # System architecture documentation
├── PROJECT_STRUCTURE.md         # This file
├── API-SPECIFICATION.md        # API endpoints documentation
├── README.md                    # Project overview
├── order-service/               # Order management microservice
├── menu-service/                # Menu & restaurant browse microservice
└── logistics-service/          # Real-time logistics & analytics microservice
```

---

## order-service

**Purpose**: Order creation and payment processing  
**Port**: 8000  
**Database**: PostgreSQL (swifteats)  
**Queue**: Redis (DB 0)

### Directory Structure

```
order-service/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entrypoint
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── orders.py        # Order HTTP endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Application settings
│   │   ├── database.py           # SQLAlchemy async engine & session
│   │   └── circuit_breaker.py   # Circuit breaker for payment gateway
│   ├── models/
│   │   ├── __init__.py
│   │   └── order.py             # Order, OrderItem, PaymentAttempt models
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── order.py             # Pydantic request/response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── order_service.py     # Order business logic
│   │   └── payment_service.py   # Mock payment gateway service
│   └── workers/
│       ├── __init__.py
│       └── payment_worker.py    # Background payment processing worker
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest fixtures
│   ├── test_api_orders.py       # API endpoint tests
│   ├── test_order_service.py   # Service layer tests
│   ├── test_payment_service.py # Payment service tests
│   ├── test_payment_worker.py  # Worker tests
│   ├── test_circuit_breaker.py # Circuit breaker tests
│   └── test_models.py          # Model tests
├── docker-compose.yml          # Postgres + Redis + Service + Worker
├── Dockerfile
├── requirements.txt
├── requirements-test.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

### Key Components

- **API Layer**: FastAPI routes for order CRUD operations
- **Service Layer**: Business logic for orders and payments
- **Worker**: Background payment processing with retry logic
- **Circuit Breaker**: Resilience pattern for payment gateway failures

---

## menu-service

**Purpose**: High-performance restaurant and menu browsing  
**Port**: 8001  
**Database**: PostgreSQL (swifteats_menu)  
**Cache**: Redis (DB 1)

### Directory Structure

```
menu-service/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application (GZip enabled)
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── restaurants.py   # Restaurant & menu HTTP endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Application settings
│   │   ├── database.py           # SQLAlchemy async engine & session
│   │   └── cache.py             # Redis cache service with invalidation
│   ├── models/
│   │   ├── __init__.py
│   │   └── restaurant.py        # Restaurant, Menu, MenuCategory, MenuItem models
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── restaurant.py        # Pydantic request/response schemas
│   └── services/
│       ├── __init__.py
│       └── restaurant_service.py # Restaurant & menu business logic
├── docker-compose.yml          # Postgres + Redis + Service
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── DESIGN.md                   # Service design document
└── README.md
```

### Key Components

- **API Layer**: FastAPI routes for browsing and operator updates
- **Cache Layer**: Multi-layer caching with event-based invalidation
- **Service Layer**: Business logic with cache integration
- **Operator API**: Endpoints for restaurant operators to update data

---

## logistics-service

**Purpose**: Real-time driver location tracking and analytics  
**Port**: 8002  
**Database**: TimescaleDB (swifteats_logistics)  
**Queue**: Redis Streams (DB 2)

### Directory Structure

```
logistics-service/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── locations.py     # Location HTTP + WebSocket endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Application settings
│   │   ├── database.py          # TimescaleDB async engine & session
│   │   └── redis_streams.py     # Reliable Redis Streams processor
│   ├── models/
│   │   ├── __init__.py
│   │   └── location.py          # DriverLocation time-series model
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── location.py          # Pydantic request/response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   └── location_service.py  # Location business logic (Redis GEO, Pub/Sub)
│   └── workers/
│       ├── __init__.py
│       └── location_processor.py # Reliable queue processing worker
├── docker-compose.yml          # TimescaleDB + Redis + Service + Worker
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── DESIGN.md                   # Service design document
└── README.md
```

### Key Components

- **API Layer**: FastAPI routes + WebSocket for real-time updates
- **Stream Processor**: Reliable Redis Streams with consumer groups
- **Location Service**: Current location, historical data, GEO queries, Pub/Sub
- **Worker**: Processes location updates with guaranteed delivery

---

## Common Patterns

### Application Structure

All services follow a consistent structure:

```
app/
├── main.py              # FastAPI app initialization
├── api/routes/          # HTTP/WebSocket endpoints
├── core/                # Configuration, database, shared utilities
├── models/              # SQLAlchemy database models
├── schemas/             # Pydantic validation schemas
├── services/            # Business logic layer
└── workers/             # Background processing (if applicable)
```

### Configuration

Each service uses:
- `app/core/config.py` - Pydantic Settings for configuration
- `.env.example` - Environment variable template
- Environment-based configuration (12-factor app)

### Database

- **Order Service**: PostgreSQL with async SQLAlchemy
- **Menu Service**: PostgreSQL with async SQLAlchemy
- **Logistics Service**: TimescaleDB (PostgreSQL extension) with async SQLAlchemy

### Testing

- `tests/` directory with pytest
- `conftest.py` for shared fixtures
- Unit tests for services, integration tests for APIs

### Docker

Each service includes:
- `Dockerfile` - Container image definition
- `docker-compose.yml` - Local development environment
- Separate containers for database, cache, service, and workers

---

## Infrastructure Components

### Databases

| Service | Database | Port | Purpose |
|---------|----------|------|---------|
| Order | PostgreSQL | 5432 | Order persistence |
| Menu | PostgreSQL | 5433 | Restaurant/menu data |
| Logistics | TimescaleDB | 5434 | Time-series location data |

### Redis Instances

| Service | Redis DB | Port | Purpose |
|---------|----------|------|---------|
| Order | DB 0 | 6379 | Payment queue |
| Menu | DB 1 | 6380 | Cache layer |
| Logistics | DB 2 | 6381 | Streams, GEO, Pub/Sub |

### Service Ports

| Service | Port | Health Check |
|---------|------|--------------|
| Order | 8000 | GET /health |
| Menu | 8001 | GET /health |
| Logistics | 8002 | GET /health |

---

## Dependencies

### Common Dependencies

All services share:
- `fastapi==0.104.1` - Web framework
- `uvicorn[standard]==0.24.0` - ASGI server
- `pydantic==2.5.0` - Data validation
- `sqlalchemy[asyncio]==2.0.23` - Async ORM
- `asyncpg==0.29.0` - PostgreSQL async driver
- `redis==5.0.1` - Redis client

### Service-Specific

- **Order Service**: Standard Redis client
- **Menu Service**: Standard Redis client
- **Logistics Service**: `redis[hiredis]==5.0.1` (async support)

---

## Development Workflow

### Local Development

1. Navigate to service directory
2. Copy `.env.example` to `.env`
3. Start infrastructure: `docker-compose up -d postgres redis`
4. Run service: `uvicorn app.main:app --reload`
5. Run workers (if applicable): `python -m app.workers.<worker>`

### Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests
pytest

# With coverage
pytest --cov=app --cov-report=html
```

### Docker Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f <service-name>

# Stop services
docker-compose down
```

---

## File Naming Conventions

- **Python modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **API routes**: Resource-based (`orders.py`, `restaurants.py`, `locations.py`)

---

## Documentation

- **ARCHITECTURE.md**: System-wide architecture and design decisions
- **API-SPECIFICATION.md**: Complete API endpoint documentation
- **PROJECT_STRUCTURE.md**: This file - project organization
- **README.md**: Project overview and getting started
- **Service READMEs**: Service-specific documentation in each service directory
- **DESIGN.md**: Service-specific design documents (menu-service, logistics-service)

