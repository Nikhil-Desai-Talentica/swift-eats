# Swift Eats

A scalable, resilient, and high-performance backend for a modern food delivery service built with microservices architecture.

## Project Structure

```
swift-eats/
├── order-service/          # Order management microservice
│   ├── app/                # Application code
│   ├── docker-compose.yml  # Local development setup
│   └── README.md           # Service documentation
├── menu-service/           # Menu & restaurant browse microservice
│   ├── app/                # Application code
│   ├── docker-compose.yml  # Local development setup
│   └── README.md           # Service documentation
├── logistics-service/      # Real-time logistics & analytics microservice
│   ├── app/                # Application code
│   ├── docker-compose.yml  # Local development setup
│   └── README.md           # Service documentation
└── README.md               # This file
```

## Services

### Order Service

The first microservice in the platform, designed to handle order acceptance at scale (500+ orders/minute) with resilience against third-party service failures.

**Key Features:**
- High-throughput order acceptance
- Asynchronous payment processing
- Circuit breaker pattern for resilience
- Automatic retry with exponential backoff
- Mocked payment gateway for testing

See [order-service/README.md](./order-service/README.md) for detailed documentation.

### Menu Service

High-performance microservice for browsing restaurants and menus. Designed to achieve P99 response times under 200ms even under heavy user load.

**Key Features:**
- P99 < 200ms response time
- Multi-layer caching with Redis
- Event-based cache invalidation
- Operator API for status and menu updates
- Optimized database queries and indexes

See [menu-service/README.md](./menu-service/README.md) for detailed documentation.

### Logistics Service

Real-time logistics and analytics microservice for driver location tracking. Designed to handle 10,000 concurrent drivers sending GPS updates every 5 seconds.

**Key Features:**
- Reliable message processing (no message loss)
- Real-time WebSocket updates for customers
- Geospatial queries for nearby drivers
- Historical analytics with TimescaleDB
- Handles 2,000+ events/second

See [logistics-service/README.md](./logistics-service/README.md) for detailed documentation.

## Getting Started

Each service has its own README with setup instructions. Start with the order service:

```bash
cd order-service
docker-compose up -d
```

## Architecture

The system is designed as a collection of independent microservices, each responsible for a specific domain:

- **Order Service**: Handles order creation and payment processing
- **Menu Service**: High-performance restaurant and menu browsing
- **Logistics Service**: Real-time driver location tracking and analytics
- More services to be added...

## Development

Each service can be developed and deployed independently. See individual service READMEs for development setup.
