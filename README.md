# Swift Eats

A scalable, resilient, and high-performance backend for a modern food delivery service built with microservices architecture.

## Project Structure

```
swift-eats/
├── order-service/          # Order management microservice
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

## Getting Started

Each service has its own README with setup instructions. Start with the order service:

```bash
cd order-service
docker-compose up -d
```

## Architecture

The system is designed as a collection of independent microservices, each responsible for a specific domain:

- **Order Service**: Handles order creation and payment processing
- More services to be added...

## Development

Each service can be developed and deployed independently. See individual service READMEs for development setup.
