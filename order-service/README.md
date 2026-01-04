# Order Service

A scalable, resilient microservice for handling food delivery orders at scale. Designed to handle peak loads of 500 orders per minute while maintaining reliability even when third-party services (like payment gateways) are unavailable.

## Features

- **High Throughput**: Handles 500+ orders per minute with async processing
- **Resilient Design**: Circuit breaker pattern protects against third-party service failures
- **Asynchronous Processing**: Orders are accepted immediately and processed in the background
- **Retry Logic**: Automatic retry with exponential backoff for failed payments
- **Scalable Architecture**: Stateless API service that can scale horizontally
- **Mocked Payment Gateway**: Built-in payment gateway simulation for testing

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
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
       │
       └──► Redis Queue (Payment Queue)
              │
              ▼
       ┌──────────────────────────┐
       │  Payment Worker          │
       │  - Process payment       │
       │  - Update order status   │
       │  - Mock payment gateway  │
       └──────────────────────────┘
```

## Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **PostgreSQL**: Reliable relational database for order persistence
- **Redis**: In-memory data store for message queuing
- **SQLAlchemy**: Async ORM for database operations
- **Pydantic**: Data validation using Python type annotations

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Running with Docker Compose

1. Clone the repository and navigate to the order-service directory:
```bash
cd order-service
```

2. Start all services:
```bash
docker-compose up -d
```

This will start:
- PostgreSQL database (port 5432)
- Redis (port 6379)
- Order Service API (port 8000)
- Payment Worker

3. Check service health:
```bash
curl http://localhost:8000/health
```

4. Access API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

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

5. Run the payment worker (in a separate terminal):
```bash
python -m app.workers.payment_worker
```

## API Endpoints

### Create Order
```bash
POST /api/v1/orders
Content-Type: application/json

{
  "customer_id": "customer_123",
  "items": [
    {
      "item_id": "item_001",
      "item_name": "Burger",
      "quantity": 2,
      "price": 12.99
    }
  ]
}
```

**Response**: `202 Accepted` with order details

### Get Order
```bash
GET /api/v1/orders/{order_id}
```

### List Orders
```bash
GET /api/v1/orders?customer_id=customer_123&status=confirmed&page=1&page_size=20
```

## Order Flow

1. **Order Submission**: Client submits order via POST `/api/v1/orders`
2. **Immediate Response**: API returns `202 Accepted` with order ID
3. **Database Persistence**: Order saved to PostgreSQL with status `PENDING`
4. **Queue Enqueue**: Order added to Redis queue for payment processing
5. **Status Update**: Order status changed to `PAYMENT_PROCESSING`
6. **Worker Processing**: Payment worker picks up order from queue
7. **Payment Processing**: Worker processes payment through mocked gateway
8. **Status Update**: Order status updated to `CONFIRMED` or `PAYMENT_FAILED`

## Order Statuses

- `PENDING`: Order created, awaiting payment processing
- `PAYMENT_PROCESSING`: Payment is being processed
- `CONFIRMED`: Payment successful, order confirmed
- `PAYMENT_FAILED`: Payment failed after all retry attempts
- `CANCELLED`: Order cancelled

## Resilience Features

### Circuit Breaker
- Automatically opens after 5 consecutive failures
- Prevents cascading failures when payment gateway is down
- Attempts recovery after 60 seconds

### Retry Logic
- Automatic retry with exponential backoff
- Configurable retry attempts (default: 3)
- Each retry attempt is logged in the database

### Asynchronous Processing
- Orders are accepted immediately
- Payment processing happens in background
- System remains responsive even under high load

## Configuration

Key configuration options in `.env`:

- `PAYMENT_MOCK_SUCCESS_RATE`: Payment success rate (default: 0.95)
- `PAYMENT_MOCK_LATENCY_MIN/MAX`: Simulated payment latency in ms
- `CIRCUIT_BREAKER_FAILURE_THRESHOLD`: Failures before opening circuit
- `PAYMENT_RETRY_ATTEMPTS`: Number of retry attempts

## Monitoring

- Health check endpoint: `/health`
- Logs are output to stdout (configure logging level via `LOG_LEVEL`)
- All payment attempts are tracked in the database

## Performance

- **Throughput**: Designed for 500+ orders/minute
- **Response Time**: < 50ms for order acceptance
- **Database**: Connection pooling (20-50 connections)
- **Queue**: Redis handles thousands of operations per second

## Testing

Example order creation:
```bash
curl -X POST "http://localhost:8000/api/v1/orders" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "customer_123",
    "items": [
      {
        "item_id": "item_001",
        "item_name": "Burger",
        "quantity": 2,
        "price": 12.99
      }
    ]
  }'
```

## Testing

The service includes a comprehensive test suite covering all components.

### Running Tests

1. Install test dependencies:
```bash
pip install -r requirements-test.txt
```

2. Run all tests:
```bash
pytest
```

3. Run with coverage:
```bash
pytest --cov=app --cov-report=html
```

4. Run specific test file:
```bash
pytest tests/test_api_orders.py
```

### Test Coverage

- **API Endpoints**: All order routes (create, get, list)
- **Business Logic**: Order service operations
- **Payment Processing**: Mocked gateway and retry logic
- **Circuit Breaker**: Failure handling and recovery
- **Payment Worker**: Background processing
- **Database Models**: All models and relationships

See [tests/README.md](./tests/README.md) for detailed test documentation.

## Future Enhancements

- [ ] Real payment gateway integration
- [ ] Webhook notifications for order status changes
- [ ] Metrics and monitoring (Prometheus/Grafana)
- [ ] Distributed tracing
- [ ] Rate limiting
- [ ] Order cancellation API
- [ ] Idempotency keys for duplicate prevention

## License

Part of the Swift Eats food delivery platform.

