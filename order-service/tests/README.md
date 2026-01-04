# Test Suite

Comprehensive test suite for the Order Service microservice.

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_api_orders.py       # API endpoint tests
├── test_order_service.py     # Business logic tests
├── test_payment_service.py  # Payment service tests
├── test_payment_worker.py   # Payment worker tests
├── test_circuit_breaker.py  # Circuit breaker tests
└── test_models.py           # Database model tests
```

## Running Tests

### Install test dependencies
```bash
pip install -r requirements-test.txt
```

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=app --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_api_orders.py
```

### Run specific test
```bash
pytest tests/test_api_orders.py::test_create_order_success
```

### Run with verbose output
```bash
pytest -v
```

## Test Coverage

The test suite covers:

- **API Endpoints**: All order API routes (create, get, list)
- **Business Logic**: Order service operations
- **Payment Processing**: Mocked payment gateway and retry logic
- **Circuit Breaker**: Failure handling and recovery
- **Payment Worker**: Background processing
- **Database Models**: Order, OrderItem, PaymentAttempt models

## Test Database

Tests use an in-memory SQLite database (`sqlite+aiosqlite:///:memory:`) for fast, isolated testing. Each test gets a fresh database session.

## Mocking

- Redis is mocked for API tests
- Payment gateway is mocked with configurable success rates
- Circuit breaker behavior is tested in isolation

## Continuous Integration

These tests are designed to run in CI/CD pipelines without external dependencies (PostgreSQL, Redis).

