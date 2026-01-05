# API Specification

All services use JSON over HTTP. Base prefixes:

- Order Service:        /api/v1
- Menu Service:         /api/v1
- Logistics Service:    /api/v1

Health & root endpoints per service (not repeated below):

- GET /health
- GET /

---

## Order Service (order-service, port 8000)

Base: /api/v1/orders

### POST /api/v1/orders

- **Description**: Create a new order, enqueue payment processing.
- **Status**: 202 Accepted
- **Body**:
  - customer_id: string
  - items: [{ item_id, item_name, quantity, price }]
- **Response**: Order object (status initially `payment_processing`).

### GET /api/v1/orders/{order_id}

- **Description**: Get order by ID.
- **Status**: 200 OK / 404 Not Found
- **Response**: Order object with items and payment_attempts.

### GET /api/v1/orders

- **Description**: List orders with filters and pagination.
- **Query params**:
  - customer_id: string (optional)
  - status: one of [pending, payment_processing, confirmed, payment_failed, cancelled]
  - page: int (default 1)
  - page_size: int (default 20, max 100)
- **Status**: 200 OK
- **Response**: { orders: [...], total, page, page_size }

---

## Menu Service (menu-service, port 8001)

Base: /api/v1/restaurants

### GET /api/v1/restaurants

- **Description**: List restaurants with filters and pagination.
- **Query params**:
  - cuisine_type: enum CuisineType (optional)
  - status: enum RestaurantStatus (optional)
  - min_rating: float 0–5 (optional)
  - city: string (optional)
  - page: int (default 1)
  - page_size: int (default 20, max 100)
- **Status**: 200 OK
- **Response**: { restaurants: [...], total, page, page_size }

### GET /api/v1/restaurants/{restaurant_id}

- **Description**: Get restaurant details.
- **Status**: 200 OK / 404 Not Found
- **Response**: Restaurant object.

### GET /api/v1/restaurants/{restaurant_id}/menu

- **Description**: Get full menu for restaurant (categories + items).
- **Status**: 200 OK / 404 Not Found
- **Response**: MenuResponse (restaurant_id, categories, updated_at).

### GET /api/v1/restaurants/{restaurant_id}/status

- **Description**: Get current restaurant status and capacity.
- **Status**: 200 OK / 404 Not Found
- **Response**: { restaurant_id, status, current_capacity, max_capacity, updated_at }.

### GET /api/v1/restaurants/{restaurant_id}/menu-with-status

- **Description**: Combined restaurant, menu, and status.
- **Status**: 200 OK / 404 Not Found
- **Response**: { restaurant, menu, status }.

### PUT /api/v1/restaurants/{restaurant_id}/status

- **Description**: Operator: update restaurant status and capacity.
- **Body**:
  - status: enum RestaurantStatus
  - current_capacity: int (optional)
- **Status**: 200 OK / 404 Not Found
- **Response**: { success, message, restaurant_id, status }.

### PUT /api/v1/restaurants/{restaurant_id}/menu/items/{item_id}/availability

- **Description**: Operator: update a single menu item availability/in_stock.
- **Body**:
  - available: bool
  - in_stock: bool
- **Status**: 200 OK / 404 Not Found
- **Response**: { success, message, restaurant_id, item_id, available, in_stock }.

### PUT /api/v1/restaurants/{restaurant_id}/menu/items/availability

- **Description**: Operator: bulk update multiple menu items availability.
- **Body**:
  - items: [{ item_id, available, in_stock }]
- **Status**: 200 OK / 404 Not Found
- **Response**: { success, message, restaurant_id, updated_items: [{ item_id, available, in_stock }] }.

---

## Logistics Service (logistics-service, port 8002)

Base: /api/v1/drivers

### POST /api/v1/drivers/{driver_id}/location

- **Description**: Ingest GPS update from driver (reliable queue).
- **Status**: 202 Accepted / 500 Internal Server Error
- **Body**:
  - latitude: float (-90..90)
  - longitude: float (-180..180)
  - speed: float (optional)
  - heading: float 0–360 (optional)
  - accuracy: float (optional)
  - order_id: string (optional)
- **Response**: { status: "accepted", message_id, driver_id }.

### GET /api/v1/drivers/{driver_id}/location

- **Description**: Get current driver location (from Redis).
- **Status**: 200 OK / 404 Not Found
- **Response**: LocationResponse
  - driver_id, latitude, longitude, speed?, heading?, accuracy?, order_id?, timestamp.

### GET /api/v1/drivers/{driver_id}/route

- **Description**: Get driver route history (from TimescaleDB).
- **Query params**:
  - start_time: ISO datetime (optional, default: now - 1h)
  - end_time: ISO datetime (optional, default: now)
  - limit: int (default 1000)
- **Status**: 200 OK / 404 Not Found
- **Response**: { driver_id, locations: [LocationResponse], start_time, end_time }.

### GET /api/v1/drivers/nearby

- **Description**: Find nearby drivers using Redis GEO.
- **Query params**:
  - lat: float
  - lng: float
  - radius: float (meters, default 1000)
  - limit: int (default 50)
- **Status**: 200 OK
- **Response**: { drivers: [{ driver_id, latitude, longitude, distance }], center_latitude, center_longitude, radius }.

### GET /api/v1/drivers/orders/{order_id}/driver-location

- **Description**: (Placeholder) Get driver location by order ID.
- **Status**: 501 Not Implemented.

### WebSocket /api/v1/drivers/orders/{order_id}/driver-location

- **Description**: Real-time driver location stream for a given order.
- **Connect**: ws://localhost:8002/api/v1/drivers/orders/{order_id}/driver-location
- **Server → Client**:
  - On connect: `{ "type": "connected", "order_id", "message" }`
  - Location updates: `{ "type": "location_update", "data": { driver_id, latitude, longitude, speed?, heading?, timestamp, order_id? } }`
- **Client → Server**:
  - `{ "action": "unsubscribe" }` to stop streaming.

---