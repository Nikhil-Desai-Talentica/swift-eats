import pytest
from httpx import AsyncClient
from unittest.mock import patch


@pytest.mark.asyncio
async def test_update_location_accepted(client: AsyncClient):
    with patch("app.core.redis_streams.stream_processor.add_location_message", return_value="1-0") as mock_add:
        payload = {
            "latitude": 40.0,
            "longitude": -70.0,
            "speed": 30.0,
            "heading": 90.0,
            "accuracy": 5.0,
            "order_id": "order_1",
        }
        resp = await client.post("/api/v1/drivers/123/location", json=payload)
        assert resp.status_code == 202
        data = resp.json()
        assert data["driver_id"] == 123
        mock_add.assert_called_once()


@pytest.mark.asyncio
async def test_get_current_location_not_found(client: AsyncClient):
    resp = await client.get("/api/v1/drivers/999/location")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_current_location(client: AsyncClient):
    from app.services.location_service import location_service

    async def fake_get_current_location(driver_id: int):
        return {
            "driver_id": "123",
            "latitude": "40.0",
            "longitude": "-70.0",
            "timestamp": "2024-01-01T00:00:00Z",
            "speed": "30.0",
        }

    with patch.object(location_service, "get_current_location", side_effect=fake_get_current_location):
        resp = await client.get("/api/v1/drivers/123/location")
        assert resp.status_code == 200
        data = resp.json()
        assert data["driver_id"] == 123
        assert data["latitude"] == 40.0
        assert data["longitude"] == -70.0
        assert data["speed"] == 30.0