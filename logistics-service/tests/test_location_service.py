import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.location_service import LocationService, location_service
from app.models.location import DriverLocation


@pytest.mark.asyncio
async def test_update_and_get_current_location():
    fake_redis = AsyncMock()
    store = {}

    async def setex(key, ttl, value):
        store[key] = value

    async def get(key):
        return store.get(key)

    async def geoadd(name, values):
        return 1

    fake_redis.setex.side_effect = setex
    fake_redis.get.side_effect = get
    fake_redis.geoadd.side_effect = geoadd

    svc = LocationService(redis_client=fake_redis)

    ok = await svc.update_current_location(
        driver_id=1, latitude=1.0, longitude=2.0, speed=10.0, heading=90.0
    )
    assert ok

    loc = await svc.get_current_location(1)
    assert loc is not None
    assert float(loc["latitude"]) == 1.0
    assert float(loc["longitude"]) == 2.0


@pytest.mark.asyncio
async def test_store_and_get_history(db_session: AsyncSession):
    svc = location_service

    await svc.store_historical_location(
        db_session, driver_id=1, latitude=1.0, longitude=2.0
    )
    await svc.store_historical_location(
        db_session, driver_id=1, latitude=1.1, longitude=2.1
    )

    end = datetime.utcnow()
    start = end - timedelta(hours=1)

    history = await svc.get_location_history(db_session, driver_id=1, start_time=start, end_time=end)
    assert len(history) >= 2
    assert all(isinstance(loc, DriverLocation) for loc in history)
    assert all(loc.driver_id == 1 for loc in history)