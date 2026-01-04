"""Tests for circuit breaker implementation."""
import pytest
import asyncio
import time
from app.core.circuit_breaker import CircuitBreaker, CircuitState


def test_circuit_breaker_initial_state():
    """Test circuit breaker starts in CLOSED state."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_circuit_breaker_successful_call():
    """Test circuit breaker with successful calls."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
    
    def success_func():
        return "success"
    
    result = cb.call(success_func)
    assert result == "success"
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_circuit_breaker_failure_counting():
    """Test circuit breaker counts failures."""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
    
    def fail_func():
        raise ValueError("Test error")
    
    # First failure
    with pytest.raises(ValueError):
        cb.call(fail_func)
    assert cb.failure_count == 1
    assert cb.state == CircuitState.CLOSED
    
    # Second failure
    with pytest.raises(ValueError):
        cb.call(fail_func)
    assert cb.failure_count == 2
    assert cb.state == CircuitState.CLOSED
    
    # Third failure - should open circuit
    with pytest.raises(ValueError):
        cb.call(fail_func)
    assert cb.failure_count == 3
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_opens_after_threshold():
    """Test circuit breaker opens after reaching failure threshold."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)
    
    def fail_func():
        raise Exception("Test error")
    
    # Cause failures
    for _ in range(2):
        with pytest.raises(Exception):
            cb.call(fail_func)
    
    assert cb.state == CircuitState.OPEN
    
    # Next call should fail immediately
    with pytest.raises(Exception) as exc_info:
        cb.call(fail_func)
    assert "Circuit breaker is OPEN" in str(exc_info.value)


def test_circuit_breaker_recovery():
    """Test circuit breaker attempts recovery after timeout."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    
    def fail_func():
        raise Exception("Test error")
    
    # Open the circuit
    for _ in range(2):
        with pytest.raises(Exception):
            cb.call(fail_func)
    
    assert cb.state == CircuitState.OPEN
    
    # Wait for recovery timeout
    time.sleep(1.1)
    
    # Attempt a call - should go to HALF_OPEN
    def success_func():
        return "success"
    
    result = cb.call(success_func)
    assert result == "success"
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_circuit_breaker_half_open_failure():
    """Test circuit breaker reopens on failure in HALF_OPEN state."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    
    def fail_func():
        raise Exception("Test error")
    
    # Open the circuit
    for _ in range(2):
        with pytest.raises(Exception):
            cb.call(fail_func)
    
    assert cb.state == CircuitState.OPEN
    
    # Wait for recovery timeout
    time.sleep(1.1)
    
    # Attempt a call that fails - should reopen
    with pytest.raises(Exception):
        cb.call(fail_func)
    
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_async():
    """Test circuit breaker with async functions."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)
    
    async def success_func():
        await asyncio.sleep(0.01)
        return "success"
    
    result = await cb.call_async(success_func)
    assert result == "success"
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_async_failure():
    """Test circuit breaker with async function failures."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)
    
    async def fail_func():
        await asyncio.sleep(0.01)
        raise Exception("Test error")
    
    # Cause failures
    for _ in range(2):
        with pytest.raises(Exception):
            await cb.call_async(fail_func)
    
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_reset():
    """Test manual circuit breaker reset."""
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)
    
    def fail_func():
        raise Exception("Test error")
    
    # Open the circuit
    for _ in range(2):
        with pytest.raises(Exception):
            cb.call(fail_func)
    
    assert cb.state == CircuitState.OPEN
    
    # Reset
    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0
    assert cb.last_failure_time is None

