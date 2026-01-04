"""Circuit breaker implementation for resilient third-party service calls."""
import time
from typing import Callable, Any, Optional
from enum import Enum
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = None,
        recovery_timeout: int = None,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold or settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD
        self.recovery_timeout = recovery_timeout or settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
        self.expected_exception = expected_exception or settings.CIRCUIT_BREAKER_EXPECTED_EXCEPTION
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        # Check if we should attempt recovery
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info("Circuit breaker: Attempting recovery (HALF_OPEN)")
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        # Check if we should attempt recovery
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info("Circuit breaker: Attempting recovery (HALF_OPEN)")
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        # Attempt the call
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker: Service recovered (CLOSED)")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.warning(
                    f"Circuit breaker: Opening circuit after {self.failure_count} failures"
                )
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker: Recovery failed, reopening circuit")
            self.state = CircuitState.OPEN
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker: Manually reset")


# Global circuit breaker instance for payment gateway
payment_circuit_breaker = CircuitBreaker()

