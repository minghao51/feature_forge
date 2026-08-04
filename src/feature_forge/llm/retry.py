"""Tenacity-based retry logic for LLM API calls.

Provides exponential backoff with jitter for transient failures
(rate limits, server errors). Configured via ``RetryConfig``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from feature_forge.exceptions import LLMError, TransientLLMError
from feature_forge.observability.structlog_config import get_logger

if TYPE_CHECKING:
    from feature_forge.config import RetryConfig

logger = get_logger(__name__)


def is_transient_llm_error(exc: BaseException) -> bool:
    """Return whether an LLM failure is explicitly safe to retry."""
    if not isinstance(exc, LLMError):
        return False
    if isinstance(exc, TransientLLMError):
        return True
    current: BaseException | None = exc
    while current is not None:
        name = type(current).__name__.lower()
        status = getattr(current, "status_code", None)
        if status in {408, 409, 429, 500, 502, 503, 504}:
            return True
        if any(
            token in name
            for token in (
                "apitimeout",
                "ratelimit",
                "apiconnection",
                "serviceunavailable",
            )
        ):
            return True
        current = current.__cause__
    return False


def build_async_retry(config: RetryConfig) -> AsyncRetrying:
    """Build an ``AsyncRetrying`` instance from retry config.

    Args:
        config: Retry configuration from settings.

    Returns:
        Configured tenacity async retry controller.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(config.max_retries + 1),
        wait=wait_exponential(
            multiplier=config.backoff_base,
            max=config.backoff_max,
            exp_base=config.backoff_exponent,
        ),
        retry=retry_if_exception(is_transient_llm_error),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
