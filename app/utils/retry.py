from collections.abc import Callable

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def with_retry(max_attempts: int = 3) -> Callable:
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
