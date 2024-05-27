import logging
import typing as t
from dataclasses import dataclass

from tenacity import (
    AsyncRetrying,
    Retrying,
    WrappedFn,
    after_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log
)
from tenacity.after import after_nothing

import logging

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(threadName)s] - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

@dataclass
class RunConfig:
    """
    Configuration for a timeouts and retries.
    """

    timeout: int = 25
    max_retries: int = 5
    max_wait: int = 5
    max_workers: int = 4
    exception_types: t.Union[
        t.Type[BaseException],
        t.Tuple[t.Type[BaseException], ...],
    ] = (Exception,)
    log_tenacity: bool = False


def add_retry(fn: WrappedFn, run_config: RunConfig) -> WrappedFn:
    # configure tenacity's after section wtih logger
    if run_config.log_tenacity is not None:
        logger = logging.getLogger(f"ragas.retry.{fn.__name__}")
        tenacity_logger = after_log(logger, logging.DEBUG)
    else:
        tenacity_logger = after_nothing

    r = Retrying(
        wait=wait_random_exponential(multiplier=1, max=run_config.max_wait),
        stop=stop_after_attempt(run_config.max_retries),
        retry=retry_if_exception_type(run_config.exception_types),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        after=tenacity_logger,
    )
    return r.wraps(fn)


def add_async_retry(fn: WrappedFn, run_config: RunConfig) -> WrappedFn:
    """
    Decorator for retrying a function if it fails.
    """
    # configure tenacity's after section wtih logger
    if run_config.log_tenacity is not None:
        logger = logging.getLogger(f"TENACITYRetry[{fn.__name__}]")
        tenacity_logger = after_log(logger, logging.DEBUG)
    else:
        tenacity_logger = after_nothing

    r = AsyncRetrying(
        wait=wait_random_exponential(multiplier=1, max=run_config.max_wait),
        stop=stop_after_attempt(run_config.max_retries),
        retry=retry_if_exception_type(run_config.exception_types),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        after=tenacity_logger
    )
    return r.wraps(fn)
