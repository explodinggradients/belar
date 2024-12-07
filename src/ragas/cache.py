import functools
import hashlib
import inspect
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel


class CacheInterface(ABC):
    @abstractmethod
    def get(self, key: str) -> Any:
        pass

    @abstractmethod
    def set(self, key: str, value) -> None:
        pass

    @abstractmethod
    def has_key(self, key: str) -> bool:
        pass


class DiskCacheBackend(CacheInterface):
    def __init__(self, cache_dir: str):
        try:
            from diskcache import Cache
        except ImportError:
            raise ImportError(
                "For using the diskcache backend, please install it with `pip install diskcache`."
            )

        self.cache = Cache(cache_dir)

    def get(self, key: str) -> Any:
        return self.cache.get(key)

    def set(self, key: str, value) -> None:
        self.cache.set(key, value)

    def has_key(self, key: str) -> bool:
        return key in self.cache

    def __del__(self):
        if hasattr(self, "cache"):
            self.cache.close()


def _make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple(_make_hashable(e) for e in o)
    elif isinstance(o, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in o.items()))
    elif isinstance(o, set):
        return tuple(sorted(_make_hashable(e) for e in o))
    elif isinstance(o, BaseModel):
        return _make_hashable(o.model_dump())
    else:
        return o


EXCLUDE_KEYS = ["callbacks", "llm"]


def _generate_cache_key(func, args, kwargs):
    if inspect.ismethod(func):
        args = args[1:]

    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in EXCLUDE_KEYS}

    key_data = {
        "function": func.__qualname__,
        "args": _make_hashable(args),
        "kwargs": _make_hashable(filtered_kwargs),
    }

    key_string = json.dumps(key_data, sort_keys=True, default=str)
    cache_key = hashlib.sha256(key_string.encode("utf-8")).hexdigest()
    return cache_key


def cacher(cache_backend: Optional[CacheInterface] = None):
    if cache_backend is None:
        cache_backend = DiskCacheBackend(
            cache_dir=".cache",
        )

    def decorator(func):
        def is_caching_enabled():
            # Read from environment variable, default to 0 (no caching) if not set
            return os.environ.get("RAGAS_CACHE_ENABLED", "0") == "1"

        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not is_caching_enabled():
                return await func(*args, **kwargs)

            cache_key = _generate_cache_key(func, args, kwargs)

            if cache_backend.has_key(cache_key):
                return cache_backend.get(cache_key)

            result = await func(*args, **kwargs)
            cache_backend.set(cache_key, result)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not is_caching_enabled():
                return func(*args, **kwargs)

            cache_key = _generate_cache_key(func, args, kwargs)

            if cache_backend.has_key(cache_key):
                return cache_backend.get(cache_key)

            result = func(*args, **kwargs)
            cache_backend.set(cache_key, result)
            return result

        return async_wrapper if is_async else sync_wrapper

    return decorator