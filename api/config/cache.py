from cachetools import cached, TTLCache
import asyncio

class AsyncCache:
    def __init__(self, 
                 maxsize: int = 100,
                 ttl: int = 3600):
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._locks = {}
    
    def get_cache_key(self, 
                      func: str, 
                      args: tuple, 
                      kwargs: dict) -> str:
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        return f"{func}:{hash((args_str + kwargs_str))}"

    async def get_or_set(self,
                         key: str,
                         coro_unc: Callable,
                         *args,
                         **kwargs) -> Any:
        if key in self._cache:
            return self._cache[key]
        if key not in self_locks:
            self._locks[key] = asyncio.Lock()
        
        async with self._locks[key]:
            if key in self._cache:
                return self._cache[key]

        result = await coro_unc(*args, **kwargs)
        self._cache[key] = result
        return result


discussion_cache = AsyncCache(maxsize=100, ttl=3600)
organization_cache = AsyncCache(maxsize=100, ttl=3600)
project_cache = AsyncCache(maxsize=100, ttl=3600)

def async_cached(cache: TTLCache):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = cache.get_cache_key(func.__name__, args, kwargs)
            return await cache.get_or_set(key, func, *args, **kwargs)
        return wrapper
    return decorator
