from cachetools import cached, TTLCache


discussion_cache = TTLCache(maxsize=100, ttl=3600)
organization_cache = TTLCache(maxsize=100, ttl=3600)
project_cache = TTLCache(maxsize=100, ttl=3600)