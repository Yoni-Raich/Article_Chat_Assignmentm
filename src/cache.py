"""
Query Response Cache

This module implements an in-memory cache for chat query responses with TTL (Time To Live)
and LRU (Least Recently Used) eviction policies. This ensures that repeated identical
queries return cached responses, improving performance and meeting assignment requirements.
"""

import time
import hashlib
import json
from typing import Optional, Dict, Any
from .logger import logger


class QueryCache:
    """
    In-memory cache for query responses with TTL and LRU eviction.

    Features:
    - TTL (Time To Live) for automatic expiration
    - LRU (Least Recently Used) eviction when cache is full
    - Cache key generation based on query text and parameters
    - Statistics tracking for monitoring
    """

    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600):
        """
        Initialize the query cache.

        Args:
            max_size: Maximum number of entries to store
            ttl_seconds: Time to live for cached entries (default: 1 hour)
        """
        self.cache = {}
        self.timestamps = {}
        self.access_times = {}  # For LRU tracking
        self.max_size = max_size
        self.ttl = ttl_seconds

        # Statistics
        self.stats_data = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }

        logger.info("QueryCache initialized: max_size=%s, ttl=%ss", max_size, ttl_seconds)

    def _generate_key(self, query: str, max_articles: int = 5) -> str:
        """
        Generate a consistent cache key from query and parameters.

        Args:
            query: The user's query text
            max_articles: Maximum articles parameter

        Returns:
            MD5 hash of the normalized query data
        """
        # Normalize the query for consistent caching
        normalized_query = query.lower().strip()
        cache_data = {
            "query": normalized_query,
            "max_articles": max_articles
        }

        # Generate consistent hash
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()

    def get(self, query: str, max_articles: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get cached response if it exists and hasn't expired.

        Args:
            query: The user's query text
            max_articles: Maximum articles parameter

        Returns:
            Cached response data or None if not found/expired
        """
        key = self._generate_key(query, max_articles)
        current_time = time.time()

        # Check if key exists
        if key not in self.cache:
            self.stats_data["misses"] += 1
            return None

        # Check if expired
        if current_time - self.timestamps[key] >= self.ttl:
            logger.debug("Cache entry expired: %s...", key[:8])
            self._remove(key)
            self.stats_data["misses"] += 1
            self.stats_data["expirations"] += 1
            return None

        # Update access time for LRU
        self.access_times[key] = current_time
        self.stats_data["hits"] += 1

        logger.debug("Cache HIT: %s... (query: '%s...')", key[:8], query[:50])
        return self.cache[key]

    def set(self, query: str, response: Dict[str, Any], max_articles: int = 5):
        """
        Cache a response for the given query.

        Args:
            query: The user's query text
            response: The response data to cache
            max_articles: Maximum articles parameter
        """
        key = self._generate_key(query, max_articles)
        current_time = time.time()

        # Check if cache is full and we need to evict
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()

        # Store the response
        self.cache[key] = response
        self.timestamps[key] = current_time
        self.access_times[key] = current_time

        logger.debug("Cache SET: %s... (query: '%s...')", key[:8], query[:50])

    def _remove(self, key: str):
        """
        Remove a specific key from all cache structures.

        Args:
            key: The cache key to remove
        """
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
        if key in self.access_times:
            del self.access_times[key]

    def _evict_lru(self):
        """
        Evict the least recently used entry from the cache.
        """
        if not self.access_times:
            return

        # Find the key with the oldest access time
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])

        logger.debug("Cache LRU eviction: %s...", lru_key[:8])
        self._remove(lru_key)
        self.stats_data["evictions"] += 1

    def clear(self):
        """
        Clear all cached entries.
        """
        entries_cleared = len(self.cache)
        self.cache.clear()
        self.timestamps.clear()
        self.access_times.clear()

        logger.info("Cache cleared: %s entries removed", entries_cleared)

    def cleanup_expired(self):
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = []

        for key, timestamp in self.timestamps.items():
            if current_time - timestamp >= self.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            self._remove(key)

        if expired_keys:
            logger.info("Cache cleanup: %s expired entries removed", len(expired_keys))
            self.stats_data["expirations"] += len(expired_keys)

        return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        current_time = time.time()

        # Count valid (non-expired) entries
        valid_entries = 0
        expired_entries = 0

        for timestamp in self.timestamps.values():
            if current_time - timestamp < self.ttl:
                valid_entries += 1
            else:
                expired_entries += 1

        # Calculate hit rate
        total_requests = self.stats_data["hits"] + self.stats_data["misses"]
        hit_rate = (self.stats_data["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "usage_percentage": (len(self.cache) / self.max_size * 100) if self.max_size > 0 else 0,
            "hit_rate_percentage": round(hit_rate, 2),
            "statistics": self.stats_data.copy()
        }

    def get_cache_info(self, query: str, max_articles: int = 5) -> Dict[str, Any]:
        """
        Get information about a specific query's cache status.

        Args:
            query: The user's query text
            max_articles: Maximum articles parameter

        Returns:
            Cache information for the query
        """
        key = self._generate_key(query, max_articles)
        current_time = time.time()

        if key not in self.cache:
            return {
                "cached": False,
                "key": key[:8],
                "status": "not_cached"
            }

        age = current_time - self.timestamps[key]
        time_until_expiry = self.ttl - age

        return {
            "cached": True,
            "key": key[:8],
            "status": "expired" if age >= self.ttl else "valid",
            "age_seconds": round(age, 2),
            "time_until_expiry_seconds": round(time_until_expiry, 2)
        }
