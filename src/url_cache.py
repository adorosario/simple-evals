"""
URL Content Caching System

Provides persistent caching for fetched URL content to avoid re-downloading during development.
Uses file-based caching with optional force refresh.
"""

import os
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict
import pickle

from .url_fetcher import FetchResult

logger = logging.getLogger(__name__)


class URLCache:
    """
    File-based cache for URL fetch results with configurable expiration and force refresh.
    """

    def __init__(self,
                 cache_dir: str = "cache/url_cache",
                 default_ttl: int = 7 * 24 * 3600,  # 7 days default
                 max_cache_size: int = 1000):  # Max number of cached items
        """
        Initialize URL cache.

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default time-to-live in seconds
            max_cache_size: Maximum number of cached items (LRU cleanup)
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file for tracking cache entries
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Cache metadata corrupted, starting fresh")
                return {}
        return {}

    def _save_metadata(self):
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given key"""
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_expired(self, cache_key: str, ttl: Optional[int] = None) -> bool:
        """Check if cached item is expired"""
        if cache_key not in self.metadata:
            return True

        ttl = ttl or self.default_ttl
        cached_time = self.metadata[cache_key].get('timestamp', 0)
        return time.time() - cached_time > ttl

    def _cleanup_old_entries(self):
        """Remove old cache entries if cache is too large"""
        if len(self.metadata) <= self.max_cache_size:
            return

        # Sort by access time (LRU)
        sorted_entries = sorted(
            self.metadata.items(),
            key=lambda x: x[1].get('last_access', 0)
        )

        # Remove oldest entries
        entries_to_remove = len(self.metadata) - self.max_cache_size
        for cache_key, _ in sorted_entries[:entries_to_remove]:
            self._remove_cache_entry(cache_key)

        logger.info(f"Cleaned up {entries_to_remove} old cache entries")

    def _remove_cache_entry(self, cache_key: str):
        """Remove a specific cache entry"""
        cache_path = self._get_cache_path(cache_key)
        try:
            if cache_path.exists():
                cache_path.unlink()
            if cache_key in self.metadata:
                del self.metadata[cache_key]
        except Exception as e:
            logger.error(f"Failed to remove cache entry {cache_key}: {e}")

    def get(self, url: str, ttl: Optional[int] = None, force: bool = False) -> Optional[FetchResult]:
        """
        Get cached result for URL.

        Args:
            url: URL to check
            ttl: Custom time-to-live in seconds
            force: If True, ignore cache and return None

        Returns:
            Cached FetchResult or None if not cached/expired/force
        """
        if force:
            return None

        cache_key = self._get_cache_key(url)

        # Check if expired
        if self._is_expired(cache_key, ttl):
            return None

        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            # Clean up metadata for missing file
            if cache_key in self.metadata:
                del self.metadata[cache_key]
            return None

        try:
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)

            # Update access time
            self.metadata[cache_key]['last_access'] = time.time()

            logger.debug(f"Cache hit for {url}")
            return result

        except Exception as e:
            logger.error(f"Failed to load cached result for {url}: {e}")
            self._remove_cache_entry(cache_key)
            return None

    def put(self, url: str, result: FetchResult):
        """
        Cache a fetch result.

        Args:
            url: URL that was fetched
            result: FetchResult to cache
        """
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)

            # Update metadata
            current_time = time.time()
            self.metadata[cache_key] = {
                'url': url,
                'timestamp': current_time,
                'last_access': current_time,
                'success': result.success,
                'content_size': len(result.content) if result.content else 0,
                'content_type': result.content_type
            }

            # Cleanup if needed
            self._cleanup_old_entries()

            # Save metadata
            self._save_metadata()

            logger.debug(f"Cached result for {url}")

        except Exception as e:
            logger.error(f"Failed to cache result for {url}: {e}")

    def clear(self, url: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            url: If provided, clear only this URL. Otherwise clear all.
        """
        if url:
            cache_key = self._get_cache_key(url)
            self._remove_cache_entry(cache_key)
            self._save_metadata()
            logger.info(f"Cleared cache for {url}")
        else:
            # Clear all
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete {cache_file}: {e}")

            self.metadata.clear()
            self._save_metadata()
            logger.info("Cleared all cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.metadata)
        successful_entries = sum(1 for entry in self.metadata.values() if entry.get('success', False))
        total_size = sum(entry.get('content_size', 0) for entry in self.metadata.values())

        # Calculate cache directory size
        dir_size = sum(f.stat().st_size for f in self.cache_dir.glob("*") if f.is_file())

        return {
            'total_entries': total_entries,
            'successful_entries': successful_entries,
            'failed_entries': total_entries - successful_entries,
            'total_content_size': total_size,
            'cache_directory_size': dir_size,
            'cache_directory': str(self.cache_dir),
            'oldest_entry': min((entry.get('timestamp', float('inf')) for entry in self.metadata.values()), default=None),
            'newest_entry': max((entry.get('timestamp', 0) for entry in self.metadata.values()), default=None)
        }

    def list_cached_urls(self, successful_only: bool = True) -> list[str]:
        """
        List all cached URLs.

        Args:
            successful_only: If True, only return URLs with successful fetches

        Returns:
            List of cached URLs
        """
        urls = []
        for entry in self.metadata.values():
            if not successful_only or entry.get('success', False):
                urls.append(entry['url'])
        return sorted(urls)


class CachedURLFetcher:
    """
    URL fetcher with integrated caching support.
    """

    def __init__(self,
                 fetcher_kwargs: Optional[Dict] = None,
                 cache_kwargs: Optional[Dict] = None,
                 force_refresh: bool = False,
                 cache_ttl: Optional[int] = None):
        """
        Initialize cached URL fetcher.

        Args:
            fetcher_kwargs: Arguments for URLFetcher
            cache_kwargs: Arguments for URLCache
            force_refresh: If True, ignore cache for all requests
            cache_ttl: Custom cache TTL in seconds
        """
        from .url_fetcher import URLFetcher

        self.fetcher = URLFetcher(**(fetcher_kwargs or {}))
        self.cache = URLCache(**(cache_kwargs or {}))
        self.force_refresh = force_refresh
        self.cache_ttl = cache_ttl

        # Track statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'fetches': 0
        }

    def fetch(self, url: str, force: Optional[bool] = None) -> FetchResult:
        """
        Fetch URL with caching.

        Args:
            url: URL to fetch
            force: Override force_refresh setting for this request

        Returns:
            FetchResult (from cache or fresh fetch)
        """
        force = force if force is not None else self.force_refresh

        # Try cache first
        if not force:
            cached_result = self.cache.get(url, ttl=self.cache_ttl)
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache hit for {url}")
                return cached_result

        # Cache miss - fetch fresh
        self.stats['cache_misses'] += 1
        self.stats['fetches'] += 1

        logger.debug(f"Cache miss for {url}, fetching...")
        result = self.fetcher.fetch(url)

        # Cache the result (even failures, to avoid repeated attempts)
        self.cache.put(url, result)

        return result

    def fetch_multiple(self, urls: list[str],
                      delay_between_requests: float = 0.1,
                      force: Optional[bool] = None) -> list[FetchResult]:
        """
        Fetch multiple URLs with caching.

        Args:
            urls: List of URLs to fetch
            delay_between_requests: Delay between fresh fetches (not cached)
            force: Override force_refresh setting

        Returns:
            List of FetchResult objects
        """
        results = []
        force = force if force is not None else self.force_refresh

        logger.info(f"Fetching {len(urls)} URLs (force={force})...")

        for i, url in enumerate(urls):
            result = self.fetch(url, force=force)
            results.append(result)

            # Only delay for fresh fetches, not cache hits
            if not force and delay_between_requests > 0 and i < len(urls) - 1:
                if self.stats['cache_misses'] > 0:  # Had at least one fresh fetch
                    time.sleep(delay_between_requests)

            # Progress logging
            if (i + 1) % 10 == 0 or i + 1 == len(urls):
                successful = sum(1 for r in results if r.success)
                cache_hit_rate = self.stats['cache_hits'] / (i + 1) if i >= 0 else 0
                logger.info(f"Progress: {i+1}/{len(urls)} URLs processed "
                           f"({successful} successful, {cache_hit_rate:.1%} cache hit rate)")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get combined fetcher and cache statistics"""
        cache_stats = self.cache.get_stats()

        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = self.stats['cache_hits'] / total_requests if total_requests > 0 else 0

        return {
            'session_stats': self.stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_stats': cache_stats
        }

    def clear_cache(self, url: Optional[str] = None):
        """Clear cache entries"""
        self.cache.clear(url)

    def close(self):
        """Close fetcher and save cache"""
        self.fetcher.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()