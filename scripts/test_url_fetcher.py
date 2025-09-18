#!/usr/bin/env python3
"""
Test script for URL Fetcher component with caching

This script tests the URL fetcher with a small sample of URLs from the build-rag/urls.txt file.
It demonstrates caching functionality and provides performance comparisons.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.url_fetcher import URLFetcher
from src.url_cache import CachedURLFetcher


def load_sample_urls(num_urls: int = 10) -> list[str]:
    """Load a sample of URLs from the build-rag/urls.txt file"""
    urls_file = Path(__file__).parent.parent / "build-rag" / "urls.txt"

    if not urls_file.exists():
        print(f"Error: URLs file not found at {urls_file}")
        return []

    urls = []
    with open(urls_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_urls:
                break
            url = line.strip()
            if url:
                urls.append(url)

    print(f"Loaded {len(urls)} sample URLs from {urls_file}")
    return urls


def test_basic_fetcher():
    """Test basic URL fetcher without caching"""
    print("\n" + "="*60)
    print("TESTING BASIC URL FETCHER (NO CACHING)")
    print("="*60)

    urls = load_sample_urls(5)  # Start with just 5 URLs
    if not urls:
        return

    print(f"Testing with {len(urls)} URLs...")

    with URLFetcher(timeout=10, max_retries=2) as fetcher:
        start_time = time.time()
        results = fetcher.fetch_multiple(urls, delay_between_requests=0.5)
        end_time = time.time()

        # Print results
        for i, result in enumerate(results):
            status = "✓" if result.success else "✗"
            size = len(result.content) if result.content else 0
            print(f"{status} {result.url[:60]}... "
                  f"({size} bytes, {result.response_time:.2f}s)")

        # Print statistics
        stats = fetcher.get_stats(results)
        print(f"\nStatistics:")
        print(f"  Total time: {end_time - start_time:.2f}s")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Average response time: {stats.get('avg_response_time', 0):.2f}s")
        print(f"  Total content: {stats.get('total_content_size', 0):,} bytes")


def test_cached_fetcher():
    """Test cached URL fetcher"""
    print("\n" + "="*60)
    print("TESTING CACHED URL FETCHER")
    print("="*60)

    urls = load_sample_urls(10)  # Test with more URLs
    if not urls:
        return

    print(f"Testing with {len(urls)} URLs...")

    # First run - populate cache
    print("\nFirst run (populating cache):")
    with CachedURLFetcher(cache_kwargs={'cache_dir': 'cache/test_cache'}) as cached_fetcher:
        start_time = time.time()
        results1 = cached_fetcher.fetch_multiple(urls, delay_between_requests=0.2)
        end_time = time.time()

        first_run_time = end_time - start_time
        stats1 = cached_fetcher.get_stats()

        print(f"  Time: {first_run_time:.2f}s")
        print(f"  Cache hit rate: {stats1['cache_hit_rate']:.1%}")
        print(f"  Successful fetches: {sum(1 for r in results1 if r.success)}")

    # Second run - use cache
    print("\nSecond run (using cache):")
    with CachedURLFetcher(cache_kwargs={'cache_dir': 'cache/test_cache'}) as cached_fetcher:
        start_time = time.time()
        results2 = cached_fetcher.fetch_multiple(urls, delay_between_requests=0.1)
        end_time = time.time()

        second_run_time = end_time - start_time
        stats2 = cached_fetcher.get_stats()

        print(f"  Time: {second_run_time:.2f}s")
        print(f"  Cache hit rate: {stats2['cache_hit_rate']:.1%}")
        print(f"  Successful fetches: {sum(1 for r in results2 if r.success)}")

        # Performance improvement
        if first_run_time > 0:
            speedup = first_run_time / second_run_time
            print(f"  Speedup: {speedup:.1f}x faster")

        # Cache statistics
        cache_stats = stats2['cache_stats']
        print(f"\nCache Statistics:")
        print(f"  Total entries: {cache_stats['total_entries']}")
        print(f"  Successful entries: {cache_stats['successful_entries']}")
        print(f"  Cache directory size: {cache_stats['cache_directory_size']:,} bytes")


def test_force_refresh():
    """Test force refresh functionality"""
    print("\n" + "="*60)
    print("TESTING FORCE REFRESH")
    print("="*60)

    urls = load_sample_urls(3)  # Just a few URLs for force refresh test
    if not urls:
        return

    print(f"Testing force refresh with {len(urls)} URLs...")

    cache_dir = 'cache/test_force_refresh'

    # First run - populate cache
    print("\n1. Populating cache:")
    with CachedURLFetcher(cache_kwargs={'cache_dir': cache_dir}) as cached_fetcher:
        results1 = cached_fetcher.fetch_multiple(urls)
        stats1 = cached_fetcher.get_stats()
        print(f"   Cache hit rate: {stats1['cache_hit_rate']:.1%}")

    # Second run - use cache
    print("\n2. Using cache:")
    with CachedURLFetcher(cache_kwargs={'cache_dir': cache_dir}) as cached_fetcher:
        results2 = cached_fetcher.fetch_multiple(urls)
        stats2 = cached_fetcher.get_stats()
        print(f"   Cache hit rate: {stats2['cache_hit_rate']:.1%}")

    # Third run - force refresh
    print("\n3. Force refresh (ignoring cache):")
    with CachedURLFetcher(
        cache_kwargs={'cache_dir': cache_dir},
        force_refresh=True
    ) as cached_fetcher:
        results3 = cached_fetcher.fetch_multiple(urls)
        stats3 = cached_fetcher.get_stats()
        print(f"   Cache hit rate: {stats3['cache_hit_rate']:.1%}")
        print(f"   Fresh fetches: {stats3['session_stats']['fetches']}")


def test_error_handling():
    """Test error handling with various URL types"""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)

    # Test URLs with different expected outcomes
    test_urls = [
        "https://httpbin.org/status/200",  # Should succeed
        "https://httpbin.org/status/404",  # Should fail with 404
        "https://httpbin.org/delay/1",     # Should succeed (with delay)
        "https://invalid-domain-12345.com",  # Should fail with connection error
        "not-a-url",                       # Should fail with invalid URL
    ]

    print(f"Testing error handling with {len(test_urls)} test URLs...")

    with CachedURLFetcher(
        fetcher_kwargs={'timeout': 5, 'max_retries': 1},
        cache_kwargs={'cache_dir': 'cache/test_errors'}
    ) as cached_fetcher:
        results = cached_fetcher.fetch_multiple(test_urls, delay_between_requests=0.1)

        print("\nResults:")
        for result in results:
            status = "✓" if result.success else "✗"
            error_msg = result.error_message or "No error"
            size = len(result.content) if result.content else 0

            print(f"{status} {result.url}")
            print(f"   Status: {result.status_code}, Size: {size} bytes")
            if not result.success:
                print(f"   Error: {error_msg}")

        stats = cached_fetcher.get_stats()
        print(f"\nError handling stats:")
        print(f"  Success rate: {sum(1 for r in results if r.success) / len(results):.1%}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")


def test_large_batch():
    """Test with a larger batch of URLs"""
    print("\n" + "="*60)
    print("TESTING LARGE BATCH (50 URLs)")
    print("="*60)

    urls = load_sample_urls(50)
    if not urls:
        return

    print(f"Testing with {len(urls)} URLs...")

    with CachedURLFetcher(
        fetcher_kwargs={'timeout': 10, 'max_retries': 2},
        cache_kwargs={'cache_dir': 'cache/test_large_batch'},
        force_refresh=False
    ) as cached_fetcher:
        start_time = time.time()
        results = cached_fetcher.fetch_multiple(urls, delay_between_requests=0.1)
        end_time = time.time()

        total_time = end_time - start_time
        successful = sum(1 for r in results if r.success)
        stats = cached_fetcher.get_stats()

        print(f"\nLarge batch results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Successful fetches: {successful}/{len(urls)} ({successful/len(urls):.1%})")
        print(f"  Average time per URL: {total_time/len(urls):.2f}s")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")

        # Show cache statistics
        cache_stats = stats['cache_stats']
        print(f"\nCache statistics:")
        print(f"  Total cached entries: {cache_stats['total_entries']}")
        print(f"  Cache directory size: {cache_stats['cache_directory_size']:,} bytes")


def main():
    """Run all tests"""
    print("URL Fetcher Component Test Suite")
    print("=" * 60)

    try:
        # Run tests in order
        test_basic_fetcher()
        test_cached_fetcher()
        test_force_refresh()
        test_error_handling()

        # Ask user if they want to run large batch test
        response = input("\nRun large batch test with 50 URLs? (y/N): ").strip().lower()
        if response in ('y', 'yes'):
            test_large_batch()

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)

    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise


if __name__ == "__main__":
    main()