#!/usr/bin/env python3
"""
Quick manual test of URL fetcher without pytest dependencies
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.url_fetcher import URLFetcher

def quick_test():
    """Quick test of URL fetcher functionality"""
    print("Quick URL Fetcher Test")
    print("=" * 40)

    # Test URLs
    test_urls = [
        "https://httpbin.org/status/200",
        "https://httpbin.org/html",
        "https://example.com",
    ]

    with URLFetcher(timeout=10, max_retries=1) as fetcher:
        for url in test_urls:
            print(f"\nTesting: {url}")
            result = fetcher.fetch(url)

            if result.success:
                size = len(result.content) if result.content else 0
                print(f"  ✓ Success - {size} bytes, {result.content_type}")
            else:
                print(f"  ✗ Failed - {result.error_message}")

    print("\nBasic URL fetcher functionality working!")

if __name__ == "__main__":
    quick_test()