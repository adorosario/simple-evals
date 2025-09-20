#!/usr/bin/env python3
"""
Rebuild cache metadata from existing cache files.

This script fixes cases where cache files exist but metadata entries are missing,
which can cause cache lookups to fail and result in unnecessary re-fetching.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.url_cache import URLCache


def main():
    """Rebuild cache metadata"""
    print("🔧 REBUILDING CACHE METADATA")
    print("=" * 50)

    cache = URLCache('cache/url_cache')

    print(f"📦 Cache directory: {cache.cache_dir}")
    print(f"📊 Before rebuild:")
    print(f"   Cache files on disk: {len(list(cache.cache_dir.glob('*.pkl')))}")
    print(f"   Metadata entries: {len(cache.metadata)}")

    rebuilt_count = cache.rebuild_metadata()

    print(f"📊 After rebuild:")
    print(f"   Metadata entries: {len(cache.metadata)}")
    print(f"   Rebuilt entries: {rebuilt_count}")

    if rebuilt_count > 0:
        print(f"✅ Successfully rebuilt {rebuilt_count} missing metadata entries!")
        print("   Cache lookups should now work properly.")
    else:
        print("✅ Cache metadata was already complete - no rebuild needed.")

    # Verify cache is working
    stats = cache.get_stats()
    print(f"📈 Cache statistics:")
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Successful entries: {stats['successful_entries']}")
    print(f"   Failed entries: {stats['failed_entries']}")


if __name__ == "__main__":
    main()