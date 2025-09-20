#!/usr/bin/env python3
"""
Cache and Knowledge Base Utilities for Hamza

This script provides utilities to interact with the URL cache and knowledge base documents.
Allows searching by URL, reading cached content, and accessing processed documents.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class CacheAndKBManager:
    """Manager for URL cache and knowledge base documents"""

    def __init__(self, cache_dir: str = "cache/url_cache", kb_dir: str = "knowledge_base_full"):
        self.cache_dir = Path(cache_dir)
        self.kb_dir = Path(kb_dir)

        print(f"📁 Cache directory: {self.cache_dir}")
        print(f"📚 Knowledge base directory: {self.kb_dir}")

        if not self.cache_dir.exists():
            print(f"⚠️  Warning: Cache directory not found: {self.cache_dir}")
        if not self.kb_dir.exists():
            print(f"⚠️  Warning: KB directory not found: {self.kb_dir}")

    def get_cache_key(self, url: str) -> str:
        """Generate cache key for URL (same as URLCache)"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()

    def get_doc_filename(self, url: str) -> str:
        """Generate document filename for URL (same as ContentProcessor)"""
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()[:12]
        return f"doc_{url_hash}.txt"

    def get_cached_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached content for a URL"""
        cache_key = self.get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            print(f"❌ No cache entry found for: {url}")
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            print(f"✅ Found cached content for: {url}")
            print(f"   Cached at: {cached_data.get('timestamp', 'Unknown')}")
            print(f"   Content length: {len(cached_data.get('content', ''))} chars")
            return cached_data

        except Exception as e:
            print(f"❌ Error reading cache file: {e}")
            return None

    def get_processed_document(self, url: str) -> Optional[str]:
        """Get processed document content for a URL"""
        doc_filename = self.get_doc_filename(url)
        doc_file = self.kb_dir / doc_filename

        if not doc_file.exists():
            print(f"❌ No processed document found for: {url}")
            print(f"   Expected file: {doc_filename}")
            return None

        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                content = f.read()

            print(f"✅ Found processed document: {doc_filename}")
            print(f"   Content length: {len(content)} chars")
            return content

        except Exception as e:
            print(f"❌ Error reading document file: {e}")
            return None

    def search_urls_in_cache(self, search_term: str, limit: int = 10) -> List[str]:
        """Search for URLs containing a term in the cache"""
        matches = []

        print(f"🔍 Searching cache for URLs containing: '{search_term}'")

        try:
            for cache_file in self.cache_dir.glob("*.json"):
                if len(matches) >= limit:
                    break

                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)

                    url = cached_data.get('url', '')
                    if search_term.lower() in url.lower():
                        matches.append(url)

                except Exception:
                    continue

            print(f"✅ Found {len(matches)} matching URLs")
            return matches

        except Exception as e:
            print(f"❌ Error searching cache: {e}")
            return []

    def list_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache"""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        stats = {
            "total_cached_urls": len(cache_files),
            "total_cache_size_mb": total_size / (1024 * 1024),
            "cache_directory": str(self.cache_dir)
        }

        print(f"📊 Cache Statistics:")
        print(f"   Total cached URLs: {stats['total_cached_urls']:,}")
        print(f"   Total cache size: {stats['total_cache_size_mb']:.1f} MB")

        return stats

    def list_kb_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        kb_files = list(self.kb_dir.glob("doc_*.txt"))
        total_size = sum(f.stat().st_size for f in kb_files if f.exists())

        # Read build metadata if available
        metadata_file = self.kb_dir / "build_metadata.json"
        build_stats = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    build_stats = metadata.get('build_stats', {})
            except Exception:
                pass

        stats = {
            "total_documents": len(kb_files),
            "total_kb_size_mb": total_size / (1024 * 1024),
            "kb_directory": str(self.kb_dir),
            "build_stats": build_stats
        }

        print(f"📚 Knowledge Base Statistics:")
        print(f"   Total documents: {stats['total_documents']:,}")
        print(f"   Total KB size: {stats['total_kb_size_mb']:.1f} MB")
        if build_stats:
            print(f"   Total words: {build_stats.get('total_words', 'Unknown'):,}")
            print(f"   Avg words per doc: {build_stats.get('avg_words_per_doc', 'Unknown'):.0f}")

        return stats

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='Cache and Knowledge Base Utilities')
    parser.add_argument('--cache-dir', default='cache/url_cache', help='Cache directory path')
    parser.add_argument('--kb-dir', default='knowledge_base_full', help='Knowledge base directory path')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show cache and KB statistics')

    # Get cached content command
    cache_parser = subparsers.add_parser('cache', help='Get cached content for URL')
    cache_parser.add_argument('url', help='URL to get cached content for')

    # Get processed document command
    doc_parser = subparsers.add_parser('doc', help='Get processed document for URL')
    doc_parser.add_argument('url', help='URL to get processed document for')

    # Search URLs command
    search_parser = subparsers.add_parser('search', help='Search for URLs in cache')
    search_parser.add_argument('term', help='Search term')
    search_parser.add_argument('--limit', type=int, default=10, help='Max results to show')

    # Both command (cache + doc)
    both_parser = subparsers.add_parser('both', help='Get both cached content and processed doc for URL')
    both_parser.add_argument('url', help='URL to get both cache and doc for')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize manager
    manager = CacheAndKBManager(args.cache_dir, args.kb_dir)

    if args.command == 'stats':
        print("\n" + "="*50)
        manager.list_cache_stats()
        print("\n" + "="*50)
        manager.list_kb_stats()

    elif args.command == 'cache':
        print(f"\n🔍 Getting cached content for: {args.url}")
        print("="*50)
        cached_content = manager.get_cached_content(args.url)
        if cached_content:
            print(f"\n📄 Raw cached content (first 500 chars):")
            content = cached_content.get('content', '')
            print(content[:500] + ("..." if len(content) > 500 else ""))

    elif args.command == 'doc':
        print(f"\n🔍 Getting processed document for: {args.url}")
        print("="*50)
        doc_content = manager.get_processed_document(args.url)
        if doc_content:
            print(f"\n📄 Processed document (first 500 chars):")
            print(doc_content[:500] + ("..." if len(doc_content) > 500 else ""))

    elif args.command == 'search':
        print(f"\n🔍 Searching for URLs containing: {args.term}")
        print("="*50)
        matches = manager.search_urls_in_cache(args.term, args.limit)
        for i, url in enumerate(matches, 1):
            print(f"{i:2d}. {url}")

    elif args.command == 'both':
        print(f"\n🔍 Getting both cache and doc for: {args.url}")
        print("="*50)
        print("\n📦 CACHED CONTENT:")
        cached_content = manager.get_cached_content(args.url)
        print("\n📚 PROCESSED DOCUMENT:")
        doc_content = manager.get_processed_document(args.url)

if __name__ == "__main__":
    main()