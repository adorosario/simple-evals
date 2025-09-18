#!/usr/bin/env python3
"""
Test script for Knowledge Base Builder component
Tests the complete pipeline: URL fetching → content extraction → processing → file preparation
"""

import sys
import os
import tempfile
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.knowledge_base_builder import KnowledgeBaseBuilder
from src.url_fetcher import URLFetcher
from src.content_extractor import ContentExtractor
from src.content_processor import ContentProcessor


def test_small_url_set():
    """Test with a small set of URLs"""
    print("\n" + "="*60)
    print("Testing Knowledge Base Builder with Small URL Set")
    print("="*60)

    # Create test URLs
    test_urls = [
        "https://httpbin.org/html",
        "https://example.com",
        "https://httpbin.org/json"
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create URL file
        url_file = os.path.join(temp_dir, 'test_urls.txt')
        with open(url_file, 'w') as f:
            f.write('\n'.join(test_urls))

        # Initialize builder
        builder = KnowledgeBaseBuilder(
            output_dir=os.path.join(temp_dir, 'knowledge_base'),
            cache_dir=os.path.join(temp_dir, 'cache'),
            min_document_words=20,
            use_cache=True,
            force_refresh=False
        )

        # Build knowledge base
        with builder:
            result = builder.build_from_url_file(url_file)

        # Display results
        print(f"Build Results:")
        print(f"  Total URLs: {result.total_urls}")
        print(f"  Successful fetches: {result.successful_fetches}")
        print(f"  Successful extractions: {result.successful_extractions}")
        print(f"  Documents created: {result.documents_created}")
        print(f"  Output directory: {result.output_directory}")

        if result.build_stats:
            print(f"  Total words: {result.build_stats.get('total_words', 'N/A')}")
            print(f"  Average words per doc: {result.build_stats.get('avg_words_per_doc', 'N/A'):.1f}")
            print(f"  Success rate: {result.build_stats.get('success_rate', 'N/A'):.1%}")

        if result.errors:
            print(f"  Errors: {len(result.errors)}")
            for i, error in enumerate(result.errors[:3]):  # Show first 3 errors
                print(f"    {i+1}. {error}")
            if len(result.errors) > 3:
                print(f"    ... and {len(result.errors) - 3} more")

        # Check output files
        files = builder.get_document_list_for_openai()
        print(f"\nGenerated {len(files)} text files for OpenAI:")
        for i, filepath in enumerate(files[:5]):  # Show first 5 files
            filename = os.path.basename(filepath)
            file_size = os.path.getsize(filepath)
            print(f"  {i+1}. {filename} ({file_size} bytes)")

        # Show sample file content
        if files:
            sample_file = files[0]
            print(f"\nSample content from {os.path.basename(sample_file)}:")
            print("-" * 40)
            with open(sample_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content[:300] + "..." if len(content) > 300 else content)
            print("-" * 40)

def test_url_file_creation():
    """Test creating and processing a URL file"""
    print("\n" + "="*60)
    print("Testing URL File Creation and Processing")
    print("="*60)

    test_urls = [
        "https://example.com",
        "# This is a comment line",
        "",  # Empty line
        "https://httpbin.org/html",
        "invalid-url",  # Should be skipped
        "https://httpbin.org/user-agent",
        "http://example.org"  # Different protocol
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create URL file with mixed content
        url_file = os.path.join(temp_dir, 'mixed_urls.txt')
        with open(url_file, 'w') as f:
            f.write('\n'.join(test_urls))

        print(f"Created URL file with {len(test_urls)} lines")

        # Test URL loading
        builder = KnowledgeBaseBuilder(
            output_dir=os.path.join(temp_dir, 'output'),
            use_cache=False,
            min_document_words=10
        )

        try:
            urls = builder._load_urls_from_file(url_file)
            print(f"✓ Loaded {len(urls)} valid URLs from file")
            for i, url in enumerate(urls):
                print(f"  {i+1}. {url}")
        except Exception as e:
            print(f"✗ Failed to load URLs: {e}")

def test_caching_behavior():
    """Test URL caching functionality"""
    print("\n" + "="*60)
    print("Testing URL Caching Behavior")
    print("="*60)

    test_urls = [
        "https://httpbin.org/html",
        "https://example.com"
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = os.path.join(temp_dir, 'cache')
        output_dir = os.path.join(temp_dir, 'output')

        # First build - should fetch from web
        print("First build (should fetch from web):")
        builder1 = KnowledgeBaseBuilder(
            output_dir=output_dir + "_1",
            cache_dir=cache_dir,
            min_document_words=10,
            use_cache=True,
            force_refresh=False
        )

        with builder1:
            result1 = builder1.build_from_urls(test_urls)
            print(f"  Documents created: {result1.documents_created}")

        # Check cache directory
        if os.path.exists(cache_dir):
            cache_files = os.listdir(cache_dir)
            print(f"  Cache files created: {len(cache_files)}")

        # Second build - should use cache
        print("\nSecond build (should use cache):")
        builder2 = KnowledgeBaseBuilder(
            output_dir=output_dir + "_2",
            cache_dir=cache_dir,
            min_document_words=10,
            use_cache=True,
            force_refresh=False
        )

        with builder2:
            result2 = builder2.build_from_urls(test_urls)
            print(f"  Documents created: {result2.documents_created}")

        # Third build - force refresh
        print("\nThird build (force refresh):")
        builder3 = KnowledgeBaseBuilder(
            output_dir=output_dir + "_3",
            cache_dir=cache_dir,
            min_document_words=10,
            use_cache=True,
            force_refresh=True
        )

        with builder3:
            result3 = builder3.build_from_urls(test_urls)
            print(f"  Documents created: {result3.documents_created}")

def test_error_handling():
    """Test error handling with problematic URLs"""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)

    # Mix of good and bad URLs
    test_urls = [
        "https://httpbin.org/html",  # Should work
        "https://httpbin.org/status/404",  # 404 error
        "https://httpbin.org/status/500",  # Server error
        "https://nonexistent-domain-12345.com",  # DNS error
        "https://example.com",  # Should work
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        builder = KnowledgeBaseBuilder(
            output_dir=os.path.join(temp_dir, 'error_test'),
            use_cache=False,
            min_document_words=10
        )

        with builder:
            result = builder.build_from_urls(test_urls)

        print(f"Results with error handling:")
        print(f"  Total URLs: {result.total_urls}")
        print(f"  Successful fetches: {result.successful_fetches}")
        print(f"  Documents created: {result.documents_created}")
        print(f"  Errors encountered: {len(result.errors)}")

        if result.errors:
            print(f"\nError details:")
            for i, error in enumerate(result.errors[:3]):
                print(f"  {i+1}. {error}")

def test_build_metadata():
    """Test build metadata generation"""
    print("\n" + "="*60)
    print("Testing Build Metadata Generation")
    print("="*60)

    test_urls = ["https://example.com", "https://httpbin.org/html"]

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, 'metadata_test')

        builder = KnowledgeBaseBuilder(
            output_dir=output_dir,
            use_cache=False,
            min_document_words=10
        )

        with builder:
            result = builder.build_from_urls(test_urls)

        # Check metadata file
        metadata_file = os.path.join(output_dir, 'build_metadata.json')
        if os.path.exists(metadata_file):
            print(f"✓ Metadata file created: {metadata_file}")

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            print(f"Metadata contents:")
            print(f"  Build stats keys: {list(metadata.get('build_stats', {}).keys())}")
            print(f"  Total errors: {metadata.get('total_errors', 0)}")
            print(f"  Config saved: {'build_config' in metadata}")

            # Show sample build stats
            build_stats = metadata.get('build_stats', {})
            if build_stats:
                print(f"  Total documents: {build_stats.get('total_documents')}")
                print(f"  Total words: {build_stats.get('total_words')}")
                print(f"  Success rate: {build_stats.get('success_rate', 0):.1%}")
        else:
            print(f"✗ Metadata file not found")

def test_max_documents_limit():
    """Test max documents limit"""
    print("\n" + "="*60)
    print("Testing Max Documents Limit")
    print("="*60)

    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://httpbin.org/json",
        "https://httpbin.org/user-agent"
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Limit to 2 documents
        builder = KnowledgeBaseBuilder(
            output_dir=os.path.join(temp_dir, 'limited'),
            use_cache=False,
            min_document_words=5,
            max_documents=2
        )

        with builder:
            result = builder.build_from_urls(test_urls)

        print(f"Results with max_documents=2:")
        print(f"  Input URLs: {len(test_urls)}")
        print(f"  Processed URLs: {result.total_urls}")
        print(f"  Documents created: {result.documents_created}")

        # Should have processed only first 2 URLs
        assert result.total_urls == 2

def main():
    """Run all knowledge base builder tests"""
    print("Knowledge Base Builder Test Suite")
    print("=" * 70)

    try:
        # Run all tests
        test_url_file_creation()
        test_small_url_set()
        test_caching_behavior()
        test_error_handling()
        test_build_metadata()
        test_max_documents_limit()

        print("\n" + "="*70)
        print("Knowledge Base Builder Test Suite Completed Successfully!")
        print("Ready for OpenAI vector store upload preparation!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())