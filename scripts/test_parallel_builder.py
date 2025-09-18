#!/usr/bin/env python3
"""
Test the parallelized Knowledge Base Builder
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.knowledge_base_builder import KnowledgeBaseBuilder


def test_parallel_performance():
    """Test parallel processing performance"""
    print("Testing Parallelized Knowledge Base Builder")
    print("=" * 60)

    # Test with 50 URLs to see parallel speedup
    start_time = time.time()

    builder = KnowledgeBaseBuilder(
        output_dir="knowledge_base_parallel_test",
        cache_dir="cache/url_cache",
        min_document_words=20,
        max_documents=50,       # Test with 50 URLs
        max_workers=20,         # 20 parallel workers
        timeout=10,             # 10s timeout
        max_retries=2,          # 2 retries max
        use_cache=True,
        force_refresh=False
    )

    print(f"Configuration:")
    print(f"  Max workers: {builder.max_workers}")
    print(f"  Timeout: {builder.timeout}s")
    print(f"  Max retries: {builder.max_retries}")
    print(f"  Max documents: {builder.max_documents}")
    print(f"  Cache enabled: {builder.use_cache}")

    try:
        with builder:
            result = builder.build_from_url_file('build-rag/urls.txt')

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n=== PARALLEL PERFORMANCE RESULTS ===")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.2f} minutes)")
        print(f"Total URLs: {result.total_urls}")
        print(f"Documents created: {result.documents_created}")
        print(f"Errors: {len(result.errors)}")
        print(f"")
        print(f"Performance:")
        print(f"  {result.total_urls/duration:.1f} URLs/second")
        print(f"  {duration/result.total_urls:.2f} seconds per URL")
        print(f"  Success rate: {result.documents_created/result.total_urls:.1%}")
        print(f"")
        print(f"Extrapolation to 11,000 URLs:")
        estimated_11k = (11000 / result.total_urls) * duration
        print(f"  Estimated time: {estimated_11k:.0f} seconds ({estimated_11k/3600:.2f} hours)")
        print(f"  Expected documents: {int(11000 * result.documents_created/result.total_urls)}")

        # Show speedup vs sequential
        sequential_estimate = result.total_urls * 2.93  # From previous test
        speedup = sequential_estimate / duration
        print(f"")
        print(f"Speedup analysis:")
        print(f"  Sequential estimate: {sequential_estimate:.1f}s")
        print(f"  Parallel actual: {duration:.1f}s")
        print(f"  Speedup: {speedup:.1f}x")

        if result.errors:
            print(f"\nError summary (first 5):")
            for i, error in enumerate(result.errors[:5]):
                print(f"  {i+1}. {error[:80]}...")

        return 0

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Run parallel performance test"""
    return test_parallel_performance()


if __name__ == "__main__":
    sys.exit(main())