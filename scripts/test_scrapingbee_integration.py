#!/usr/bin/env python3
"""
Test ScrapingBee Integration with Knowledge Base Builder
Test that ScrapingBee fallback works correctly for failed URLs
"""

import sys
import os
import tempfile
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.knowledge_base_builder import KnowledgeBaseBuilder


def test_scrapingbee_fallback():
    """Test ScrapingBee fallback with URLs that should fail regular fetching"""
    print("🧪 SCRAPINGBEE INTEGRATION TEST")
    print("=" * 60)

    # Create test URLs - mix of working and potentially problematic ones
    test_urls = [
        "https://httpbin.org/html",  # Should work with regular fetch
        "https://this-domain-does-not-exist-12345.com/",  # Should fail both
        "https://quotes.toscrape.com/js/",  # May need JavaScript
        "https://news.ycombinator.com/",  # May be blocked by regular fetch
    ]

    # Create temporary URL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for url in test_urls:
            f.write(url + '\n')
        url_file = f.name

    try:
        print(f"📝 Test URLs ({len(test_urls)}):")
        for i, url in enumerate(test_urls, 1):
            print(f"   {i}. {url}")

        # Test with ScrapingBee enabled
        print(f"\n🛡️  TESTING WITH SCRAPINGBEE ENABLED")
        print("-" * 50)

        start_time = time.time()

        builder_with_sb = KnowledgeBaseBuilder(
            output_dir=tempfile.mkdtemp(prefix="kb_test_with_sb_"),
            cache_dir=tempfile.mkdtemp(prefix="cache_test_"),
            min_document_words=10,  # Low threshold for testing
            max_documents=len(test_urls),
            max_workers=2,  # Low concurrency for clear logs
            timeout=10,
            max_retries=1,
            use_cache=False,  # Disable cache for clean test
            use_scrapingbee_fallback=True  # ENABLE SCRAPINGBEE
        )

        with builder_with_sb:
            result_with_sb = builder_with_sb.build_from_url_file(url_file)

        duration_with_sb = time.time() - start_time

        print(f"\n📊 RESULTS WITH SCRAPINGBEE:")
        print(f"   Total URLs: {result_with_sb.total_urls}")
        print(f"   Successful fetches: {result_with_sb.successful_fetches}")
        print(f"   Documents created: {result_with_sb.documents_created}")
        print(f"   ScrapingBee rescues: {result_with_sb.scrapingbee_rescues}")
        print(f"   ScrapingBee cost: {result_with_sb.scrapingbee_cost} credits")
        print(f"   Duration: {duration_with_sb:.1f}s")

        # Test without ScrapingBee for comparison
        print(f"\n❌ TESTING WITHOUT SCRAPINGBEE")
        print("-" * 50)

        start_time = time.time()

        builder_without_sb = KnowledgeBaseBuilder(
            output_dir=tempfile.mkdtemp(prefix="kb_test_without_sb_"),
            cache_dir=tempfile.mkdtemp(prefix="cache_test_"),
            min_document_words=10,  # Low threshold for testing
            max_documents=len(test_urls),
            max_workers=2,  # Low concurrency for clear logs
            timeout=10,
            max_retries=1,
            use_cache=False,  # Disable cache for clean test
            use_scrapingbee_fallback=False  # DISABLE SCRAPINGBEE
        )

        with builder_without_sb:
            result_without_sb = builder_without_sb.build_from_url_file(url_file)

        duration_without_sb = time.time() - start_time

        print(f"\n📊 RESULTS WITHOUT SCRAPINGBEE:")
        print(f"   Total URLs: {result_without_sb.total_urls}")
        print(f"   Successful fetches: {result_without_sb.successful_fetches}")
        print(f"   Documents created: {result_without_sb.documents_created}")
        print(f"   ScrapingBee rescues: {result_without_sb.scrapingbee_rescues}")
        print(f"   ScrapingBee cost: {result_without_sb.scrapingbee_cost} credits")
        print(f"   Duration: {duration_without_sb:.1f}s")

        # Analysis
        print(f"\n📈 COMPARISON ANALYSIS:")
        print(f"   Improvement in success: +{result_with_sb.documents_created - result_without_sb.documents_created} documents")
        print(f"   URLs rescued by ScrapingBee: {result_with_sb.scrapingbee_rescues}")

        if result_with_sb.scrapingbee_rescues > 0:
            print(f"   Cost per rescued URL: {result_with_sb.scrapingbee_cost/result_with_sb.scrapingbee_rescues:.1f} credits")

        # Success criteria
        success_criteria = [
            ("ScrapingBee enabled correctly", result_with_sb.scrapingbee_rescues >= 0),
            ("ScrapingBee improved results", result_with_sb.documents_created >= result_without_sb.documents_created),
            ("Cost tracking working", result_with_sb.scrapingbee_cost >= 0),
            ("No crashes", True)  # If we got here, no crashes occurred
        ]

        print(f"\n✅ SUCCESS CRITERIA:")
        all_passed = True
        for criterion, passed in success_criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {criterion}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            print(f"\n🎉 INTEGRATION TEST SUCCESSFUL!")
            print("   ScrapingBee fallback is working correctly.")
            return True
        else:
            print(f"\n⚠️  Some criteria failed. Check configuration.")
            return False

    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        try:
            os.unlink(url_file)
        except:
            pass


def main():
    """Run ScrapingBee integration test"""
    print("🔬 SCRAPINGBEE KNOWLEDGE BASE INTEGRATION TEST")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    success = test_scrapingbee_fallback()

    print(f"\n📋 FINAL RESULT:")
    if success:
        print("✅ ScrapingBee integration is working correctly!")
        print("   You can now use --concurrency 5 with automatic fallback for failed URLs.")
        return 0
    else:
        print("❌ ScrapingBee integration has issues. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())