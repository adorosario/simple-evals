#!/usr/bin/env python3
"""
Standalone ScrapingBee Test with TDD
Test ScrapingBee API independently before integrating into KB builder
"""

import sys
import os
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scrapingbee_fetcher import ScrapingBeeFetcher, ScrapingBeeResult


def test_scrapingbee_environment():
    """Test ScrapingBee environment setup"""
    print("🔍 SCRAPINGBEE ENVIRONMENT TEST")
    print("=" * 50)

    api_key = os.environ.get('SCRAPINGBEE_API_KEY')
    if api_key:
        masked_key = f"{api_key[:8]}...{api_key[-8:]}" if len(api_key) > 16 else "***"
        print(f"   ✅ API Key: {masked_key} (length: {len(api_key)})")
        return True
    else:
        print("   ❌ SCRAPINGBEE_API_KEY not found")
        return False


def test_scrapingbee_credits():
    """Test ScrapingBee API access and check credits"""
    print("\\n💰 SCRAPINGBEE CREDITS TEST")
    print("=" * 50)

    try:
        fetcher = ScrapingBeeFetcher()
        credits = fetcher.get_remaining_credits()

        if credits is not None:
            print(f"   ✅ Remaining credits: {credits:,}")
            if credits < 100:
                print("   ⚠️  Warning: Low credit balance")
            return True
        else:
            print("   ❌ Could not check credits")
            return False

    except Exception as e:
        print(f"   ❌ Error checking credits: {e}")
        return False


def test_scrapingbee_simple_fetch():
    """Test simple URL fetch without JavaScript"""
    print("\\n🌐 SIMPLE FETCH TEST")
    print("=" * 50)

    test_url = "https://httpbin.org/html"

    try:
        # Test without JavaScript first (cheaper)
        fetcher = ScrapingBeeFetcher(enable_js=False, premium_proxy=False)

        print(f"   Testing: {test_url}")
        start_time = time.time()

        result = fetcher.fetch(test_url)

        end_time = time.time()
        duration = end_time - start_time

        print(f"   Response time: {duration:.2f}s")
        print(f"   Success: {result.success}")
        print(f"   Status code: {result.status_code}")
        print(f"   API cost: {result.api_cost} credits")

        if result.success:
            print(f"   Content length: {len(result.content)} chars")
            print(f"   Content preview: {result.content[:100]}...")
            print("   ✅ Simple fetch successful")
            return True
        else:
            print(f"   ❌ Fetch failed: {result.error_message}")
            return False

    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def test_scrapingbee_javascript_fetch():
    """Test JavaScript-enabled fetch"""
    print("\\n🚀 JAVASCRIPT FETCH TEST")
    print("=" * 50)

    # Test with a site that requires JavaScript
    test_url = "https://quotes.toscrape.com/js/"

    try:
        # Test with JavaScript enabled
        fetcher = ScrapingBeeFetcher(enable_js=True, premium_proxy=True)

        print(f"   Testing: {test_url}")
        print("   Features: JavaScript ON, Premium Proxy ON")

        start_time = time.time()
        result = fetcher.fetch(test_url)
        end_time = time.time()

        duration = end_time - start_time

        print(f"   Response time: {duration:.2f}s")
        print(f"   Success: {result.success}")
        print(f"   Status code: {result.status_code}")
        print(f"   API cost: {result.api_cost} credits")

        if result.success:
            print(f"   Content length: {len(result.content)} chars")

            # Check if JavaScript rendered content properly
            if "quote" in result.content.lower() and "author" in result.content.lower():
                print("   ✅ JavaScript rendering successful (found quotes)")
                return True
            else:
                print("   ⚠️  JavaScript may not have rendered properly")
                print(f"   Content preview: {result.content[:200]}...")
                return False
        else:
            print(f"   ❌ JavaScript fetch failed: {result.error_message}")
            return False

    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def test_scrapingbee_difficult_site():
    """Test with a potentially difficult site that might block regular requests"""
    print("\\n🛡️  DIFFICULT SITE TEST")
    print("=" * 50)

    # Test with a site that often blocks scrapers
    test_url = "https://news.ycombinator.com/"

    try:
        # Use maximum features for difficult sites
        fetcher = ScrapingBeeFetcher(
            enable_js=True,
            premium_proxy=True,
            stealth_proxy=False  # Start with premium, can escalate to stealth
        )

        print(f"   Testing: {test_url}")
        print("   Features: JavaScript ON, Premium Proxy ON")

        start_time = time.time()
        result = fetcher.fetch(test_url)
        end_time = time.time()

        duration = end_time - start_time

        print(f"   Response time: {duration:.2f}s")
        print(f"   Success: {result.success}")
        print(f"   Status code: {result.status_code}")
        print(f"   API cost: {result.api_cost} credits")

        if result.success:
            print(f"   Content length: {len(result.content)} chars")

            # Check for Hacker News specific content
            if "hacker news" in result.content.lower() or "ycombinator" in result.content.lower():
                print("   ✅ Difficult site successfully scraped")
                return True
            else:
                print("   ⚠️  Got content but may not be the expected page")
                print(f"   Content preview: {result.content[:200]}...")
                return False
        else:
            print(f"   ❌ Difficult site failed: {result.error_message}")
            return False

    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def test_scrapingbee_error_handling():
    """Test error handling with invalid URLs"""
    print("\\n❌ ERROR HANDLING TEST")
    print("=" * 50)

    invalid_urls = [
        "https://this-domain-definitely-does-not-exist-12345.com/",
        "https://httpbin.org/status/404",
        "https://httpbin.org/status/500"
    ]

    fetcher = ScrapingBeeFetcher(enable_js=False, premium_proxy=False)

    error_handling_working = True

    for url in invalid_urls:
        try:
            print(f"   Testing error case: {url}")
            result = fetcher.fetch(url)

            if not result.success:
                print(f"   ✅ Correctly handled error: {result.error_message}")
            else:
                print(f"   ⚠️  Expected error but got success")
                error_handling_working = False

        except Exception as e:
            print(f"   ❌ Unexpected exception: {e}")
            error_handling_working = False

    return error_handling_working


def main():
    """Run comprehensive ScrapingBee TDD tests"""
    print("🧪 SCRAPINGBEE STANDALONE TDD TESTS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    tests = [
        ("Environment Setup", test_scrapingbee_environment),
        ("Credits Check", test_scrapingbee_credits),
        ("Simple Fetch", test_scrapingbee_simple_fetch),
        ("JavaScript Fetch", test_scrapingbee_javascript_fetch),
        ("Difficult Site", test_scrapingbee_difficult_site),
        ("Error Handling", test_scrapingbee_error_handling),
    ]

    results = []
    total_cost = 0

    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\\n📋 TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print(f"\\n🎯 OVERALL RESULT: {passed}/{total} tests passed")

    if passed == total:
        print("\\n🎉 ALL TESTS PASSED! ScrapingBee is ready for integration.")
        return 0
    else:
        print(f"\\n⚠️  {total - passed} tests failed. Check configuration before integration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())