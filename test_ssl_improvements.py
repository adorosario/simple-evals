#!/usr/bin/env python3
"""
Test script to demonstrate SSL bypassing and user agent rotation improvements
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.url_fetcher import URLFetcher

def test_ssl_bypass():
    """Test SSL certificate bypass functionality"""
    print("🧪 Testing SSL Certificate Bypass & User Agent Rotation")
    print("=" * 60)

    # Test URL that might have SSL issues
    test_url = "http://eyeway.org.in/?q=ronnie-lee-milsap"

    # Create fetcher with SSL bypass enabled
    fetcher = URLFetcher(
        timeout=10,
        verify_ssl=False,  # Bypass SSL errors
        use_random_user_agents=True,  # Use random user agents
        max_retries=1
    )

    print(f"🌐 Testing URL: {test_url}")
    print(f"📋 SSL Verification: {fetcher.verify_ssl}")
    print(f"🎭 Random User Agents: {fetcher.use_random_user_agents}")
    print(f"🔄 Current User Agent: {fetcher.session.headers.get('User-Agent')}")

    result = fetcher.fetch(test_url)

    print(f"\n📊 Results:")
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Status Code: {result.status_code}")
        print(f"   Content Size: {len(result.content)} bytes")
        print(f"   Content Type: {result.content_type}")
        print(f"   Final URL: {result.final_url}")
    else:
        print(f"   Error: {result.error_message}")

    return result.success

if __name__ == "__main__":
    success = test_ssl_bypass()
    print(f"\n✅ Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)