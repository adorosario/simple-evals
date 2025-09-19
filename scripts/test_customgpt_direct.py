#!/usr/bin/env python3
"""
Direct CustomGPT API Test
Test CustomGPT API independently to isolate any issues
"""

import os
import sys
import json
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sampler.customgpt_sampler import CustomGPTSampler


def test_environment():
    """Test environment variable access"""
    print("🔍 ENVIRONMENT DIAGNOSTIC")
    print("=" * 50)

    # Check all CustomGPT related env vars
    env_vars = [
        'CUSTOMGPT_API_KEY',
        'CUSTOMGPT_PROJECT',
        'CUSTOMGPT_MODEL_NAME'
    ]

    for var in env_vars:
        value = os.environ.get(var)
        if value:
            # Show first/last few chars for security
            if len(value) > 10:
                masked = f"{value[:4]}...{value[-4:]}"
            else:
                masked = "***"
            print(f"   {var}: {masked} (length: {len(value)})")
        else:
            print(f"   {var}: NOT SET")

    print()


def test_customgpt_sampler():
    """Test CustomGPT sampler directly"""
    print("🚀 CUSTOMGPT SAMPLER TEST")
    print("=" * 50)

    try:
        # Set model name explicitly
        os.environ["CUSTOMGPT_MODEL_NAME"] = "customgpt-rag"

        print("📝 Creating CustomGPT sampler from environment...")
        sampler = CustomGPTSampler.from_env()

        print("✅ Sampler created successfully")
        print(f"   Config: {sampler.to_config()}")

        # Test simple question
        test_question = "What is the capital of France?"
        message_list = [{"role": "user", "content": test_question}]

        print(f"\n📤 Testing with question: {test_question}")

        start_time = time.time()
        response = sampler(message_list)
        end_time = time.time()

        duration = end_time - start_time

        print(f"📥 Response received in {duration:.2f}s:")
        print(f"   Length: {len(response)} characters")
        print(f"   Content: {response[:200]}...")

        if response and len(response.strip()) > 0:
            print("✅ CustomGPT test SUCCESSFUL!")
            return True
        else:
            print("❌ CustomGPT test FAILED - empty response")
            return False

    except Exception as e:
        print(f"❌ CustomGPT test FAILED with exception:")
        print(f"   Error: {str(e)}")
        print(f"   Type: {type(e).__name__}")

        # Additional debugging
        import traceback
        print("\n🔍 Full traceback:")
        traceback.print_exc()

        return False


def test_raw_api_call():
    """Test raw API call to CustomGPT to isolate issues"""
    print("\n🌐 RAW API TEST")
    print("=" * 50)

    try:
        import requests

        api_key = os.environ.get('CUSTOMGPT_API_KEY')
        project_id = os.environ.get('CUSTOMGPT_PROJECT')

        if not api_key or not project_id:
            print("❌ Missing API key or project ID for raw test")
            return False

        # Test endpoint accessibility
        url = f"https://app.customgpt.ai/api/v1/projects/{project_id}/conversations"

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        # Create conversation
        print("📝 Creating conversation...")
        conversation_data = {
            'name': f'Test Conversation {datetime.now().strftime("%H:%M:%S")}'
        }

        response = requests.post(url, headers=headers, json=conversation_data, timeout=10)

        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")

        if response.status_code == 200 or response.status_code == 201:
            print("✅ Raw API test SUCCESSFUL!")
            return True
        else:
            print(f"❌ Raw API test FAILED - status {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Raw API test FAILED:")
        print(f"   Error: {str(e)}")
        return False


def main():
    """Run comprehensive CustomGPT testing"""
    print("🧪 CUSTOMGPT INDEPENDENT TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python version: {sys.version}")
    print("=" * 60)

    # Test environment
    test_environment()

    # Test sampler
    sampler_success = test_customgpt_sampler()

    # Test raw API if sampler fails
    if not sampler_success:
        raw_success = test_raw_api_call()
    else:
        raw_success = True

    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 30)
    print(f"Sampler Test: {'✅ PASS' if sampler_success else '❌ FAIL'}")
    print(f"Raw API Test: {'✅ PASS' if raw_success else '❌ FAIL'}")

    if sampler_success:
        print("\n🎉 CustomGPT is working correctly!")
        print("   The issue may be in the benchmark integration.")
    else:
        print("\n🔍 CustomGPT has issues:")
        if raw_success:
            print("   - Raw API works, issue is in sampler implementation")
        else:
            print("   - API connectivity issues (auth, network, etc.)")

    return 0 if sampler_success else 1


if __name__ == "__main__":
    sys.exit(main())