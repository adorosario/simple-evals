#!/usr/bin/env python3
"""
CustomGPT Knowledge Base Test
Test with questions that should be in our knowledge base
"""

import os
import sys
import json
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sampler.customgpt_sampler import CustomGPTSampler


def test_knowledge_questions():
    """Test CustomGPT with questions from our knowledge base"""
    print("🧠 CUSTOMGPT KNOWLEDGE BASE TEST")
    print("=" * 60)

    try:
        # Set model name explicitly
        os.environ["CUSTOMGPT_MODEL_NAME"] = "customgpt-rag"

        print("📝 Creating CustomGPT sampler...")
        sampler = CustomGPTSampler.from_env()

        # Test questions that should be in our space observatory knowledge base
        test_questions = [
            "What is GRANAT?",
            "What was GRANAT's mission?",
            "When was GRANAT launched?",
            "What type of observatory is GRANAT?",
            "Tell me about the GRANAT space observatory"
        ]

        successful_tests = 0

        for i, question in enumerate(test_questions, 1):
            print(f"\n📤 Test {i}/{len(test_questions)}: {question}")

            message_list = [{"role": "user", "content": question}]

            start_time = time.time()
            response = sampler(message_list)
            end_time = time.time()

            duration = end_time - start_time

            print(f"⏱️  Response time: {duration:.2f}s")
            print(f"📄 Response length: {len(response)} characters")
            print(f"📝 Response: {response[:200]}...")

            # Check if we got a real response (not the default "I don't know")
            if response and "I don't know" not in response and len(response) > 100:
                print("✅ Knowledge-based response received!")
                successful_tests += 1
            elif response and len(response) > 10:
                print("⚠️  Generic response (knowledge base may not contain this info)")
            else:
                print("❌ Empty or failed response")

            print("-" * 60)

            # Small delay between requests
            time.sleep(1)

        print(f"\n📊 KNOWLEDGE TEST SUMMARY:")
        print(f"   Questions asked: {len(test_questions)}")
        print(f"   Knowledge-based responses: {successful_tests}")
        print(f"   Success rate: {successful_tests/len(test_questions)*100:.1f}%")

        if successful_tests > 0:
            print("✅ CustomGPT knowledge base is responding!")
        else:
            print("⚠️  CustomGPT may not have the expected knowledge base loaded")

        return successful_tests > 0

    except Exception as e:
        print(f"❌ Knowledge base test failed:")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run knowledge base test"""
    print("🔬 CUSTOMGPT KNOWLEDGE BASE VALIDATION")
    print("=" * 70)
    print(f"Testing with questions from our uploaded knowledge base")
    print("=" * 70)

    success = test_knowledge_questions()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())