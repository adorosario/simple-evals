#!/usr/bin/env python3
"""
Direct test of OpenAI Responses API with vector store
Tests with specific content we know is in the vector store
"""

import os
import sys
import time
from openai import OpenAI

def test_responses_api():
    """Test Responses API directly with vector store"""

    vector_store_id = os.environ.get('OPENAI_VECTOR_STORE_ID')
    if not vector_store_id:
        print("❌ OPENAI_VECTOR_STORE_ID not set")
        return False

    print(f"🧪 TESTING RESPONSES API")
    print(f"Vector Store: {vector_store_id}")

    # Question about content we know is in the vector store
    test_question = "What was GRANAT and when was it launched?"
    print(f"❓ Question: {test_question}")

    client = OpenAI()

    # First check if responses API exists
    print(f"\n🔍 Checking OpenAI client capabilities:")
    print(f"Available methods: {[attr for attr in dir(client) if not attr.startswith('_')]}")

    if hasattr(client, 'responses'):
        print("✅ Responses API found!")

        try:
            print(f"\n🚀 Testing Responses API...")
            start_time = time.time()

            response = client.responses.create(
                input=test_question,
                model="gpt-4o-mini",
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                }]
            )

            end_time = time.time()
            print(f"✅ Response received in {end_time - start_time:.1f}s")
            print(f"📄 Response: {response}")

            return True

        except Exception as e:
            print(f"❌ Responses API failed: {e}")
            return False

    else:
        print("❌ Responses API not found in client")
        print(f"OpenAI version: {client.__module__}")

        # Try to check what version we have
        import openai
        print(f"OpenAI library version: {openai.__version__}")

        return False

def test_assistants_api_simple():
    """Test Assistants API with minimal approach"""

    vector_store_id = os.environ.get('OPENAI_VECTOR_STORE_ID')
    test_question = "What was GRANAT and when was it launched?"

    print(f"\n🧪 TESTING ASSISTANTS API (simplified)")
    print(f"❓ Question: {test_question}")

    client = OpenAI()

    try:
        print(f"🔧 Creating assistant...")
        assistant = client.beta.assistants.create(
            name="Test Assistant",
            instructions="You are a helpful assistant. Use the knowledge base to answer questions.",
            model="gpt-4o-mini",
            tools=[{"type": "file_search"}],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        )
        print(f"✅ Assistant created: {assistant.id}")

        print(f"🗨️ Creating thread...")
        thread = client.beta.threads.create(
            messages=[{
                "role": "user",
                "content": test_question
            }]
        )
        print(f"✅ Thread created: {thread.id}")

        print(f"🏃 Running assistant...")
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        print(f"✅ Run started: {run.id}")

        # Wait for completion with timeout
        max_wait = 30
        start_time = time.time()

        while time.time() - start_time < max_wait:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

            print(f"⏳ Status: {run_status.status}")

            if run_status.status == "completed":
                print(f"✅ Run completed!")

                # Get response
                messages = client.beta.threads.messages.list(
                    thread_id=thread.id,
                    order="desc",
                    limit=1
                )

                if messages.data and messages.data[0].role == "assistant":
                    response_content = ""
                    for content_block in messages.data[0].content:
                        if content_block.type == "text":
                            response_content += content_block.text.value

                    print(f"📄 Response: {response_content}")

                    # Cleanup
                    try:
                        client.beta.assistants.delete(assistant.id)
                        client.beta.threads.delete(thread.id)
                        print(f"🧹 Cleaned up resources")
                    except:
                        pass

                    return True

            elif run_status.status in ["failed", "cancelled", "expired"]:
                print(f"❌ Run failed: {run_status.status}")
                break
            else:
                time.sleep(1)

        print(f"⏰ Timeout after {max_wait}s")
        return False

    except Exception as e:
        print(f"❌ Assistants API failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_search_direct():
    """Test direct vector search if available"""

    vector_store_id = os.environ.get('OPENAI_VECTOR_STORE_ID')
    test_question = "What was GRANAT and when was it launched?"

    print(f"\n🧪 TESTING DIRECT VECTOR SEARCH")
    print(f"❓ Question: {test_question}")

    client = OpenAI()

    try:
        # Check if direct search is available
        if hasattr(client.beta.vector_stores, 'search'):
            print(f"🔍 Direct search available")

            search_results = client.beta.vector_stores.search(
                vector_store_id=vector_store_id,
                query=test_question
            )

            print(f"📄 Search results: {search_results}")
            return True
        else:
            print(f"❌ Direct search not available")
            return False

    except Exception as e:
        print(f"❌ Direct search failed: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 DIRECT API TESTING WITH VECTOR STORE")
    print("=" * 60)

    results = {
        "responses_api": test_responses_api(),
        "assistants_api": test_assistants_api_simple(),
        "direct_search": test_vector_search_direct()
    }

    print(f"\n📊 RESULTS:")
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")

    if any(results.values()):
        print(f"\n🎉 At least one API method works!")
        return 0
    else:
        print(f"\n💥 All API methods failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())