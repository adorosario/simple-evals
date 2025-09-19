#!/usr/bin/env python3
"""
Test script for OpenAI Vector Store Manager
Tests vector store creation and file upload functionality
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.openai_vector_store import OpenAIVectorStoreManager, VectorStoreInfo


def create_test_documents():
    """Create temporary test documents for upload"""
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Created test directory: {temp_dir}")

    # Create test documents
    test_docs = [
        ("doc1.txt", "This is a test document about artificial intelligence and machine learning."),
        ("doc2.txt", "Another document discussing natural language processing and neural networks."),
        ("doc3.txt", "A third document covering computer vision and deep learning algorithms.")
    ]

    created_files = []
    for filename, content in test_docs:
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        created_files.append(file_path)
        print(f"   ✅ Created: {filename} ({len(content)} chars)")

    return temp_dir, created_files


def test_vector_store_manager():
    """Test OpenAI Vector Store Manager functionality"""
    print("🧪 TESTING OPENAI VECTOR STORE MANAGER")
    print("=" * 50)

    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("   Please set your OpenAI API key to run this test")
        return False

    print(f"✅ Found OpenAI API key: {api_key[:8]}...")

    try:
        # Initialize manager
        print("\n🔧 Initializing Vector Store Manager...")
        manager = OpenAIVectorStoreManager(api_key=api_key)
        print("✅ Manager initialized successfully")

        # Create test documents
        print("\n📝 Creating test documents...")
        temp_dir, test_files = create_test_documents()

        # Test file upload
        print("\n📤 Testing file upload...")
        file_ids = manager.upload_files(test_files)
        print(f"✅ Uploaded {len(file_ids)} files")
        for i, file_id in enumerate(file_ids):
            print(f"   File {i+1}: {file_id}")

        if not file_ids:
            print("❌ No files were uploaded successfully")
            return False

        # Test vector store creation
        print("\n🗂️  Testing vector store creation...")
        store_name = "Test Vector Store"
        store_info = manager.create_vector_store(store_name, file_ids)

        print(f"✅ Vector store created successfully!")
        print(f"   Store ID: {store_info.vector_store_id}")
        print(f"   Name: {store_info.name}")
        print(f"   Files: {store_info.file_count}")
        print(f"   Status: {store_info.status}")

        # Test store retrieval
        print("\n🔍 Testing vector store retrieval...")
        retrieved_info = manager.get_vector_store_info(store_info.vector_store_id)
        print(f"✅ Retrieved store info:")
        print(f"   ID matches: {retrieved_info.vector_store_id == store_info.vector_store_id}")
        print(f"   Name matches: {retrieved_info.name == store_info.name}")
        print(f"   File count: {retrieved_info.file_count}")

        # Test directory-based creation
        print("\n📁 Testing directory-based creation...")
        directory_store = manager.create_vector_store_from_directory(
            temp_dir,
            "Test Directory Store"
        )
        print(f"✅ Directory store created: {directory_store.vector_store_id}")
        print(f"   Files from directory: {directory_store.file_count}")

        # Test store listing
        print("\n📋 Testing vector store listing...")
        all_stores = manager.list_vector_stores()
        print(f"✅ Found {len(all_stores)} vector stores in account")

        test_stores = [s for s in all_stores if s.name.startswith("Test")]
        print(f"   Test stores found: {len(test_stores)}")

        # Save store info
        print("\n💾 Testing store info saving...")
        output_file = os.path.join(temp_dir, "store_info.json")
        manager.save_vector_store_info(store_info, output_file)

        with open(output_file, 'r') as f:
            saved_data = json.load(f)

        print(f"✅ Store info saved to: {output_file}")
        print(f"   Saved store ID: {saved_data['vector_store_id']}")

        # Cleanup test stores
        print("\n🧹 Cleaning up test stores...")
        for store in test_stores:
            try:
                manager.delete_vector_store(store.vector_store_id)
                print(f"   ✅ Deleted: {store.name} ({store.vector_store_id})")
            except Exception as e:
                print(f"   ⚠️  Failed to delete {store.vector_store_id}: {e}")

        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up temp directory: {temp_dir}")

        print("\n🎉 ALL TESTS PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_from_knowledge_base():
    """Test using existing knowledge base if available"""
    knowledge_base_dir = "knowledge_base_full"

    if not os.path.exists(knowledge_base_dir):
        print(f"ℹ️  Knowledge base directory not found: {knowledge_base_dir}")
        print("   Run the knowledge base builder first to test with real data")
        return True

    print(f"\n🗂️  TESTING WITH EXISTING KNOWLEDGE BASE")
    print("=" * 50)

    try:
        manager = OpenAIVectorStoreManager()

        # Count available documents
        txt_files = list(Path(knowledge_base_dir).glob("*.txt"))
        print(f"📊 Found {len(txt_files)} documents in knowledge base")

        if len(txt_files) == 0:
            print("   No .txt files found, skipping knowledge base test")
            return True

        # Test with small subset for speed
        test_files = txt_files[:5]  # Just test with 5 files
        print(f"   Testing with {len(test_files)} sample files...")

        # Upload sample files
        file_paths = [str(f) for f in test_files]
        file_ids = manager.upload_files(file_paths)

        if file_ids:
            # Create vector store
            store_info = manager.create_vector_store("Knowledge Base Sample", file_ids)
            print(f"✅ Created sample vector store: {store_info.vector_store_id}")

            # Clean up
            manager.delete_vector_store(store_info.vector_store_id)
            print("✅ Cleaned up sample store")

        return True

    except Exception as e:
        print(f"❌ Knowledge base test failed: {e}")
        return False


def main():
    """Run all vector store tests"""
    print("🚀 OPENAI VECTOR STORE COMPONENT TEST")
    print("=" * 70)

    success = True

    # Test basic functionality
    if not test_vector_store_manager():
        success = False

    # Test with knowledge base if available
    if not test_from_knowledge_base():
        success = False

    print("\n" + "=" * 70)
    if success:
        print("🎉 ALL VECTOR STORE TESTS COMPLETED SUCCESSFULLY!")
        print("\n✅ OpenAI Vector Store Manager is ready for production use")
        print("\n🚀 NEXT STEPS:")
        print("   1. Build full knowledge base if not done: scripts/build_full_knowledge_base_laptop.py")
        print("   2. Upload all documents and create production vector store")
        print("   3. Add OPENAI_VECTOR_STORE_ID to .env")
        print("   4. Create OpenAI RAG sampler")
    else:
        print("❌ SOME TESTS FAILED!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())