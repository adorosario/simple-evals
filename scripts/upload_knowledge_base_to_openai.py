#!/usr/bin/env python3
"""
Upload knowledge base documents to OpenAI and create vector store
This script takes the output from build_full_knowledge_base.py and creates a production vector store
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.openai_vector_store import OpenAIVectorStoreManager


def main():
    """Upload knowledge base to OpenAI vector store"""
    print("🚀 UPLOADING KNOWLEDGE BASE TO OPENAI VECTOR STORE")
    print("=" * 70)

    # Check for knowledge base directory
    knowledge_base_dir = "knowledge_base_full"
    if not os.path.exists(knowledge_base_dir):
        print(f"❌ Knowledge base directory not found: {knowledge_base_dir}")
        print("   Please run scripts/build_full_knowledge_base_laptop.py first")
        return 1

    # Check for build summary
    summary_file = os.path.join(knowledge_base_dir, "build_summary.json")
    build_summary = None
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            build_summary = json.load(f)
        print(f"📊 Found build summary from {time.ctime(build_summary['timestamp'])}")
        print(f"   Documents created: {build_summary['documents_created']:,}")
        print(f"   Success rate: {build_summary['success_rate']:.1%}")
        print(f"   Total size: {build_summary['total_size_mb']:.1f} MB")

    # Find all text documents
    txt_files = list(Path(knowledge_base_dir).glob("*.txt"))
    if len(txt_files) == 0:
        print(f"❌ No .txt files found in {knowledge_base_dir}")
        return 1

    print(f"\n📁 Found {len(txt_files):,} documents ready for upload")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in txt_files)
    print(f"   Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")

    # Estimate costs (rough estimate)
    # OpenAI pricing: ~$0.10 per 1M tokens for storage
    # Assume ~750 chars per token (rough estimate)
    estimated_tokens = total_size / 750
    estimated_cost = estimated_tokens * 0.10 / 1_000_000
    print(f"   Estimated tokens: {estimated_tokens:,.0f}")
    print(f"   Estimated storage cost: ${estimated_cost:.2f}/month")

    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n❌ OPENAI_API_KEY not found in environment")
        print("   Please set your OpenAI API key in .env")
        return 1

    print(f"\n✅ Found OpenAI API key: {api_key[:8]}...")

    # Confirm before proceeding
    print(f"\n⚠️  UPLOAD CONFIRMATION:")
    print(f"   Files to upload: {len(txt_files):,}")
    print(f"   Total size: {total_size/1024/1024:.1f} MB")
    print(f"   Estimated cost: ${estimated_cost:.2f}/month")
    print(f"   This operation may take 20-30 minutes")

    # For script usage, we'll proceed automatically
    # In interactive mode, you might want to add input() confirmation
    print(f"\n🏗️  STARTING UPLOAD TO OPENAI...")

    start_time = time.time()

    try:
        # Initialize manager
        manager = OpenAIVectorStoreManager(api_key=api_key)

        # Create vector store name with timestamp
        store_name = f"Simple-Evals Knowledge Base {time.strftime('%Y-%m-%d %H:%M')}"
        print(f"   Vector store name: {store_name}")

        # Upload and create vector store
        print(f"\n📤 Uploading documents and creating vector store...")
        store_info = manager.create_vector_store_from_directory(
            knowledge_base_dir,
            store_name,
            file_pattern="*.txt"
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n🎉 UPLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        print(f"📊 UPLOAD STATISTICS:")
        print(f"   Vector Store ID: {store_info.vector_store_id}")
        print(f"   Vector Store Name: {store_info.name}")
        print(f"   Files uploaded: {store_info.file_count:,}")
        print(f"   Upload duration: {duration:.0f}s ({duration/60:.1f} minutes)")
        print(f"   Processing status: {store_info.status}")
        print(f"   Created at: {time.ctime(store_info.created_at)}")

        # Save vector store info
        output_file = os.path.join(knowledge_base_dir, "vector_store_info.json")
        manager.save_vector_store_info(store_info, output_file)
        print(f"   Store info saved: {output_file}")

        # Update .env file instructions
        env_file = ".env"
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Add to {env_file}:")
        print(f"      OPENAI_VECTOR_STORE_ID={store_info.vector_store_id}")
        print(f"   2. Create OpenAI RAG sampler")
        print(f"   3. Run three-way benchmark!")

        # Auto-update .env if possible
        if os.path.exists(env_file):
            try:
                # Read current .env
                with open(env_file, 'r') as f:
                    env_content = f.read()

                # Check if OPENAI_VECTOR_STORE_ID already exists
                if 'OPENAI_VECTOR_STORE_ID=' in env_content:
                    # Update existing line
                    lines = env_content.split('\n')
                    updated_lines = []
                    for line in lines:
                        if line.startswith('OPENAI_VECTOR_STORE_ID='):
                            updated_lines.append(f'OPENAI_VECTOR_STORE_ID={store_info.vector_store_id}')
                            print(f"   ✅ Updated existing OPENAI_VECTOR_STORE_ID in {env_file}")
                        else:
                            updated_lines.append(line)

                    with open(env_file, 'w') as f:
                        f.write('\n'.join(updated_lines))
                else:
                    # Append new line
                    with open(env_file, 'a') as f:
                        if not env_content.endswith('\n'):
                            f.write('\n')
                        f.write(f'OPENAI_VECTOR_STORE_ID={store_info.vector_store_id}\n')
                    print(f"   ✅ Added OPENAI_VECTOR_STORE_ID to {env_file}")

            except Exception as e:
                print(f"   ⚠️  Could not auto-update {env_file}: {e}")
                print(f"   Please manually add: OPENAI_VECTOR_STORE_ID={store_info.vector_store_id}")

        print(f"\n✅ VECTOR STORE READY FOR RAG QUERIES!")
        print(f"   Store ID: {store_info.vector_store_id}")
        print(f"   Ready to implement OpenAI RAG sampler")

        return 0

    except KeyboardInterrupt:
        print(f"\n⏹️  Upload interrupted by user")
        end_time = time.time()
        partial_duration = end_time - start_time
        print(f"   Partial duration: {partial_duration:.0f}s ({partial_duration/60:.1f} minutes)")
        print(f"   Note: Uploaded files may still be processing in OpenAI")
        return 130

    except Exception as e:
        print(f"\n💥 Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())