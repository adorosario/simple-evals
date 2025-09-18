#!/usr/bin/env python3
"""
Build a small knowledge base from the actual URLs for demonstration
This shows the complete pipeline working with real data
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.knowledge_base_builder import KnowledgeBaseBuilder


def main():
    """Build small knowledge base for demonstration"""
    print("Building Small Knowledge Base for OpenAI Vector Store")
    print("=" * 60)

    # Check if urls.txt exists
    urls_file = "build-rag/urls.txt"
    if not os.path.exists(urls_file):
        print(f"✗ URL file not found: {urls_file}")
        print("This script expects the build-rag/urls.txt file to exist")
        return 1

    print(f"✓ Found URL file: {urls_file}")

    # Create knowledge base builder
    builder = KnowledgeBaseBuilder(
        output_dir="knowledge_base_small",
        cache_dir="cache/url_cache",  # Persistent cache in repo
        min_document_words=50,        # Skip very short documents
        max_documents=20,             # Build small demo set
        use_cache=True,               # Enable caching for development
        force_refresh=False           # Use cache if available
    )

    try:
        print(f"Building knowledge base with first 20 URLs...")
        print(f"Using cache: {builder.use_cache}")
        print(f"Cache directory: {builder.cache_dir}")
        print(f"Output directory: {builder.output_dir}")

        # Build knowledge base
        with builder:
            result = builder.build_from_url_file(urls_file)

        # Display comprehensive results
        print("\n" + "="*60)
        print("BUILD RESULTS")
        print("="*60)

        print(f"📊 PROCESSING STATISTICS:")
        print(f"   Total URLs processed: {result.total_urls}")
        print(f"   Successful fetches: {result.successful_fetches} ({result.successful_fetches/result.total_urls:.1%})")
        print(f"   Successful extractions: {result.successful_extractions}")
        print(f"   Final documents created: {result.documents_created}")

        if result.build_stats:
            stats = result.build_stats
            print(f"\n📈 CONTENT STATISTICS:")
            print(f"   Total words: {stats.get('total_words', 0):,}")
            print(f"   Total characters: {stats.get('total_characters', 0):,}")
            print(f"   Average words per document: {stats.get('avg_words_per_doc', 0):.1f}")
            print(f"   Document size range: {stats.get('min_words', 0)}-{stats.get('max_words', 0)} words")
            print(f"   Overall success rate: {stats.get('success_rate', 0):.1%}")

        print(f"\n📁 OUTPUT:")
        print(f"   Documents saved to: {result.output_directory}")

        # List generated files
        files = builder.get_document_list_for_openai()
        print(f"   Generated {len(files)} text files ready for OpenAI:")

        total_size = 0
        for i, filepath in enumerate(files[:10]):  # Show first 10
            filename = os.path.basename(filepath)
            file_size = os.path.getsize(filepath)
            total_size += file_size
            print(f"     {i+1:2d}. {filename} ({file_size:,} bytes)")

        if len(files) > 10:
            for filepath in files[10:]:
                total_size += os.path.getsize(filepath)
            print(f"     ... and {len(files) - 10} more files")

        print(f"   Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")

        # Show errors if any
        if result.errors:
            print(f"\n⚠️  ERRORS ENCOUNTERED ({len(result.errors)}):")
            for i, error in enumerate(result.errors[:5]):  # Show first 5
                print(f"   {i+1}. {error}")
            if len(result.errors) > 5:
                print(f"   ... and {len(result.errors) - 5} more errors")

        # Show sample document content
        if files:
            print(f"\n📄 SAMPLE DOCUMENT CONTENT:")
            sample_file = files[0]
            print(f"From {os.path.basename(sample_file)}:")
            print("-" * 50)
            with open(sample_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Show first 400 characters
                if len(content) > 400:
                    print(content[:400] + "...")
                else:
                    print(content)
            print("-" * 50)

        # Final instructions
        print(f"\n🚀 NEXT STEPS FOR OPENAI VECTOR STORE:")
        print(f"   1. Upload the {len(files)} text files in '{result.output_directory}/' to OpenAI")
        print(f"   2. Create vector store using OpenAI Files API")
        print(f"   3. Get vector store ID and add to .env as OPENAI_VECTOR_STORE_ID")
        print(f"   4. Run three-way RAG benchmark with all models")

        print(f"\n✅ Knowledge base build completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Knowledge base build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())