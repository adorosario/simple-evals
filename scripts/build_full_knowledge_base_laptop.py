#!/usr/bin/env python3
"""
Build the complete knowledge base from all 11,000+ URLs - LAPTOP VERSION
Reduced concurrency to avoid overwhelming the system
"""

import sys
import os
import time
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.knowledge_base_builder import KnowledgeBaseBuilder


def main():
    """Build complete knowledge base for OpenAI vector store - laptop optimized"""
    print("🚀 BUILDING COMPLETE KNOWLEDGE BASE (LAPTOP VERSION)")
    print("=" * 70)

    start_time = time.time()

    # Laptop-optimized configuration
    builder = KnowledgeBaseBuilder(
        output_dir="knowledge_base_full",
        cache_dir="cache/url_cache",  # Persistent cache
        min_document_words=30,        # Reasonable threshold
        max_documents=None,           # Process ALL URLs
        max_workers=10,               # REDUCED: 10 workers for laptop
        timeout=15,                   # INCREASED: 15s timeout (more patient)
        max_retries=2,                # 2 retries max
        use_cache=True,               # Enable caching
        force_refresh=False           # Use cache if available
    )

    print(f"⚙️  LAPTOP-OPTIMIZED CONFIGURATION:")
    print(f"   Parallel workers: {builder.max_workers} (reduced for laptop)")
    print(f"   Timeout per URL: {builder.timeout}s (increased for stability)")
    print(f"   Max retries: {builder.max_retries}")
    print(f"   Min document words: {builder.min_document_words}")
    print(f"   Output directory: {builder.output_dir}")
    print(f"   Cache directory: {builder.cache_dir}")
    print(f"   Using cache: {builder.use_cache}")

    try:
        print(f"\n🏗️  STARTING LAPTOP-OPTIMIZED BUILD...")
        print(f"   Expected time: ~3-4 hours (reduced concurrency)")
        print(f"   Expected documents: ~8,000-9,000")
        print(f"   Press Ctrl+C to safely interrupt and resume later")

        with builder:
            result = builder.build_from_url_file('build-rag/urls.txt')

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n🎉 BUILD COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        print(f"📊 FINAL STATISTICS:")
        print(f"   Total URLs processed: {result.total_urls:,}")
        print(f"   Documents created: {result.documents_created:,}")
        print(f"   Success rate: {result.documents_created/result.total_urls:.1%}")
        print(f"   Total duration: {duration:.0f}s ({duration/3600:.2f} hours)")
        print(f"   Processing rate: {result.total_urls/duration:.1f} URLs/second")

        if result.build_stats:
            stats = result.build_stats
            print(f"\n📈 CONTENT STATISTICS:")
            print(f"   Total words: {stats.get('total_words', 0):,}")
            print(f"   Total characters: {stats.get('total_characters', 0):,}")
            print(f"   Average words per document: {stats.get('avg_words_per_doc', 0):.0f}")
            print(f"   Document size range: {stats.get('min_words', 0)}-{stats.get('max_words', 0)} words")

        print(f"\n📁 OUTPUT:")
        files = builder.get_document_list_for_openai()
        total_size = sum(os.path.getsize(f) for f in files)
        print(f"   Documents ready for OpenAI: {len(files):,}")
        print(f"   Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
        print(f"   Location: {result.output_directory}")

        if result.errors:
            error_types = {}
            for error in result.errors:
                if 'DNS resolution' in error or 'Name or service not known' in error:
                    error_types['DNS'] = error_types.get('DNS', 0) + 1
                elif 'Connection refused' in error or 'Failed to establish' in error:
                    error_types['Connection'] = error_types.get('Connection', 0) + 1
                elif 'timeout' in error.lower():
                    error_types['Timeout'] = error_types.get('Timeout', 0) + 1
                elif 'HTTP error' in error:
                    error_types['HTTP'] = error_types.get('HTTP', 0) + 1
                else:
                    error_types['Other'] = error_types.get('Other', 0) + 1

            print(f"\n⚠️  ERROR BREAKDOWN ({len(result.errors)} total):")
            for error_type, count in error_types.items():
                print(f"   {error_type}: {count}")

        # Save summary for reference
        summary = {
            'timestamp': time.time(),
            'duration_hours': duration/3600,
            'total_urls': result.total_urls,
            'documents_created': result.documents_created,
            'success_rate': result.documents_created/result.total_urls,
            'build_stats': result.build_stats,
            'output_directory': result.output_directory,
            'files_ready_for_openai': len(files),
            'total_size_mb': total_size/1024/1024,
            'config_used': 'laptop_optimized'
        }

        summary_file = os.path.join(result.output_directory, 'build_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Upload {len(files)} text files to OpenAI using Files API")
        print(f"   2. Create vector store with uploaded files")
        print(f"   3. Get vector store ID and add to .env")
        print(f"   4. Run three-way RAG benchmark!")
        print(f"\n   Files are ready in: {result.output_directory}/")
        print(f"   Build summary saved to: {summary_file}")

        return 0

    except KeyboardInterrupt:
        print(f"\n⏹️  Build interrupted by user")
        end_time = time.time()
        partial_duration = end_time - start_time
        print(f"   Partial duration: {partial_duration:.0f}s ({partial_duration/3600:.2f} hours)")
        print(f"   Cache preserved for resuming later")
        print(f"   Can restart with same command - will use cached results")
        return 130

    except Exception as e:
        print(f"\n💥 Build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())