# Knowledge Base and Cache Package for Hamza

This package contains the complete knowledge base, URL cache, and utilities for the 11K URL RAG project.

## Contents

### Core Data
- **`knowledge_base_full/`** - ~8,000 processed documents ready for OpenAI vector store
- **`cache/url_cache/`** - Cached content from 11,000+ URLs
- **`build-rag/urls.txt`** - Original list of 11,067 URLs

### Utilities & Scripts
- **`scripts/cache_and_kb_utilities.py`** - Main utility script for accessing cache and KB
- **`scripts/build_knowledge_base.py`** - Rebuild knowledge base from cache
- **`src/`** - Source code for all components (URL fetching, caching, processing)

### Docker Environment
- **`docker-compose.yml`** - Docker environment setup
- **`Dockerfile`** - Python environment with all dependencies
- **`requirements.txt`** - Python dependencies

### Configuration
- **`.env.example`** - Environment variables template
- **`CLAUDE.md`** - Instructions for Claude Code on how to work with this project

## Quick Start

### 1. Setup Environment
```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Start Docker environment
docker compose up -d
```

### 2. Access Data Using Utilities

```bash
# Get statistics about cache and knowledge base
python scripts/cache_and_kb_utilities.py stats

# Get cached content for a specific URL
python scripts/cache_and_kb_utilities.py cache "https://example.com/page"

# Get processed document for a URL
python scripts/cache_and_kb_utilities.py doc "https://example.com/page"

# Get both cache and processed doc
python scripts/cache_and_kb_utilities.py both "https://example.com/page"

# Search for URLs containing a term
python scripts/cache_and_kb_utilities.py search "wikipedia"
```

### 3. Rebuild Knowledge Base (if needed)
```bash
# Rebuild from cache (very fast since URLs are cached)
docker compose run --rm simple-evals python scripts/build_knowledge_base.py --concurrency 10

# Or rebuild just a subset for testing
docker compose run --rm simple-evals python scripts/build_knowledge_base.py --max-urls 100
```

## Key Features

### URL Cache
- **11,000+ cached URLs** from diverse sources
- **JSON format** with metadata (timestamp, content, success status)
- **MD5-based file naming** for consistent lookups
- **~3.3GB total size** of cached web content

### Knowledge Base
- **~8,000 processed documents** from successful URL extractions
- **Text format optimized for OpenAI vector stores**
- **Metadata headers** with source URL, title, word count
- **Ready for RAG applications**

### Processing Pipeline
- **Concurrent URL fetching** with configurable workers
- **Intelligent content extraction** (HTML, PDF, text)
- **Quality filtering** (minimum word count, content validation)
- **Memory-efficient streaming** for large-scale processing

## File Structure
```
simple-evals/
├── knowledge_base_full/          # Processed documents
│   ├── doc_*.txt                 # Individual documents
│   ├── build_metadata.json      # Build statistics
│   └── build_summary.json       # Summary report
├── cache/url_cache/              # URL cache
│   └── *.json                    # Cached URL content
├── build-rag/
│   └── urls.txt                  # Original URL list
├── scripts/
│   ├── cache_and_kb_utilities.py # Main utility script
│   └── build_knowledge_base.py   # Knowledge base builder
├── src/                          # Source code
│   ├── url_fetcher.py           # URL fetching logic
│   ├── url_cache.py             # Caching system
│   ├── content_extractor.py     # Content extraction
│   ├── content_processor.py     # Document processing
│   └── knowledge_base_builder.py # Main orchestrator
├── docker-compose.yml           # Docker environment
├── Dockerfile                   # Container definition
└── requirements.txt             # Python dependencies
```

## Advanced Usage

### Custom Processing
```python
# Load and process specific URLs
from src.knowledge_base_builder import KnowledgeBaseBuilder

builder = KnowledgeBaseBuilder(
    output_dir="custom_output",
    cache_dir="cache/url_cache",
    max_workers=5,
    use_cache=True
)

# Process custom URL list
with builder:
    result = builder.build_from_url_file("custom_urls.txt")
```

### Cache Management
```python
# Direct cache access
from src.url_cache import URLCache

cache = URLCache("cache/url_cache")
cached_content = cache.get("https://example.com")
```

## Statistics (Current Build)
- **Total URLs**: 11,067
- **Cached URLs**: 11,000+ (99.4% cache hit rate)
- **Processed Documents**: ~8,000
- **Total Word Count**: ~65M words
- **Average Document Size**: ~8,000 words
- **Success Rate**: ~72%

## Performance Notes
- **Cache-based rebuilds**: ~20-30 minutes for full 11K URLs
- **Memory usage**: <500MB (streaming processing)
- **Concurrent processing**: 10-15 workers optimal
- **Storage requirements**: ~4GB total (cache + KB)

## Support
For questions about the data or utilities, refer to the source code in `src/` or the utility script examples.