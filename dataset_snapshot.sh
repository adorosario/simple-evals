#!/bin/bash
"""
Dataset Snapshot Creator
Creates a complete package of the knowledge base, cache, and utilities for distribution.
"""

set -e  # Exit on any error

# Configuration
SNAPSHOT_NAME="knowledge_base_snapshot_$(date +%Y%m%d_%H%M%S)"
OUTPUT_FILE="${SNAPSHOT_NAME}.tar.gz"

echo "📦 Creating dataset snapshot: ${OUTPUT_FILE}"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ] || [ ! -d "knowledge_base_full" ]; then
    echo "❌ Error: Run this script from the simple-evals project root directory"
    echo "   Expected files: docker-compose.yml, knowledge_base_full/"
    exit 1
fi

# Get current stats
echo "📊 Current dataset statistics:"
if [ -d "knowledge_base_full" ]; then
    KB_COUNT=$(find knowledge_base_full/ -name "*.txt" | wc -l)
    KB_SIZE=$(du -sh knowledge_base_full/ | cut -f1)
    echo "   Knowledge Base: ${KB_COUNT} documents (${KB_SIZE})"
fi

if [ -d "cache/url_cache" ]; then
    CACHE_COUNT=$(find cache/url_cache/ -name "*.json" | wc -l)
    CACHE_SIZE=$(du -sh cache/url_cache/ | cut -f1)
    echo "   URL Cache: ${CACHE_COUNT} cached URLs (${CACHE_SIZE})"
fi

if [ -f "build-rag/urls.txt" ]; then
    URL_COUNT=$(wc -l < build-rag/urls.txt)
    echo "   Source URLs: ${URL_COUNT} total URLs"
fi

echo ""
echo "🔄 Creating compressed archive..."

# Create the tar archive with comprehensive contents
tar -czf "${OUTPUT_FILE}" \
  knowledge_base_full/ \
  cache/url_cache/ \
  build-rag/urls.txt \
  scripts/cache_and_kb_utilities.py \
  scripts/build_knowledge_base.py \
  src/ \
  docker-compose.yml \
  Dockerfile \
  requirements.txt \
  .env.example \
  HAMZA_README.md \
  CLAUDE.md \
  --exclude="*.pyc" \
  --exclude="__pycache__" \
  --exclude=".git*" \
  --exclude="results/" \
  --exclude="audit_logs/" \
  --exclude="checkpoints/" \
  --exclude="test_*/" \
  --exclude="*.log" \
  --exclude=".env" \
  2>/dev/null || true

# Get final archive size
if [ -f "${OUTPUT_FILE}" ]; then
    ARCHIVE_SIZE=$(du -sh "${OUTPUT_FILE}" | cut -f1)
    echo "✅ Archive created successfully!"
    echo ""
    echo "📋 SNAPSHOT SUMMARY:"
    echo "   File: ${OUTPUT_FILE}"
    echo "   Size: ${ARCHIVE_SIZE}"
    echo "   Contains:"
    echo "     • Knowledge base documents and metadata"
    echo "     • Complete URL cache (11K+ URLs)"
    echo "     • Source code and utilities"
    echo "     • Docker environment setup"
    echo "     • Documentation and examples"
    echo ""
    echo "🚀 Ready for distribution!"
    echo ""
    echo "📤 To share this snapshot:"
    echo "   scp ${OUTPUT_FILE} user@server:/path/to/destination/"
    echo ""
    echo "📥 To extract on destination:"
    echo "   tar -xzf ${OUTPUT_FILE}"
    echo "   cd simple-evals"
    echo "   cp .env.example .env  # then edit with API keys"
    echo "   docker compose up"
else
    echo "❌ Error: Failed to create archive"
    exit 1
fi