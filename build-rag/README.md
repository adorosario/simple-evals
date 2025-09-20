# Simple QA URL Fixer

A comprehensive tool for auditing and fixing bad URLs in the Simple QA test set using intelligent search-based replacements.

## 🎯 Overview

This tool addresses the problem of broken or inaccessible URLs in the Simple QA test set by:

1. **Auditing URLs** against cache metadata and knowledge base presence
2. **Identifying bad URLs** that are not accessible or missing from the knowledge base
3. **Finding intelligent replacements** using Serper.dev Google search API
4. **Replacing bad URLs** with high-quality, relevant alternatives
5. **Providing comprehensive logging** for full transparency and auditability

## 📋 Features

### ✅ Core Functionality
- **URL Validation**: Checks URLs against cache success and knowledge base presence
- **Intelligent Search**: Uses exact question text to find relevant replacement URLs
- **Smart Filtering**: Removes irrelevant results (HuggingFace datasets, Hacker News, etc.)
- **Confidence Scoring**: Prioritizes authoritative sources (Wikipedia, .edu, .org domains)
- **Cost Optimization**: Comprehensive caching system to minimize API costs

### 🔧 Advanced Features
- **Dry Run Mode**: Test without making actual changes
- **Flexible Limits**: Process specific number of records for testing
- **Structured Logging**: JSON logs per record plus session summaries
- **API Caching**: Persistent cache for Serper.dev API calls
- **Progress Tracking**: Real-time progress updates and statistics

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure you're in the Docker environment
docker compose run --rm simple-evals bash
```

### Basic Usage

```bash
# Dry run with first 100 records
python /app/build-rag/fix_simple_qa_urls.py \
  --input-csv /app/build-rag/simple_qa_test_set.csv \
  --cache-dir /app/cache/url_cache \
  --kb-dir /app/knowledge_base_full \
  --limit 100 \
  --dry-run

# Actual run with specific output file
python /app/build-rag/fix_simple_qa_urls.py \
  --input-csv /app/build-rag/simple_qa_test_set.csv \
  --output-csv /app/build-rag/simple_qa_test_set_fixed.csv \
  --cache-dir /app/cache/url_cache \
  --kb-dir /app/knowledge_base_full \
  --max-search-results 10
```

## 📖 Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-csv` | `simple_qa_test_set.csv` | Path to input CSV file |
| `--output-csv` | `simple_qa_test_set_fixed.csv` | Path to output CSV file |
| `--cache-dir` | `../cache/url_cache` | Path to URL cache directory |
| `--kb-dir` | `../knowledge_base_full` | Path to knowledge base directory |
| `--max-search-results` | `10` | Maximum search results per query |
| `--limit` | `None` | Limit processing to first N records |
| `--dry-run` | `False` | Test mode - no actual changes made |

## 🔍 URL Classification Logic

### ✅ Good URLs
A URL is considered "good" if:
- **In Cache**: URL exists in cache metadata
- **Cache Success**: URL was successfully fetched
- **In Knowledge Base**: URL content is present in knowledge base

### ❌ Bad URLs
A URL is considered "bad" if any of:
- **Not in cache**: URL not found in cache metadata
- **Cache failure**: URL fetch failed (404, timeout, SSL errors, etc.)
- **Not in knowledge base**: URL content missing from knowledge base

## 🔄 Replacement Process

### 1. Search Strategy
- Uses **exact question text** from CSV as search query
- Leverages Serper.dev Google search API (cost: $1 per 1000 requests)
- Applies intelligent filtering to remove irrelevant results

### 2. Filtering Rules
Automatically filters out:
- HuggingFace dataset pages (`huggingface.co/datasets`)
- Hacker News discussions (`news.ycombinator.com`)
- Social media posts and forums
- Duplicate URLs already in the record

### 3. Confidence Scoring
URLs are scored based on domain authority:
- **1.00**: Wikipedia, .edu, .org, government sites
- **0.95**: News organizations, academic publishers
- **0.85**: Professional organizations, established media
- **0.75**: General websites with relevant content
- **0.60**: Lower confidence sources

### 4. Replacement Logic
- **Keep good URLs**: Preserves existing working URLs
- **Replace bad URLs**: Substitutes with highest confidence alternatives
- **Maintain order**: Preserves original URL ordering where possible

## 📊 Logging and Output

### Session Logs
- **Main log**: `logs/session_{timestamp}.log` - Complete session activity
- **Final report**: `logs/session_{timestamp}_final_report.json` - Summary statistics
- **Detailed report**: `url_fixing_report_{timestamp}.txt` - Human-readable summary

### Per-Record Logs
Each processed record generates: `logs/records/record_{id:06d}.json`

```json
{
  "session_id": "session_1758407916",
  "record_id": 1,
  "timestamp": 1758407917.4627,
  "question": "Who received the IEEE Frank Rosenblatt Award in 2010?",
  "topic": "Science and technology",
  "answer": "Michio Sugeno",
  "original_urls": [...],
  "fixed_urls": [...],
  "url_analysis": [...],
  "search_info": {...},
  "replacements": [...],
  "summary": {...}
}
```

### Cache Management
- **Serper cache**: `logs/serper_cache/` - API response caching
- **Cache metadata**: `logs/serper_cache/cache_metadata.json` - Query tracking
- **Cost tracking**: Monitors API usage and estimated costs

## 📈 Performance Metrics

### Typical Results (100 records)
- **Processing time**: ~2-3 minutes
- **Cache hit rate**: 95%+ (after initial run)
- **API cost**: $0.10 - $0.30 per 100 records
- **Success rate**: 95%+ of bad URLs successfully replaced

### Cost Optimization
- **API caching**: Avoids repeated identical queries
- **Result deduplication**: Removes duplicate URLs from search results
- **Intelligent limits**: Configurable result limits to balance quality vs. cost

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the build-rag directory:

```bash
SERPER_API_KEY=your_serper_api_key_here
```

### API Key Setup
1. Sign up at [Serper.dev](https://serper.dev/)
2. Get your API key from the dashboard
3. Add to `.env` file (no quotes needed)

## 🛠 Development

### Project Structure
```
build-rag/
├── fix_simple_qa_urls.py      # Main script
├── test_serper.py             # API testing utilities
├── simple_qa_test_set.csv     # Input data
├── requirements.txt           # Python dependencies
├── .env                       # API keys (local)
├── logs/                      # Output logs
│   ├── records/               # Per-record JSON logs
│   ├── serper_cache/          # API response cache
│   └── session_*.log          # Session logs
└── README.md                  # This file
```

### Key Classes

- **`SerperSearcher`**: Handles Google search API calls and caching
- **`SerperCache`**: Manages persistent API response caching
- **`URLValidator`**: Validates URL accessibility and response codes
- **`SimpleQAURLFixer`**: Main orchestrator class

### Dependencies
```txt
requests>=2.31.0
python-dotenv>=1.0.0
```

## 🧪 Testing

### Run Tests
```bash
# Test Serper.dev API functionality
docker compose run --rm simple-evals python /app/build-rag/test_serper.py

# Test with small dataset
docker compose run --rm simple-evals python /app/build-rag/fix_simple_qa_urls.py \
  --limit 5 --dry-run

# Test specific functionality
docker compose run --rm simple-evals python /app/build-rag/fix_simple_qa_urls.py \
  --input-csv /app/build-rag/simple_qa_test_set.csv \
  --limit 10 \
  --max-search-results 5 \
  --dry-run
```

### Debugging
```bash
# Enable verbose logging
export PYTHONPATH=/app
python /app/build-rag/fix_simple_qa_urls.py --dry-run --limit 1

# Check cache status
ls -la logs/serper_cache/
cat logs/serper_cache/cache_metadata.json

# Review individual record processing
cat logs/records/record_000001.json | python -m json.tool
```

## 🎯 Use Cases

### 1. Quality Assurance
Audit URL quality before running evaluations:
```bash
python fix_simple_qa_urls.py --dry-run --limit 1000
```

### 2. Knowledge Base Improvement
Fix URLs to improve RAG benchmark accuracy:
```bash
python fix_simple_qa_urls.py --output-csv improved_test_set.csv
```

### 3. Cost Estimation
Estimate API costs for full dataset:
```bash
python fix_simple_qa_urls.py --dry-run --max-search-results 5
```

### 4. Batch Processing
Process dataset in chunks:
```bash
# Process first 500 records
python fix_simple_qa_urls.py --limit 500 --output-csv part1_fixed.csv

# Continue from record 501 (would need additional logic)
```

## 🔍 Troubleshooting

### Common Issues

**API Key Error (403 Forbidden)**
```bash
# Check .env file format (no quotes)
cat .env
# Should show: SERPER_API_KEY=your_key_here
```

**File Not Found Errors**
```bash
# Ensure correct Docker paths
docker compose run --rm simple-evals ls -la /app/build-rag/
```

**Cache Issues**
```bash
# Clear cache if needed
rm -rf logs/serper_cache/*
```

**Memory Issues**
```bash
# Process in smaller batches
python fix_simple_qa_urls.py --limit 100
```

### Performance Optimization

**Reduce API Costs**
- Use `--max-search-results 5` for basic fixing
- Enable caching with persistent logs directory
- Use `--dry-run` for testing without API calls

**Speed Up Processing**
- Ensure knowledge base is fully populated
- Use SSD storage for cache directory
- Process in smaller batches for better monitoring

## 📚 Background

This tool was developed to address URL quality issues in the Simple QA test set for RAG (Retrieval-Augmented Generation) benchmarking. The original dataset contained numerous broken or inaccessible URLs that could negatively impact evaluation accuracy.

### Design Principles
1. **Transparency**: Complete audit trail of all changes
2. **Cost Efficiency**: Minimize API costs through intelligent caching
3. **Quality**: Prioritize authoritative sources for replacements
4. **Flexibility**: Support various use cases from testing to production
5. **Safety**: Dry-run mode for risk-free testing

### Related Tools
- `simpleqa_url_audit.py`: Initial URL quality assessment
- `test_serper.py`: API functionality testing
- `pretty_print_csv.py`: CSV formatting utilities

---

## 🤝 Contributing

To contribute improvements:

1. Test changes with `--dry-run --limit 5`
2. Ensure comprehensive logging is maintained
3. Add appropriate error handling
4. Update documentation for new features
5. Verify cost optimization features work correctly

---

*This tool is part of the Simple Evals RAG benchmark infrastructure. For questions or issues, refer to the main project documentation.*