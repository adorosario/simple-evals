# SimpleQA Dataset KB Coverage Audit Plan

## Overview
Create a script `scripts/simpleqa_kb_coverage_audit.py` to audit the ~4,333 SimpleQA questions and identify rows where all URLs are missing from the knowledge base.

## Implementation Strategy

### 1. Data Sources Analysis
- **Input**: `build-rag/simple_qa_test_set.csv` (4,332 questions + header)
- **Knowledge Base**: `knowledge_base_full/` (8,009 documents from 11,067 URLs, 72.4% success rate)
- **URL Mapping**: `cache/url_cache/cache_metadata.json` (contains URL→hash mappings with success/failure status)

### 2. Core Logic
- Parse CSV and extract URLs from metadata column (JSON format)
- For each URL, check if it exists in knowledge base by:
  - Looking up URL hash in cache metadata
  - Verifying cache success status
  - Cross-referencing with actual document files in KB
- Add `rag_should_abstain` column (1 if ALL URLs missing, 0 otherwise)

### 3. Key Features
- **Input/Output Parameters**:
  - `--input` (default: `./build-rag/simple_qa_test_set.csv`)
  - `--output` (default: `build-rag/simple_qa_test_set_enhanced.csv`)
  - `--stats` flag for statistical analysis
- **URL Validation**: Use existing hash-based approach from `simpleqa_url_audit.py`
- **Robust Parsing**: Handle malformed JSON in metadata fields
- **Performance**: Process 4K+ rows efficiently with minimal memory usage

### 4. Statistical Analysis (when `--stats` enabled)
- Distribution of `rag_should_abstain` column (0s vs 1s)
- URL coverage statistics:
  - Total URLs processed
  - URLs found in KB vs missing
  - Success rate per question
  - Average URLs per question
- Failure pattern analysis by topic/answer_type
- Export summary statistics to JSON file

### 5. Output Format
Enhanced CSV with original columns plus:
- `rag_should_abstain`: Binary flag (0/1)
- `urls_in_kb`: Count of URLs found in KB
- `total_urls`: Total URLs for the question
- `kb_coverage_ratio`: Percentage of URLs found

### 6. Error Handling
- Graceful handling of malformed JSON metadata
- Skip invalid rows with warnings
- Comprehensive logging of processing issues
- Validation of input/output file paths

This approach leverages existing infrastructure (cache metadata, KB structure) while providing the specific audit functionality requested.

## Usage Examples

```bash
# Basic audit with default files
docker compose run --rm simple-evals python scripts/simpleqa_kb_coverage_audit.py

# Custom input/output with statistics
docker compose run --rm simple-evals python scripts/simpleqa_kb_coverage_audit.py \
  --input ./build-rag/simple_qa_test_set.csv \
  --output ./build-rag/simple_qa_test_set_enhanced.csv \
  --stats

# Statistics only mode
docker compose run --rm simple-evals python scripts/simpleqa_kb_coverage_audit.py --stats
```

## Expected Outcomes

Based on the knowledge base containing 8,009 documents from 11,067 URLs (72.4% success rate), we expect:
- Approximately 27.6% of URLs to be missing from the KB
- Questions with multiple URLs will have varying coverage ratios
- Some questions may have all URLs missing (requiring RAG abstention)
- Statistical analysis will reveal patterns in URL availability by topic/type

## Implementation Status

- [x] Plan documented
- [ ] Script implementation
- [ ] Testing with actual data
- [ ] Statistical analysis validation