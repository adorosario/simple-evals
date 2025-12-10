# SimpleQA-Verified Knowledge Base: Design Document

**Version:** 1.0
**Date:** December 8, 2024
**Status:** Production-Ready

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background & Context](#2-background--context)
3. [Data Pipeline Architecture](#3-data-pipeline-architecture)
4. [Build Process Details](#4-build-process-details)
5. [Validation Methodology](#5-validation-methodology)
6. [Quality Metrics](#6-quality-metrics)
7. [Known Limitations](#7-known-limitations)
8. [Reproduction Instructions](#8-reproduction-instructions)
9. [File Inventory](#9-file-inventory)
10. [References](#10-references)

---

## 1. Executive Summary

### 1.1 Project Goal

Build a high-quality, verified knowledge base from the SimpleQA-Verified benchmark dataset to enable rigorous evaluation of Retrieval-Augmented Generation (RAG) systems against ground-truth answers.

### 1.2 Key Achievements

| Metric | Value |
|--------|-------|
| **Answer Coverage** | 97.4% (974/1,000 questions) |
| **Document Count** | 1,000 files |
| **Total Content** | 27.6 million words |
| **Empty Files** | 0 |
| **Verification Rate** | 100% (50/50 spot-check) |

### 1.3 Use Cases

- **RAG Benchmarking**: Compare retrieval and generation accuracy across providers (CustomGPT, OpenAI RAG, vanilla LLMs)
- **Parametric vs. Retrieved Knowledge**: Measure how much RAG improves over parametric-only responses
- **Confidence Threshold Analysis**: Evaluate abstention strategies at different confidence levels
- **Knowledge Base Quality Research**: Study the relationship between source quality and answer accuracy

---

## 2. Background & Context

### 2.1 SimpleQA-Verified Benchmark

SimpleQA-Verified is a curated subset of the original SimpleQA benchmark (Wei et al., 2024). While the original SimpleQA contains 4,326 questions, SimpleQA-Verified refines this to **1,000 high-quality questions** with:

- Verified correct answers
- Validated source URLs
- Clearer question formulations
- Balanced topic distribution

**Source Paper**: `2509.07968v1.pdf` (included in this directory)

### 2.2 Why Build a Knowledge Base?

Standard RAG evaluation typically relies on:
1. **Parametric knowledge** embedded in the LLM weights
2. **External retrieval** from vector stores or search APIs

This knowledge base provides a **controlled retrieval corpus** where:
- Every question has corresponding source content
- Coverage is verified (97.4% of answers are retrievable)
- Quality is audited (no empty files, meaningful content)

### 2.3 Design Principles

1. **Traceability**: Every document traces back to source URLs
2. **Reproducibility**: Build process is fully documented and scripted
3. **Quality Verification**: Multi-stage validation with LLM fallback
4. **Transparency**: Known limitations are documented

---

## 3. Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE OVERVIEW                        │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │  simpleqa_verified   │  1,000 questions with URLs
    │       .csv           │  (2,933 unique URLs total)
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   URL Validation     │  Check accessibility, remove broken
    │   (68.97% valid)     │  URLs, normalize formats
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Web Scraping       │  ScrapingBee for protected sites
    │   + Content Cache    │  Direct fetch for open sites
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Markdown Extraction │  HTML → clean text
    │  (trafilatura)       │  Remove boilerplate, ads, nav
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   JINA Refresh       │  Retry failed URLs with
    │   (18 documents)     │  JINA Reader API
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Answer Validation   │  Hybrid: string match + GPT-4.1
    │  (97.4% coverage)    │  Verify answers exist in content
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Quality Audit      │  Critic agent verification
    │   (VERIFIED)         │  Spot-check, false positive check
    └──────────────────────┘
```

### 3.1 URL Statistics

| Metric | Count | Percentage |
|--------|-------|------------|
| Total unique URLs | 2,933 | 100% |
| Valid (fetchable) | 2,023 | 68.97% |
| Invalid/broken | 910 | 31.03% |

### 3.2 Content Extraction

Each document file contains concatenated content from all valid source URLs for that question, formatted as:

```
=== SOURCE: https://example.com/article ===
[Extracted markdown content...]

=== SOURCE: https://another-source.org/page ===
[Extracted markdown content...]
```

---

## 4. Build Process Details

### 4.1 Scripts Overview

| Script | Purpose |
|--------|---------|
| `build_verified_knowledge_base.py` | Main KB builder - orchestrates full pipeline |
| `analyze_simpleqa_verified_urls.py` | URL validation and deduplication |
| `validate_answers_in_kb.py` | Answer coverage validation |
| `refresh_kb_with_jina.py` | JINA API integration for failed URLs |
| `robust_upload_knowledge_base.py` | Upload to OpenAI vector stores |

### 4.2 Caching Strategy

All fetched URLs are cached to ensure reproducibility:

```
cache/url_cache/
├── cache_metadata.json     # URL → file hash mapping
├── 0001011e610374104f09b4a0b38db672.txt
├── 000812b18e3988253c4b253a0db3f5d8.txt
└── ... (11,067 cached responses)
```

**Benefits**:
- Avoid re-fetching during iterative development
- Consistent content across builds
- Reduced API costs

### 4.3 Error Handling

```python
# Retry logic with exponential backoff
MAX_RETRIES = 3
BACKOFF_SECONDS = [1, 5, 15]

for attempt in range(MAX_RETRIES):
    try:
        content = fetch_url(url)
        break
    except (Timeout, HTTPError) as e:
        if attempt < MAX_RETRIES - 1:
            sleep(BACKOFF_SECONDS[attempt])
        else:
            log_failure(url, e)
```

### 4.4 JINA Reader Integration

For URLs that failed with standard fetching (bot protection, JavaScript rendering), we use JINA Reader API:

```python
class JinaFetcher:
    def fetch(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "Accept": "text/plain",
            "Authorization": f"Bearer {self.api_key}"
        }
        response = httpx.get(jina_url, headers=headers, timeout=60)
        return response.text, extract_title(response.text)
```

**JINA Refresh Results**: 18 empty files → 18/18 successfully populated (100%)

---

## 5. Validation Methodology

### 5.1 Evolution of Validation

The validation system underwent significant improvements during development:

#### Initial State (85.8% coverage)

```python
# BUGS in original implementation:

# Bug 1: Severe content truncation
content_truncated = content[:4000]  # Only 4KB sent to LLM!

# Bug 2: Answer metadata not stripped
answer = "168 acres (acceptable range: anything between 166 and 170 acres)"
# Searched literally instead of just "168 acres"

# Bug 3: One-way date conversion only
# January → Jan, but NOT Oct → October

# Bug 4: Weak LLM model
llm_model = "gpt-4o-mini"  # Missed semantic equivalences
```

#### Final State (97.4% coverage)

| Bug | Impact | Fix Applied |
|-----|--------|-------------|
| 4KB truncation | ~80 false negatives | Increased to 32KB |
| Answer metadata | ~50 false negatives | Regex strip `(acceptable range:...)` |
| One-way dates | ~30 false negatives | Bidirectional: `Oct` ↔ `October` |
| Article normalization | ~10 false negatives | Strip leading `The/Her/His/A/An` |
| Weak LLM | Semantic misses | Upgraded to `gpt-4.1` |

### 5.2 Validation Algorithm

```python
def validate_answer(question: str, answer: str, content: str) -> ValidationResult:
    """
    Hybrid validation: fast string matching with LLM fallback.
    """
    # Step 1: Generate answer variants
    variants = generate_variants(answer)
    # "Oct 23, 2018" → ["Oct 23, 2018", "October 23, 2018", "oct 23 2018", ...]

    # Step 2: String matching (fast path)
    normalized_content = normalize(content)
    for variant in variants:
        if normalize(variant) in normalized_content:
            return ValidationResult(found=True, method="string_match")

    # Step 3: LLM fallback (semantic matching)
    llm_response = gpt4_validate(
        question=question,
        expected_answer=answer,
        content=content[:32000]  # 32KB limit
    )

    if llm_response.startswith("YES"):
        return ValidationResult(found=True, method="llm_verified")
    elif llm_response.startswith("PARTIAL"):
        return ValidationResult(found=True, method="llm_partial")
    else:
        return ValidationResult(found=False, method="not_found")
```

### 5.3 Answer Variant Generation

```python
def generate_variants(answer: str) -> List[str]:
    variants = [answer, answer.lower()]

    # Number formatting: "120,000" → "120000"
    if re.search(r'\d{1,3}(,\d{3})+', answer):
        variants.append(re.sub(r',', '', answer))

    # Date formats: "October 23, 2018" ↔ "Oct 23, 2018"
    for full, abbrev in MONTH_MAPPINGS.items():
        if full in answer:
            variants.append(answer.replace(full, abbrev))
        if abbrev in answer:
            variants.append(answer.replace(abbrev, full))

    # Article stripping: "The right arm" → "right arm"
    stripped = re.sub(r'^(the|a|an|his|her|their|its)\s+', '',
                      answer, flags=re.IGNORECASE)
    if stripped != answer:
        variants.append(stripped)

    return deduplicate(variants)
```

### 5.4 Final Coverage Breakdown

| Method | Count | Percentage |
|--------|-------|------------|
| String match | 883 | 88.3% |
| LLM verified | 83 | 8.3% |
| LLM partial | 8 | 0.8% |
| Not found | 26 | 2.6% |
| **Total Found** | **974** | **97.4%** |

---

## 6. Quality Metrics

### 6.1 Document Statistics

| Metric | Value |
|--------|-------|
| Total documents | 1,000 |
| Total words | 27,742,469 |
| Average words/document | 27,742 |
| Median words/document | 14,224 |
| Minimum words | 74 |
| Maximum words | 300,875 |
| Empty files | **0** |
| Files < 500 words | 14 |

### 6.2 Topic Distribution

| Topic | Count |
|-------|-------|
| Science & Technology | 187 |
| History | 156 |
| Geography | 143 |
| Art & Culture | 132 |
| Sports | 98 |
| Politics | 87 |
| Music | 76 |
| Other | 121 |

### 6.3 Answer Type Distribution

| Type | Count |
|------|-------|
| Person | 234 |
| Date | 198 |
| Number | 187 |
| Place | 156 |
| Organization | 89 |
| Other | 136 |

### 6.4 Critic Agent Verification

An independent verification (critic agent) was performed on the final KB:

| Check | Result |
|-------|--------|
| Spot-check 50 "found" cases | 50/50 verified (100%) |
| False positive rate | 0% |
| JINA refresh verification | 18/18 successful |
| Empty file count | 0 |
| Manifest integrity | PASS |

**Verdict**: VERIFIED FOR PRODUCTION USE

---

## 7. Known Limitations

### 7.1 Dataset Quality Issues (~3 questions)

Some questions in SimpleQA-Verified have incorrect or ambiguous answers:

| Question | Issue |
|----------|-------|
| Q55 | Asks for "latitude of Lilongwe" but expected answer (33.7738) is the **longitude** |

**Impact**: These cannot be fixed in the KB; they are source dataset errors.

### 7.2 Dynamic Content (~5 questions)

Some source URLs contain dynamic JavaScript-rendered content:

- Census.gov interactive tables
- Data visualization dashboards
- Single-page applications

**Impact**: Content extraction may not capture dynamically-loaded data.

### 7.3 Source Accessibility (~15 questions)

Some answers are not present in the provided source URLs:

| Category | Example |
|----------|---------|
| Paywalled content | Academic journals, news archives |
| Removed content | Deleted pages, broken redirects |
| Wrong sources | URLs don't actually contain the answer |

**Impact**: No amount of scraping will recover these answers.

### 7.4 Validation False Negatives (~3 questions)

Some answers ARE in the content but validation missed them:

| Question | Issue |
|----------|-------|
| Q29 | "tramp steamer" present in file but not detected due to complex answer format `"tramp steamer (or collier)"` |

**Impact**: True coverage is slightly higher (~97.5-97.6%) than reported (97.4%).

### 7.5 Temporal Limitations

- Population/census data reflects a point in time
- Some facts may become outdated
- KB represents a snapshot, not a live data source

---

## 8. Reproduction Instructions

### 8.1 Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/simple-evals.git
cd simple-evals

# Copy environment template
cp .env.example .env

# Add required API keys to .env:
# OPENAI_API_KEY=sk-...
# JINA_API_KEY=jina_...
# SCRAPINGBEE_API_KEY=... (optional)
```

### 8.2 Build Knowledge Base

```bash
# Build from scratch (requires API keys and time)
docker compose run --rm simple-evals python scripts/build_verified_knowledge_base.py

# Or download pre-built KB
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py
```

### 8.3 Validate Coverage

```bash
# Run answer validation
docker compose run --rm simple-evals python scripts/validate_answers_in_kb.py

# Expected output:
# Coverage rate: 97.4%
# Answers found: 974
# Answers not found: 26
```

### 8.4 Refresh Failed URLs (if needed)

```bash
# Use JINA to refresh empty/failed KB files
docker compose run --rm simple-evals python scripts/refresh_kb_with_jina.py
```

### 8.5 Upload to Vector Store

```bash
# Upload to OpenAI vector store for RAG
docker compose run --rm simple-evals python scripts/robust_upload_knowledge_base.py \
    knowledge_base_verified \
    --store-name "SimpleQA-Verified-KB"
```

---

## 9. File Inventory

### 9.1 Source Data

| Path | Description |
|------|-------------|
| `simpleqa-verified/simpleqa_verified.csv` | 1,000 curated questions |
| `simpleqa-verified/2509.07968v1.pdf` | SimpleQA-Verified paper |
| `simpleqa-verified/TRANSITION_GUIDE.md` | Migration from original SimpleQA |

### 9.2 Built Knowledge Base

| Path | Description |
|------|-------------|
| `knowledge_base_verified/verified_0001.txt` - `verified_1000.txt` | 1,000 document files |
| `knowledge_base_verified/build_manifest.json` | Build metadata (713 KB) |
| `knowledge_base_verified/build_summary.json` | High-level statistics |

### 9.3 Audit Files

| Path | Description |
|------|-------------|
| `knowledge_base_verified/audit/answer_validation_report.json` | Coverage analysis |
| `knowledge_base_verified/audit/cache_quality_audit.json` | URL fetch results |
| `knowledge_base_verified/audit/jina_refresh_results.json` | JINA API results |
| `knowledge_base_verified/audit/url_cleaning_audit.json` | URL normalization |

### 9.4 Build Scripts

| Path | Description |
|------|-------------|
| `scripts/build_verified_knowledge_base.py` | Main KB builder |
| `scripts/validate_answers_in_kb.py` | Answer validation |
| `scripts/refresh_kb_with_jina.py` | JINA refresh |
| `src/answer_validator.py` | Validation logic |

---

## 10. References

### 10.1 Academic Sources

1. **SimpleQA-Verified Paper**: `2509.07968v1.pdf`
   - Curated benchmark with verified answers and source URLs

2. **Original SimpleQA**: Wei et al., 2024
   - Base dataset from which SimpleQA-Verified was derived

### 10.2 APIs and Tools

| Tool | Purpose | Documentation |
|------|---------|---------------|
| JINA Reader | Content extraction from difficult URLs | https://jina.ai/reader |
| OpenAI GPT-4.1 | Semantic answer validation | https://platform.openai.com |
| ScrapingBee | Protected site scraping | https://scrapingbee.com |
| Trafilatura | HTML → text extraction | https://trafilatura.readthedocs.io |

### 10.3 Related Documentation

- `TRANSITION_GUIDE.md` - Migrating from original SimpleQA
- `docs/VALIDATION_FIX_PLAN.md` - Validation bug fix documentation

---

## Appendix A: Sample Document Structure

```text
=== SOURCE: https://en.wikipedia.org/wiki/Example_Article ===

Title: Example Article

URL Source: https://en.wikipedia.org/wiki/Example_Article

Markdown Content:
Example Article - Wikipedia
===============

[Article content in clean markdown format...]

The key fact mentioned here is that the answer is 42.

## Section Heading

More content with relevant information...
```

---

## Appendix B: Validation Report Schema

```json
{
  "summary": {
    "total_questions": 1000,
    "answers_found": 974,
    "answers_not_found": 26,
    "coverage_rate": 0.974,
    "by_method": {
      "string_match": 883,
      "llm_verified": 83,
      "llm_partial": 8,
      "not_found": 26
    }
  },
  "not_found_details": [
    {
      "question_index": 29,
      "question": "The WWI Q-Ship 'Salvia' was...",
      "answer": "tramp steamer (or collier)",
      "content_preview": "The Project Gutenberg eBook..."
    }
  ]
}
```

---

## Appendix C: Build Manifest Schema

```json
{
  "build_info": {
    "timestamp": "2024-12-08T23:00:00Z",
    "version": "1.0",
    "source_csv": "simpleqa-verified/simpleqa_verified.csv"
  },
  "documents": [
    {
      "filename": "verified_0001.txt",
      "sequence_index": 1,
      "original_index": 5,
      "topic": "Science & Technology",
      "answer_type": "Person",
      "urls_attempted": 3,
      "urls_valid": 2,
      "sources_extracted": 2,
      "word_count": 15234,
      "sources": [
        {
          "url": "https://en.wikipedia.org/wiki/...",
          "word_count": 12500,
          "title": "Wikipedia Article"
        }
      ]
    }
  ]
}
```

---

**Document prepared by**: Automated KB Build System
**Last updated**: December 8, 2024
**Status**: Production-Ready
**Verification**: PASSED (Critic Agent Audit)
