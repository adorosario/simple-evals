# OpenAI RAG Cost Calculation: A Technical Analysis

**Document Version:** 1.0
**Last Updated:** December 14, 2025
**Authors:** Automated Analysis System
**Review Status:** Pending academic review

## Abstract

This document provides a comprehensive technical analysis of OpenAI's Responses API cost calculation for RAG (Retrieval-Augmented Generation) applications using GPT-5.1 with File Search. Through empirical analysis, we verify that OpenAI's token reporting is straightforward and transparent, with all costs properly captured in the standard `input_tokens` and `output_tokens` fields.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background](#2-background)
3. [Token Categories in OpenAI Responses API](#3-token-categories-in-openai-responses-api)
4. [Cost Calculation Methodology](#4-cost-calculation-methodology)
5. [Empirical Analysis](#5-empirical-analysis)
6. [Comparison with Google Gemini](#6-comparison-with-google-gemini)
7. [Vector Store Costs](#7-vector-store-costs)
8. [Implementation](#8-implementation)
9. [Recommendations](#9-recommendations)
10. [References](#10-references)

---

## 1. Executive Summary

### Key Findings

| Finding | Details |
|---------|---------|
| Token reporting is transparent | All tokens included in `input_tokens` and `output_tokens` |
| RAG context included in input | ~17,000 tokens per query (varies by retrieval) |
| Reasoning tokens included in output | ~30-50 tokens per query (adaptive reasoning) |
| No hidden costs | Unlike Gemini, no "hidden" token categories |
| Vector store storage | $0.10/GB/day (first 1GB free) |

### Cost Formula

```
Total Cost = Input Cost + Output Cost + Storage Cost

Where:
  Input Cost   = input_tokens / 1,000,000 × $1.25
  Output Cost  = output_tokens / 1,000,000 × $10.00
  Storage Cost = (GB_stored - 1) × $0.10 / day  (if > 1GB)
```

---

## 2. Background

### 2.1 OpenAI Responses API

The Responses API is OpenAI's modern interface for building AI applications with tool use, including File Search for RAG capabilities. It replaces the older Assistants API for many use cases.

### 2.2 GPT-5.1 Model

GPT-5.1, released November 12, 2025, features:
- **Adaptive reasoning**: Dynamically decides when to "think" before responding
- **1M token context window**: Supports very large document retrieval
- **Improved coding and math**: Significant benchmark improvements over GPT-5

**Source:** [GPT-5.1 Release Notes](https://openai.com/index/gpt-5-1/)

### 2.3 Pricing Model

As of December 2025, GPT-5.1 pricing (per million tokens):

| Token Type | Price |
|------------|-------|
| Input | $1.25 |
| Output | $10.00 |
| Cached Input | $0.3125 (75% discount) |

**Source:** [OpenAI API Pricing](https://openai.com/api/pricing/)

---

## 3. Token Categories in OpenAI Responses API

### 3.1 Usage Object Structure

The Responses API returns a comprehensive usage object:

```python
response.usage = {
    "input_tokens": 17650,
    "output_tokens": 65,
    "total_tokens": 17715,
    "input_tokens_details": {
        "cached_tokens": 0
    },
    "output_tokens_details": {
        "reasoning_tokens": 31
    }
}
```

### 3.2 Token Type Breakdown

| Field | Description | Billing |
|-------|-------------|---------|
| `input_tokens` | User query + RAG retrieved context | $1.25/M |
| `output_tokens` | Response text + reasoning tokens | $10.00/M |
| `reasoning_tokens` | Internal thinking (INCLUDED in output_tokens) | Already counted |
| `cached_tokens` | Tokens served from cache | $0.3125/M (75% discount) |

### 3.3 Key Insight: No Hidden Tokens

Unlike Google Gemini, OpenAI's token reporting is straightforward:

```
input_tokens = user_query + RAG_context
output_tokens = visible_response + reasoning_tokens

total_billed = input_tokens + output_tokens
```

**There are no "hidden" tokens that need to be inferred from `total_tokens`.**

---

## 4. Cost Calculation Methodology

### 4.1 Standard Cost Calculation

```python
def calculate_openai_cost(input_tokens, output_tokens, cached_tokens=0):
    """
    Calculate OpenAI RAG cost.

    Args:
        input_tokens: Total input tokens (includes RAG context)
        output_tokens: Total output tokens (includes reasoning)
        cached_tokens: Tokens served from cache (subset of input_tokens)

    Returns:
        Total cost in USD
    """
    INPUT_RATE = 1.25      # $/M tokens
    OUTPUT_RATE = 10.00    # $/M tokens
    CACHED_RATE = 0.3125   # $/M tokens (75% discount)

    # Uncached input tokens
    uncached_input = input_tokens - cached_tokens

    # Calculate costs
    uncached_input_cost = (uncached_input / 1_000_000) * INPUT_RATE
    cached_input_cost = (cached_tokens / 1_000_000) * CACHED_RATE
    output_cost = (output_tokens / 1_000_000) * OUTPUT_RATE

    return uncached_input_cost + cached_input_cost + output_cost
```

### 4.2 Example Calculation

For a typical RAG query:

```
Input tokens:  17,650 (includes ~17,500 RAG context)
Output tokens:     65 (includes 31 reasoning tokens)
Cached tokens:      0

Cost = (17,650 / 1M × $1.25) + (65 / 1M × $10.00)
     = $0.022063 + $0.000650
     = $0.022713
```

---

## 5. Empirical Analysis

### 5.1 Test Configuration

```yaml
Model: gpt-5.1
API: OpenAI Responses API
Vector Store: vs_6938645787788191bcff16ba2f298d45
Dataset: SimpleQA Verified (5 questions)
```

### 5.2 Token Distribution

From 5 RAG queries:

```
Token Category Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input Tokens:     17,650 ± 500 (avg)
  - User query:      ~50 tokens
  - RAG context: ~17,600 tokens

Output Tokens:        70 ± 20 (avg)
  - Reasoning:    ~35 tokens (50%)
  - Visible:      ~35 tokens (50%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.3 Cost Analysis

| Query | Input | Output | Reasoning | Cost |
|-------|-------|--------|-----------|------|
| simpleqa_0000 | 17,889 | 83 | 45 | $0.0232 |
| simpleqa_0001 | 17,650 | 65 | 31 | $0.0227 |
| simpleqa_0002 | 17,523 | 94 | 52 | $0.0228 |
| simpleqa_0003 | 17,801 | 78 | 40 | $0.0230 |
| simpleqa_0004 | 17,645 | 71 | 38 | $0.0228 |

**Average cost per query: $0.0229**

---

## 6. Comparison with Google Gemini

### 6.1 Token Reporting Differences

| Aspect | OpenAI GPT-5.1 | Google Gemini 3 Pro |
|--------|----------------|---------------------|
| RAG context | Included in `input_tokens` | Hidden in `total_tokens` |
| Reasoning tokens | Included in `output_tokens` | Separate `thoughts_tokens` |
| Hidden tokens | None | ~91% of tokens hidden |
| Transparency | High | Requires calculation |

### 6.2 Cost Transparency

**OpenAI:**
```
Cost = (input_tokens × input_rate) + (output_tokens × output_rate)
     = Direct calculation from API response
```

**Google Gemini:**
```
hidden_tokens = total - prompt - completion - thoughts
input_tokens = prompt + hidden_tokens
output_tokens = completion + thoughts
Cost = Complex calculation required
```

### 6.3 Pricing Comparison

| Provider | Input Rate | Output Rate | Avg Query Cost |
|----------|------------|-------------|----------------|
| OpenAI GPT-5.1 | $1.25/M | $10.00/M | ~$0.023 |
| Gemini 3 Pro | $2.00/M | $12.00/M | ~$0.007 |

Note: OpenAI retrieves more context (~17,000 tokens) vs Gemini (~2,500 tokens), resulting in higher but more comprehensive RAG responses.

---

## 7. Vector Store Costs

### 7.1 Storage Pricing

| Tier | Price |
|------|-------|
| First 1 GB | Free |
| Additional storage | $0.10/GB/day |

### 7.2 Our Benchmark Storage

```
Vector Store: vs_6938645787788191bcff16ba2f298d45
Estimated size: ~500MB (within free tier)
Monthly storage cost: $0.00
```

### 7.3 File Search Query Costs

File Search queries are **included in the model token costs**. There is no additional per-query fee for retrieval. The retrieved documents appear as tokens in `input_tokens`.

**Source:** [OpenAI Developer Community](https://community.openai.com/t/how-file-search-works-and-pricing/805817)

---

## 8. Implementation

### 8.1 Token Capture (Python)

```python
# File: sampler/audited_openai_rag_sampler.py

if hasattr(response, 'usage') and response.usage:
    input_tokens = response.usage.input_tokens or 0
    output_tokens = response.usage.output_tokens or 0

    # Extract detailed token breakdowns
    reasoning_tokens = 0
    cached_tokens = 0

    if hasattr(response.usage, 'output_tokens_details'):
        reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens or 0

    if hasattr(response.usage, 'input_tokens_details'):
        cached_tokens = response.usage.input_tokens_details.cached_tokens or 0

    self._last_usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cached_tokens": cached_tokens,
        "visible_output_tokens": output_tokens - reasoning_tokens
    }
```

### 8.2 Cost Calculation

```python
# File: pricing_config.py

MODEL_PRICING = {
    "gpt-5.1": {"input": 1.25, "output": 10.00},
}

def calculate_cost(model, prompt_tokens, completion_tokens):
    pricing = MODEL_PRICING.get(model)
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
```

---

## 9. Recommendations

### 9.1 For Implementers

1. **Use the standard formula**: `input_tokens` and `output_tokens` contain all billable tokens
2. **Track reasoning tokens** for optimization insights
3. **Monitor cached tokens** to measure cache efficiency
4. **No hidden token calculation needed** unlike Gemini

### 9.2 For Cost Optimization

1. **Enable caching**: Cached tokens are 75% cheaper
2. **Optimize retrieval**: Reduce chunk size if responses include too much irrelevant context
3. **Use batch API**: 50% discount for async processing
4. **Consider GPT-5.1-mini**: Cheaper option for simpler queries

### 9.3 Comparison Note

OpenAI RAG appears more expensive per query (~$0.023) than Gemini RAG (~$0.007) because:
- OpenAI retrieves more context (~17,000 vs ~2,500 tokens)
- This may result in higher quality/more comprehensive answers
- Consider accuracy-per-dollar metrics for true comparison

---

## 10. References

1. OpenAI API Pricing
   https://openai.com/api/pricing/

2. OpenAI Responses API Documentation
   https://platform.openai.com/docs/api-reference/responses

3. GPT-5.1 Release Announcement
   https://openai.com/index/gpt-5-1/

4. File Search Pricing Discussion
   https://community.openai.com/t/how-file-search-works-and-pricing/805817

5. Vector Store Pricing
   https://community.openai.com/t/vector-stores-search-pricing-in-the-api/1251975

6. Reasoning Tokens in Responses API
   https://cookbook.openai.com/examples/responses_api/reasoning_items

---

## Appendix: Raw Test Data

```json
{
  "test_query": "What year was Glipa andamana described?",
  "response": "Glipa andamana was described in 1941.",
  "usage": {
    "prompt_tokens": 17650,
    "completion_tokens": 65,
    "total_tokens": 17715,
    "reasoning_tokens": 31,
    "cached_tokens": 0,
    "visible_output_tokens": 34
  },
  "cost_calculation": {
    "input_cost": "$0.022063",
    "output_cost": "$0.000650",
    "total_cost": "$0.022713"
  }
}
```

---

*This document was generated through empirical analysis of OpenAI API responses. For the most current pricing, always refer to OpenAI's official documentation.*
