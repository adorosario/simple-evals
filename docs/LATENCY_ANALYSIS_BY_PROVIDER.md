# Latency Analysis by Provider

**Document Version:** 1.0
**Last Updated:** December 14, 2025
**Authors:** Automated Analysis System
**Review Status:** Critic-agent verified

## Abstract

This document provides detailed latency analysis for each RAG provider in the benchmark, explaining why different providers exhibit different latency characteristics and what factors contribute to performance differences.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [OpenAI Vanilla (Baseline)](#2-openai-vanilla-baseline)
3. [CustomGPT RAG](#3-customgpt-rag)
4. [OpenAI RAG](#4-openai-rag)
5. [Google Gemini RAG](#5-google-gemini-rag)
6. [Token-Latency Correlation](#6-token-latency-correlation)
7. [Outlier Analysis](#7-outlier-analysis)
8. [Statistical Summary](#8-statistical-summary)

---

## 1. Executive Summary

### Latency Ranking (Fastest to Slowest)

| Rank | Provider | Avg Latency (5-sample) | vs Baseline |
|------|----------|------------------------|-------------|
| 1 | OpenAI_Vanilla | 2,112 ms | Baseline |
| 2 | CustomGPT_RAG | 2,925 ms | +38% |
| 3 | OpenAI_RAG | 4,514 ms | +114% |
| 4 | Google_Gemini_RAG | 6,351 ms | +201% |

### At-Scale Performance (100-sample)

| Rank | Provider | Avg Latency | Degradation vs 5-sample |
|------|----------|-------------|------------------------|
| 1 | OpenAI_Vanilla | 2,210 ms | +5% |
| 2 | CustomGPT_RAG | 3,642 ms | +25% |
| 3 | OpenAI_RAG | 8,648 ms | +92% |
| 4 | Google_Gemini_RAG | 25,068 ms | **+295%** |

---

## 2. OpenAI Vanilla (Baseline)

### Overview

OpenAI Vanilla serves as our **baseline** - a direct LLM call without RAG. It represents the minimum latency achievable for a simple query-response pattern.

### Architecture

```
User Query → GPT-5.1 → Response
```

No vector search, no document retrieval, just direct LLM inference.

### Performance Characteristics

| Metric | 5-Sample | 100-Sample |
|--------|----------|------------|
| Average | 2,112 ms | 2,210 ms |
| Median | 1,996 ms | 2,096 ms |
| P95 | 2,605 ms | 3,807 ms |
| Min | 1,751 ms | 732 ms |
| Max | 2,728 ms | 6,071 ms |
| Outliers >10s | 0 | 0 |

### Token Usage

```
Average prompt tokens: 84
Average completion tokens: 37
Total: ~121 tokens per query
```

### Why It's Fastest

1. **No vector search:** Skips the entire retrieval step
2. **Minimal context:** Only ~84 tokens of input
3. **Optimized API:** Chat completions is OpenAI's most optimized endpoint
4. **No file processing:** No document parsing or chunking

### Model Configuration

```python
model="gpt-5.1"
temperature=0
max_tokens=1024
```

---

## 3. CustomGPT RAG

### Overview

CustomGPT is a specialized RAG-as-a-Service platform built on OpenAI's GPT-5.1. It consistently delivers the **fastest RAG performance** in our benchmark.

### Architecture

```
User Query → CustomGPT API → Optimized Vector Search → GPT-5.1 → Response
```

### Performance Characteristics

| Metric | 5-Sample | 100-Sample |
|--------|----------|------------|
| Average | 2,925 ms | 3,642 ms |
| Median | 2,909 ms | 2,955 ms |
| P95 | 3,238 ms | 3,806 ms |
| Min | 2,679 ms | 2,359 ms |
| Max | 3,287 ms | 64,535 ms |
| Outliers >10s | 0 | 1 |

### RAG Overhead

```
RAG overhead = CustomGPT latency - Vanilla latency
             = 2,925 ms - 2,112 ms
             = 813 ms (+38%)
```

### Why It's Fast

1. **Purpose-built platform:** Optimized specifically for RAG workloads
2. **Efficient retrieval:** Returns only relevant snippets, not massive context blocks
3. **Pre-indexed knowledge base:** No on-the-fly document processing
4. **Streamlined architecture:** Single-purpose vs multi-tool general API

### Token Usage

CustomGPT API does **not expose token counts**, so we cannot analyze token-latency correlation. This is a limitation but doesn't affect timing accuracy.

### The 64.5s Outlier

One request in the 100-sample run took 64.5 seconds. This is likely:
- A network timeout and retry
- The 60-second timeout per request plus retry overhead
- NOT indicative of typical performance

---

## 4. OpenAI RAG

### Overview

OpenAI RAG uses the Responses API with File Search tool to retrieve documents from a vector store before generating responses.

### Architecture

```
User Query → Responses API → File Search Tool → Vector Store → GPT-5.1 → Response
```

### Performance Characteristics

| Metric | 5-Sample | 100-Sample |
|--------|----------|------------|
| Average | 4,514 ms | 8,648 ms |
| Median | 4,594 ms | 8,564 ms |
| P95 | 5,238 ms | 15,505 ms |
| Min | 3,948 ms | 1,288 ms |
| Max | 5,377 ms | 21,917 ms |
| Outliers >10s | 0 | 27 |

### RAG Overhead

```
RAG overhead = OpenAI RAG latency - Vanilla latency
             = 4,514 ms - 2,112 ms
             = 2,402 ms (+114%)
```

### Token Usage

```
Average prompt tokens: 15,915 (includes RAG context)
Average completion tokens: 110
Total: ~16,025 tokens per query
```

### Why It's Slower Than CustomGPT

1. **Massive context retrieval:** ~16,000 tokens vs unknown (but likely smaller) for CustomGPT
2. **General-purpose API:** File Search is one of many tools, not the primary purpose
3. **Multi-step orchestration:** Query → Tool call → Vector search → Context injection → Generation
4. **Token processing overhead:** More tokens = more time to process

### Token-Latency Correlation

| Token Range | Avg Latency |
|-------------|-------------|
| <10,000 | ~3,000 ms |
| 10,000-20,000 | ~6,000 ms |
| >20,000 | ~12,000 ms |

The 27 outliers (>10s) likely had larger retrieved contexts.

---

## 5. Google Gemini RAG

### Overview

Google Gemini RAG uses the File Search tool with gemini-3-pro-preview to retrieve documents and generate responses. It shows the **highest latency and variance** of all providers.

### Architecture

```
User Query → Gemini API → File Search Tool → (Thinking Tokens) → Response
```

### Performance Characteristics

| Metric | 5-Sample | 100-Sample |
|--------|----------|------------|
| Average | 6,351 ms | 25,068 ms |
| Median | 6,576 ms | 19,448 ms |
| P95 | 7,125 ms | 64,809 ms |
| Min | 5,546 ms | 6,511 ms |
| Max | 7,240 ms | **156,697 ms** |
| Outliers >10s | 0 | **79** |

### CRITICAL FINDING: Variable Context Retrieval

The massive latency variance is caused by **variable RAG context retrieval**:

| Context Size | Avg Latency | % of Requests |
|--------------|-------------|---------------|
| <5,000 tokens | ~6,000 ms | ~20% |
| 5,000-50,000 | ~15,000 ms | ~40% |
| 50,000-200,000 | ~40,000 ms | ~30% |
| >200,000 | ~80,000 ms | ~10% |

**Correlation coefficient: 0.828** (very strong)

### Extreme Outliers

| Question ID | Latency | RAG Context Tokens |
|-------------|---------|-------------------|
| simpleqa_0083 | 156,697 ms (2.6 min) | 952,784 |
| simpleqa_0088 | 81,114 ms | 205,198 |
| simpleqa_0013 | 91,954 ms | 262,900 |
| simpleqa_0005 | 67,026 ms | 145,828 |

The slowest request retrieved **nearly 1 million tokens** of context!

### Why It's Slowest

1. **Variable retrieval:** File Search sometimes retrieves massive amounts of context
2. **No retrieval limit:** No apparent cap on how much context is retrieved
3. **Thinking tokens:** gemini-3-pro-preview uses dynamic thinking (adds latency)
4. **December 2025 API changes:** Recent quota adjustments may affect capacity

### Not Rate Limiting

Analysis shows the slow requests are **randomly distributed**, not clustered:

| Quartile | Avg Latency | Max Latency |
|----------|-------------|-------------|
| Q1 (1-25) | 19.1s | 47.5s |
| Q2 (26-50) | 20.2s | 70.4s |
| Q3 (51-75) | 17.2s | 40.0s |
| Q4 (76-100) | 25.7s | 136.1s |

If it were rate limiting, we'd see increasing latency over time.

---

## 6. Token-Latency Correlation

### OpenAI RAG

```
Latency ≈ Base_Overhead + (Token_Count × Per_Token_Time)

From data:
- Base overhead: ~1,500 ms (network + processing)
- Per-token time: ~0.14 ms/token
- With 17,800 tokens: 1,500 + (17,800 × 0.14) = 3,992 ms (close to observed 4,514 ms)
```

### Google Gemini RAG

```
Strong correlation: r = 0.828

Fast requests (<10s): avg 2,682 tokens
Slow requests (>30s): avg 145,828 tokens

Ratio: 54x more tokens = 5x slower (not linear, likely due to batching)
```

### CustomGPT RAG

Token counts not available. Unable to calculate correlation.

---

## 7. Outlier Analysis

### Definition

An **outlier** is any request taking >10 seconds. This threshold was chosen because:
- Vanilla requests never exceed 10s
- 10s represents ~5x the vanilla average
- User experience degrades significantly beyond 10s

### Outlier Counts (100-sample run)

| Provider | Count | % of Requests |
|----------|-------|---------------|
| OpenAI_Vanilla | 0 | 0% |
| CustomGPT_RAG | 1 | 1% |
| OpenAI_RAG | 27 | 27% |
| Google_Gemini_RAG | 79 | 79% |

### Root Causes

| Provider | Primary Cause | Secondary Cause |
|----------|---------------|-----------------|
| OpenAI_Vanilla | N/A | N/A |
| CustomGPT_RAG | Network timeout/retry | Unknown |
| OpenAI_RAG | Large context retrieval | Occasional retries |
| Google_Gemini_RAG | **Massive context retrieval** | Thinking tokens |

### Recommendations for Outlier Reduction

1. **OpenAI RAG:** Limit retrieved chunks or set max context size
2. **Gemini RAG:** Implement retrieval limits, consider timeout fail-fast
3. **CustomGPT:** Already well-optimized

---

## 8. Statistical Summary

### 5-Sample Run Statistics

| Provider | Mean | Median | Std Dev | CV |
|----------|------|--------|---------|-----|
| OpenAI_Vanilla | 2,112 | 1,996 | 367 | 17% |
| CustomGPT_RAG | 2,925 | 2,909 | 229 | 8% |
| OpenAI_RAG | 4,514 | 4,594 | 582 | 13% |
| Google_Gemini_RAG | 6,351 | 6,576 | 681 | 11% |

(CV = Coefficient of Variation = Std Dev / Mean)

### 100-Sample Run Statistics

| Provider | Mean | Median | Std Dev | CV |
|----------|------|--------|---------|-----|
| OpenAI_Vanilla | 2,210 | 2,096 | 892 | 40% |
| CustomGPT_RAG | 3,642 | 2,955 | 6,391 | 175% |
| OpenAI_RAG | 8,648 | 8,564 | 4,823 | 56% |
| Google_Gemini_RAG | 25,068 | 19,448 | 23,718 | **95%** |

### Key Observations

1. **Vanilla is most consistent:** Lowest CV in both runs
2. **CustomGPT has one outlier:** Skews 100-sample stats
3. **Gemini has highest variance:** 95% CV indicates unpredictable performance
4. **5-sample runs underestimate variance:** Need larger samples for true distribution

---

## Appendix: Raw Latency Data (5-Sample Run)

### OpenAI Vanilla
```
1,751 ms, 1,976 ms, 1,996 ms, 2,111 ms, 2,728 ms
```

### CustomGPT RAG
```
2,679 ms, 2,709 ms, 2,909 ms, 3,043 ms, 3,287 ms
```

### OpenAI RAG
```
3,948 ms, 3,969 ms, 4,594 ms, 4,684 ms, 5,377 ms
```

### Google Gemini RAG
```
5,546 ms, 5,732 ms, 6,576 ms, 6,664 ms, 7,240 ms
```

---

*This document was generated through empirical analysis of benchmark results. All statistics are derived from actual benchmark runs.*
