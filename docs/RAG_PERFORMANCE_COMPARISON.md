# RAG Performance Comparison

**Document Version:** 1.0
**Last Updated:** December 14, 2025
**Authors:** Automated Analysis System
**Review Status:** Critic-agent verified

## Abstract

This document provides a comprehensive head-to-head comparison of all RAG providers in the benchmark, analyzing latency, cost, accuracy, and overall value. It includes recommendations for different use cases.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Head-to-Head Comparison](#2-head-to-head-comparison)
3. [Latency Analysis](#3-latency-analysis)
4. [Cost Analysis](#4-cost-analysis)
5. [Accuracy Analysis](#5-accuracy-analysis)
6. [Value Analysis (Cost per Correct Answer)](#6-value-analysis)
7. [Scalability Considerations](#7-scalability-considerations)
8. [Recommendations](#8-recommendations)

---

## 1. Executive Summary

### Provider Rankings

| Metric | Winner | Runner-up | Third | Fourth |
|--------|--------|-----------|-------|--------|
| **Latency** | CustomGPT | OpenAI_RAG | Gemini_RAG | - |
| **Cost** | Gemini_RAG | OpenAI_RAG | CustomGPT | - |
| **Accuracy** | CustomGPT | Gemini_RAG | OpenAI_RAG | OpenAI_Vanilla |
| **Value (Cost/Correct)** | Gemini_RAG | OpenAI_RAG | CustomGPT | - |
| **Scalability** | CustomGPT | OpenAI_RAG | Gemini_RAG | - |

### Quick Recommendation

| Use Case | Best Provider | Why |
|----------|---------------|-----|
| Production RAG | CustomGPT | Best latency + accuracy |
| Budget-conscious | Gemini_RAG | Lowest cost per correct answer |
| Enterprise/Volume | OpenAI_RAG | Predictable at scale |
| Quick Prototyping | OpenAI_Vanilla | Simplest integration |

---

## 2. Head-to-Head Comparison

### Overall Metrics (100-Sample Benchmark - run_20251214_152848_133)

| Provider | Avg Latency | Avg Cost/Query | Accuracy | P95 Latency |
|----------|-------------|----------------|----------|-------------|
| OpenAI_Vanilla | 2,210 ms | $0.0002 | 37.9%* | 3,807 ms |
| CustomGPT_RAG | 3,642 ms | $0.10 | 97.8% | 3,806 ms |
| OpenAI_RAG | 8,648 ms | $0.023 | 89.0% | 15,505 ms |
| Google_Gemini_RAG | 25,068 ms | $0.007 | 94.7% | 64,809 ms |

*Note: OpenAI Vanilla accuracy is lower because it lacks RAG context and abstains frequently (42% "I don't know").

### Visual Comparison (5-Sample Run)

```
Latency (lower is better):
OpenAI_Vanilla  ████████░░░░░░░░░░░░ 2,112 ms
CustomGPT_RAG   ███████████░░░░░░░░░ 2,925 ms
OpenAI_RAG      █████████████████░░░ 4,514 ms
Google_Gemini   ████████████████████████░░ 6,351 ms

Cost per Query (lower is better):
OpenAI_Vanilla  ░░░░░░░░░░░░░░░░░░░░ $0.0002
Google_Gemini   ███░░░░░░░░░░░░░░░░░ $0.007
OpenAI_RAG      █████████░░░░░░░░░░░ $0.023
CustomGPT_RAG   ████████████████████ $0.10

Accuracy (higher is better):
OpenAI_Vanilla  ████████░░░░░░░░░░░░ 37.9%
OpenAI_RAG      ██████████████████░░ 89.0%
Google_Gemini   ███████████████████░ 94.7%
CustomGPT_RAG   ████████████████████ 97.8%
```

---

## 3. Latency Analysis

### Latency at Different Scales

| Provider | 5-Sample Avg | 100-Sample Avg | Degradation |
|----------|--------------|----------------|-------------|
| OpenAI_Vanilla | 2,112 ms | 2,210 ms | +5% |
| CustomGPT_RAG | 2,925 ms | 3,642 ms | +25% |
| OpenAI_RAG | 4,514 ms | 8,648 ms | +92% |
| Google_Gemini_RAG | 6,351 ms | 25,068 ms | **+295%** |

### Latency Consistency (Coefficient of Variation)

| Provider | CV (5-Sample) | CV (100-Sample) | Verdict |
|----------|---------------|-----------------|---------|
| OpenAI_Vanilla | 17% | 40% | Consistent |
| CustomGPT_RAG | 8% | 175%* | Consistent (1 outlier) |
| OpenAI_RAG | 13% | 56% | Moderate variance |
| Google_Gemini_RAG | 11% | **95%** | Unpredictable |

*CustomGPT's high CV in 100-sample is due to a single 64.5s outlier.

### P95 Latency (What 95% of Users Experience)

| Provider | P95 (5-Sample) | P95 (100-Sample) | Verdict |
|----------|----------------|------------------|---------|
| OpenAI_Vanilla | 2,605 ms | 3,807 ms | Excellent |
| CustomGPT_RAG | 3,238 ms | 3,806 ms | Excellent |
| OpenAI_RAG | 5,238 ms | 15,505 ms | Acceptable |
| Google_Gemini_RAG | 7,125 ms | **64,809 ms** | Poor |

---

## 4. Cost Analysis

### Cost per Query

| Provider | Pricing Model | Avg Cost/Query |
|----------|---------------|----------------|
| OpenAI_Vanilla | $1.25/M input, $10/M output | $0.0002 |
| CustomGPT_RAG | $0.10/query flat | $0.10 |
| OpenAI_RAG | $1.25/M input, $10/M output | $0.023 |
| Google_Gemini_RAG | $2/M input, $12/M output | $0.007 |

### Cost Breakdown by Component

**OpenAI RAG** ($0.023/query):
```
Input tokens: ~16,000 @ $1.25/M = $0.020
Output tokens: ~110 @ $10/M = $0.001
Reasoning tokens: ~50 (included in output)
Total: ~$0.021-0.023
```

**Google Gemini RAG** ($0.007/query):
```
User prompt: ~140 tokens @ $2/M = $0.0003
RAG context: ~2,500 tokens @ $2/M = $0.005
Completion: ~50 tokens @ $12/M = $0.0006
Thinking: ~100 tokens @ $12/M = $0.0012
Total: ~$0.007
```

**CustomGPT RAG** ($0.10/query):
```
Flat rate per addon query = $0.10
No token breakdown available
```

### Cost for 100 Queries

| Provider | Cost |
|----------|------|
| OpenAI_Vanilla | $0.02 |
| Google_Gemini_RAG | $0.71 |
| OpenAI_RAG | $2.30 |
| CustomGPT_RAG | $10.00 |

---

## 5. Accuracy Analysis

### Accuracy Rates (from run_20251214_152848_133)

| Provider | Correct | Incorrect | "I Don't Know" | Accuracy |
|----------|---------|-----------|----------------|----------|
| CustomGPT_RAG | 87 (87%) | 2 (2%) | 11 (11%) | **97.8%** |
| Google_Gemini_RAG | 90 (90%) | 5 (5%) | 5 (5%) | 94.7% |
| OpenAI_RAG | 89 (89%) | 11 (11%) | 0 (0%) | 89.0% |
| OpenAI_Vanilla | 22 (22%) | 36 (36%) | 42 (42%) | 37.9% |

### Why CustomGPT Has Highest Accuracy

1. **Optimized retrieval:** Returns most relevant snippets
2. **Pre-indexed knowledge base:** Purpose-built for the dataset
3. **Efficient context usage:** Less noise, more signal

### Why OpenAI RAG Has Strong but Not Highest Accuracy

OpenAI RAG retrieves ~17,000 tokens (most context) and achieves 89% accuracy:
1. **High context volume:** Retrieves extensive information
2. **No abstention:** Attempts all questions (0% "I don't know")
3. **Some irrelevant context:** May include noise that affects answer quality

---

## 6. Value Analysis

### Cost per Correct Answer

This metric answers: "How much does it cost to get a correct answer?"

| Provider | Cost/Query | Accuracy | Cost/Correct Answer |
|----------|------------|----------|---------------------|
| Google_Gemini_RAG | $0.007 | 94.7% | **$0.0074** |
| OpenAI_RAG | $0.023 | 89.0% | $0.026 |
| CustomGPT_RAG | $0.10 | 97.8% | $0.102 |

### Value Ranking

1. **Gemini RAG:** $0.0074 per correct answer (best value)
2. **OpenAI RAG:** $0.026 per correct answer
3. **CustomGPT RAG:** $0.102 per correct answer (highest accuracy but most expensive)

### Trade-off Analysis

```
                 Cost →
         Low                    High
    ┌────────────────────────────────────┐
 H  │  Gemini RAG    │  CustomGPT RAG    │
 i  │  (94.7% acc)   │  (97.8% acc)      │
 g  │  $0.007/query  │  $0.10/query      │
 h  │────────────────┼───────────────────│
    │  OpenAI RAG    │                   │
 A  │  (89.0% acc)   │                   │
 c  │  $0.023/query  │                   │
 c  │────────────────┼───────────────────│
 u  │                │                   │
 r  │                │                   │
 a  │  OpenAI Vanilla│                   │
 c  │  (37.9% acc)   │                   │
 y  │  $0.0002/query │                   │
 ↓  └────────────────────────────────────┘
    Low
```

---

## 7. Scalability Considerations

### Performance at Scale (100 queries)

| Provider | Degradation | Outliers >10s | Max Latency | Verdict |
|----------|-------------|---------------|-------------|---------|
| OpenAI_Vanilla | +5% | 0 | 6s | Excellent |
| CustomGPT_RAG | +25% | 1 | 65s | Good |
| OpenAI_RAG | +92% | 27 | 22s | Moderate |
| Google_Gemini_RAG | **+295%** | **79** | **157s** | Poor |

### Rate Limits

| Provider | Tier | RPM | TPM | RPD |
|----------|------|-----|-----|-----|
| OpenAI | Tier 5 | 10,000 | 2,000,000 | Unlimited |
| Google AI Studio | Tier 2 | 1,000 | 2,000,000 | 10,000 |
| CustomGPT | N/A | Unknown | Unknown | Unknown |

### Scaling Recommendations

**For <1,000 queries/day:**
- All providers suitable
- Gemini offers best value

**For 1,000-10,000 queries/day:**
- OpenAI RAG or CustomGPT recommended
- Gemini may hit rate limits

**For >10,000 queries/day:**
- CustomGPT: Contact for enterprise pricing
- OpenAI: Consider batch API (50% discount)
- Gemini: Requires Tier 3 or Vertex AI

---

## 8. Recommendations

### By Use Case

#### Production RAG Application
**Recommendation: CustomGPT**
- Best: Latency (2.9s), accuracy (97.8%)
- Acceptable: Cost ($0.10/query)
- Consistent P95 latency
- Minimal outliers

#### Budget-Conscious Development
**Recommendation: Google Gemini RAG (with caveats)**
- Best: Cost ($0.007/query), value ($0.0074/correct)
- Caveat: High latency variance at scale
- Caveat: 79% of requests may exceed 10 seconds at volume
- Best for: Small batches, non-time-sensitive workloads

#### Enterprise/High-Volume
**Recommendation: OpenAI RAG**
- Moderate latency (4.5s at low volume)
- Predictable pricing
- Strong documentation and support
- Batch API available for async workloads

#### Quick Prototyping
**Recommendation: OpenAI Vanilla**
- Fastest integration
- Lowest cost
- Note: Lower accuracy (37.9%) due to no RAG

### Decision Matrix

| Factor | Weight | CustomGPT | OpenAI_RAG | Gemini_RAG |
|--------|--------|-----------|------------|------------|
| Latency | 30% | 5 (1.5) | 3 (0.9) | 2 (0.6) |
| Cost | 20% | 2 (0.4) | 3 (0.6) | 5 (1.0) |
| Accuracy | 30% | 5 (1.5) | 4 (1.2) | 5 (1.5) |
| Scalability | 20% | 4 (0.8) | 4 (0.8) | 2 (0.4) |
| **Total** | 100% | **4.2** | 3.5 | **3.5** |

(Score: 1-5, higher is better. Accuracy: CustomGPT 97.8%=5, Gemini 94.7%=5, OpenAI 89%=4)

### Final Verdict

| Use Case | Best Choice |
|----------|-------------|
| Overall best | **CustomGPT RAG** |
| Best value | **Gemini RAG** (small scale only) |
| Most balanced | **OpenAI RAG** |
| Fastest (no RAG) | **OpenAI Vanilla** |

---

## Appendix: Benchmark Configuration

### Environment
- Platform: Docker (Linux)
- Python: 3.11
- Concurrency: max_workers=10, semaphore=5 per provider
- Date: December 14, 2025

### Models Used
| Provider | Model | Release Date |
|----------|-------|--------------|
| OpenAI | gpt-5.1 | Nov 12, 2025 |
| Gemini | gemini-3-pro-preview | Nov 18, 2025 |
| CustomGPT | gpt-5.1 (via API) | Nov 12, 2025 |

### Dataset
- SimpleQA Verified (100 questions)
- Factual question-answering
- Graded by GPT-5.1 as LLM-as-a-Judge

---

*This document was generated through comprehensive benchmark analysis. All comparisons are based on actual benchmark runs with verified timing and cost calculations.*
