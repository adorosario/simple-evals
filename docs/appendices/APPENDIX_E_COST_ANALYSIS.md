# Appendix E: Cost Analysis

**Run ID:** `20251214_152848_133`
**Analysis Date:** December 2025

---

## E.1 Pricing Models

### Token-Based Pricing (OpenAI, Gemini)

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gpt-5.1 | $1.25 | $10.00 |
| gpt-5-nano | $0.10 | $0.40 |
| gemini-3-pro-preview | $2.00 | $12.00 |

### Per-Query Pricing (CustomGPT)

| Provider | Cost per Query | Notes |
|----------|----------------|-------|
| CustomGPT | $0.10 | Subscription addon pricing |

---

## E.2 Provider Costs (100 Questions)

### CustomGPT RAG

| Metric | Value |
|--------|-------|
| Queries | 100 |
| Cost per query | $0.10 |
| **Total cost** | **$10.00** |
| Token usage | Not tracked |

### OpenAI RAG

| Metric | Value |
|--------|-------|
| Total tokens | 1,602,500 |
| Avg tokens/request | 16,025 |
| Avg prompt tokens | 15,915 |
| Avg completion tokens | 110 |
| Input cost | $1.99 (15,915 × 100 × $1.25/M) |
| Output cost | $0.11 (110 × 100 × $10/M) |
| **Total cost** | **$2.10** |

### Google Gemini RAG

| Metric | Value |
|--------|-------|
| Total tokens | 1,686,772 |
| Avg tokens/request | 16,868 |
| Avg prompt tokens | 144 |
| Avg completion tokens | 88 |
| Input cost | $0.03 |
| Output cost | $0.03 |
| **Total cost** | **$0.06** |

*Note: Gemini token counts may not include file search context.*

### OpenAI Vanilla

| Metric | Value |
|--------|-------|
| Total tokens | 9,533 |
| Avg tokens/request | 95 |
| Avg prompt tokens | 67 |
| Avg completion tokens | 29 |
| Input cost | $0.008 |
| Output cost | $0.029 |
| **Total cost** | **$0.04** |

---

## E.3 Judge Costs

### Intent Classification (GPT-5-nano)

| Metric | Value |
|--------|-------|
| Classifications | 400 |
| Est. tokens/classification | 150 |
| Total tokens | 60,000 |
| Input cost | $0.006 |
| Output cost | $0.008 |
| **Total cost** | **$0.01** |

### Judge Grading (GPT-5.1)

| Metric | Value |
|--------|-------|
| Evaluations | 400 |
| Est. tokens/evaluation | 500 |
| Total tokens | 200,000 |
| Input cost | $0.19 |
| Output cost | $0.25 |
| **Total cost** | **$0.44** |

### Consistency Validation

| Metric | Value |
|--------|-------|
| Validations | 40 × 3 = 120 |
| Est. tokens/validation | 500 |
| Total tokens | 60,000 |
| **Total cost** | **$0.08** |

---

## E.4 Total Benchmark Cost

| Component | Cost |
|-----------|------|
| CustomGPT RAG (100q) | $10.00 |
| OpenAI RAG (100q) | $2.10 |
| Google Gemini RAG (100q) | $0.06 |
| OpenAI Vanilla (100q) | $0.04 |
| Intent classifier | $0.01 |
| Judge grading | $0.44 |
| Consistency validation | $0.08 |
| **Total** | **$12.73** |

---

## E.5 Cost-Effectiveness Analysis

### Cost per Query

| Provider | Cost/Query | Rank |
|----------|------------|------|
| OpenAI Vanilla | $0.00037 | 1st |
| Google Gemini RAG | $0.00060 | 2nd |
| OpenAI RAG | $0.02100 | 3rd |
| CustomGPT RAG | $0.10000 | 4th |

### Cost per Correct Answer

| Provider | Correct | Cost | Cost/Correct |
|----------|---------|------|--------------|
| Google Gemini RAG | 90 | $0.06 | $0.00067 |
| OpenAI Vanilla | 22 | $0.04 | $0.00182 |
| OpenAI RAG | 89 | $2.10 | $0.02360 |
| CustomGPT RAG | 87 | $10.00 | $0.11494 |

### Effective Cost (Including Error Penalties)

In high-stakes applications, errors have costs. Assuming a $5 business cost per error:

| Provider | API Cost | Error Cost (n × $5) | Effective Cost |
|----------|----------|---------------------|----------------|
| CustomGPT RAG | $10.00 | $10.00 (2 errors) | $20.00 |
| Google Gemini RAG | $0.06 | $25.00 (5 errors) | $25.06 |
| OpenAI RAG | $2.10 | $55.00 (11 errors) | $57.10 |
| OpenAI Vanilla | $0.04 | $180.00 (36 errors) | $180.04 |

**Key Insight:** When error costs exceed ~$10, CustomGPT RAG becomes the most cost-effective option despite highest per-query cost.

---

## E.6 Scaling Projections

### 1,000 Questions/Day

| Provider | Daily Cost | Monthly Cost | Annual Cost |
|----------|------------|--------------|-------------|
| OpenAI Vanilla | $0.37 | $11 | $135 |
| Google Gemini RAG | $0.60 | $18 | $219 |
| OpenAI RAG | $21.00 | $630 | $7,665 |
| CustomGPT RAG | $100.00 | $3,000 | $36,500 |

### 10,000 Questions/Day

| Provider | Daily Cost | Monthly Cost | Annual Cost |
|----------|------------|--------------|-------------|
| OpenAI Vanilla | $3.70 | $111 | $1,351 |
| Google Gemini RAG | $6.00 | $180 | $2,190 |
| OpenAI RAG | $210.00 | $6,300 | $76,650 |
| CustomGPT RAG | $1,000.00 | $30,000 | $365,000 |

---

## E.7 Break-Even Analysis

### Error Cost Threshold

At what error cost does CustomGPT RAG become cheaper than alternatives?

**CustomGPT vs Gemini:**
- API difference: $10.00 - $0.06 = $9.94
- Error difference: 5 - 2 = 3 errors
- Break-even: $9.94 / 3 = **$3.31 per error**

**CustomGPT vs OpenAI RAG:**
- API difference: $10.00 - $2.10 = $7.90
- Error difference: 11 - 2 = 9 errors
- Break-even: $7.90 / 9 = **$0.88 per error**

**Interpretation:** If each error costs more than $3.31, CustomGPT RAG is the most cost-effective option.

---

## E.8 Hidden Costs

### Context Window Costs

OpenAI RAG averages 16,000 tokens/request due to retrieved document context. This is 170× more than vanilla (95 tokens/request).

### Latency Costs

| Provider | Avg Latency | Queries/Min (Theoretical) |
|----------|-------------|---------------------------|
| OpenAI Vanilla | 2.2s | 27 |
| CustomGPT RAG | 3.6s | 17 |
| OpenAI RAG | 8.6s | 7 |
| Google Gemini RAG | 25.1s | 2 |

For latency-sensitive applications, Gemini's 25-second average may be prohibitive.

### Rate Limit Costs

| Provider | Concurrent Limit | Impact |
|----------|------------------|--------|
| CustomGPT | 5 | May bottleneck high-volume workloads |
| OpenAI | Unlimited* | Best for parallel processing |
| Gemini | 5 | May bottleneck high-volume workloads |

*Subject to token-per-minute limits.

---

## E.9 Recommendations by Use Case

| Use Case | Recommended Provider | Reasoning |
|----------|---------------------|-----------|
| Cost-sensitive, low stakes | Gemini RAG | Best quality/cost ratio |
| Cost-sensitive, moderate stakes | OpenAI RAG | Balance of cost and quality |
| Quality-critical, high stakes | CustomGPT RAG | Lowest error rate (2%) |
| Latency-critical | OpenAI Vanilla | Fastest response (2.2s) |
| High volume (>10K/day) | OpenAI RAG or Gemini | Better cost scaling |

---

## E.10 Cost Tracking Implementation

**pricing_config.py:**
```python
MODEL_PRICING = {
    "gpt-5.1": {"input": 1.25, "output": 10.00},  # per million
    "gpt-5-nano": {"input": 0.10, "output": 0.40},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    "customgpt": {"per_query": 0.10},
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model == "customgpt":
        return MODEL_PRICING[model]["per_query"]
    pricing = MODEL_PRICING[model]
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
```

---

*Appendix E: Cost Analysis | Run 20251214_152848_133*
