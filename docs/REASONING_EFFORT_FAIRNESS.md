# Reasoning/Thinking Fairness Configuration

This document establishes the fair comparison configuration for RAG provider benchmarking, ensuring apples-to-apples evaluation across different model families.

## Final Configuration (January 2026)

After extensive testing, the following configuration provides the fairest comparison:

| Provider | Model | Reasoning/Thinking Setting | Avg Latency | Cost per 1M tokens |
|----------|-------|---------------------------|-------------|-------------------|
| **OpenAI RAG** | `gpt-5.1` | `reasoning_effort: none` (default) | ~8,700ms | $1.25 in / $10 out |
| **OpenAI Vanilla** | `gpt-5.1` | `reasoning_effort: none` (default) | ~2,700ms | $1.25 in / $10 out |
| **CustomGPT RAG** | `gpt-5.1` (via API) | `reasoning_effort: none` (inherited) | ~3,300ms | $0.10/query (subscription) |
| **Google Gemini RAG** | `gemini-3-flash-preview` | `thinking_level: MINIMAL` | ~4,700ms | $0.50 in / $3 out |

### Why This Configuration?

1. **GPT-5.1 defaults to `reasoning_effort: none`** - No extended reasoning unless explicitly requested
2. **Gemini 3 Pro cannot disable thinking** - Even `thinking_level: LOW` produces 2000-4000+ thinking tokens
3. **Gemini 3 Flash with `MINIMAL`** - Closest to "no thinking" available, matches GPT-5.1's behavior

## The Problem We Solved

### Initial Unfair Configuration (Discovered January 2026)

| Provider | Model | Default Thinking | Latency | Issue |
|----------|-------|-----------------|---------|-------|
| OpenAI RAG | gpt-5.1 | **NONE** | ~8,900ms | Fair baseline |
| Gemini RAG | gemini-3-pro-preview | **HIGH** (default) | ~29,000ms | 🚨 Massive advantage |

Gemini was using 2000-4000+ thinking tokens per request while OpenAI used zero reasoning tokens.

### Testing Different Configurations

| Gemini Configuration | Latency | vs CustomGPT (3.3s) | Fairness |
|---------------------|---------|---------------------|----------|
| Pro + HIGH (default) | 29,003ms | 8.8x slower | ❌ Unfair - heavy thinking |
| Pro + LOW | 18,625ms | 5.6x slower | ⚠️ Still thinking significantly |
| **Flash + MINIMAL** | **4,720ms** | **1.4x slower** | ✅ Fair - minimal thinking |

## Implementation Details

### OpenAI GPT-5.1 Configuration

```python
# No special configuration needed - defaults to no reasoning
sampler = AuditedOpenAIRAGSampler(
    model="gpt-5.1",
    vector_store_id=vector_store_id,
    temperature=0,
    # reasoning_effort defaults to "none"
)
```

GPT-5.1 supports: `none` (default), `low`, `medium`, `high`

### Google Gemini Configuration

```python
# Use Flash with MINIMAL thinking for fair comparison
sampler = AuditedGeminiRAGSampler(
    store_name=store_name,
    model="gemini-3-flash-preview",  # Flash supports MINIMAL
    temperature=0.0,
    thinking_level="MINIMAL",  # Closest to "no thinking"
)
```

Gemini 3 thinking levels:
- **Pro**: `LOW`, `HIGH` (default) - cannot disable
- **Flash**: `MINIMAL`, `LOW`, `MEDIUM`, `HIGH` (default)

### Code Location

The configuration is set in `scripts/confidence_threshold_benchmark.py`:
- Lines 113-130: Gemini sampler setup with `thinking_level="MINIMAL"`

## Cost Comparison

| Provider | Model | Input/1M | Output/1M | Notes |
|----------|-------|----------|-----------|-------|
| OpenAI | gpt-5.1 | $1.25 | $10.00 | Reasoning tokens billed as output |
| Gemini | gemini-3-pro-preview | $2.00 | $12.00 | Thinking tokens billed as output |
| Gemini | gemini-3-flash-preview | $0.50 | $3.00 | 75% cheaper than Pro |
| CustomGPT | gpt-5.1 | $0.10/query | - | Subscription model |

## Alternative: High Reasoning Comparison

If you want to compare models WITH reasoning enabled:

```python
# OpenAI with reasoning
ChatCompletionSampler(model="gpt-5.1", reasoning_effort="low")

# Gemini with thinking
AuditedGeminiRAGSampler(model="gemini-3-pro-preview", thinking_level="LOW")
```

Note: Even at `LOW`, Gemini Pro still produces significant thinking tokens.

## Verification

To verify fair configuration, check latencies in benchmark output:
- All RAG providers should be within 2-3x of each other
- If Gemini is 5x+ slower, thinking is likely too high

## 100-Sample Benchmark Results (January 2026)

Official results with fair configuration (run_20260122_152001_873):

| Provider | Quality Score | Volume Score | Accuracy | Avg Latency | Total Cost |
|----------|---------------|--------------|----------|-------------|------------|
| **Google_Gemini_RAG** | **0.85** | **0.97** | 97.0% | 6,474ms | $0.17 |
| **CustomGPT_RAG** | 0.78 | 0.86 | 97.7% | 3,632ms | $10.00 |
| **OpenAI_RAG** | 0.60 | 0.92 | 92.0% | 13,169ms | $2.14 |
| **OpenAI_Vanilla** | -1.26 | 0.22 | 37.3% | 4,305ms | $0.04 |

### Key Findings

1. **Latency Fairness Achieved**: Gemini at 6,474ms is now 1.8x CustomGPT (was 8x with Pro+HIGH)
2. **Quality Leader**: Google Gemini RAG with 0.85 quality score (penalty-aware)
3. **Volume Leader**: Google Gemini RAG with 0.97 volume score (traditional)
4. **Cost Efficiency**: Gemini Flash at $0.17 total vs $2.14 for OpenAI RAG

### Penalty Breakdown

| Provider | Correct | Incorrect | Abstained | Penalty Count |
|----------|---------|-----------|-----------|---------------|
| Google_Gemini_RAG | 97 | 3 | 0 | 3 |
| CustomGPT_RAG | 86 | 2 | 12 | 2 |
| OpenAI_RAG | 92 | 8 | 0 | 8 |
| OpenAI_Vanilla | 22 | 37 | 41 | 37 |

### Latency Comparison (Before vs After Fair Configuration)

| Configuration | Gemini Latency | vs CustomGPT | Status |
|---------------|----------------|--------------|--------|
| **Before** (Pro + HIGH) | 29,003ms | 8.8x slower | ❌ Unfair |
| **After** (Flash + MINIMAL) | 6,474ms | 1.8x slower | ✅ Fair |

## References

- [GPT-5.1 Model | OpenAI API](https://platform.openai.com/docs/models/gpt-5.1)
- [Reasoning models | OpenAI API](https://platform.openai.com/docs/guides/reasoning)
- [Gemini thinking | Google AI for Developers](https://ai.google.dev/gemini-api/docs/thinking)
- [Gemini 3 Flash | Google Cloud](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-flash)
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing)

## Changelog

- **2026-01-22**: Added 100-sample benchmark results with fair configuration
- **2026-01-22**: Switched from Gemini 3 Pro to Gemini 3 Flash with `thinking_level="MINIMAL"` for fair comparison
- **2026-01-22**: Documented the thinking/reasoning fairness issue and resolution
- **2026-01-12**: Initial benchmark with unfair Gemini Pro HIGH configuration
