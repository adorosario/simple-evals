# RAG Benchmark: Executive Summary

**Evaluation Date:** December 2025
**Providers Tested:** CustomGPT RAG, OpenAI RAG, Google Gemini RAG, OpenAI Vanilla (baseline)
**Questions Evaluated:** 100 factual queries from SimpleQA-Verified

---

## Key Findings

### 1. RAG Eliminates 94% of Hallucination Errors

| Metric | Vanilla LLM | Best RAG | Improvement |
|--------|-------------|----------|-------------|
| Error Rate | 36% | 2% (CustomGPT) | **94% reduction** |
| Accuracy When Attempting | 37.9% | 97.8% (CustomGPT) | **+60 points** |

**Bottom Line:** RAG technology delivers production-ready accuracy for enterprise applications.

---

### 2. Provider Leaderboard

| Rank | Provider | Quality Score | Error Rate | Cost/Query |
|------|----------|---------------|------------|------------|
| 1 | **CustomGPT RAG** | **0.79** | **2%** | $0.10 |
| 2 | Google Gemini RAG | 0.70 | 5% | $0.0006 |
| 3 | OpenAI RAG | 0.45 | 11% | $0.021 |
| 4 | OpenAI Vanilla | -1.22 | 36% | $0.0004 |

*Quality Score uses penalty-aware evaluation (Correct=+1, Wrong=-4, Abstain=0)*

---

### 3. Recommendation by Use Case

| Use Case | Recommended Provider | Rationale |
|----------|---------------------|-----------|
| **High-stakes** (medical, legal, finance) | CustomGPT RAG | Lowest error rate (2%), best calibration |
| **Cost-optimized** (high volume, low stakes) | Gemini RAG | Best quality/cost ratio ($0.0006/query) |
| **Balanced** (moderate stakes, controlled budget) | OpenAI RAG | Strong accuracy (89%) at mid-tier pricing |
| **Latency-critical** (real-time) | OpenAI Vanilla | Fastest response (2.2s avg) |

---

## Why This Evaluation Matters

Traditional accuracy metrics reward guessing over uncertainty. A system that says "I don't know" is penalized the same as one that gives a wrong answer.

**Our evaluation penalizes confident errors 4:1.** This reveals which systems are truly reliable vs. which are overconfident.

**Key Insight:** OpenAI RAG has the highest raw correct answers (89) but the third-best quality score (0.45) because it never abstains—producing 11 confident errors.

---

## Cost-Benefit Analysis

### When Error Costs Exceed $3.31

CustomGPT RAG costs more per query ($0.10) but has fewer errors (2 vs 5 for Gemini).

**Break-even calculation:**
- If each error costs >$3.31 in business impact, CustomGPT is more cost-effective
- For customer-facing products, regulatory compliance, or liability-sensitive applications, this threshold is easily exceeded

### 100-Query Benchmark Costs

| Provider | API Cost | Error-Adjusted Cost* |
|----------|----------|---------------------|
| CustomGPT RAG | $10.00 | $20.00 |
| Gemini RAG | $0.06 | $25.06 |
| OpenAI RAG | $2.10 | $57.10 |
| OpenAI Vanilla | $0.04 | $180.04 |

*Assuming $5 business cost per error*

---

## Technical Validation

| Validation Check | Result |
|------------------|--------|
| Judge Consistency | 100% (40/40 samples identical across 3 re-evaluations) |
| Dataset Coverage | 100% (100/100 questions evaluated per provider) |
| Blind Evaluation | Providers anonymized during judging |
| Audit Trail | Complete logging of all 400 API calls and judge decisions |

---

## Action Items

### For Technical Teams

1. **Adopt penalty-aware evaluation** for RAG deployments
2. **Configure abstention behavior** in RAG prompts ("If unsure, say 'I don't know'")
3. **Monitor accuracy-when-attempting** as primary quality metric

### For Decision Makers

1. **CustomGPT RAG is recommended** for quality-critical deployments
2. **Factor in error costs** when evaluating provider pricing
3. **Request audit logs** from vendors for compliance

---

## Full Documentation

| Document | Purpose |
|----------|---------|
| [Technical Report](RAG_BENCHMARK_TECHNICAL_REPORT.md) | Complete methodology and results |
| [Blog Post](RAG_BENCHMARK_BLOG_POST.md) | Accessible summary |
| [Statistical Analysis](appendices/APPENDIX_A_STATISTICAL_ANALYSIS.md) | Confidence intervals, significance tests |
| [Cost Analysis](appendices/APPENDIX_E_COST_ANALYSIS.md) | Detailed pricing breakdown |
| [Reproducibility](appendices/APPENDIX_D_REPRODUCIBILITY.md) | How to run the benchmark |

---

**Contact:** CustomGPT.ai Research
**Run ID:** `20251214_152848_133`

---

*This evaluation implements OpenAI's confidence threshold framework from "Why Language Models Hallucinate" (arXiv:2509.04664v1). All data and methodology are publicly available for independent verification.*
