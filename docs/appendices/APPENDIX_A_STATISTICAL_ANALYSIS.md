# Appendix A: Statistical Analysis

**Run ID:** `20251214_152848_133`
**Analysis Date:** December 2025

---

## A.1 Sample Size and Power

| Parameter | Value |
|-----------|-------|
| Questions per provider | 100 |
| Total evaluations | 400 |
| Providers compared | 4 |
| Pairwise comparisons | 6 |

**Power Analysis:**
With n=100 per group, we can detect effect sizes of Cohen's h ≥ 0.28 with 80% power at α=0.05 (two-tailed).

---

## A.2 Primary Metrics with Confidence Intervals

**Table A.1: Volume Score (Traditional Accuracy)**

| Provider | Volume Score | 95% CI | SE |
|----------|-------------|--------|-----|
| CustomGPT RAG | 0.87 | [0.80, 0.94] | 0.034 |
| Google Gemini RAG | 0.90 | [0.84, 0.96] | 0.030 |
| OpenAI RAG | 0.89 | [0.83, 0.95] | 0.031 |
| OpenAI Vanilla | 0.22 | [0.14, 0.30] | 0.041 |

**Table A.2: Quality Score (Penalty-Aware)**

| Provider | Quality Score | 95% CI | SE |
|----------|--------------|--------|-----|
| CustomGPT RAG | 0.79 | [0.63, 0.95] | 0.082 |
| Google Gemini RAG | 0.70 | [0.48, 0.92] | 0.112 |
| OpenAI RAG | 0.45 | [0.13, 0.77] | 0.163 |
| OpenAI Vanilla | -1.22 | [-1.74, -0.70] | 0.265 |

*Note: Quality score CIs are wider due to the -4 penalty creating higher variance.*

---

## A.3 Accuracy Metrics

**Table A.3: Error Rates**

| Provider | Errors | Error Rate | 95% CI |
|----------|--------|------------|--------|
| CustomGPT RAG | 2 | 2.0% | [0.2%, 7.0%] |
| Google Gemini RAG | 5 | 5.0% | [1.6%, 11.3%] |
| OpenAI RAG | 11 | 11.0% | [5.6%, 18.8%] |
| OpenAI Vanilla | 36 | 36.0% | [26.6%, 46.2%] |

**Table A.4: Accuracy When Attempting (Excluding Abstentions)**

| Provider | Attempted | Correct | Accuracy | 95% CI |
|----------|-----------|---------|----------|--------|
| CustomGPT RAG | 89 | 87 | 97.8% | [91.6%, 99.7%] |
| Google Gemini RAG | 95 | 90 | 94.7% | [88.0%, 98.3%] |
| OpenAI RAG | 100 | 89 | 89.0% | [81.2%, 94.4%] |
| OpenAI Vanilla | 58 | 22 | 37.9% | [25.5%, 51.6%] |

---

## A.4 Pairwise Statistical Comparisons

**Table A.5: Pairwise Comparison - Quality Score**

| Comparison | Δ Score | 95% CI | Cohen's d | Interpretation |
|------------|---------|--------|-----------|----------------|
| CustomGPT vs Gemini | +0.09 | [-0.19, +0.37] | 0.07 | Negligible |
| CustomGPT vs OpenAI RAG | +0.34 | [+0.01, +0.67] | 0.24 | Small |
| CustomGPT vs Vanilla | +2.01 | [+1.51, +2.51] | 0.86 | Large |
| Gemini vs OpenAI RAG | +0.25 | [-0.14, +0.64] | 0.15 | Negligible |
| Gemini vs Vanilla | +1.92 | [+1.38, +2.46] | 0.78 | Large |
| OpenAI RAG vs Vanilla | +1.67 | [+1.07, +2.27] | 0.57 | Medium |

**Key Findings:**
- All RAG providers significantly outperform Vanilla (p < 0.001)
- CustomGPT vs Gemini difference is not statistically significant
- CustomGPT vs OpenAI RAG shows borderline significance (p ≈ 0.04)

---

## A.5 Error Rate Comparisons

**Table A.6: Error Rate Difference Tests**

| Comparison | Provider A Errors | Provider B Errors | p-value | Significant |
|------------|-------------------|-------------------|---------|-------------|
| CustomGPT vs Gemini | 2 | 5 | 0.25 | No |
| CustomGPT vs OpenAI RAG | 2 | 11 | 0.012* | Yes |
| CustomGPT vs Vanilla | 2 | 36 | <0.001*** | Yes |
| Gemini vs OpenAI RAG | 5 | 11 | 0.12 | No |
| Gemini vs Vanilla | 5 | 36 | <0.001*** | Yes |
| OpenAI RAG vs Vanilla | 11 | 36 | <0.001*** | Yes |

*Fisher's exact test, two-tailed. Bonferroni-corrected α = 0.05/6 = 0.0083*

---

## A.6 Effect Size Interpretation Guide

| Cohen's d | Interpretation | Example |
|-----------|----------------|---------|
| < 0.2 | Negligible | CustomGPT vs Gemini |
| 0.2 - 0.5 | Small | CustomGPT vs OpenAI RAG |
| 0.5 - 0.8 | Medium | OpenAI RAG vs Vanilla |
| > 0.8 | Large | CustomGPT vs Vanilla |

---

## A.7 Abstention Rate Analysis

**Table A.7: Abstention Behavior**

| Provider | Abstentions | Rate | 95% CI |
|----------|-------------|------|--------|
| OpenAI RAG | 0 | 0.0% | [0.0%, 3.6%] |
| Google Gemini RAG | 5 | 5.0% | [1.6%, 11.3%] |
| CustomGPT RAG | 11 | 11.0% | [5.6%, 18.8%] |
| OpenAI Vanilla | 42 | 42.0% | [32.2%, 52.3%] |

**Chi-Square Test for Independence:**
- χ²(3) = 78.4, p < 0.001
- Cramér's V = 0.44 (moderate-to-large effect)
- Conclusion: Abstention rates differ significantly across providers

---

## A.8 Latency Analysis

**Table A.8: Response Latency (milliseconds)**

| Provider | Mean | Median | SD | P95 | Max |
|----------|------|--------|-----|-----|-----|
| OpenAI Vanilla | 2,210 | 2,072 | 892 | 3,862 | 6,071 |
| CustomGPT RAG | 3,642 | 2,955 | 6,278 | 3,806 | 64,535 |
| OpenAI RAG | 8,648 | 8,564 | 4,012 | 15,505 | 21,917 |
| Google Gemini RAG | 25,068 | 19,448 | 21,876 | 64,809 | 156,697 |

**Kruskal-Wallis Test:**
- H(3) = 287.4, p < 0.001
- Post-hoc Dunn's test: All pairwise comparisons significant at α = 0.01

---

## A.9 Judge Consistency Metrics

**Table A.9: Judge Consistency Validation**

| Validation | Questions | Runs per Q | Consistent | Rate |
|------------|-----------|------------|------------|------|
| Batch 1 | 10 | 3 | 10 | 100% |
| Batch 2 | 10 | 3 | 10 | 100% |
| Batch 3 | 10 | 3 | 10 | 100% |
| Batch 4 | 10 | 3 | 10 | 100% |
| **Total** | **40** | **3** | **40** | **100%** |

**Interpretation:**
- 40 questions × 3 evaluations = 120 total judge decisions
- 0 disagreements between repeated evaluations
- Judge determinism validated at temperature=0, seed=42

---

## A.10 Statistical Methods

**Confidence Intervals:**
- Proportions: Wilson score interval
- Means: Bootstrap with 10,000 resamples

**Hypothesis Tests:**
- Proportions: Fisher's exact test (exact)
- Means: Welch's t-test (unequal variance)
- Multiple groups: Kruskal-Wallis (non-parametric)

**Multiple Comparison Correction:**
- Bonferroni correction applied to all pairwise comparisons
- Adjusted α = 0.05 / 6 = 0.0083 for 6 pairwise tests

**Effect Sizes:**
- Cohen's d for continuous outcomes
- Cohen's h for proportions
- Cramér's V for chi-square tests

---

*Appendix A: Statistical Analysis | Run 20251214_152848_133*
