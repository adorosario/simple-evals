# Architecture Overview

System design for the SimpleQA RAG Benchmark Framework v3.0.0.

## System Diagram

```
+------------------------------------------------------------------+
|                    confidence_threshold_benchmark.py              |
|                         (Entry Point)                             |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                      Provider Samplers                            |
|  +---------------+  +---------------+  +---------------+          |
|  | CustomGPT_RAG |  | OpenAI_RAG    |  | Gemini_RAG    |          |
|  | (gpt-5.1)     |  | (gpt-5.1)     |  | (gemini-3-pro)|          |
|  +---------------+  +---------------+  +---------------+          |
|                           +                                       |
|                   +---------------+                               |
|                   | OpenAI_Vanilla|                               |
|                   | (baseline)    |                               |
|                   +---------------+                               |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|              confidence_threshold_simpleqa_eval.py                |
|                    (Evaluation Pipeline)                          |
|                                                                   |
|  1. Intent Classification (GPT-5-nano)                           |
|     - Classifies: ATTEMPT vs ABSTENTION                          |
|                                                                   |
|  2. Judge Grading (GPT-5.1)                                      |
|     - Grades: A (correct) or B (incorrect)                       |
|     - Provides reasoning explanation                              |
|                                                                   |
|  3. Penalty Calculation                                          |
|     - Volume Score: Correct=+1, Wrong=0, Abstain=0               |
|     - Quality Score: Correct=+1, Wrong=-4, Abstain=0             |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                     Report Generation                             |
|  +---------------+  +---------------+  +---------------+          |
|  | Brand Kit     |  | HTML Reports  |  | Audit Logs   |          |
|  | (brand_kit.py)|  | (dashboards)  |  | (JSONL)      |          |
|  +---------------+  +---------------+  +---------------+          |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                    results/run_YYYYMMDD_HHMMSS/                   |
|  - index.html (dashboard hub)                                     |
|  - quality_benchmark_report.html                                  |
|  - statistical_analysis_report.html                               |
|  - *.jsonl (audit logs)                                          |
+------------------------------------------------------------------+
```

## Key Components

### 1. Entry Point

**File:** `scripts/confidence_threshold_benchmark.py`

Orchestrates the entire benchmark:
- Initializes all provider samplers
- Runs evaluations in parallel (ThreadPoolExecutor)
- Generates reports and audit logs
- Handles errors and retries

```bash
# Command-line options
--debug          # 5 questions (quick test)
--examples N     # Custom question count
--max-workers N  # Parallel threads (default: 8)
--flex-tier      # Use GPT-5 Flex tier (50% cost savings)
--dry-run        # Validate config without running
```

### 2. Sampler Architecture

**Directory:** `sampler/`

All samplers inherit from `AuditedSamplerBase` and implement:

```python
class AuditedSamplerBase:
    def __call__(self, messages, question_id=None, return_metrics=False):
        """
        Args:
            messages: List of chat messages
            question_id: For audit logging
            return_metrics: If True, return (response, metrics) tuple

        Returns:
            response: str (or tuple if return_metrics=True)
            metrics: {latency_ms, token_usage, estimated_cost_usd}
        """
```

| Sampler | Provider | Model | Rate Limit |
|---------|----------|-------|------------|
| `AuditedCustomGPTSampler` | CustomGPT | gpt-5.1 | 5 concurrent |
| `AuditedOpenAIRAGSampler` | OpenAI | gpt-5.1 | Unlimited |
| `AuditedOpenAIVanillaSampler` | OpenAI | gpt-5.1 | Unlimited |
| `AuditedGeminiRAGSampler` | Google | gemini-3-pro | 5 concurrent |

### 3. Evaluation Pipeline

**File:** `confidence_threshold_simpleqa_eval.py`

Three-stage evaluation:

#### Stage 1: Intent Classification
```python
def classify_response_intent(response: str) -> str:
    """
    Uses GPT-5-nano to classify response as:
    - "ATTEMPT": Provider attempted to answer
    - "ABSTENTION": Provider said "I don't know"
    """
```

#### Stage 2: Judge Grading
```python
def grade_sample_with_explanation(question, target, response) -> dict:
    """
    Uses GPT-5.1 to grade response:
    - Grade A: Correct answer
    - Grade B: Incorrect answer
    - Includes reasoning explanation
    """
```

#### Stage 3: Penalty Calculation
```python
def calculate_scores(grade, intent):
    """
    Volume Score (traditional):
        Correct = +1, Wrong = 0, Abstain = 0

    Quality Score (penalty-aware, 80% threshold):
        Correct = +1, Wrong = -4, Abstain = 0
    """
```

### 4. Scoring System

Based on OpenAI's ["Why Language Models Hallucinate"](https://openai.com/index/why-language-models-hallucinate/) research.

| Response Type | Volume Score | Quality Score |
|--------------|--------------|---------------|
| Correct (A) | +1 | +1 |
| Incorrect (B) | 0 | **-4** |
| Abstention ("I don't know") | 0 | 0 |

**Why penalty-aware scoring?**
- Traditional scoring rewards guessing (wrong = 0 penalty)
- Penalty scoring rewards appropriate uncertainty
- Abstaining is better than being confidently wrong
- 80% confidence threshold filters borderline judge decisions

### 5. Audit Logging

**File:** `audit_logger.py`

Complete traceability with 6 log file types:

| File | Contents |
|------|----------|
| `provider_requests.jsonl` | All provider API calls with latency, tokens, cost |
| `judge_evaluations.jsonl` | All judge decisions with reasoning |
| `abstention_classifications.jsonl` | Intent classification decisions |
| `judge_validation_overrides.jsonl` | Audit flags for consistency |
| `judge_consistency_validation.jsonl` | Judge determinism checks |
| `run_metadata.json` | Run configuration and summary |

Example log entry:
```json
{
  "timestamp": "2025-12-21T14:30:22.123Z",
  "provider": "CustomGPT",
  "question_id": "simpleqa_0001",
  "question": "What is the capital of France?",
  "response": {"content": "Paris", "latency_ms": 1234},
  "metadata": {"token_usage": {"total": 150}, "estimated_cost_usd": 0.10}
}
```

### 6. Report Generation

**Files:**
- `brand_kit.py` - Apple-inspired CSS design system
- `scripts/report_generators.py` - HTML report templates
- `scripts/generate_main_dashboard.py` - Hub page
- `leaderboard_generator.py` - Publication-ready tables

Reports generated:
- `index.html` - Main dashboard with navigation
- `quality_benchmark_report.html` - Provider comparison
- `statistical_analysis_report.html` - Academic statistics
- `<provider>_penalty_analysis/` - Forensic reports for failures

### 7. Pricing System

**File:** `pricing_config.py`

```python
MODEL_PRICING = {
    # Token-based (per million tokens)
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},

    # Per-query
    "customgpt": {"per_query": 0.10},
}
```

## Data Flow

```
SimpleQA-Verified Dataset (1,000 questions)
              |
              v
    +-------------------+
    | Load N questions  |
    +-------------------+
              |
    +---------+---------+---------+
    |         |         |         |
    v         v         v         v
CustomGPT  OpenAI   OpenAI   Gemini
   RAG      RAG     Vanilla    RAG
    |         |         |         |
    +---------+---------+---------+
              |
              v
    +-------------------+
    | Intent Classifier |
    | (GPT-5-nano)      |
    +-------------------+
              |
              v
    +-------------------+
    | Judge Grading     |
    | (GPT-5.1)         |
    +-------------------+
              |
              v
    +-------------------+
    | Penalty Scoring   |
    | (80% threshold)   |
    +-------------------+
              |
              v
    +-------------------+
    | Report Generation |
    +-------------------+
              |
              v
    results/run_YYYYMMDD/
```

## Results Directory Structure

```
results/run_YYYYMMDD_HHMMSS_mmm/
|
+-- index.html                        # Main dashboard hub
+-- quality_benchmark_report_*.html   # Provider comparison
+-- statistical_analysis_run_*.html   # Academic statistics
+-- quality_benchmark_results.json    # Machine-readable results
|
+-- provider_requests.jsonl           # Audit: all provider API calls
+-- judge_evaluations.jsonl           # Audit: all judge decisions
+-- abstention_classifications.jsonl  # Audit: intent classifications
+-- judge_consistency_validation.jsonl
+-- run_metadata.json                 # Run configuration
|
+-- customgpt_penalty_analysis/       # Forensics (if penalties)
|   +-- forensic_dashboard.html
|   +-- question_reports/
|   |   +-- simpleqa_0001.html
|   |   +-- ...
|   +-- forensic_data.json
|
+-- openai_rag_penalty_analysis/      # Forensics (if penalties)
    +-- ...
```

## Key Design Patterns

### Blind Evaluation
Providers are anonymized during judging to prevent bias. Mapping revealed after analysis.

### Thread-Safe Metrics
Samplers return metrics atomically to avoid race conditions in parallel execution.

```python
response, metrics = sampler(messages, return_metrics=True)
# metrics captured at call time, not after
```

### API Error vs Abstention
- **API Errors** (timeouts, rate limits): Excluded from scoring
- **Abstentions** ("I don't know"): Score = 0 (not penalized)

### Post-hoc Evaluation
Providers respond naturally without knowledge of confidence thresholds. The framework applies thresholds during grading, not generation.

---

*SimpleQA RAG Benchmark Framework v3.0.0*
