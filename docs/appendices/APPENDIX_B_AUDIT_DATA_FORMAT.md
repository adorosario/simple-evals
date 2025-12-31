# Appendix B: Audit Data Format

**Version:** 2.0
**Run ID:** `20251214_152848_133`

---

## B.1 Overview

The benchmark generates complete audit trails in JSONL (JSON Lines) format. Each line is a valid JSON object, enabling streaming processing and append-only logging.

**Log Files Generated:**

| File | Records | Purpose |
|------|---------|---------|
| `provider_requests.jsonl` | 400 | All provider API calls |
| `judge_evaluations.jsonl` | 400+ | All judge grading decisions |
| `abstention_classifications.jsonl` | 400 | Intent classifications |
| `judge_consistency_validation.jsonl` | 4 | Consistency checks |
| `run_metadata.json` | 1 | Run configuration |

---

## B.2 Provider Requests Schema

**File:** `provider_requests.jsonl`

```json
{
  "timestamp": "2025-12-14T15:29:12.628200",
  "run_id": "20251214_152848_133",
  "question_id": "simpleqa_0007",
  "provider": "CustomGPT_RAG",
  "request": {
    "question": "Who crowned Reita Faria, the winner of Miss World 1966?",
    "system_message": "You are a helpful assistant..."
  },
  "response": {
    "content": "Reita Faria was crowned by Lesley Langley...",
    "raw_response": { /* provider-specific response object */ }
  },
  "metrics": {
    "latency_ms": 2925.29,
    "token_usage": {
      "prompt_tokens": 150,
      "completion_tokens": 45,
      "total_tokens": 195
    },
    "estimated_cost_usd": 0.10
  },
  "metadata": {
    "provider_type": "customgpt_rag",
    "project_id": "88141",
    "session_id": "uuid-here",
    "citation_count": 1
  }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO 8601 | Request completion time |
| `run_id` | string | Unique benchmark run identifier |
| `question_id` | string | SimpleQA question identifier |
| `provider` | string | Provider name (anonymized during eval) |
| `request.question` | string | The question asked |
| `response.content` | string | Provider's answer text |
| `metrics.latency_ms` | float | Round-trip time in milliseconds |
| `metrics.token_usage` | object | Token counts (null for CustomGPT) |
| `metrics.estimated_cost_usd` | float | Estimated API cost |

---

## B.3 Abstention Classifications Schema

**File:** `abstention_classifications.jsonl`

```json
{
  "timestamp": "2025-12-14T15:29:12.682670",
  "run_id": "20251214_152848_133",
  "question_id": "simpleqa_0002",
  "question": "What was the name of the choir that performed at the 2019 Cape Town International Jazz Festival?",
  "provider_response": "I don't know.",
  "provider_name": "Provider_01",
  "classifier": {
    "model": "gpt-5-nano",
    "classification": "abstention",
    "confidence": 0.62,
    "reasoning": "The provider explicitly says 'I don't know,' which matches an abstention signal per the guidelines."
  },
  "metadata": {
    "real_provider_name": "OpenAI_Vanilla"
  }
}
```

**Classification Values:**

| Value | Meaning |
|-------|---------|
| `attempt` | Provider attempted to answer the question |
| `abstention` | Provider declined to answer ("I don't know") |

**Confidence Interpretation:**

| Range | Interpretation |
|-------|----------------|
| 0.0 - 0.5 | Low confidence (borderline case) |
| 0.5 - 0.7 | Moderate confidence |
| 0.7 - 1.0 | High confidence |

---

## B.4 Judge Evaluations Schema

**File:** `judge_evaluations.jsonl`

```json
{
  "timestamp": "2025-12-14T15:30:15.123456",
  "run_id": "20251214_152848_133",
  "question_id": "simpleqa_0017",
  "question": "What was the full name of the first Prime Minister of the Democratic Republic of the Congo?",
  "target_answer": "Patrice Émery Lumumba",
  "provider_response": "The first Prime Minister was Patrice Émery Lumumba.",
  "provider_name": "Provider_02",
  "judge": {
    "model": "gpt-5.1",
    "grade": "A",
    "reasoning": "The predicted answer exactly matches the gold target name: Patrice Émery Lumumba. There are no contradictions.",
    "confidence": 0.95,
    "latency_ms": 1777.03
  },
  "scoring": {
    "volume_score": 1.0,
    "quality_score": 1.0,
    "penalty_applied": false,
    "threshold_used": 0.8,
    "penalty_ratio": 4.0
  },
  "metadata": {
    "real_provider_name": "CustomGPT_RAG",
    "intent_classification": "attempt"
  }
}
```

**Grade Values:**

| Grade | Meaning | Volume Score | Quality Score |
|-------|---------|--------------|---------------|
| `A` | Correct answer | +1.0 | +1.0 |
| `B` | Incorrect answer | 0.0 | -4.0 |
| `C` | Not attempted (abstention) | 0.0 | 0.0 |
| `E` | API error (excluded) | null | null |

---

## B.5 Judge Consistency Validation Schema

**File:** `judge_consistency_validation.jsonl`

```json
{
  "timestamp": "2025-12-14T15:34:10.896868",
  "run_id": "20251214_152848_133",
  "validation_type": "judge_consistency",
  "summary": {
    "total_responses_tested": 10,
    "consistent_responses": 10,
    "inconsistent_responses": 0,
    "consistency_rate": 1.0,
    "runs_per_response": 3,
    "expected_consistency_rate": 1.0,
    "consistency_gap": 0.0,
    "detailed_results": [
      {
        "question_id": "simpleqa_0017",
        "question": "What was the full name of the first PM of DRC?",
        "target": "Patrice Émery Lumumba",
        "predicted_answer": "The first PM was Patrice Émery Lumumba.",
        "evaluations": [
          {"run": 1, "grade": "A", "reasoning": "...", "latency_ms": 1777.0},
          {"run": 2, "grade": "A", "reasoning": "...", "latency_ms": 2018.1},
          {"run": 3, "grade": "A", "reasoning": "...", "latency_ms": 1871.6}
        ],
        "is_consistent": true,
        "unique_grades": ["A"],
        "grade_distribution": {"A": 3}
      }
    ]
  },
  "validation_metadata": {
    "temperature_expected": 0.0,
    "deterministic_expected": true,
    "critical_threshold": 1.0
  }
}
```

---

## B.6 Run Metadata Schema

**File:** `run_metadata.json`

```json
{
  "run_id": "20251214_152848_133",
  "start_time": "2025-12-14T15:28:48.133527",
  "end_time": "2025-12-14T15:39:48.766408",
  "status": "completed",
  "providers": [
    {
      "name": "CustomGPT_RAG",
      "config": {
        "model": "gpt-5.1",
        "temperature": 0,
        "max_tokens": 1024,
        "project_id": "88141"
      },
      "registered_at": "2025-12-14T15:28:48.255042"
    }
  ],
  "total_questions": 100,
  "completed_questions": 100,
  "errors": [],
  "results": {
    "total_providers": 4,
    "successful_evaluations": 4,
    "failed_evaluations": 0,
    "samples_per_provider": 100,
    "confidence_threshold": "80% (Conservative)",
    "expected_total_evaluations": 400,
    "actual_total_evaluations": 400,
    "evaluation_coverage_complete": true
  }
}
```

---

## B.7 Data Dictionary

### Common Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | ISO 8601 datetime with microseconds |
| `run_id` | string | Format: `YYYYMMDD_HHMMSS_mmm` |
| `question_id` | string | Format: `simpleqa_NNNN` |
| `provider_name` | string | Anonymized: `Provider_NN` |

### Metrics Fields

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `latency_ms` | float | milliseconds | API response time |
| `prompt_tokens` | int | tokens | Input token count |
| `completion_tokens` | int | tokens | Output token count |
| `estimated_cost_usd` | float | USD | Estimated API cost |

### Scoring Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `volume_score` | float | 0.0 or 1.0 | Traditional accuracy |
| `quality_score` | float | -4.0 to 1.0 | Penalty-aware score |
| `confidence` | float | 0.0 to 1.0 | Judge confidence |

---

## B.8 Example: Reading Audit Logs

**Python:**
```python
import json

with open('judge_evaluations.jsonl', 'r') as f:
    for line in f:
        record = json.loads(line)
        print(f"{record['question_id']}: {record['judge']['grade']}")
```

**jq (command line):**
```bash
# Count grades by type
jq -r '.judge.grade' judge_evaluations.jsonl | sort | uniq -c

# Filter to incorrect answers
jq -r 'select(.judge.grade == "B")' judge_evaluations.jsonl

# Calculate average latency
jq -s 'map(.judge.latency_ms) | add / length' judge_evaluations.jsonl
```

---

*Appendix B: Audit Data Format | Schema Version 2.0*
