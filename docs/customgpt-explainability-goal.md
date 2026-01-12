# GoalSpec: CustomGPT Explainability Post-Mortem System

## Objective (North Star)

Build an **automated post-mortem investigation system** that uses CustomGPT's new Explainability API to analyze every failed query from the SimpleQA benchmark, producing actionable engineering reports that identify root causes of failures and guide improvements toward **85% quality metric**.

> Transform opaque failures into transparent, actionable insights. Know *why* CustomGPT failed, not just *that* it failed.

---

## Team

| Role | Name | GitHub | Slack |
|------|------|--------|-------|
| CEO / Product Owner | Alden Do Rosario | @adorosario | @Alden Do Rosario |
| Engineering (CustomGPT) | Marko | - | @Marko |
| Engineering (CustomGPT) | Andrew | - | @Andrew |
| Engineering (CustomGPT) | Abdallah Yasser | - | @Abdallah Yasser |

---

## Roadmap

### Phase 1: Explainability Integration (Immediate)

**Goal**: Integrate the Explainability API endpoints into the post-mortem pipeline.

**Deliverables**:
- Script to fetch claims and trust scores for failed queries
- Parse and structure explainability data into forensic reports
- Handle async analysis (may need polling/retry logic)

**Success Criteria**:
- Can retrieve claims breakdown for any failed query using stored `project_id`, `session_id`, `prompt_id`
- Claims data includes: claim text, source attribution, flagged status
- Trust scores retrievable and parsed

### Phase 2: Root Cause Classification

**Goal**: Classify failures into actionable categories using explainability data.

**Failure Taxonomy**:
| Category | Description | Explainability Signal |
|----------|-------------|----------------------|
| **Hallucination** | Claim made without KB source | Flagged claims with no source |
| **Partial Knowledge** | KB has related but incomplete info | Some claims sourced, key claim unsourced |
| **KB Gap** | Information doesn't exist in KB | No citations, no sources found |
| **Retrieval Miss** | KB has info but wasn't retrieved | Low trust score despite KB coverage |
| **Reasoning Error** | Retrieved right info, concluded wrongly | Sources present but conclusion incorrect |
| **Specificity Failure** | Got general answer, needed specific | Partial answer (e.g., "July 2023" vs "15 July 2023") |

**Success Criteria**:
- Each failed query classified into 1+ categories
- Classification backed by explainability evidence

### Phase 3: Engineering Report Generation

**Goal**: Produce actionable reports for CustomGPT engineering team.

**Report Structure**:
1. **Executive Summary**: Total failures, category breakdown, top improvement opportunities
2. **Per-Failure Analysis**: Question, expected answer, actual answer, claims breakdown, trust score, root cause, stakeholder flags
3. **Pattern Analysis**: Common failure patterns, KB coverage gaps, retrieval quality issues
4. **Recommendations**: Prioritized list of improvements with expected impact

**Success Criteria**:
- HTML + JSON reports generated automatically
- Each failure has full audit trail from question to explainability data
- Patterns aggregated across failures

### Phase 4: Quality Improvement Tracking

**Goal**: Track progress toward 85% quality metric.

**Metrics**:
- Current quality score vs target (85%)
- Failure rate by category over time
- KB coverage improvement tracking
- Retrieval precision trends

---

## Acceptance Function (AF) - "Done?"

AF is true only when **ALL** criteria are satisfied:

### Core Functionality
- [ ] **#1** Explainability API integration: Can fetch claims + trust-score for any message
- [ ] **#2** Post-mortem script: Processes all failed queries from benchmark run
- [ ] **#3** Root cause classification: Each failure categorized with evidence
- [ ] **#4** Engineering report: HTML + JSON output with full audit trail

### Data Quality
- [ ] **#5** Claims parsing: All claim fields extracted (text, source, flagged status)
- [ ] **#6** Stakeholder analysis: 6 perspectives captured (End User, Security IT, Risk Compliance, Legal Compliance, PR, Executive)
- [ ] **#7** Trust scores: Numerical scores retrieved and integrated

### Integration
- [ ] **#8** Works with existing forensic reports: Enhances, doesn't replace
- [ ] **#9** Uses existing metadata: Leverages stored session_id, prompt_id from provider_requests.jsonl
- [ ] **#10** Cost tracking: 4 queries/analysis cost tracked

### Documentation
- [ ] **#11** How-to guide for running post-mortem analysis
- [ ] **#12** Reference documentation for report schema
- [ ] **#13** Explanation of failure taxonomy

---

## Quality Gates

- [ ] **Reviewer pass**: Report schema reviewed for completeness
- [ ] **Real testing**: Run on actual benchmark failures, not mocks
- [ ] **Engineering validation**: CustomGPT team confirms reports are actionable
- [ ] **Critic verification**: Independent verification that insights match actual failures

---

## Constraints

### API Constraints
- Explainability analysis costs **4 queries per message**
- Analysis runs **asynchronously in background** - may need polling
- Uses **Claude 4.5 Sonnet** internally (other models proved insufficient)

### Data Constraints
- Must use existing forensic data: `provider_requests.jsonl`, `judge_evaluations.jsonl`
- Explainability only available for messages with valid `project_id`, `session_id`, `prompt_id`

### Integration Constraints
- Enhance existing forensic pipeline, don't replace
- Maintain brand kit consistency for reports

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration mode | **On-demand only** | Run separately using stored session/prompt IDs - more flexible, cost-controlled |
| Report depth | **Comprehensive forensics** | Full claims breakdown, all 6 stakeholder perspectives, detailed evidence chains |
| API approach | **Real API testing** | Direct integration with CustomGPT explainability endpoints |

---

## Default Commands

| Command | Usage |
|---------|-------|
| Run benchmark | `docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug` |
| Run post-mortem | `docker compose run --rm simple-evals python scripts/explainability_postmortem.py --run-id <run_id>` |
| Single query analysis | `docker compose run --rm simple-evals python scripts/explainability_postmortem.py --question-id simpleqa_0099` |
| Test | `docker compose run --rm simple-evals pytest tests/test_explainability.py -v` |

---

## Technical Design

### Existing Data Available (per failed query)

From `provider_requests.jsonl`:
```json
{
  "provider": "CustomGPT_RAG",
  "question_id": "simpleqa_0099",
  "metadata": {
    "project_id": "81643",
    "session_id": "9c01ad8e-2a49-49b5-a33c-c8775116b1b6",
    "prompt_id": 10631255,
    "citations": [73558806]
  }
}
```

### New Data to Fetch

**Claims Endpoint**: `GET /api/v1/projects/{projectId}/conversations/{sessionId}/messages/{promptId}/claims`
- Claims made by the agent
- Source attribution per claim
- Flagged claims (no source found)
- Stakeholder assessments (6 perspectives)

**Trust Score Endpoint**: `GET /api/v1/projects/{projectId}/conversations/{sessionId}/messages/{promptId}/trust-score`
- Numerical trust score
- Confidence metrics

### Output Schema (Comprehensive Forensics)

```json
{
  "question_id": "simpleqa_0099",
  "question": "On what date was T. M. Selvaganapathy acquitted?",
  "target_answer": "4 December 2001",
  "customgpt_answer": "10 June 1999",
  "grade": "INCORRECT",
  "judge_reasoning": "Gold target date is 4 December 2001. Predicted is 10 June 1999...",
  "judge_confidence": 1.0,

  "api_context": {
    "project_id": "81643",
    "session_id": "9c01ad8e-2a49-49b5-a33c-c8775116b1b6",
    "prompt_id": 10631255,
    "citations_returned": [73558806],
    "latency_ms": 4455.68
  },

  "explainability": {
    "claims": [
      {
        "claim_id": 1,
        "text": "T. M. Selvaganapathy was acquitted by the High Court",
        "source": {"citation_id": 73558806, "snippet": "...acquitted by High Court..."},
        "flagged": false,
        "confidence": 0.95
      },
      {
        "claim_id": 2,
        "text": "The acquittal occurred on 10 June 1999",
        "source": null,
        "flagged": true,
        "confidence": 0.20
      }
    ],
    "trust_score": {
      "overall": 0.35,
      "sourced_claims_ratio": 0.5,
      "flagged_claims_count": 1
    },
    "stakeholder_analysis": {
      "end_user": {
        "status": "CONCERN",
        "assessment": "User may receive incorrect date information, impacting trust",
        "recommendations": ["Clarify uncertainty about specific date"]
      },
      "security_it": {
        "status": "OK",
        "assessment": "No security implications identified",
        "recommendations": []
      },
      "risk_compliance": {
        "status": "FLAG",
        "assessment": "Unverified factual claim could expose organization to accuracy complaints",
        "recommendations": ["Verify date against authoritative sources"]
      },
      "legal_compliance": {
        "status": "FLAG",
        "assessment": "Incorrect legal dates could have compliance implications",
        "recommendations": ["Add disclaimer about date uncertainty"]
      },
      "public_relations": {
        "status": "CONCERN",
        "assessment": "Factual errors damage credibility",
        "recommendations": ["Review KB coverage for legal/political topics"]
      },
      "executive_leadership": {
        "status": "FLAG",
        "assessment": "Pattern of date inaccuracies could indicate systemic retrieval issues",
        "recommendations": ["Audit KB coverage for temporal data"]
      }
    },
    "overall_status": "FLAGGED",
    "analysis_cost_queries": 4
  },

  "root_cause": {
    "primary_category": "hallucination",
    "secondary_categories": ["specificity_failure"],
    "evidence_chain": [
      "Claim 2 (date) flagged - no source attribution",
      "KB citation 73558806 contains 'acquitted' but not specific date",
      "Model fabricated specific date without KB support"
    ],
    "confidence": 0.90
  },

  "recommendations": {
    "kb_remediation": "Add Wikipedia article or court records with specific acquittal date",
    "retrieval_improvement": "Enhance date extraction from legal documents",
    "response_quality": "Consider abstaining when specific dates not in KB",
    "priority": "HIGH",
    "expected_impact": "Prevent similar date hallucinations in legal/political queries"
  },

  "competitor_context": {
    "OpenAI_RAG": {"answer": "4 December 2001", "grade": "CORRECT"},
    "Google_Gemini_RAG": {"answer": "December 4, 2001", "grade": "CORRECT"}
  }
}
```

---

## Implementation Plan

### Step 1: Create Explainability Client
- [ ] Build `ExplainabilityClient` class with methods for claims and trust-score endpoints
- [ ] Handle authentication (Bearer token)
- [ ] Implement retry logic for async analysis
- [ ] Add cost tracking (4 queries per analysis)

### Step 2: Create Post-Mortem Pipeline
- [ ] Read failed queries from `customgpt_penalty_analysis*.json`
- [ ] Extract `project_id`, `session_id`, `prompt_id` from `provider_requests.jsonl`
- [ ] Fetch explainability data for each failure
- [ ] Classify root cause based on claims/trust analysis

### Step 3: Generate Engineering Reports
- [ ] Create HTML report with brand kit styling
- [ ] Include per-failure breakdowns with claims visualization
- [ ] Aggregate patterns and recommendations
- [ ] Export JSON for programmatic analysis

### Step 4: Integrate with Existing Pipeline
- [ ] Add optional `--explainability` flag to penalty analyzer
- [ ] Link from existing forensic reports to explainability analysis
- [ ] Update dashboard to include explainability insights

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| CustomGPT Quality Score | ~75-80% | 85% |
| Failures with root cause | 0% | 100% |
| Actionable recommendations | 0 | 1 per failure pattern |
| Time to analyze failures | Manual | <5 min automated |

---

## Non-Goals (Phase 2+)

- **Async quality improvement**: Using explainability to improve responses before delivery (future "ninja move")
- **Real-time explainability**: Integrating into live benchmark runs
- **Automated KB remediation**: Auto-adding missing sources to KB
- **Multi-provider comparison**: Applying explainability to non-CustomGPT providers

---

## Files to Create/Modify

### New Files
1. `scripts/explainability_postmortem.py` - Main post-mortem script
2. `sampler/explainability_client.py` - API client for explainability endpoints
3. `scripts/generate_explainability_report.py` - Report generator
4. `tests/test_explainability.py` - Unit tests

### Files to Modify
1. `scripts/generate_universal_forensics.py` - Add explainability section
2. `scripts/universal_penalty_analyzer.py` - Add optional explainability enrichment

---

## Verification Plan

1. **Unit Tests**: Mock explainability API responses, verify parsing
2. **Integration Test**: Fetch real explainability for 1 failed query
3. **Full Run**: Process all failures from most recent benchmark
4. **Engineering Review**: CustomGPT team reviews report actionability
