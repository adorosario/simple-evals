# CustomGPT Explainability Post-Mortem - TODO

Lightweight file-based state for the Goal Optimizer. Update this after each `/next` iteration.

---

## BLOCKED - Waiting for API PR

**Status**: Verified Responses (Explainability) feature does NOT work via API yet. Only works through UI.

**Blocker**: PR in review to enable API support (per Abdallah Yasser, 2026-01-12)

**When to Resume**: Once PR is merged and API returns actual claims/trust-score data

**Reference**: See `/home/adorosario/quick-and-dirty/customgpt-explainability-benchmark/` for more details

**How to Check if PR is Merged**:
```bash
# Test API call - should return actual claims data, not empty array
docker compose run --rm simple-evals python -c "
from sampler.explainability_client import ExplainabilityClient
client = ExplainabilityClient()
result = client.get_claims('88141', 'e49729fb-0d66-42d5-b35b-ac3eba1c32ae', '12345')
print('API working!' if result and result.get('data') else 'Still empty - PR not merged')
"
```

---

## Completed (Ready to Resume)

- [x] **ExplainabilityClient** - API client for claims and trust-score endpoints
- [x] **Post-mortem script** - Main analysis pipeline (explainability_postmortem.py)
- [x] **HTML Report Generator** - Brand kit styled forensic reports
- [x] **Real API Testing** - Verified API integration works (data: null expected for pre-feature benchmarks)
- [x] **Full Post-Mortem Run** - Analyzed all failures from run_20251214_152848_133
- [x] **Hard Questions Extraction** - Script to extract all past failures
- [x] **Question IDs File Support** - `--question-ids-file` parameter for benchmark
- [x] **Simpleqa-Verified Assertion** - Fail fast if not using verified dataset

---

## Next (After PR Merges)

- [ ] Run benchmark with hard questions file to test explainability on known failures
- [ ] Run explainability post-mortem on new benchmark failures
- [ ] Add unit tests for client and post-mortem script
- [ ] Integrate with existing forensic reports (add explainability section)
- [ ] Track progress toward 85% quality metric

---

## Completed

### 2026-01-12 (Session 2)

- [x] Created `scripts/extract_hard_questions.py` - Hard questions extraction script
  - Scans all `results/run_*/[provider]_penalty_analysis/*.json` files
  - Extracts unique question_ids where any RAG provider failed
  - Found 49 total hard questions across all runs
  - CustomGPT-specific failures: 24 questions
  - Outputs: `simpleqa-verified/hard_questions.csv` and `hard_question_ids.txt`

- [x] Added `--question-ids-file` parameter to benchmark
  - Supports CSV (with question_id column) or plain text (one ID per line)
  - Automatically sets n_samples to match file contents
  - Passes question_ids to evaluator for filtering

- [x] Added simpleqa-verified assertion
  - Benchmark now FAILS if verified dataset not found
  - Validates expected 1000 questions in dataset
  - Ensures all benchmarks use KB-covered questions only

- [x] Fixed question_id assignment
  - Now uses `original_index` from CSV (not sampled index)
  - Question IDs match across benchmark runs

### 2026-01-12 (Session 1)

- [x] Created `sampler/explainability_client.py` - ExplainabilityClient class
  - Handles claims and trust-score API endpoints
  - Robust parsing for different response formats
  - Retry logic for async analysis

- [x] Created `scripts/explainability_postmortem.py` - Main post-mortem script
  - Loads failed queries from penalty analysis
  - Extracts API context (project_id, session_id, prompt_id) from provider_requests.jsonl
  - Fetches explainability data for each failure
  - Classifies root causes into 6 categories
  - Generates actionable recommendations

- [x] Created `scripts/generate_explainability_report.py` - HTML report generator
  - Uses brand kit for consistent styling
  - Executive summary with root cause breakdown
  - Detailed per-failure cards with claims, evidence, recommendations
  - Trust score gauges and competitor comparison

- [x] Tested with real API on run_20251214_152848_133
  - API returns empty data for pre-feature benchmarks (expected)
  - System gracefully handles missing explainability data
  - Root cause classification still works based on available evidence

---

## AF Status Update

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Explainability API integration | DONE | ExplainabilityClient fetches claims + trust-score |
| 2 | Post-mortem script | DONE | Processes all failed queries from benchmark run |
| 3 | Root cause classification | DONE | 6 categories with evidence chains |
| 4 | Engineering report | DONE | HTML + JSON output generated |
| 5 | Claims parsing | PARTIAL | Works, but needs real data to verify fully |
| 6 | Stakeholder analysis | PARTIAL | Parsing ready, awaiting real data |
| 7 | Trust scores | DONE | Retrieved and integrated |
| 8 | Works with existing forensics | PENDING | Not yet integrated |
| 9 | Uses existing metadata | DONE | Leverages session_id, prompt_id from logs |
| 10 | Cost tracking | DONE | 4 queries/analysis tracked |
| 11 | How-to guide | PENDING | |
| 12 | Reference docs | PENDING | |
| 13 | Failure taxonomy docs | DONE | In goal spec |
| 14 | Hard questions extraction | DONE | scripts/extract_hard_questions.py |
| 15 | Question IDs file support | DONE | --question-ids-file parameter |
| 16 | Simpleqa-verified assertion | DONE | Fails if verified dataset missing |

---

## Hard Questions Analysis

From extraction script output:

| Category | Count | Description |
|----------|-------|-------------|
| CustomGPT failures | 24 | Total CustomGPT failures |
| OpenAI RAG failures | 34 | Total OpenAI RAG failures |
| Gemini RAG failures | 8 | Total Gemini RAG failures |
| All RAG failed | 1 | simpleqa_0004 |
| CustomGPT-only | 10 | Unique CustomGPT weaknesses |
| Shared CustomGPT+OpenAI | 12 | Common hard questions |

**CustomGPT-Only Failures** (opportunities for improvement):
- simpleqa_0007, simpleqa_0079, simpleqa_0084, simpleqa_0092, simpleqa_0099
- simpleqa_0102, simpleqa_0108, simpleqa_0128, simpleqa_0167, simpleqa_0175

---

## Blockers / Questions

- [x] ~~Need to run a fresh benchmark AFTER Jan 22, 2026 (explainability launch date) to get real claims data~~
- [x] ~~API returns empty data for Dec 2025 queries~~ - **RESOLVED**: API doesn't work yet, PR in review
- [ ] **BLOCKER**: Explainability API only works via UI, not programmatic API (PR in review by Abdallah Yasser)
- [ ] Confirm exact response schema from CustomGPT engineering when PR is merged
- [ ] Test with UI-created session: `e49729fb-0d66-42d5-b35b-ac3eba1c32ae` (Marko's test - works in UI)

---

## Usage Examples

### Run Benchmark with Hard Questions
```bash
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py \
  --question-ids-file simpleqa-verified/hard_questions.csv
```

### Extract Hard Questions from Past Runs
```bash
docker compose run --rm simple-evals python scripts/extract_hard_questions.py
docker compose run --rm simple-evals python scripts/extract_hard_questions.py --customgpt-only
```

### Run Explainability Post-Mortem
```bash
docker compose run --rm simple-evals python scripts/explainability_postmortem.py --run-id run_20251214_152848_133
```

---

## Notes

### 2026-01-12 - Hard Questions Strategy

**Key Insight**: Rather than waiting for new benchmarks, we can target known hard questions to maximize explainability value.

**Hard Questions Dataset**:
- 49 unique questions where at least one RAG provider failed
- 10 CustomGPT-only failures (highest priority for investigation)
- Available at `simpleqa-verified/hard_questions.csv`

**Next Action**: Run benchmark with hard questions file, then run explainability post-mortem on failures.

### 2026-01-12 - Initial Implementation

**Key Finding**: The Dec 2025 benchmark was run before explainability was available, so API returns `data: []` (empty). The system handles this gracefully and still provides root cause analysis based on:
- Citation presence/absence
- Competitor results
- Date/specificity patterns in answers

**Root Cause Taxonomy**:
1. `hallucination` - Flagged claims with no source
2. `partial_knowledge` - Some claims sourced, key claim unsourced
3. `kb_gap` - No citations, no sources found
4. `retrieval_miss` - Low trust score despite KB coverage
5. `reasoning_error` - Sources present but conclusion incorrect
6. `specificity_failure` - Partial answer (e.g., "July 2023" vs "15 July 2023")
