# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### CRITICAL: Docker-Only Environment
**ALL PYTHON COMMANDS MUST RUN IN DOCKER** - The host system has no Python. Running Python directly will fail.

**Primary Benchmark (Confidence Threshold Framework):**
```bash
# Quick debug run (5 questions)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug

# Standard benchmark (10 questions per provider)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py

# Full benchmark with custom question count
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --examples 100

# Dry run to validate configuration
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --dry-run

# Use GPT-5 Flex tier for judge (50% cost savings, slower)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --flex-tier
```

**Running Tests:**
```bash
docker compose run --rm simple-evals python -m pytest <test_file> -v
docker compose run --rm simple-evals python -m pytest tests/ -v  # All tests
```

**Running Scripts:**
```bash
docker compose run --rm simple-evals python scripts/<script_name>.py
```

**Knowledge Base Management:**
```bash
# Download pre-built knowledge base (RECOMMENDED)
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py

# Upload to OpenAI vector store
docker compose run --rm simple-evals python scripts/robust_upload_knowledge_base.py knowledge_base_full --store-name "MyStore"
```

**Services:**
```bash
docker compose up api-server    # FastAPI server on port 8000
docker compose run --rm shell   # Interactive shell
```

**Environment Setup:** Copy `.env.example` to `.env` and add API keys.

## Architecture

### Core Evaluation Framework

The benchmark implements OpenAI's "Why Language Models Hallucinate" research (arXiv:2509.04664v1) using penalty-aware scoring:

```
confidence_threshold_benchmark.py          # Entry point, orchestrates parallel provider evaluation
    └── confidence_threshold_simpleqa_eval.py  # Core evaluation logic
        ├── classify_response_intent()     # GPT-5-nano classifies abstention vs attempt
        ├── grade_sample_with_explanation() # GPT-5.1 judge grades A/B with reasoning
        └── validate_judge_reasoning_structured() # GPT-5-nano validates consistency
```

**Scoring System (80% Confidence Threshold):**
- **Volume Score (Traditional):** Correct=+1, Wrong=0, Abstain=0
- **Quality Score (Penalty-Aware):** Correct=+1, Wrong=-4, Abstain=0

### Sampler Architecture

Samplers in `sampler/` implement `SamplerBase` from `custom_types.py`:

**Audited Samplers (Production):**
- `audited_customgpt_sampler.py` - CustomGPT RAG with audit logging
- `audited_openai_rag_sampler.py` - OpenAI vector store file search
- `audited_openai_vanilla_sampler.py` - OpenAI without RAG (baseline)
- `audited_gemini_rag_sampler.py` - Google Gemini File Search

**Base Classes:**
- `audited_sampler_base.py` - Adds audit logging, latency tracking, cost calculation
- `chat_completion_sampler.py` - Base OpenAI API interface

**Thread-Safe Metrics:** Samplers support `return_metrics=True` for atomic metric capture:
```python
response, metrics = sampler(messages, question_id=qid, return_metrics=True)
# metrics = {"latency_ms": float, "token_usage": dict, "estimated_cost_usd": float}
```

### Audit System

Complete traceability via `audit_logger.py`:
- `provider_requests.jsonl` - All provider API calls with latency/tokens/cost
- `judge_evaluations.jsonl` - All GPT-5.1 judge decisions with reasoning
- `abstention_classifications.jsonl` - GPT-5-nano intent classifications
- `judge_validation_overrides.jsonl` - Audit flags (no grade overrides)
- `run_metadata.json` - Run configuration and summary

### Report Generation

Brand kit system (`brand_kit.py`) provides consistent styling:
- `scripts/report_generators.py` - Quality benchmark + statistical analysis HTML
- `scripts/generate_main_dashboard.py` - Hub page linking all reports
- `scripts/generate_universal_forensics.py` - Penalty case analysis
- `leaderboard_generator.py` - Publication-ready leaderboards

### Pricing & Cost Tracking

`pricing_config.py` tracks costs per model:
- Token-based: GPT-5.1 ($1.25/$10 per M tokens), Gemini 3 Pro ($2/$12)
- Per-query: CustomGPT ($0.10/query subscription addon)

### Dataset

Uses `simpleqa-verified/simpleqa_verified.csv` - 1000 curated questions with confirmed KB coverage.

### Environment Variables

**Required:**
- `OPENAI_API_KEY` - OpenAI models, vector stores, LLM-As-A-Judge
- `OPENAI_VECTOR_STORE_ID` - Pre-built vector store for OpenAI RAG

**Provider-Specific:**
- `CUSTOMGPT_API_KEY`, `CUSTOMGPT_PROJECT` - CustomGPT RAG
- `GEMINI_API_KEY`, `GOOGLE_FILE_SEARCH_STORE_NAME` - Google Gemini RAG

**Optional:**
- `ANTHROPIC_API_KEY` - Claude models
- `SCRAPINGBEE_API_KEY` - Web content extraction for KB building

### Results Structure

```
results/run_YYYYMMDD_HHMMSS_mmm/
├── index.html                    # Main dashboard hub
├── quality_benchmark_report.html # Provider comparison
├── statistical_analysis_report.html
├── quality_benchmark_results.json
├── provider_requests.jsonl       # Audit: all API calls
├── judge_evaluations.jsonl       # Audit: all grades
├── abstention_classifications.jsonl
└── <provider>_penalty_analysis/  # Forensic reports for wrong answers
```

### Key Patterns

- **Blind Evaluation:** Providers anonymized during judging, revealed after
- **Parallel Execution:** Provider-level + question-level parallelism
- **API Error vs Abstention:** Technical failures excluded from scoring; abstentions score 0
- **Judge Consistency Validation:** Re-evaluates sample responses to detect non-determinism
- **Post-hoc Evaluation:** Providers respond naturally; confidence framework applied during grading