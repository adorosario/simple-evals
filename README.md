# SimpleQA RAG Benchmark

A confidence threshold RAG benchmark framework for evaluating factual question-answering accuracy across multiple providers. Implements OpenAI's penalty-aware scoring methodology from ["Why Language Models Hallucinate"](https://openai.com/index/why-language-models-hallucinate/) (arXiv:2509.04664v1).

**Version:** 3.0.0

## Quick Start

```bash
# 1. Clone and configure
git clone <repo>
cd simple-evals
cp .env.example .env
# Edit .env with your API keys (see Environment Variables below)

# 2. Run benchmark (5-question debug mode)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug

# 3. Open results
# Results saved to results/run_YYYYMMDD_HHMMSS/index.html
```

## Benchmark Results

100-sample benchmark using SimpleQA-Verified dataset (December 2025):

| Provider | Accuracy | Avg Latency | Cost/Query | Quality Score |
|----------|----------|-------------|------------|---------------|
| CustomGPT_RAG | **97.8%** | 3,642ms | $0.10 | Best |
| Google_Gemini_RAG | 94.7% | 25,068ms | $0.007 | High |
| OpenAI_RAG | 89.0% | 8,648ms | $0.023 | Medium |
| OpenAI_Vanilla | 37.9%* | 2,210ms | $0.0002 | Low |

*OpenAI Vanilla lacks RAG context; abstains on 42% of questions.

**Models:** GPT-5.1 (OpenAI providers), Gemini 3 Pro Preview (Google)
**Judge:** GPT-5.1 with 80% confidence threshold

### Provider Selection Guide

| Use Case | Best Provider | Why |
|----------|---------------|-----|
| Production RAG | CustomGPT | Best accuracy + consistent latency |
| Budget-conscious | Gemini RAG | Lowest cost per correct answer |
| Enterprise/Volume | OpenAI RAG | Predictable scaling |
| Quick Prototyping | OpenAI Vanilla | Simplest integration |

## Key Features

- **4 RAG Providers:** CustomGPT, OpenAI RAG, OpenAI Vanilla, Google Gemini RAG
- **Penalty-Aware Scoring:** Based on OpenAI's research (Correct=+1, Wrong=-4, Abstain=0)
- **80% Confidence Threshold:** Optimal balance between volume and quality strategies
- **Complete Audit Trail:** Full traceability of all API calls, judge decisions, and evaluations
- **Publication-Ready Reports:** Apple-inspired HTML dashboards with statistical analysis
- **Cost & Latency Tracking:** Per-request metrics with aggregated statistics

## Architecture

```
confidence_threshold_benchmark.py          # Entry point
    |
    +-- Audited Samplers (sampler/audited_*.py)
    |       +-- CustomGPT RAG
    |       +-- OpenAI RAG (vector store)
    |       +-- OpenAI Vanilla (baseline)
    |       +-- Gemini RAG (file search)
    |
    +-- Evaluation Pipeline (confidence_threshold_simpleqa_eval.py)
    |       +-- Intent Classification (GPT-5-nano)
    |       +-- Judge Grading (GPT-5.1)
    |       +-- Penalty Calculation
    |
    +-- Report Generation
            +-- Brand Kit (brand_kit.py)
            +-- HTML Dashboards
            +-- Audit Logs (JSONL)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed documentation.

## Environment Variables

Copy `.env.example` to `.env` and configure:

**Required:**
```bash
OPENAI_API_KEY=sk-...          # OpenAI models + LLM-as-Judge
OPENAI_VECTOR_STORE_ID=vs_...  # Pre-built vector store for OpenAI RAG
```

**Provider-Specific:**
```bash
CUSTOMGPT_API_KEY=...          # CustomGPT RAG
CUSTOMGPT_PROJECT=...          # CustomGPT project ID

GEMINI_API_KEY=...             # Google Gemini RAG
GOOGLE_FILE_SEARCH_STORE_NAME=...  # Gemini file search store
```

**Optional:**
```bash
ANTHROPIC_API_KEY=...          # Claude models
SCRAPINGBEE_API_KEY=...        # Web scraping for KB building
```

## Commands

All commands run in Docker (no local Python required):

```bash
# Benchmark
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug      # 5 questions
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py              # 10 questions
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --examples 100  # Custom count
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --flex-tier  # 50% cost savings

# Tests
docker compose run --rm simple-evals python -m pytest tests/ -v

# Services
docker compose up api-server     # FastAPI on port 8000
docker compose run --rm shell    # Interactive shell
```

## Results Structure

```
results/run_YYYYMMDD_HHMMSS/
+-- index.html                    # Main dashboard hub
+-- quality_benchmark_report.html # Provider comparison
+-- statistical_analysis_report.html
+-- quality_benchmark_results.json
+-- provider_requests.jsonl       # Audit: all API calls
+-- judge_evaluations.jsonl       # Audit: all grades
+-- abstention_classifications.jsonl
+-- <provider>_penalty_analysis/  # Forensic reports
```

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - 5-minute setup
- [Architecture Overview](docs/ARCHITECTURE.md) - System design
- [RAG Performance Comparison](docs/RAG_PERFORMANCE_COMPARISON.md) - Provider analysis
- [Latency Analysis](docs/LATENCY_ANALYSIS_BY_PROVIDER.md) - Performance deep-dive
- [Cost Calculation - Gemini](docs/GOOGLE_GEMINI_COST_CALCULATION.md)
- [Cost Calculation - OpenAI](docs/OPENAI_RAG_COST_CALCULATION.md)
- [Knowledge Base Design](simpleqa-verified/KNOWLEDGE_BASE_DESIGN.md)

See [docs/README.md](docs/README.md) for complete documentation index.

## Dataset

Uses **SimpleQA-Verified**: 1,000 curated factual questions with 97.4% answer coverage in knowledge base.

- Source: OpenAI SimpleQA (Wei et al., 2024), curated by DeepMind/Google Research
- Topics: Politics, Science, Art, Geography, History, Music, Sports
- Knowledge base: 1,000 documents, 27.6M words total

## Development

This is a Docker-only environment. See [CLAUDE.md](CLAUDE.md) for development instructions.

## Research Foundation

This benchmark implements the confidence threshold methodology from:

> **"Why Language Models Hallucinate"** (arXiv:2509.04664v1)
> OpenAI Research, 2025
>
> Key insight: Using an 80% confidence threshold with penalty-aware scoring
> (wrong answers = -4 points) better evaluates RAG system quality than
> traditional accuracy metrics.

## License

MIT License. See original [OpenAI simple-evals](https://github.com/openai/simple-evals) repository.

---

*Generated with the SimpleQA RAG Benchmark Framework v3.0.0*
