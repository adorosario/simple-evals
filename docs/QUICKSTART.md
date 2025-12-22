# Quick Start Guide

Get the SimpleQA RAG Benchmark running in 5 minutes.

## Prerequisites

- **Docker** and **Docker Compose** installed
- API keys for at least one provider (see below)

## Step 1: Clone and Configure

```bash
git clone <repository-url>
cd simple-evals
cp .env.example .env
```

## Step 2: Add API Keys

Edit `.env` with your API keys:

```bash
# Required for all benchmarks (OpenAI powers the judge)
OPENAI_API_KEY=sk-...

# Required for OpenAI RAG provider
OPENAI_VECTOR_STORE_ID=vs_...

# Optional: CustomGPT RAG provider
CUSTOMGPT_API_KEY=...
CUSTOMGPT_PROJECT=...

# Optional: Google Gemini RAG provider
GEMINI_API_KEY=...
GOOGLE_FILE_SEARCH_STORE_NAME=...
```

**Minimum requirement:** `OPENAI_API_KEY` is always required (powers the GPT-5.1 judge).

## Step 3: Run Your First Benchmark

```bash
# Debug mode: 5 questions, fast results
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug
```

Expected output:
```
RAG PROVIDER QUALITY BENCHMARK
======================================================================
Comparing:
  1. OpenAI Vanilla (no RAG - baseline) [gpt-5.1]
  2. OpenAI RAG (vector store file search) [gpt-5.1]
  3. CustomGPT (RAG with existing knowledge base) [gpt-5.1]
  4. Google Gemini RAG (File Search) [gemini-3-pro]
...
QUALITY BENCHMARK COMPLETE!
Results directory: results/run_YYYYMMDD_HHMMSS_mmm
```

## Step 4: View Results

Open the main dashboard in your browser:

```bash
# Find the results directory
ls results/

# Open the dashboard (example path)
open results/run_20251221_143022_123/index.html
```

The dashboard includes:
- **Provider leaderboard** with accuracy, latency, and cost
- **Statistical analysis** with confidence intervals
- **Forensic reports** for incorrect answers
- **Complete audit trail** in JSONL format

## Benchmark Modes

| Mode | Command | Questions | Use Case |
|------|---------|-----------|----------|
| Debug | `--debug` | 5 | Quick validation |
| Standard | (default) | 10 | Development testing |
| Full | `--examples 100` | 100 | Production benchmarks |
| Flex Tier | `--flex-tier` | varies | 50% cost savings (slower) |

```bash
# Standard run (10 questions per provider)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py

# Full benchmark (100 questions)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --examples 100

# Cost-optimized (uses GPT-5 Flex tier for judge)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --flex-tier
```

## Understanding Results

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct answers |
| **Quality Score** | Penalty-aware score (Correct=+1, Wrong=-4, Abstain=0) |
| **Volume Score** | Traditional score (Correct=+1, Wrong=0, Abstain=0) |
| **Attempted Rate** | Percentage of questions answered (not abstained) |
| **Latency** | Average response time in milliseconds |
| **Cost/Query** | Average cost per API call |

### Output Files

```
results/run_YYYYMMDD_HHMMSS/
+-- index.html                    # Start here - main dashboard
+-- quality_benchmark_report.html # Detailed provider comparison
+-- statistical_analysis_report.html  # Academic statistics
+-- quality_benchmark_results.json    # Machine-readable results
+-- provider_requests.jsonl       # All provider API calls
+-- judge_evaluations.jsonl       # All judge decisions
```

## Common Issues

### "No samplers were successfully initialized"

**Cause:** Missing or invalid API keys.

**Fix:** Check your `.env` file has valid keys:
```bash
# Verify .env exists and has content
cat .env | grep -v '^#' | grep -v '^$'
```

### "OPENAI_VECTOR_STORE_ID not set"

**Cause:** OpenAI RAG requires a pre-built vector store.

**Fix:** Either:
1. Set `OPENAI_VECTOR_STORE_ID` to an existing store, or
2. The benchmark will skip OpenAI RAG and test other providers

### Docker permission errors

**Fix:**
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in

# Or run with sudo
sudo docker compose run --rm simple-evals ...
```

### Rate limiting

**Cause:** Too many concurrent requests to provider APIs.

**Fix:** Reduce parallelism:
```bash
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --max-workers 2
```

## Next Steps

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Understand the system design
- **[RAG_PERFORMANCE_COMPARISON.md](RAG_PERFORMANCE_COMPARISON.md)** - Deep-dive into provider analysis
- **[../CLAUDE.md](../CLAUDE.md)** - Development and contribution guide

---

*SimpleQA RAG Benchmark Framework v3.0.0*
