# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### CRITICAL: Docker-Only Environment
**ABSOLUTELY ALL PYTHON COMMANDS MUST BE RUN IN DOCKER** - The host system has no Python installation or dependencies. Running Python directly on the host will fail.

**Running Evaluations:**
```bash
docker compose run --rm simple-evals python -m simple_evals --list-models                    # List available models
docker compose run --rm simple-evals python -m simple_evals --model <model_name>             # Run evals on specific model
docker compose run --rm simple-evals python -m simple_evals --model <model_name> --examples <num> --debug  # Debug mode with limited examples
```

**API server:**
```bash
docker compose up simple-evals                          # Start FastAPI server on port 8000
```

**Running tests:**
```bash
docker compose run --rm simple-evals python -m pytest <test_file> -v
```

**Running scripts:**
```bash
docker compose run --rm simple-evals python scripts/<script_name>.py
```

**Confidence Threshold Benchmark (Primary Framework):**
```bash
# Quick debug run with 5 questions
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug

# Full benchmark with custom question count
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --examples 100

# Standard benchmark (10 questions per provider, all confidence thresholds)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py

# Dry run to validate configuration
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --dry-run

# Custom configuration
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --examples 50 --max-workers 5 --output-dir custom_results
```

**Legacy Multi-Provider Benchmark:**
```bash
# Basic multi-provider comparison (v1.x compatibility)
docker compose run --rm simple-evals python scripts/multi_provider_benchmark.py --debug
docker compose run --rm simple-evals python scripts/multi_provider_benchmark.py --examples 100
```

**Knowledge Base Management:**
```bash
# Download pre-built knowledge base and cache (RECOMMENDED - saves hours)
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py

# Alternative: Build knowledge base from URLs (slow, only if needed)
docker compose run --rm simple-evals python scripts/build_knowledge_base.py

# Upload knowledge base to OpenAI vector store
docker compose run --rm simple-evals python scripts/robust_upload_knowledge_base.py knowledge_base_full --store-name "MyStore"

# Knowledge base utilities
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py --list        # Check asset status
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py --cache-only  # Download only cache
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py --kb-only     # Download only knowledge bases
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py --force       # Force re-download
```

**Installation/Dependencies:**
Dependencies are managed via Docker - no local installation needed or possible.
Environment setup: Copy `.env.example` to `.env` and add required API keys.

**Alternative service commands:**
```bash
docker compose up api-server                             # Start FastAPI server on port 8000
docker compose run --rm shell                            # Interactive shell access
docker compose up default                                # Interactive container session
```

## Architecture

### Core Structure

- **simple_evals.py**: Main CLI entry point for running evaluations
- **main.py**: FastAPI server for running evaluations via HTTP API
- **common.py**: Shared utilities, templates, and HTML report generation
- **custom_types.py**: Type definitions for evaluation results and messages

### Evaluation Modules

Each evaluation has its own module (e.g., `mmlu_eval.py`, `math_eval.py`, `gpqa_eval.py`, etc.) that:
- Inherits from a base evaluation pattern
- Defines evaluation-specific logic and scoring
- Uses common utilities from `common.py`

Available evaluations: `simpleqa` (only SimpleQA is supported in v2.0.0)

### Sampler Architecture

The `sampler/` directory contains model interface adapters:
- **chat_completion_sampler.py**: OpenAI API interface
- **o1_chat_completion_sampler.py**: Specialized for O1 models (no system prompt support)
- **claude_sampler.py**: Anthropic Claude API interface
- **customgpt_sampler.py**: Custom GPT implementation

All samplers implement the `SamplerBase` interface from `custom_types.py`.

### Key Design Patterns

- **Zero-shot, chain-of-thought prompting**: The library emphasizes this approach over few-shot or role-playing prompts
- **Streaming API**: The FastAPI server supports streaming results for long-running evaluations
- **Parallel processing**: Uses ThreadPool for concurrent evaluation processing
- **HTML reporting**: Generates detailed HTML reports for each evaluation run
- **Metric aggregation**: Combines multiple evaluation runs with configurable statistics (mean, std, min, max)

### Confidence Threshold RAG Framework (v3.0.0)

**Primary Framework Components:**
- **confidence_threshold_benchmark.py**: Main entry point implementing confidence threshold analysis
- **confidence_threshold_simpleqa_eval.py**: Core evaluation logic with confidence-aware scoring
- **Audit Logging System**: Complete traceability of all requests, responses, and evaluations
- **Leaderboard Generator**: Publication-ready reports with statistical analysis and confidence metrics
- **Parallel Execution**: Provider-level and question-level parallelism for efficient benchmarking

**Confidence Threshold Innovation:**
- **Volume vs Quality Strategy**: Tests multiple confidence thresholds (0.5, 0.6, 0.7, 0.8, 0.9)
- **GPT-4.1 Standardization**: All providers and LLM-As-A-Judge use gpt-4.1 for consistency
- **80% Confidence Threshold**: Optimal balance discovered through empirical testing
- **Grade-Based Analysis**: A/B/C/D/F grading system with confidence-weighted scores
- **Statistical Validation**: Confidence intervals and performance metrics per threshold

**RAG Architecture:**
- Uses OpenAI vector stores for document retrieval
- Supports ScrapingBee integration for reliable web content extraction
- Implements robust knowledge base building with caching and failure handling
- gpt-4.1 serves as automated judge with detailed explanations
- Extensible architecture supports multiple RAG providers (CustomGPT, OpenAI RAG, etc.)

### Environment Variables

Required API keys:
- `OPENAI_API_KEY`: For OpenAI models, vector stores, and LLM-As-A-Judge
- `OPENAI_VECTOR_STORE_ID`: Pre-built vector store ID for RAG evaluations
- `CUSTOMGPT_API_KEY`: For CustomGPT RAG functionality
- `CUSTOMGPT_PROJECT`: CustomGPT project ID
- `ANTHROPIC_API_KEY`: For Claude models (optional)
- `SCRAPINGBEE_API_KEY`: For web content extraction (optional)

### Results Storage

- Results saved to `results/` directory with timestamp-based subdirectories
- Each run generates both JSON metrics and HTML reports
- FastAPI server exposes results via `/results` static file endpoint