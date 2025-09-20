# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### IMPORTANT: Docker-Only Environment
**ALL COMMANDS MUST BE RUN IN DOCKER** - The host system does not have Python/dependencies installed.

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

**RAG Benchmark (NEW in v1.1.0):**
```bash
# Quick test with 5 questions (debug mode)
docker compose run --rm simple-evals python rag_benchmark.py --debug

# Full benchmark with custom question count
docker compose run --rm simple-evals python rag_benchmark.py --examples 100

# Three-way comparison (OpenAI vs CustomGPT vs OpenAI+RAG)
docker compose run --rm simple-evals python scripts/three_way_rag_benchmark.py
```

**Knowledge Base Management:**
```bash
# Build knowledge base from URLs
docker compose run --rm simple-evals python scripts/build_knowledge_base.py

# Upload knowledge base to OpenAI vector store
docker compose run --rm simple-evals python scripts/robust_upload_knowledge_base.py knowledge_base_full --store-name "MyStore"
```

**Installation/Dependencies:**
Dependencies are managed via Docker - no local installation needed.
Environment setup: Copy `.env.example` to `.env` and add required API keys.

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

Available evaluations: `simpleqa`, `mmlu`, `math`, `gpqa`, `mgsm`, `drop`

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

### RAG Framework (v1.1.0)

**Key Components:**
- **rag_benchmark.py**: Main entry point for RAG vs standard LLM comparison
- **CustomGPT Integration**: RAG-enabled model interface via CustomGPT API
- **Knowledge Base Builder**: Automated pipeline for creating vector stores from URLs
- **Three-way Benchmarks**: Compare OpenAI, CustomGPT, and OpenAI+RAG simultaneously

**RAG Architecture:**
- Uses OpenAI vector stores for document retrieval
- Supports ScrapingBee integration for reliable web content extraction
- Implements robust knowledge base building with caching and failure handling
- GPT-4o serves as automated judge for evaluation consistency

### Environment Variables

Required API keys:
- `OPENAI_API_KEY`: For OpenAI models and vector stores
- `ANTHROPIC_API_KEY`: For Claude models
- `CUSTOMGPT_API_KEY`: For CustomGPT RAG functionality
- `CUSTOMGPT_PROJECT`: CustomGPT project ID
- `SCRAPINGBEE_API_KEY`: For web content extraction (optional)

### Results Storage

- Results saved to `results/` directory with timestamp-based subdirectories
- Each run generates both JSON metrics and HTML reports
- FastAPI server exposes results via `/results` static file endpoint