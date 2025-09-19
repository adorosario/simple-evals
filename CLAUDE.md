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

**Installation/Dependencies:**
Dependencies are managed via Docker - no local installation needed.

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

### Environment Variables

Required API keys:
- `OPENAI_API_KEY`: For OpenAI models
- `ANTHROPIC_API_KEY`: For Claude models

### Results Storage

- Results saved to `results/` directory with timestamp-based subdirectories
- Each run generates both JSON metrics and HTML reports
- FastAPI server exposes results via `/results` static file endpoint