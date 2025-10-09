# Docker Guide

This project runs **exclusively in Docker**. The host system has no Python installation or dependencies.

## Prerequisites

- Docker and Docker Compose installed
- API keys configured in `.env` file (copy from `.env.example`)

## Quick Start

```bash
# 1. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 2. Build the Docker image
docker compose build

# 3. Run evaluations
docker compose run --rm simple-evals python -m simple_evals --list-models
```

## Services

The `docker-compose.yml` defines four services:

### 1. `simple-evals` (Primary Service)
Main service for running evaluations and scripts.

```bash
# Run confidence threshold benchmark
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug

# Run multi-provider benchmark
docker compose run --rm simple-evals python scripts/multi_provider_benchmark.py --examples 100

# Run tests
docker compose run --rm simple-evals python -m pytest tests/ -v

# Run any Python script
docker compose run --rm simple-evals python scripts/<script_name>.py
```

### 2. `api-server`
FastAPI server for HTTP-based evaluations.

```bash
# Start API server on port 8000
docker compose up api-server

# Access at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 3. `shell`
Interactive shell for debugging and exploration.

```bash
# Launch interactive bash shell
docker compose run --rm shell

# Inside the shell, run commands directly:
python -m simple_evals --help
pytest tests/ -v
```

### 4. `default`
Interactive container that starts with `docker compose up`.

```bash
# Start interactive session
docker compose up default
```

## Common Commands

### Running Evaluations

```bash
# List available models
docker compose run --rm simple-evals python -m simple_evals --list-models

# Run evaluation on specific model
docker compose run --rm simple-evals python -m simple_evals --model gpt-4o-mini

# Debug mode with limited examples
docker compose run --rm simple-evals python -m simple_evals --model gpt-4o-mini --examples 5 --debug
```

### Confidence Threshold Benchmark

```bash
# Quick debug run (5 questions)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --debug

# Standard benchmark (10 questions per provider, all thresholds)
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py

# Custom configuration
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --examples 100 --max-workers 5

# Dry run validation
docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py --dry-run
```

### Knowledge Base Management

```bash
# Download pre-built knowledge base (RECOMMENDED)
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py

# Build knowledge base from URLs (slow, only if needed)
docker compose run --rm simple-evals python scripts/build_knowledge_base.py

# Upload to OpenAI vector store
docker compose run --rm simple-evals python scripts/robust_upload_knowledge_base.py knowledge_base_full --store-name "MyStore"

# Knowledge base utilities
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py --list        # Check status
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py --cache-only  # Cache only
docker compose run --rm simple-evals python scripts/download_and_extract_kb.py --force       # Force re-download
```

### Running Tests

```bash
# Run all tests
docker compose run --rm simple-evals python -m pytest -v

# Run specific test file
docker compose run --rm simple-evals python -m pytest tests/test_url_fetcher.py -v

# Run with coverage
docker compose run --rm simple-evals python -m pytest --cov=. --cov-report=html
```

## Volume Mounts

All services mount the following volumes:

- `.:/app` - Live code changes (no rebuild needed)
- `./results:/app/results` - Persist evaluation results
- `./.env:/app/.env` - Environment variables

Changes to Python code are reflected immediately without rebuilding the image.

## Environment Variables

Required in `.env` file:

- `OPENAI_API_KEY` - For OpenAI models, vector stores, and LLM-As-A-Judge
- `OPENAI_VECTOR_STORE_ID` - Pre-built vector store ID for RAG evaluations
- `CUSTOMGPT_API_KEY` - For CustomGPT RAG functionality
- `CUSTOMGPT_PROJECT` - CustomGPT project ID

Optional:
- `ANTHROPIC_API_KEY` - For Claude models
- `SCRAPINGBEE_API_KEY` - For web content extraction

## Rebuilding the Image

Rebuild when:
- `requirements.txt` changes
- `Dockerfile` changes
- System dependencies change

```bash
# Rebuild image
docker compose build

# Rebuild without cache
docker compose build --no-cache

# Pull latest base image and rebuild
docker compose build --pull
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs simple-evals

# Remove old containers
docker compose down
docker compose up default
```

### Python package issues
```bash
# Rebuild without cache
docker compose build --no-cache
```

### Permission issues with results/
```bash
# Fix permissions (run on host)
sudo chown -R $USER:$USER results/
```

### Out of disk space
```bash
# Clean up Docker resources
docker system prune -a
docker volume prune
```

## Performance Tips

1. **Use `--rm` flag** - Automatically removes containers after use
2. **Pre-download knowledge base** - Saves hours vs building from scratch
3. **Adjust `--max-workers`** - Match your CPU cores for parallel processing
4. **Use `--debug` mode first** - Test with small sample before full runs

## Image Details

- **Base**: `python:3.11-slim`
- **System packages**: curl, git
- **Python packages**: Defined in `requirements.txt`
- **Working directory**: `/app`
- **Python path**: `/app` added to `PYTHONPATH`

## Alternative: Interactive Shell Workflow

For exploratory work:

```bash
# Start interactive shell
docker compose run --rm shell

# Now you're inside the container
root@abc123:/app# python -m simple_evals --list-models
root@abc123:/app# python scripts/confidence_threshold_benchmark.py --debug
root@abc123:/app# pytest tests/ -v
root@abc123:/app# exit
```
