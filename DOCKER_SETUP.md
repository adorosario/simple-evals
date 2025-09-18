# Docker Setup for RAG Benchmark

This setup provides a containerized environment to run the RAG benchmark comparing CustomGPT vs OpenAI GPT-4o.

## Prerequisites

- Docker and Docker Compose installed on your system
- API keys for OpenAI and CustomGPT

## Quick Start

1. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys
   ```

2. **Build and run the benchmark:**
   ```bash
   # Test with debug mode (5 questions)
   docker-compose run --rm simple-evals python rag_benchmark.py --debug

   # Run full benchmark (100+ questions)
   docker-compose run --rm simple-evals python rag_benchmark.py --examples 100
   ```

## Available Services

### 1. RAG Benchmark (`simple-evals`)
Main service for running the benchmark comparison:

```bash
# Run with debug mode
docker-compose run --rm simple-evals python rag_benchmark.py --debug

# Run with specific number of examples
docker-compose run --rm simple-evals python rag_benchmark.py --examples 50

# Show help
docker-compose run --rm simple-evals python rag_benchmark.py --help
```

### 2. API Server (`api-server`)
FastAPI server for web-based evaluations:

```bash
# Start the API server
docker-compose up api-server

# Access at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### 3. Interactive Shell (`shell`)
For debugging and development:

```bash
# Get an interactive shell in the container
docker-compose run --rm shell

# Once inside, you can run:
python rag_benchmark.py --debug
python -m simple_evals --list-models
python main.py  # Start API server
```

## Live Development

The current directory is mapped to `/app` in the container, so any changes you make to Python files will be immediately available inside the container.

## Output

Results are saved to the `results/` directory, which is mapped from the container to your host machine:
- `rag_benchmark_results_[timestamp]/rag_comparison_report.html` - Main comparison report
- `rag_benchmark_results_[timestamp]/rag_benchmark_results.json` - Detailed metrics

## Environment Variables

Required in your `.env` file:
- `OPENAI_API_KEY` - For OpenAI GPT-4o model and grading
- `CUSTOMGPT_API_KEY` - For CustomGPT RAG API
- `CUSTOMGPT_PROJECT` - Your CustomGPT project ID

Optional:
- `ANTHROPIC_API_KEY` - For Claude models (if needed)

## Troubleshooting

1. **Permission errors with volumes:**
   ```bash
   sudo chown -R $USER:$USER results/
   ```

2. **API key errors:**
   - Verify your `.env` file has correct values
   - Ensure no spaces around the `=` sign in environment variables

3. **Container rebuild after changes:**
   ```bash
   docker-compose build simple-evals
   ```

4. **Clean up:**
   ```bash
   docker-compose down
   docker system prune  # Remove unused containers/images
   ```

## Example Workflow

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 2. Quick test
docker-compose run --rm simple-evals python rag_benchmark.py --debug

# 3. Full benchmark
docker-compose run --rm simple-evals python rag_benchmark.py --examples 100

# 4. View results
open results/rag_benchmark_results_*/rag_comparison_report.html
```