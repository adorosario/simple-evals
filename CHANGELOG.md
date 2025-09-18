# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-01-18

### Added
- **RAG Benchmark Framework**: New `rag_benchmark.py` script for comparing RAG-enabled vs non-RAG models
- **Docker Environment**: Complete containerized setup with Docker Compose for development and testing
- **CustomGPT Integration**: Enhanced CustomGPT sampler with proper project ID handling
- **Claude Code Documentation**: Added `CLAUDE.md` for future Claude Code instances
- **Docker Documentation**: Comprehensive `DOCKER_SETUP.md` with usage instructions

### Features
- **RAG vs OpenAI Comparison**: Side-by-side evaluation of CustomGPT (with RAG) vs OpenAI GPT-4o (without RAG)
- **Automated Evaluation Pipeline**: Uses GPT-4o as judge for consistent grading
- **HTML Comparative Reports**: Generate detailed comparison reports with metrics breakdown
- **Streaming Results**: Real-time progress tracking during evaluations
- **Docker Live Development**: Volume mapping for live code changes during development
- **Multi-Service Setup**: Separate services for benchmarking, API server, and interactive shell

### Improved
- **CustomGPT Sampler**: Fixed environment variable loading for project ID configuration
- **Environment Configuration**: Enhanced `.env.example` with better documentation and all required keys
- **Error Handling**: Robust error handling and retry logic for API calls

### Technical Details
- **SimpleQA Integration**: Leverages existing SimpleQA evaluation framework for factual accuracy testing
- **Containerization**: Python 3.11-slim Docker environment with all dependencies
- **Environment Variable Management**: Proper handling of API keys and project configurations
- **Results Persistence**: Results directory mapping for persistent storage across container runs

### Performance
- OpenAI GPT-4o: ~2.4 seconds per question
- CustomGPT RAG: ~11.2 seconds per question (includes RAG lookup overhead)
- Scalable from single question debugging to full dataset evaluation

### API Requirements
- `OPENAI_API_KEY`: Required for OpenAI models and evaluation judging
- `CUSTOMGPT_API_KEY`: Required for CustomGPT RAG functionality
- `CUSTOMGPT_PROJECT`: Required project ID for CustomGPT knowledge base access

### Usage
```bash
# Quick test (5 questions)
docker compose run --rm simple-evals python rag_benchmark.py --debug

# Custom evaluation (100 questions)
docker compose run --rm simple-evals python rag_benchmark.py --examples 100

# Interactive development
docker compose run --rm shell
```

### Files Added
- `rag_benchmark.py` - Main RAG comparison script
- `Dockerfile` - Container definition
- `docker-compose.yml` - Multi-service orchestration
- `DOCKER_SETUP.md` - Docker usage documentation
- `CLAUDE.md` - Claude Code integration guide
- `CHANGELOG.md` - This changelog

### Files Modified
- `sampler/customgpt_sampler.py` - Fixed project ID environment variable loading
- `.env.example` - Enhanced with comprehensive API key documentation

## [1.0.0] - 2024-11-28

### Added
- Initial release with basic evaluation framework
- Support for multiple language model APIs (OpenAI, Claude, CustomGPT)
- SimpleQA evaluation implementation
- FastAPI server for web-based evaluations
- Basic samplers for different model providers