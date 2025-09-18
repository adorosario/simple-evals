# GitHub Issues for Three-Way RAG Benchmark

## Epic: Three-Way RAG Benchmark Implementation

**Issue Title**: Epic: Implement Three-Way RAG Benchmark (CustomGPT vs OpenAI RAG vs OpenAI Vanilla)

**Description**:
Extend the current RAG benchmark framework to include OpenAI's native RAG capabilities using the new Responses API (March 2025), creating a proper "apples-to-apples" comparison between:
- CustomGPT (RAG-enabled)
- OpenAI Responses API with file search (RAG-enabled)
- OpenAI standard API (no RAG)

**Goals**:
- Fair comparison using identical knowledge bases
- Comprehensive performance, accuracy, and cost analysis
- Production-ready codebase with full test coverage
- Complete documentation and examples

**Data Source**: 11,067 URLs in `build-rag/urls.txt` for knowledge base creation

---

## Issue 1: URL Content Fetching Pipeline

**Title**: Implement URL Content Fetching Pipeline with Caching

**Description**:
Build a robust, production-ready pipeline for fetching and processing content from 11,000+ URLs. Includes caching strategy for development efficiency and comprehensive error handling.

**Acceptance Criteria**:
- [ ] Robust URL fetcher with retries, timeouts, and error handling
- [ ] File-based caching system with TTL and force refresh options
- [ ] Content type detection and size limits
- [ ] Comprehensive test coverage (unit + integration)
- [ ] Performance monitoring and statistics
- [ ] Support for various content types (HTML, PDF, JSON, etc.)

**Sub-Issues**:

### Sub-issue 1.1: URL Fetcher Component
- **Status**: 🟡 In Progress
- **Files**: `src/url_fetcher.py`, `tests/test_url_fetcher.py`, `scripts/test_url_fetcher.py`
- **Features**:
  - Configurable timeouts, retries, and user agents
  - Request session pooling for efficiency
  - Content size limits and streaming downloads
  - Detailed error categorization and logging
  - Statistics collection and reporting

### Sub-issue 1.2: URL Caching System
- **Status**: 🟡 In Progress
- **Files**: `src/url_cache.py`, `tests/test_url_cache.py`
- **Features**:
  - File-based persistent caching with metadata
  - Configurable TTL and cache size limits
  - LRU cleanup for cache management
  - Force refresh and selective cache clearing
  - Cache statistics and monitoring

### Sub-issue 1.3: Content Extractor Component
- **Status**: ⏳ Pending
- **Files**: `src/content_extractor.py`, `tests/test_content_extractor.py`, `scripts/test_content_extractor.py`
- **Features**:
  - HTML text extraction (remove tags, preserve structure)
  - PDF text extraction with proper encoding
  - Handle multiple content types and encodings
  - Malformed content recovery
  - Metadata extraction (title, author, etc.)

### Sub-issue 1.4: Content Processor Component
- **Status**: ⏳ Pending
- **Files**: `src/content_processor.py`, `tests/test_content_processor.py`, `scripts/test_content_processor.py`
- **Features**:
  - Text cleaning and normalization
  - Intelligent chunking (sentence/paragraph boundaries)
  - Deduplication logic
  - Quality filtering (length, language detection)
  - Metadata enrichment

### Sub-issue 1.5: Knowledge Base Builder
- **Status**: ⏳ Pending
- **Files**: `src/knowledge_base_builder.py`, `tests/test_knowledge_base_builder.py`, `scripts/build_knowledge_base.py`
- **Features**:
  - Pipeline orchestration (fetch → extract → process)
  - Progress tracking and resumable processing
  - Error recovery and continuation
  - Output validation and quality metrics
  - Resource usage monitoring

---

## Issue 2: OpenAI RAG Infrastructure

**Title**: Implement OpenAI RAG Infrastructure using Responses API

**Description**:
Build infrastructure for OpenAI's native RAG capabilities using the new Responses API with file search and vector stores. Ensure knowledge base parity with CustomGPT.

**Acceptance Criteria**:
- [ ] Vector store management (create, upload, update, delete)
- [ ] Document upload pipeline with metadata
- [ ] Knowledge base synchronization with CustomGPT
- [ ] Cost monitoring and usage controls
- [ ] Complete test coverage and documentation

**Sub-Issues**:

### Sub-issue 2.1: Vector Store Manager
- **Status**: ⏳ Pending
- **Files**: `src/openai_vector_store.py`, `tests/test_openai_vector_store.py`, `scripts/test_vector_store.py`
- **Features**:
  - Create and manage OpenAI vector stores
  - Upload documents with metadata
  - Search and query vector stores
  - Delete and cleanup operations
  - Error handling for API failures

### Sub-issue 2.2: Document Uploader
- **Status**: ⏳ Pending
- **Files**: `src/document_uploader.py`, `tests/test_document_uploader.py`, `scripts/test_document_upload.py`
- **Features**:
  - Batch document upload with rate limiting
  - Progress tracking and resumable uploads
  - Upload failure recovery and retries
  - Cost estimation and monitoring
  - Metadata consistency validation

### Sub-issue 2.3: OpenAI RAG Setup Automation
- **Status**: ⏳ Pending
- **Files**: `src/openai_rag_setup.py`, `tests/test_openai_rag_setup.py`, `scripts/setup_openai_rag.py`
- **Features**:
  - Automated end-to-end RAG environment setup
  - Knowledge base parity validation
  - Setup completion verification
  - Cleanup and teardown procedures
  - Cost estimation and limits

---

## Issue 3: OpenAI RAG Sampler Integration

**Title**: Implement OpenAI RAG Sampler for Responses API

**Description**:
Create a sampler that uses OpenAI's Responses API with file search functionality, ensuring compatibility with the existing evaluation framework.

**Acceptance Criteria**:
- [ ] OpenAI RAG sampler using Responses API
- [ ] Integration with existing SimpleQA evaluation
- [ ] Consistent response formatting across all samplers
- [ ] Performance benchmarking and optimization
- [ ] Comprehensive error handling

**Sub-Issues**:

### Sub-issue 3.1: OpenAI RAG Sampler Component
- **Status**: ⏳ Pending
- **Files**: `sampler/openai_rag_sampler.py`, `tests/test_openai_rag_sampler.py`, `scripts/test_openai_rag_sampler.py`
- **Features**:
  - Responses API integration with file search
  - Vector store ID configuration
  - Response annotation parsing
  - Error handling and retries
  - Performance monitoring

### Sub-issue 3.2: Sampler Integration Testing
- **Status**: ⏳ Pending
- **Files**: `tests/test_sampler_integration.py`, `scripts/test_three_samplers.py`
- **Features**:
  - Test all three samplers with same questions
  - Response format consistency validation
  - Error handling consistency testing
  - Performance comparison benchmarking
  - Integration with existing framework

---

## Issue 4: Extended Benchmark Framework

**Title**: Extend Benchmark Framework for Three-Way Comparison

**Description**:
Enhance the existing RAG benchmark to support three-model comparison with comprehensive reporting and analysis.

**Acceptance Criteria**:
- [ ] Three-way benchmark execution engine
- [ ] Enhanced HTML reports with side-by-side-by-side comparison
- [ ] Statistical analysis and significance testing
- [ ] Cost analysis and performance breakdowns
- [ ] CLI options for flexible evaluation runs

**Sub-Issues**:

### Sub-issue 4.1: Three-Way Benchmark Engine
- **Status**: ⏳ Pending
- **Files**: `rag_benchmark_v2.py`, `tests/test_benchmark_v2.py`, `scripts/test_benchmark_small.py`
- **Features**:
  - Run all three models on identical questions
  - Graceful failure handling per model
  - Progress tracking and real-time updates
  - Configurable evaluation parameters
  - Resource usage monitoring

### Sub-issue 4.2: Enhanced Reporting System
- **Status**: ⏳ Pending
- **Files**: `src/three_way_reporter.py`, `tests/test_three_way_reporter.py`, `scripts/test_reporting.py`
- **Features**:
  - Three-way HTML comparison reports
  - Statistical significance testing
  - Cost analysis and breakdown charts
  - Performance vs accuracy trade-off analysis
  - Export capabilities (CSV, JSON)

---

## Issue 5: Integration & Deployment

**Title**: End-to-End Integration and Docker Environment Updates

**Description**:
Complete integration testing and update Docker environment to support the full three-way benchmark pipeline.

**Acceptance Criteria**:
- [ ] End-to-end integration testing
- [ ] Updated Docker environment with all dependencies
- [ ] Performance optimization and resource management
- [ ] Complete documentation updates
- [ ] Production deployment readiness

**Sub-Issues**:

### Sub-issue 5.1: End-to-End Integration Testing
- **Status**: ⏳ Pending
- **Files**: `tests/test_end_to_end.py`, `scripts/integration_test.py`
- **Features**:
  - Full pipeline test (URLs → knowledge base → benchmark)
  - Component interaction validation
  - Error recovery and rollback testing
  - Performance and resource usage validation
  - Data consistency verification

### Sub-issue 5.2: Docker Environment Updates
- **Status**: ⏳ Pending
- **Files**: `Dockerfile`, `docker-compose.yml`, `DOCKER_SETUP.md`
- **Features**:
  - Updated dependencies for all new components
  - Service configuration for three-way benchmark
  - Volume mapping for cache and results
  - Environment variable management
  - Development workflow optimization

---

## Technical Dependencies

### API Requirements
- `OPENAI_API_KEY`: OpenAI models and Responses API
- `CUSTOMGPT_API_KEY`: CustomGPT RAG functionality
- `CUSTOMGPT_PROJECT`: CustomGPT project ID

### New Python Dependencies
- `pytest==8.3.4`: Testing framework
- `responses==0.25.8`: HTTP request mocking
- Additional content extraction libraries (TBD)

### Performance Targets
- Process 11,000 URLs in <2 hours with caching
- Cache hit rate >90% during development
- Three-way benchmark completion in <30 minutes for 100 questions
- Memory usage <4GB for full pipeline

### Quality Standards
- Unit test coverage >90%
- All components have standalone test scripts
- Comprehensive error handling and logging
- Production-ready code quality and documentation