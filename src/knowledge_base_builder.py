"""
Knowledge Base Builder Component

Orchestrates the complete pipeline: URL fetching → content extraction → processing → file preparation
for OpenAI vector store upload. Handles the 11,000+ URLs with progress tracking and error recovery.
"""

import os
import json
import logging
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .url_fetcher import URLFetcher
from .url_cache import URLCache
from .content_extractor import ContentExtractor
from .content_processor import ContentProcessor, ProcessedDocument
from .scrapingbee_fetcher import ScrapingBeeFetcher

logger = logging.getLogger(__name__)

@dataclass
class BuildResult:
    """Result of knowledge base build process"""
    total_urls: int
    successful_fetches: int
    successful_extractions: int
    successful_processing: int
    documents_created: int
    output_directory: str
    build_stats: Dict[str, Any]
    errors: List[str]
    scrapingbee_rescues: int = 0  # URLs rescued by ScrapingBee
    scrapingbee_cost: int = 0     # Total ScrapingBee credits used
    cache_hits: int = 0           # URLs served from cache
    cache_misses: int = 0         # URLs fetched fresh

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_urls': self.total_urls,
            'successful_fetches': self.successful_fetches,
            'successful_extractions': self.successful_extractions,
            'successful_processing': self.successful_processing,
            'documents_created': self.documents_created,
            'output_directory': self.output_directory,
            'build_stats': self.build_stats,
            'errors': self.errors,
            'scrapingbee_rescues': self.scrapingbee_rescues,
            'scrapingbee_cost': self.scrapingbee_cost,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }


class KnowledgeBaseBuilder:
    """
    Orchestrates the complete knowledge base building pipeline.
    """

    def __init__(self,
                 output_dir: str = "knowledge_base",
                 cache_dir: str = "cache/url_cache",  # In repo, not temp
                 min_document_words: int = 50,
                 max_documents: Optional[int] = None,
                 use_cache: bool = True,
                 force_refresh: bool = False,
                 max_workers: int = 20,  # Parallel workers
                 timeout: int = 10,      # Faster timeout
                 max_retries: int = 2,   # Fewer retries
                 use_scrapingbee_fallback: bool = True):  # Enable ScrapingBee fallback
        """
        Initialize knowledge base builder.

        Args:
            output_dir: Directory to save processed documents
            cache_dir: Directory for URL caching
            min_document_words: Minimum words per document
            max_documents: Maximum documents to create (for testing)
            use_cache: Whether to use URL caching
            force_refresh: Force refresh of cached content
            max_workers: Number of parallel workers for URL processing
            timeout: Timeout per URL in seconds
            max_retries: Maximum retries per URL
            use_scrapingbee_fallback: Use ScrapingBee for failed URLs
        """
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.min_document_words = min_document_words
        self.max_documents = max_documents
        self.use_cache = use_cache
        self.force_refresh = force_refresh
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_scrapingbee_fallback = use_scrapingbee_fallback

        # Initialize components
        self.url_fetcher = URLFetcher(
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_delay=1.0,  # Faster retry
            max_content_size=10*1024*1024,  # 10MB limit
            verify_ssl=False,  # Ignore SSL certificate errors
            use_random_user_agents=True  # Rotate user agents
        )

        self.url_cache = URLCache(
            cache_dir=cache_dir,
            default_ttl=7*24*3600,  # 7 days
            max_cache_size=15000  # Cache up to 15000 URLs (for full 11K build)
        ) if use_cache else None

        self.content_extractor = ContentExtractor(
            max_text_length=1_000_000,  # 1MB text limit for OpenAI
            preserve_structure=True,
            extract_metadata=True
        )

        self.content_processor = ContentProcessor(
            min_word_count=min_document_words
        )

        # Initialize ScrapingBee fallback if enabled
        self.scrapingbee_fetcher = None
        if self.use_scrapingbee_fallback:
            try:
                self.scrapingbee_fetcher = ScrapingBeeFetcher(
                    enable_js=True,        # Enable JS for difficult sites
                    premium_proxy=True,    # Use premium proxies
                    timeout=self.timeout + 10  # Give ScrapingBee more time
                )
                logger.info("ScrapingBee fallback enabled")
            except Exception as e:
                logger.warning(f"ScrapingBee fallback disabled: {e}")
                self.use_scrapingbee_fallback = False

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Thread lock for cache access
        self._cache_lock = threading.Lock()

        # Counters for ScrapingBee usage
        self._scrapingbee_rescues = 0
        self._scrapingbee_cost = 0
        self._scrapingbee_lock = threading.Lock()

        # Counters for cache usage
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_stats_lock = threading.Lock()

    def build_from_url_file(self, url_file_path: str) -> BuildResult:
        """
        Build knowledge base from a file containing URLs.

        Args:
            url_file_path: Path to file containing URLs (one per line)

        Returns:
            BuildResult with statistics and output information
        """
        logger.info(f"Starting knowledge base build from {url_file_path}")

        # Read URLs
        urls = self._load_urls_from_file(url_file_path)
        logger.info(f"Loaded {len(urls)} URLs from file")

        # Randomize URL order to ensure unprocessed URLs get a chance
        random.shuffle(urls)
        logger.info(f"🎲 Randomized URL order to avoid processing same URLs every run")

        if self.max_documents:
            urls = urls[:self.max_documents]
            logger.info(f"Limited to first {len(urls)} URLs for testing")

        return self.build_from_urls(urls)

    def build_from_urls(self, urls: List[str]) -> BuildResult:
        """
        Build knowledge base from a list of URLs.

        Args:
            urls: List of URLs to process

        Returns:
            BuildResult with statistics and output information
        """
        # Randomize URL order to ensure different URLs are processed each run
        urls = urls.copy()  # Don't modify the original list
        random.shuffle(urls)
        logger.info(f"🎲 Randomized URL processing order")

        # Apply max_documents limit if specified
        if self.max_documents:
            urls = urls[:self.max_documents]
            logger.info(f"Limited to first {len(urls)} URLs for testing")

        logger.info(f"Building knowledge base from {len(urls)} URLs using {self.max_workers} workers")

        # Initialize tracking
        total_urls = len(urls)
        documents = []
        errors = []

        # Process URLs in parallel
        with self.url_fetcher:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all URL processing jobs
                future_to_url = {
                    executor.submit(self._process_single_url, url): url
                    for url in urls
                }

                # Process completed jobs
                completed = 0
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    completed += 1

                    try:
                        result = future.result()
                        if result['document']:
                            documents.append(result['document'])
                        if result['error']:
                            errors.append(result['error'])

                        # Progress reporting
                        if completed % 50 == 0 or completed == total_urls:
                            logger.info(f"Progress: {completed}/{total_urls} URLs processed, "
                                       f"{len(documents)} documents created")

                    except Exception as e:
                        error_msg = f"Unexpected error processing {url}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)

        # Calculate final stats
        successful_fetches = total_urls - len([e for e in errors if 'Fetch failed' in e])
        successful_extractions = total_urls - len([e for e in errors if 'Extraction failed' in e])
        successful_processing = len(documents)

        # Save documents to files
        logger.info(f"Saving {len(documents)} documents to {self.output_dir}")
        self.content_processor.save_documents_to_directory(documents, self.output_dir)

        # Generate build statistics
        build_stats = self._generate_build_stats(documents, urls)

        # Save build metadata
        self._save_build_metadata(build_stats, errors)

        result = BuildResult(
            total_urls=total_urls,
            successful_fetches=successful_fetches,
            successful_extractions=successful_extractions,
            successful_processing=successful_processing,
            documents_created=len(documents),
            output_directory=self.output_dir,
            build_stats=build_stats,
            errors=errors,
            scrapingbee_rescues=self._scrapingbee_rescues,
            scrapingbee_cost=self._scrapingbee_cost,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses
        )

        logger.info(f"Knowledge base build complete: {len(documents)} documents created")
        return result

    def _load_urls_from_file(self, url_file_path: str) -> List[str]:
        """Load URLs from file"""
        urls = []

        if not os.path.exists(url_file_path):
            raise FileNotFoundError(f"URL file not found: {url_file_path}")

        with open(url_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                url = line.strip()
                if url and not url.startswith('#'):  # Skip empty lines and comments
                    # Basic URL validation
                    if url.startswith(('http://', 'https://')):
                        urls.append(url)
                    else:
                        logger.warning(f"Invalid URL on line {line_num}: {url}")

        if not urls:
            raise ValueError(f"No valid URLs found in {url_file_path}")

        return urls

    def _process_single_url(self, url: str) -> Dict[str, Any]:
        """Process a single URL (thread-safe)"""
        try:
            # Fetch content (with thread-safe caching)
            cache_hit = False
            if self.url_cache:
                with self._cache_lock:
                    fetch_result = self.url_cache.get(url, force=self.force_refresh)

                if fetch_result:
                    cache_hit = True
                    with self._cache_stats_lock:
                        self._cache_hits += 1
                    logger.info(f"📦 Cache HIT: {url}")
                else:
                    with self._cache_stats_lock:
                        self._cache_misses += 1
                    logger.info(f"🌐 Cache MISS: {url} - fetching...")
                    fetch_result = self.url_fetcher.fetch(url)
                    if fetch_result.success:
                        with self._cache_lock:
                            self.url_cache.put(url, fetch_result)
                        logger.info(f"💾 Cached: {url}")
            else:
                with self._cache_stats_lock:
                    self._cache_misses += 1
                logger.info(f"🌐 No cache: {url} - fetching...")
                fetch_result = self.url_fetcher.fetch(url)

            if not fetch_result.success:
                # Try ScrapingBee fallback if enabled
                if self.use_scrapingbee_fallback and self.scrapingbee_fetcher:
                    logger.info(f"Regular fetch failed for {url}, trying ScrapingBee fallback...")

                    try:
                        scrapingbee_result = self.scrapingbee_fetcher.fetch(url)

                        if scrapingbee_result.success:
                            logger.info(f"ScrapingBee rescue successful for {url} ({scrapingbee_result.api_cost} credits)")

                            # Track ScrapingBee usage
                            with self._scrapingbee_lock:
                                self._scrapingbee_rescues += 1
                                if scrapingbee_result.api_cost:
                                    self._scrapingbee_cost += scrapingbee_result.api_cost

                            # Convert ScrapingBee result to fetch result format
                            from .url_fetcher import FetchResult
                            fetch_result = FetchResult(
                                url=url,
                                success=True,
                                content=scrapingbee_result.content.encode('utf-8'),
                                content_type=scrapingbee_result.content_type,
                                encoding='utf-8',
                                status_code=scrapingbee_result.status_code,
                                response_time=scrapingbee_result.response_time,
                                final_url=url
                            )

                            # Cache the rescued result
                            if self.url_cache:
                                with self._cache_lock:
                                    self.url_cache.put(url, fetch_result)
                        else:
                            logger.warning(f"ScrapingBee also failed for {url}: {scrapingbee_result.error_message}")

                            # Cache the failed result with 7-day TTL to avoid retrying soon
                            if self.url_cache:
                                with self._cache_lock:
                                    self.url_cache.put(url, fetch_result, ttl=604800)  # 7 day TTL for failures
                                logger.info(f"🚫 Cached failed URL (7d TTL): {url}")

                            return {
                                'document': None,
                                'error': f"Both regular and ScrapingBee fetch failed for {url}: {fetch_result.error_message} | ScrapingBee: {scrapingbee_result.error_message}"
                            }
                    except Exception as e:
                        logger.error(f"ScrapingBee fallback error for {url}: {e}")
                        return {
                            'document': None,
                            'error': f"Fetch failed for {url}: {fetch_result.error_message} | ScrapingBee error: {str(e)}"
                        }
                else:
                    # Cache the failed result with 7-day TTL to avoid retrying soon
                    if self.url_cache:
                        with self._cache_lock:
                            self.url_cache.put(url, fetch_result, ttl=604800)  # 7 day TTL for failures
                        logger.info(f"🚫 Cached failed URL (7d TTL): {url}")

                    return {
                        'document': None,
                        'error': f"Fetch failed for {url}: {fetch_result.error_message}"
                    }

            # Extract content
            extract_result = self.content_extractor.extract_from_fetch_result(fetch_result)

            if not extract_result.success:
                return {
                    'document': None,
                    'error': f"Extraction failed for {url}: {extract_result.error_message}"
                }

            # Process into document
            document = self.content_processor.process_content(extract_result)

            if not document:
                # Not an error - just filtered out (too short, etc.)
                return {'document': None, 'error': None}

            return {'document': document, 'error': None}

        except Exception as e:
            error_msg = f"Unexpected error processing {url}: {str(e)}"
            logger.error(error_msg)
            return {'document': None, 'error': error_msg}

    def _generate_build_stats(self, documents: List[ProcessedDocument], urls: List[str]) -> Dict[str, Any]:
        """Generate comprehensive build statistics"""
        if not documents:
            return {'error': 'No documents created'}

        total_words = sum(doc.word_count for doc in documents)
        total_chars = sum(len(doc.content) for doc in documents)

        # Content type analysis from URLs
        content_types = {}
        for url in urls:
            ext = url.split('.')[-1].lower() if '.' in url else 'html'
            content_types[ext] = content_types.get(ext, 0) + 1

        stats = {
            'total_documents': len(documents),
            'total_words': total_words,
            'total_characters': total_chars,
            'avg_words_per_doc': total_words / len(documents),
            'avg_chars_per_doc': total_chars / len(documents),
            'min_words': min(doc.word_count for doc in documents),
            'max_words': max(doc.word_count for doc in documents),
            'url_content_types': content_types,
            'success_rate': len(documents) / len(urls),
            'ready_for_openai_upload': True
        }

        return stats

    def _save_build_metadata(self, build_stats: Dict[str, Any], errors: List[str]):
        """Save build metadata to JSON file"""
        metadata = {
            'build_stats': build_stats,
            'errors': errors[:100],  # Save first 100 errors
            'total_errors': len(errors),
            'build_config': {
                'output_dir': self.output_dir,
                'cache_dir': self.cache_dir,
                'min_document_words': self.min_document_words,
                'max_documents': self.max_documents,
                'use_cache': self.use_cache,
                'force_refresh': self.force_refresh
            }
        }

        metadata_path = os.path.join(self.output_dir, 'build_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Build metadata saved to {metadata_path}")

    def get_document_list_for_openai(self) -> List[str]:
        """
        Get list of text files ready for OpenAI vector store upload.

        Returns:
            List of file paths in the output directory
        """
        if not os.path.exists(self.output_dir):
            return []

        files = []
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.txt') and not filename.startswith('.'):
                filepath = os.path.join(self.output_dir, filename)
                files.append(filepath)

        logger.info(f"Found {len(files)} text files ready for OpenAI upload")
        return files

    def cleanup_cache(self):
        """Clean up URL cache"""
        if self.url_cache:
            self.url_cache.cleanup()
            logger.info("Cache cleanup completed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if hasattr(self.url_fetcher, '__exit__'):
            self.url_fetcher.__exit__(exc_type, exc_val, exc_tb)