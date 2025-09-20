#!/usr/bin/env python3
"""
Simple QA URL Fixing Script

This script identifies bad URLs in the Simple QA test set and replaces them with
good URLs found via Serper.dev Google search API.

The script:
1. Audits the test set to identify records with bad URLs
2. Uses Serper.dev to search for replacement URLs based on question content
3. Validates replacement URLs before substitution
4. Creates a fixed version of the test set with detailed logging

Usage:
    python fix_simple_qa_urls.py [options]

Example:
    python fix_simple_qa_urls.py --dry-run
    python fix_simple_qa_urls.py --input-csv simple_qa_test_set.csv --output-csv fixed_test_set.csv
"""

import os
import json
import csv
import hashlib
import time
import logging
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import ast
import requests
from dotenv import load_dotenv


# We'll configure logging later after creating the logs directory
logger = logging.getLogger(__name__)


@dataclass
class URLStatus:
    """Status information for a URL"""
    url: str
    in_cache: bool
    cache_success: bool
    in_knowledge_base: bool
    is_bad: bool
    failure_reason: Optional[str] = None


@dataclass
class ReplacementURL:
    """Information about a replacement URL"""
    original_url: str
    new_url: str
    title: str
    snippet: str
    rank: int
    confidence_score: float


@dataclass
class FixResult:
    """Result of fixing URLs for a record"""
    record_id: int
    question: str
    original_urls: List[str]
    fixed_urls: List[str]
    replacements: List[ReplacementURL]
    all_urls_fixed: bool
    num_urls_replaced: int


class SerperCache:
    """Cache for Serper.dev API calls to avoid repeated requests and costs"""

    def __init__(self, cache_dir: str = "logs/serper_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Serper cache metadata: {e}")

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    def get(self, query: str) -> Optional[Dict]:
        """Get cached result for query"""
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_key in self.metadata and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    result = json.load(f)

                # Update access time
                self.metadata[cache_key]['last_access'] = time.time()
                self._save_metadata()

                logger.debug(f"Serper cache hit for query: {query}")
                return result
            except Exception as e:
                logger.error(f"Failed to load cached Serper result: {e}")

        return None

    def put(self, query: str, result: Dict):
        """Cache a search result"""
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)

            current_time = time.time()
            self.metadata[cache_key] = {
                'query': query,
                'timestamp': current_time,
                'last_access': current_time,
                'results_count': len(result.get('organic', [])),
                'credits_used': result.get('credits', 1)
            }

            self._save_metadata()
            logger.debug(f"Cached Serper result for query: {query}")

        except Exception as e:
            logger.error(f"Failed to cache Serper result: {e}")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_queries = len(self.metadata)
        total_credits = sum(entry.get('credits_used', 1) for entry in self.metadata.values())

        return {
            'total_cached_queries': total_queries,
            'total_credits_saved': total_credits,
            'cache_directory': str(self.cache_dir)
        }


class SerperSearcher:
    """Handles Google search via Serper.dev API with caching"""

    def __init__(self, api_key: str, cache_dir: str = "logs/serper_cache"):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
        self.request_count = 0
        self.rate_limit_delay = 1  # 1 second between requests
        self.cache = SerperCache(cache_dir)

    def search(self, query: str, num_results: int = 10) -> dict:
        """Perform Google search using Serper.dev API with caching"""

        # Try cache first
        cache_key = f"{query}_{num_results}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info(f"🔄 Using cached result for: {query}")
            return cached_result

        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        payload = {
            'q': query
        }
        # Only add num parameter if it's different from default
        if num_results != 10:
            payload['num'] = num_results

        try:
            # Rate limiting
            if self.request_count > 0:
                time.sleep(self.rate_limit_delay)

            logger.info(f"🌐 Making API call to Serper.dev for: {query}")
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            self.request_count += 1
            result = response.json()

            # Cache the result
            self.cache.put(cache_key, result)

            logger.debug(f"Search request #{self.request_count}: {query}")
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching for '{query}': {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing search results for '{query}': {e}")
            return {}

    def find_replacement_urls(self, question: str, topic: str, answer: str, num_results: int = 10) -> List[ReplacementURL]:
        """Find replacement URLs by searching for the exact question"""
        replacement_urls = []

        logger.info(f"=" * 80)
        logger.info(f"SEARCHING FOR REPLACEMENT URLs")
        logger.info(f"Question: '{question}'")
        logger.info(f"Topic: {topic}")
        logger.info(f"Answer: {answer}")
        logger.info(f"=" * 80)

        # PRIMARY STRATEGY: Search for the exact question
        # This is what the user specifically requested
        query = question
        logger.info(f"Searching Serper.dev for: '{query}'")
        results = self.search(query, num_results=num_results)

        if results:
            logger.info(f"Serper.dev returned {len(results.get('organic', []))} organic results")
            replacement_urls = self._extract_urls_from_results(results, query, "exact_question")
            logger.info(f"After filtering, found {len(replacement_urls)} valid replacement URLs")
        else:
            logger.warning(f"No search results returned for: '{query}'")

        # Remove duplicates and sort by confidence
        unique_urls = {}
        for url_info in replacement_urls:
            # Only keep URLs with positive confidence scores
            if url_info.confidence_score > 0.0:
                if url_info.new_url not in unique_urls or url_info.confidence_score > unique_urls[url_info.new_url].confidence_score:
                    unique_urls[url_info.new_url] = url_info

        # Sort by confidence score (higher is better)
        sorted_urls = sorted(unique_urls.values(), key=lambda x: x.confidence_score, reverse=True)

        logger.info(f"Total unique URLs found after filtering: {len(sorted_urls)}")
        for i, url_info in enumerate(sorted_urls[:num_results], 1):
            logger.info(f"  {i}. {url_info.new_url} (confidence: {url_info.confidence_score:.2f})")

        return sorted_urls[:num_results]

    def _extract_urls_from_results(self, results: dict, query: str, strategy: str) -> List[ReplacementURL]:
        """Extract URLs from search results and assign confidence scores"""
        urls = []

        if not results or 'organic' not in results:
            return urls

        organic_results = results['organic']

        for i, result in enumerate(organic_results):
            url = result.get('link', '')
            title = result.get('title', '')
            snippet = result.get('snippet', '')

            if not url:
                continue

            # Calculate confidence score based on various factors
            confidence = self._calculate_confidence_score(url, title, snippet, i, strategy)

            replacement = ReplacementURL(
                original_url="",  # Will be set later
                new_url=url,
                title=title,
                snippet=snippet,
                rank=i + 1,
                confidence_score=confidence
            )

            urls.append(replacement)

        return urls

    def _calculate_confidence_score(self, url: str, title: str, snippet: str, rank: int, strategy: str) -> float:
        """Calculate confidence score for a URL based on various factors"""

        # CRITICAL: Filter out irrelevant results
        # Check for blacklisted domains/patterns that indicate irrelevant results
        blacklist_patterns = [
            'huggingface.co/datasets',  # Dataset pages
            'simpleqa',  # SimpleQA-related content
            'expectedparrot.com',  # Demo sites
            'github.com/openai/simple-evals',  # Code repositories
            'news.ycombinator.com',  # Discussion forums about datasets
            '/pvduy/',  # Specific dataset author
            'dataset',  # Generic dataset references
        ]

        # Check URL, title, and snippet for blacklisted patterns
        content_to_check = f"{url} {title} {snippet}".lower()
        for pattern in blacklist_patterns:
            if pattern in content_to_check:
                logger.warning(f"Filtered out irrelevant result: {url} (contains '{pattern}')")
                return 0.0  # Completely reject irrelevant results

        score = 0.0

        # Base score inversely proportional to rank (1st result gets highest base score)
        base_score = max(0.1, 1.0 - (rank * 0.1))
        score += base_score

        # Bonus for trusted domains (much higher bonuses)
        trusted_domains = [
            'wikipedia.org', 'britannica.com', 'edu', 'gov',
            'archive.org', 'jstor.org', 'springer.com', 'nature.com',
            'sciencedirect.com', 'ieee.org', 'acm.org', 'nih.gov',
            'ncbi.nlm.nih.gov', 'doi.org'
        ]

        domain_bonus = 0.0
        for domain in trusted_domains:
            if domain in url.lower():
                if domain == 'wikipedia.org':
                    domain_bonus = 0.5  # Highest bonus for Wikipedia
                elif domain in ['edu', 'gov', 'ieee.org', 'acm.org']:
                    domain_bonus = 0.4  # High bonus for academic/official sources
                else:
                    domain_bonus = 0.3  # Good bonus for other trusted sources
                break

        score += domain_bonus

        # Bonus for HTTPS
        if url.startswith('https://'):
            score += 0.05

        # Strategy-based bonuses
        if strategy == "exact_question":
            score += 0.3  # Exact question searches are most reliable (user requested)
        elif strategy == "main_entity":
            score += 0.2  # Main entity searches
        elif strategy == "answer_topic":
            score += 0.15  # Answer + topic
        elif strategy == "specific_terms":
            score += 0.1

        # Title and snippet quality (check for relevance)
        if title and len(title) > 10:
            score += 0.05
        if snippet and len(snippet) > 50:
            score += 0.05

        # Penalty for very low scores (filter out weak matches)
        if score < 0.3:
            return 0.0

        return min(1.0, score)  # Cap at 1.0

    def _extract_main_entity(self, question: str, answer: str) -> str:
        """Extract the main entity/subject from question and answer"""
        import re

        # For awards, institutions, etc., look for proper nouns
        # Common patterns for main entities
        patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+ Award)',  # "Frank Rosenblatt Award"
            r'([A-Z]{2,} [A-Z][a-z]+ [A-Z][a-z]+ Award)',  # "IEEE Frank Rosenblatt Award"
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Person names like "Michio Sugeno"
            r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}(?:\s+(?:University|College|Institute|Organization))?)'  # Institutions
        ]

        # Try answer first (often contains the main entity)
        for pattern in patterns:
            matches = re.findall(pattern, answer)
            if matches:
                return matches[0]

        # Try question
        for pattern in patterns:
            matches = re.findall(pattern, question)
            if matches:
                return matches[0]

        return ""

    def _extract_specific_terms(self, question: str) -> List[str]:
        """Extract specific, meaningful terms from question (avoiding question words)"""
        import re

        # Remove question words and common words
        stop_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were',
            'the', 'a', 'an', 'did', 'do', 'does', 'in', 'on', 'at', 'for', 'of', 'with',
            'received', 'got', 'won', 'awarded', 'given', 'name', 'called'
        }

        # Extract meaningful terms (proper nouns, numbers, specific words)
        words = re.findall(r'\b[A-Za-z]{3,}\b', question)
        specific_terms = []

        for word in words:
            if word.lower() not in stop_words:
                # Keep proper nouns and meaningful terms
                if word[0].isupper() or len(word) > 5:
                    specific_terms.append(word)

        return specific_terms[:5]  # Limit to avoid overly long queries


class URLValidator:
    """Validates URLs by attempting to fetch them"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def is_url_accessible(self, url: str, timeout: int = 10) -> Tuple[bool, str]:
        """Check if URL is accessible"""
        try:
            response = self.session.head(url, timeout=timeout, allow_redirects=True)
            if response.status_code == 200:
                return True, "OK"
            elif response.status_code in [301, 302, 303, 307, 308]:
                return True, f"Redirect ({response.status_code})"
            else:
                return False, f"HTTP {response.status_code}"

        except requests.exceptions.Timeout:
            return False, "Timeout"
        except requests.exceptions.ConnectionError:
            return False, "Connection Error"
        except requests.exceptions.RequestException as e:
            return False, f"Request Error: {str(e)[:50]}"
        except Exception as e:
            return False, f"Unknown Error: {str(e)[:50]}"

    def close(self):
        """Close the session"""
        self.session.close()


class SimpleQAURLFixer:
    """Main class for fixing URLs in Simple QA test set"""

    def __init__(self,
                 input_csv: str,
                 output_csv: str,
                 cache_dir: str,
                 kb_dir: str,
                 serper_api_key: str,
                 max_search_results: int = 10,
                 limit: int = None,
                 dry_run: bool = False):

        self.input_csv = input_csv
        self.output_csv = output_csv
        self.cache_dir = Path(cache_dir)
        self.kb_dir = Path(kb_dir)
        self.max_search_results = max_search_results
        self.limit = limit
        self.dry_run = dry_run

        # Setup logging infrastructure
        self.session_id = f"session_{int(time.time())}"
        self.logs_dir = Path("logs")
        self.setup_logging()

        # Initialize components
        self.searcher = SerperSearcher(serper_api_key, cache_dir="logs/serper_cache")
        self.validator = URLValidator()

        # Load data
        self.cache_metadata = self._load_cache_metadata()
        self.kb_urls = self._load_knowledge_base_urls()

        # Statistics
        self.stats = {
            'session_id': self.session_id,
            'start_time': time.time(),
            'total_records': 0,
            'records_with_bad_urls': 0,
            'records_fixed': 0,
            'total_urls_replaced': 0,
            'search_requests': 0,
            'cached_requests': 0,
            'dry_run': dry_run
        }

    def setup_logging(self):
        """Setup logging infrastructure with logs directory"""
        # Create logs directory structure
        self.logs_dir.mkdir(exist_ok=True)
        (self.logs_dir / "records").mkdir(exist_ok=True)
        (self.logs_dir / "serper_cache").mkdir(exist_ok=True)

        # Setup main log file
        log_file = self.logs_dir / f"{self.session_id}.log"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True  # Override any existing handlers
        )

        logger.info(f"Starting URL fixing session: {self.session_id}")
        logger.info(f"Logs directory: {self.logs_dir}")
        logger.info(f"Main log file: {log_file}")

    def _load_cache_metadata(self) -> Dict:
        """Load URL cache metadata"""
        cache_metadata_path = self.cache_dir / "cache_metadata.json"
        try:
            with open(cache_metadata_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Cache metadata file not found at {cache_metadata_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cache metadata: {e}")
            return {}

    def _load_knowledge_base_urls(self) -> Set[str]:
        """Load URLs from knowledge base"""
        kb_urls = set()

        if not self.kb_dir.exists():
            logger.warning(f"Knowledge base directory not found at {self.kb_dir}")
            return kb_urls

        # Try to load from build metadata first
        build_metadata_path = self.kb_dir / "build_metadata.json"
        if build_metadata_path.exists():
            try:
                with open(build_metadata_path, 'r') as f:
                    metadata = json.load(f)
                if 'document_sources' in metadata:
                    for doc_info in metadata['document_sources'].values():
                        if 'source_url' in doc_info:
                            kb_urls.add(doc_info['source_url'])
                    return kb_urls
            except Exception as e:
                logger.warning(f"Failed to load build metadata: {e}")

        # Fallback: scan document files
        doc_files = list(self.kb_dir.glob("doc_*.txt"))
        logger.info(f"Scanning {len(doc_files)} knowledge base documents for URLs...")

        for doc_file in doc_files:
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i > 10:  # Only check first 10 lines
                            break
                        line = line.strip()
                        if line.startswith(('Source:', 'URL:')):
                            url = line.split(':', 1)[1].strip()
                            kb_urls.add(url)
                            break
                        elif line.startswith('http'):
                            kb_urls.add(line)
                            break
            except Exception as e:
                logger.debug(f"Failed to read {doc_file}: {e}")

        return kb_urls

    def _get_url_hash(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()

    def _analyze_url(self, url: str) -> URLStatus:
        """Analyze the status of a single URL"""
        url_hash = self._get_url_hash(url)

        # Check cache status
        in_cache = url_hash in self.cache_metadata
        cache_success = False
        failure_reason = None

        if in_cache:
            cache_entry = self.cache_metadata[url_hash]
            cache_success = cache_entry.get('success', False)
            if not cache_success:
                failure_reason = "Cache indicates fetch failure"
        else:
            failure_reason = "URL not found in cache"

        # Check knowledge base status
        in_knowledge_base = url in self.kb_urls

        # Determine if URL is "bad"
        is_bad = not cache_success or not in_knowledge_base

        if is_bad and not failure_reason:
            if not in_knowledge_base:
                failure_reason = "URL not present in knowledge base"

        return URLStatus(
            url=url,
            in_cache=in_cache,
            cache_success=cache_success,
            in_knowledge_base=in_knowledge_base,
            is_bad=is_bad,
            failure_reason=failure_reason
        )

    def _save_record_json_log(self, record_id: int, fix_result: FixResult, serper_query: str = None, serper_results_count: int = 0, was_cached: bool = False):
        """Save detailed JSON log for a single record"""
        record_log = {
            'session_id': self.session_id,
            'record_id': record_id,
            'timestamp': time.time(),
            'question': fix_result.question,
            'topic': None,  # Will be set by caller
            'answer': None,  # Will be set by caller
            'original_urls': fix_result.original_urls,
            'fixed_urls': fix_result.fixed_urls,
            'url_analysis': [],
            'search_info': {
                'query_sent_to_serper': serper_query,
                'serper_results_count': serper_results_count,
                'was_cached': was_cached
            },
            'replacements': [
                {
                    'original_url': rep.original_url,
                    'new_url': rep.new_url,
                    'title': rep.title,
                    'snippet': rep.snippet,
                    'confidence_score': rep.confidence_score,
                    'rank': rep.rank
                } for rep in fix_result.replacements
            ],
            'summary': {
                'total_urls': len(fix_result.original_urls),
                'bad_urls_found': len(fix_result.original_urls) - len([u for u in fix_result.original_urls if u in fix_result.fixed_urls]),
                'urls_replaced': fix_result.num_urls_replaced,
                'all_urls_fixed': fix_result.all_urls_fixed,
                'action_taken': 'none' if fix_result.num_urls_replaced == 0 else 'replaced_urls'
            }
        }

        # Save to individual record file
        record_log_file = self.logs_dir / "records" / f"record_{record_id:06d}.json"
        try:
            with open(record_log_file, 'w') as f:
                json.dump(record_log, f, indent=2)
            logger.debug(f"Saved record log: {record_log_file}")
        except Exception as e:
            logger.error(f"Failed to save record log: {e}")

        return record_log

    def _parse_metadata_field(self, metadata_str: str) -> Dict:
        """Parse metadata field from CSV"""
        try:
            return ast.literal_eval(metadata_str)
        except Exception as e:
            logger.warning(f"Failed to parse metadata: {metadata_str[:100]}... Error: {e}")
            return {}

    def fix_urls_for_record(self, record_id: int, question: str, answer: str, topic: str, urls: List[str]) -> FixResult:
        """Fix URLs for a single record"""

        logger.info(f"\n" + "=" * 100)
        logger.info(f"PROCESSING RECORD {record_id}")
        logger.info(f"Question: {question}")
        logger.info(f"Answer: {answer}")
        logger.info(f"Topic: {topic}")
        logger.info(f"Total URLs in record: {len(urls)}")
        logger.info(f"=" * 100)

        # Analyze current URLs
        logger.info(f"ANALYZING URLs FOR RECORD {record_id}:")
        url_statuses = []
        for i, url in enumerate(urls, 1):
            status = self._analyze_url(url)
            url_statuses.append(status)

            logger.info(f"  URL {i}: {url}")
            logger.info(f"    In Cache: {status.in_cache}")
            logger.info(f"    Cache Success: {status.cache_success}")
            logger.info(f"    In Knowledge Base: {status.in_knowledge_base}")
            logger.info(f"    Status: {'✅ GOOD' if not status.is_bad else '❌ BAD'}")
            if status.is_bad:
                logger.info(f"    Failure Reason: {status.failure_reason}")

        bad_urls = [status for status in url_statuses if status.is_bad]
        good_urls = len(urls) - len(bad_urls)

        logger.info(f"\nURL ANALYSIS SUMMARY:")
        logger.info(f"  Good URLs: {good_urls}/{len(urls)}")
        logger.info(f"  Bad URLs: {len(bad_urls)}/{len(urls)}")

        if not bad_urls:
            logger.info(f"✅ Record {record_id}: ALL URLs are good - NO ACTION NEEDED")
            fix_result = FixResult(
                record_id=record_id,
                question=question,
                original_urls=urls,
                fixed_urls=urls,
                replacements=[],
                all_urls_fixed=True,
                num_urls_replaced=0
            )

            # Save JSON log
            record_log = self._save_record_json_log(record_id, fix_result)
            record_log['topic'] = topic
            record_log['answer'] = answer
            record_log['url_analysis'] = [
                {
                    'url': status.url,
                    'in_cache': status.in_cache,
                    'cache_success': status.cache_success,
                    'in_knowledge_base': status.in_knowledge_base,
                    'is_bad': status.is_bad,
                    'failure_reason': status.failure_reason
                } for status in url_statuses
            ]

            # Re-save with complete info
            record_log_file = self.logs_dir / "records" / f"record_{record_id:06d}.json"
            with open(record_log_file, 'w') as f:
                json.dump(record_log, f, indent=2)

            return fix_result

        logger.info(f"🔧 Record {record_id}: Found {len(bad_urls)} bad URLs - FIXING NEEDED")

        # Search for replacement URLs
        search_query = question
        search_was_cached = self.searcher.cache.get(f"{search_query}_{self.max_search_results}") is not None

        replacement_candidates = self.searcher.find_replacement_urls(
            question, topic, answer, self.max_search_results
        )

        # Update stats
        if search_was_cached:
            self.stats['cached_requests'] += 1
        else:
            self.stats['search_requests'] += 1

        if not replacement_candidates:
            logger.warning(f"Record {record_id}: No replacement URLs found")
            return FixResult(
                record_id=record_id,
                question=question,
                original_urls=urls,
                fixed_urls=urls,
                replacements=[],
                all_urls_fixed=False,
                num_urls_replaced=0
            )

        # Validate replacement URLs
        validated_replacements = []
        for candidate in replacement_candidates:
            is_accessible, reason = self.validator.is_url_accessible(candidate.new_url)
            if is_accessible:
                validated_replacements.append(candidate)
                logger.debug(f"Validated replacement URL: {candidate.new_url}")
            else:
                logger.debug(f"Rejected replacement URL {candidate.new_url}: {reason}")

        if not validated_replacements:
            logger.warning(f"Record {record_id}: No valid replacement URLs found")
            return FixResult(
                record_id=record_id,
                question=question,
                original_urls=urls,
                fixed_urls=urls,
                replacements=[],
                all_urls_fixed=False,
                num_urls_replaced=0
            )

        # Replace bad URLs with validated replacements
        logger.info(f"\nURL REPLACEMENT PROCESS:")
        logger.info(f"Available validated replacements: {len(validated_replacements)}")
        for i, replacement in enumerate(validated_replacements, 1):
            logger.info(f"  Replacement {i}: {replacement.new_url} (confidence: {replacement.confidence_score:.2f})")

        fixed_urls = urls.copy()
        replacements = []
        replacements_used = 0

        logger.info(f"\nReplacing bad URLs:")
        for i, status in enumerate(url_statuses):
            if status.is_bad and replacements_used < len(validated_replacements):
                replacement = validated_replacements[replacements_used]
                replacement.original_url = status.url

                fixed_urls[i] = replacement.new_url
                replacements.append(replacement)
                replacements_used += 1

                logger.info(f"✅ REPLACED:")
                logger.info(f"    Original: {status.url}")
                logger.info(f"    New:      {replacement.new_url}")
                logger.info(f"    Title:    {replacement.title}")
                logger.info(f"    Confidence: {replacement.confidence_score:.2f}")
                logger.info(f"    Reason original was bad: {status.failure_reason}")
            elif status.is_bad:
                logger.info(f"❌ COULD NOT REPLACE: {status.url} (no more replacement candidates)")
            else:
                logger.info(f"✅ KEPT GOOD URL: {status.url}")

        logger.info(f"\nREPLACEMENT SUMMARY:")
        logger.info(f"  URLs replaced: {replacements_used}")
        logger.info(f"  URLs kept: {len(urls) - len(bad_urls)}")
        logger.info(f"  URLs still bad: {len(bad_urls) - replacements_used}")

        all_fixed = all(not self._analyze_url(url).is_bad for url in fixed_urls)

        fix_result = FixResult(
            record_id=record_id,
            question=question,
            original_urls=urls,
            fixed_urls=fixed_urls,
            replacements=replacements,
            all_urls_fixed=all_fixed,
            num_urls_replaced=replacements_used
        )

        # Save comprehensive JSON log
        record_log = self._save_record_json_log(
            record_id,
            fix_result,
            serper_query=search_query,
            serper_results_count=len(replacement_candidates) if replacement_candidates else 0,
            was_cached=search_was_cached
        )
        record_log['topic'] = topic
        record_log['answer'] = answer
        record_log['url_analysis'] = [
            {
                'url': status.url,
                'in_cache': status.in_cache,
                'cache_success': status.cache_success,
                'in_knowledge_base': status.in_knowledge_base,
                'is_bad': status.is_bad,
                'failure_reason': status.failure_reason
            } for status in url_statuses
        ]

        # Re-save with complete info
        record_log_file = self.logs_dir / "records" / f"record_{record_id:06d}.json"
        with open(record_log_file, 'w') as f:
            json.dump(record_log, f, indent=2)

        return fix_result

    def process_test_set(self) -> List[FixResult]:
        """Process the entire test set and fix bad URLs"""
        fix_results = []

        try:
            with open(self.input_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for i, row in enumerate(reader):
                    # Check limit
                    if self.limit is not None and i >= self.limit:
                        logger.info(f"Reached limit of {self.limit} records, stopping processing")
                        break

                    try:
                        self.stats['total_records'] += 1

                        # Parse record
                        metadata = self._parse_metadata_field(row['metadata'])
                        urls = metadata.get('urls', [])
                        topic = metadata.get('topic', 'Unknown')
                        answer_type = metadata.get('answer_type', 'Unknown')
                        question = row['problem']
                        answer = row['answer']

                        # Check if record has any bad URLs
                        url_statuses = [self._analyze_url(url) for url in urls]
                        has_bad_urls = any(status.is_bad for status in url_statuses)

                        if not has_bad_urls:
                            # No bad URLs, add original record unchanged
                            fix_result = FixResult(
                                record_id=i + 1,
                                question=question,
                                original_urls=urls,
                                fixed_urls=urls,
                                replacements=[],
                                all_urls_fixed=True,
                                num_urls_replaced=0
                            )
                        else:
                            self.stats['records_with_bad_urls'] += 1

                            # Fix URLs for this record
                            fix_result = self.fix_urls_for_record(
                                record_id=i + 1,
                                question=question,
                                answer=answer,
                                topic=topic,
                                urls=urls
                            )

                            if fix_result.num_urls_replaced > 0:
                                self.stats['records_fixed'] += 1
                                self.stats['total_urls_replaced'] += fix_result.num_urls_replaced

                        fix_results.append(fix_result)

                        # Progress logging
                        if (i + 1) % 100 == 0:
                            logger.info(f"Processed {i + 1} records...")

                    except Exception as e:
                        logger.error(f"Error processing record {i + 1}: {e}")

        except FileNotFoundError:
            logger.error(f"Input file not found: {self.input_csv}")
            return []

        return fix_results

    def write_fixed_csv(self, fix_results: List[FixResult]):
        """Write the fixed CSV file"""
        if self.dry_run:
            logger.info("DRY RUN: Would write fixed CSV file")
            return

        # Create backup of original file
        backup_path = f"{self.input_csv}.backup_{int(time.time())}"
        shutil.copy2(self.input_csv, backup_path)
        logger.info(f"Created backup: {backup_path}")

        # Read original CSV to preserve structure
        with open(self.input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            original_rows = list(reader)

        # Write fixed CSV
        with open(self.output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, fix_result in enumerate(fix_results):
                if i < len(original_rows):
                    row = original_rows[i].copy()

                    # Update URLs in metadata if they were changed
                    if fix_result.num_urls_replaced > 0:
                        metadata = self._parse_metadata_field(row['metadata'])
                        metadata['urls'] = fix_result.fixed_urls
                        row['metadata'] = str(metadata)

                    writer.writerow(row)

        logger.info(f"Fixed CSV written to: {self.output_csv}")

    def generate_final_json_report(self, fix_results: List[FixResult]):
        """Generate final JSON report with comprehensive statistics"""
        self.stats['end_time'] = time.time()
        self.stats['duration_seconds'] = self.stats['end_time'] - self.stats['start_time']

        # Get Serper cache stats
        serper_stats = self.searcher.cache.get_stats()

        final_report = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': self.stats['start_time'],
                'end_time': self.stats['end_time'],
                'duration_seconds': self.stats['duration_seconds'],
                'dry_run': self.dry_run
            },
            'input_info': {
                'input_csv': self.input_csv,
                'output_csv': self.output_csv,
                'cache_dir': str(self.cache_dir),
                'kb_dir': str(self.kb_dir)
            },
            'processing_stats': {
                'total_records_processed': self.stats['total_records'],
                'records_with_bad_urls': self.stats['records_with_bad_urls'],
                'records_successfully_fixed': self.stats['records_fixed'],
                'total_urls_replaced': self.stats['total_urls_replaced'],
                'serper_api_calls_made': self.stats['search_requests'],
                'serper_cached_requests': self.stats['cached_requests'],
                'total_serper_requests': self.stats['search_requests'] + self.stats['cached_requests']
            },
            'serper_cache_stats': serper_stats,
            'cost_analysis': {
                'estimated_serper_cost_usd': self.stats['search_requests'] * 0.001,  # $1 per 1000 requests
                'credits_saved_by_caching': serper_stats.get('total_credits_saved', 0),
                'cache_hit_rate': self.stats['cached_requests'] / max(1, self.stats['search_requests'] + self.stats['cached_requests'])
            },
            'summary_by_action': {
                'no_action_needed': len([r for r in fix_results if r.num_urls_replaced == 0 and r.all_urls_fixed]),
                'partial_fixes': len([r for r in fix_results if 0 < r.num_urls_replaced < len(r.original_urls)]),
                'complete_fixes': len([r for r in fix_results if r.num_urls_replaced > 0 and r.all_urls_fixed]),
                'unfixable_records': len([r for r in fix_results if r.num_urls_replaced == 0 and not r.all_urls_fixed])
            }
        }

        # Save final report
        final_report_path = self.logs_dir / f"{self.session_id}_final_report.json"
        with open(final_report_path, 'w') as f:
            json.dump(final_report, f, indent=2)

        logger.info(f"Final JSON report saved to: {final_report_path}")
        return final_report

    def generate_report(self, fix_results: List[FixResult]):
        """Generate detailed report of fixes"""
        report_path = f"url_fixing_report_{int(time.time())}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SIMPLE QA URL FIXING REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total records processed: {self.stats['total_records']}\n")
            f.write(f"Records with bad URLs: {self.stats['records_with_bad_urls']}\n")
            f.write(f"Records successfully fixed: {self.stats['records_fixed']}\n")
            f.write(f"Total URLs replaced: {self.stats['total_urls_replaced']}\n")
            f.write(f"Search API requests made: {self.searcher.request_count}\n\n")

            # Records that were fixed
            fixed_records = [r for r in fix_results if r.num_urls_replaced > 0]
            f.write(f"DETAILED FIXES ({len(fixed_records)} records)\n")
            f.write("-" * 80 + "\n")

            for fix_result in fixed_records:
                f.write(f"Record {fix_result.record_id}:\n")
                f.write(f"Question: {fix_result.question}\n")
                f.write(f"URLs replaced: {fix_result.num_urls_replaced}\n\n")

                for replacement in fix_result.replacements:
                    f.write(f"  Original: {replacement.original_url}\n")
                    f.write(f"  New:      {replacement.new_url}\n")
                    f.write(f"  Title:    {replacement.title}\n")
                    f.write(f"  Confidence: {replacement.confidence_score:.2f}\n\n")

                f.write("-" * 40 + "\n")

        logger.info(f"Detailed report saved to: {report_path}")

    def close(self):
        """Clean up resources"""
        self.validator.close()

    def run(self):
        """Main execution method"""
        logger.info("Starting Simple QA URL fixing process...")
        logger.info(f"Input: {self.input_csv}")
        logger.info(f"Output: {self.output_csv}")
        logger.info(f"Dry run: {self.dry_run}")

        try:
            # Process test set
            fix_results = self.process_test_set()

            if not fix_results:
                logger.error("No records processed. Exiting.")
                return

            # Write fixed CSV
            self.write_fixed_csv(fix_results)

            # Generate JSON and text reports
            final_report = self.generate_final_json_report(fix_results)
            self.generate_report(fix_results)

            # Print comprehensive summary
            logger.info("\n" + "=" * 80)
            logger.info("🎯 FINAL SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Session ID: {self.session_id}")
            logger.info(f"Duration: {final_report['session_info']['duration_seconds']:.1f} seconds")
            logger.info(f"Total records processed: {self.stats['total_records']}")
            logger.info(f"Records with bad URLs: {self.stats['records_with_bad_urls']}")
            logger.info(f"Records successfully fixed: {self.stats['records_fixed']}")
            logger.info(f"Total URLs replaced: {self.stats['total_urls_replaced']}")
            logger.info(f"Serper API calls made: {self.stats['search_requests']}")
            logger.info(f"Serper cached requests: {self.stats['cached_requests']}")
            logger.info(f"Cache hit rate: {final_report['cost_analysis']['cache_hit_rate']:.1%}")
            logger.info(f"Estimated cost: ${final_report['cost_analysis']['estimated_serper_cost_usd']:.3f}")

            # Summary by action type
            summary = final_report['summary_by_action']
            logger.info(f"\nBreakdown by action:")
            logger.info(f"  ✅ No action needed: {summary['no_action_needed']}")
            logger.info(f"  🔧 Partial fixes: {summary['partial_fixes']}")
            logger.info(f"  🎯 Complete fixes: {summary['complete_fixes']}")
            logger.info(f"  ❌ Unfixable: {summary['unfixable_records']}")

            logger.info(f"\n📁 Logs saved to: {self.logs_dir}")
            logger.info(f"📊 Final report: {self.logs_dir}/{self.session_id}_final_report.json")
            logger.info(f"📄 Record logs: {self.logs_dir}/records/")

            if not self.dry_run:
                logger.info(f"📝 Fixed CSV: {self.output_csv}")
            else:
                logger.info("🔍 DRY RUN: No changes were made")

        finally:
            self.close()


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Fix bad URLs in Simple QA test set using Serper.dev search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_simple_qa_urls.py --dry-run
  python fix_simple_qa_urls.py --input-csv test_set.csv --output-csv fixed_test_set.csv
  python fix_simple_qa_urls.py --max-search-results 15 --cache-dir ../cache
        """
    )

    parser.add_argument('--input-csv', default='simple_qa_test_set.csv',
                        help='Path to input CSV file (default: simple_qa_test_set.csv)')
    parser.add_argument('--output-csv', default='simple_qa_test_set_fixed.csv',
                        help='Path to output CSV file (default: simple_qa_test_set_fixed.csv)')
    parser.add_argument('--cache-dir', default='../cache/url_cache',
                        help='Path to URL cache directory (default: ../cache/url_cache)')
    parser.add_argument('--kb-dir', default='../knowledge_base_full',
                        help='Path to knowledge base directory (default: ../knowledge_base_full)')
    parser.add_argument('--max-search-results', type=int, default=10,
                        help='Maximum search results to consider per query (default: 10)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit processing to first N records (default: process all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    serper_api_key = os.getenv('SERPER_API_KEY')
    if not serper_api_key:
        logger.error("SERPER_API_KEY not found in environment variables")
        logger.error("Please set your Serper.dev API key in the .env file")
        return 1

    try:
        # Initialize fixer
        fixer = SimpleQAURLFixer(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            cache_dir=args.cache_dir,
            kb_dir=args.kb_dir,
            serper_api_key=serper_api_key,
            max_search_results=args.max_search_results,
            limit=args.limit,
            dry_run=args.dry_run
        )

        # Run the fixing process
        fixer.run()

        logger.info("URL fixing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"URL fixing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())