"""
Cache Quality Auditor for SimpleQA-Verified Knowledge Base

Validates that cached URL content meets quality standards:
- Cache entry exists and is successful
- Content is non-empty
- Content is extractable (not JS-only shell)
- Extracted text meets minimum word count
- Content is not an error page (404, 403, etc.)
"""

import hashlib
import json
import pickle
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class URLQualityResult:
    """Quality assessment for a single URL."""
    url: str
    cache_key: str
    checks: Dict[str, bool] = field(default_factory=dict)
    word_count: int = 0
    content_size: int = 0
    content_type: Optional[str] = None
    extracted_text_preview: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """URL passes all quality checks."""
        return all(self.checks.values())

    @property
    def passed_checks(self) -> int:
        return sum(1 for v in self.checks.values() if v)

    @property
    def failed_checks(self) -> List[str]:
        return [k for k, v in self.checks.items() if not v]


@dataclass
class CacheQualityReport:
    """Aggregate quality report for cache audit."""
    total_urls: int = 0
    valid_urls: int = 0
    invalid_urls: int = 0
    results: List[URLQualityResult] = field(default_factory=list)
    check_summary: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'summary': {
                'total_urls': self.total_urls,
                'valid_urls': self.valid_urls,
                'invalid_urls': self.invalid_urls,
                'validity_rate': self.valid_urls / max(self.total_urls, 1),
            },
            'check_breakdown': self.check_summary,
            'invalid_url_details': [
                {
                    'url': r.url,
                    'cache_key': r.cache_key,
                    'failed_checks': r.failed_checks,
                    'word_count': r.word_count,
                    'error': r.error,
                }
                for r in self.results if not r.is_valid
            ],
            'statistics': {
                'word_count_distribution': self._get_word_count_stats(),
                'content_size_distribution': self._get_content_size_stats(),
            }
        }

    def _get_word_count_stats(self) -> dict:
        word_counts = [r.word_count for r in self.results if r.word_count > 0]
        if not word_counts:
            return {}
        return {
            'min': min(word_counts),
            'max': max(word_counts),
            'avg': sum(word_counts) / len(word_counts),
            'median': sorted(word_counts)[len(word_counts) // 2],
            'below_50': sum(1 for w in word_counts if w < 50),
            'below_100': sum(1 for w in word_counts if w < 100),
        }

    def _get_content_size_stats(self) -> dict:
        sizes = [r.content_size for r in self.results if r.content_size > 0]
        if not sizes:
            return {}
        return {
            'min_kb': min(sizes) / 1024,
            'max_kb': max(sizes) / 1024,
            'avg_kb': sum(sizes) / len(sizes) / 1024,
        }


class CacheQualityAuditor:
    """
    Audits cached URL content quality for SimpleQA-Verified dataset.
    """

    # Error page indicators
    ERROR_PAGE_PATTERNS = [
        r'page not found',
        r'404\s*(error|not found)',
        r'403\s*(forbidden|error)',
        r'access denied',
        r'server error',
        r'internal server error',
        r'service unavailable',
        r'this page (doesn\'t|does not) exist',
        r'requested page (is|was) not found',
        r'couldn\'t find what you\'re looking for',
        r'sorry, we couldn\'t find',
    ]

    # JavaScript shell indicators (page requires JS to render)
    JS_SHELL_PATTERNS = [
        r'enable javascript',
        r'javascript (is )?required',
        r'please enable javascript',
        r'you need to enable javascript',
        r'noscript',
    ]

    def __init__(
        self,
        cache_dir: str = "cache/url_cache",
        min_words: int = 50,
        min_content_size: int = 500,  # bytes
        max_workers: int = 10,
    ):
        """
        Initialize cache quality auditor.

        Args:
            cache_dir: Directory containing cache files
            min_words: Minimum word count for valid content
            min_content_size: Minimum content size in bytes
            max_workers: Max threads for parallel processing
        """
        self.cache_dir = Path(cache_dir)
        self.min_words = min_words
        self.min_content_size = min_content_size
        self.max_workers = max_workers

        # Load cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

        # Lazy import content extractor
        self._extractor = None

    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Cache metadata not found or corrupted")
                return {}
        return {}

    @property
    def extractor(self):
        """Lazy load content extractor."""
        if self._extractor is None:
            from .content_extractor import ContentExtractor
            self._extractor = ContentExtractor()
        return self._extractor

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_fetch_result(self, cache_key: str):
        """Load FetchResult from cache."""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache {cache_key}: {e}")
            return None

    def _is_error_page(self, text: str) -> bool:
        """Check if text looks like an error page."""
        text_lower = text.lower()
        for pattern in self.ERROR_PAGE_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def _is_js_shell(self, text: str) -> bool:
        """Check if text is just a JS shell with no real content."""
        text_lower = text.lower()
        for pattern in self.JS_SHELL_PATTERNS:
            if re.search(pattern, text_lower):
                return True

        # Also check if content is suspiciously short despite cache success
        words = text.split()
        if len(words) < 20 and any(w in text_lower for w in ['script', 'javascript', 'loading']):
            return True

        return False

    def audit_url(self, url: str) -> URLQualityResult:
        """
        Audit a single URL's cache quality.

        Args:
            url: URL to audit

        Returns:
            URLQualityResult with detailed checks
        """
        cache_key = self._get_cache_key(url)
        result = URLQualityResult(url=url, cache_key=cache_key)

        # Check 1: Cache file exists
        cache_path = self._get_cache_path(cache_key)
        result.checks['cache_exists'] = cache_path.exists()
        if not result.checks['cache_exists']:
            result.error = "Cache file not found"
            return result

        # Check 2: Can load from cache
        fetch_result = self._load_fetch_result(cache_key)
        result.checks['cache_loadable'] = fetch_result is not None
        if not result.checks['cache_loadable']:
            result.error = "Failed to load cache file"
            return result

        # Check 3: Fetch was successful
        result.checks['fetch_success'] = fetch_result.success
        result.content_type = fetch_result.content_type
        if not result.checks['fetch_success']:
            result.error = f"Fetch failed: {fetch_result.error_message}"
            return result

        # Check 4: Content non-empty
        content_size = len(fetch_result.content) if fetch_result.content else 0
        result.content_size = content_size
        result.checks['content_non_empty'] = content_size > 0
        if not result.checks['content_non_empty']:
            result.error = "Empty content"
            return result

        # Check 5: Content size above minimum
        result.checks['min_size'] = content_size >= self.min_content_size

        # Check 6: Extract text
        try:
            extracted = self.extractor.extract_from_fetch_result(fetch_result)
            result.checks['extractable'] = extracted.success and bool(extracted.text)

            if extracted.success and extracted.text:
                text = extracted.text
                result.word_count = len(text.split())
                result.extracted_text_preview = text[:200] + "..." if len(text) > 200 else text

                # Check 7: Minimum word count
                result.checks['min_words'] = result.word_count >= self.min_words

                # Check 8: Not an error page
                result.checks['not_error_page'] = not self._is_error_page(text)

                # Check 9: Not a JS shell
                result.checks['not_js_shell'] = not self._is_js_shell(text)
            else:
                result.checks['extractable'] = False
                result.error = f"Extraction failed: {extracted.error_message}"
                result.checks['min_words'] = False
                result.checks['not_error_page'] = True  # Unknown
                result.checks['not_js_shell'] = True  # Unknown

        except Exception as e:
            result.checks['extractable'] = False
            result.error = f"Extraction error: {str(e)}"
            result.checks['min_words'] = False
            result.checks['not_error_page'] = True
            result.checks['not_js_shell'] = True

        return result

    def audit_urls(self, urls: List[str], progress_callback=None) -> CacheQualityReport:
        """
        Audit multiple URLs in parallel.

        Args:
            urls: List of URLs to audit
            progress_callback: Optional callback(current, total)

        Returns:
            CacheQualityReport with all results
        """
        report = CacheQualityReport()
        report.total_urls = len(urls)

        # Initialize check summary
        check_names = [
            'cache_exists', 'cache_loadable', 'fetch_success',
            'content_non_empty', 'min_size', 'extractable',
            'min_words', 'not_error_page', 'not_js_shell'
        ]
        report.check_summary = {name: {'passed': 0, 'failed': 0} for name in check_names}

        # Audit in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {executor.submit(self.audit_url, url): url for url in urls}

            for future in as_completed(future_to_url):
                result = future.result()
                report.results.append(result)

                # Update counts
                if result.is_valid:
                    report.valid_urls += 1
                else:
                    report.invalid_urls += 1

                # Update check summary
                for check_name, passed in result.checks.items():
                    if check_name in report.check_summary:
                        if passed:
                            report.check_summary[check_name]['passed'] += 1
                        else:
                            report.check_summary[check_name]['failed'] += 1

                completed += 1
                if progress_callback:
                    progress_callback(completed, report.total_urls)

        return report

    def save_report(self, report: CacheQualityReport, output_path: Path) -> None:
        """Save audit report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

    def get_valid_urls(self, urls: List[str]) -> List[str]:
        """
        Quick check to return only valid URLs from list.

        Args:
            urls: URLs to check

        Returns:
            List of URLs that pass quality checks
        """
        report = self.audit_urls(urls)
        return [r.url for r in report.results if r.is_valid]

    def get_invalid_urls(self, urls: List[str]) -> List[str]:
        """
        Return URLs that fail quality checks.

        Args:
            urls: URLs to check

        Returns:
            List of URLs that fail quality checks
        """
        report = self.audit_urls(urls)
        return [r.url for r in report.results if not r.is_valid]


if __name__ == "__main__":
    # Test with a sample URL
    import sys

    logging.basicConfig(level=logging.INFO)

    auditor = CacheQualityAuditor(cache_dir="cache/url_cache")

    # Test URLs
    test_urls = [
        "https://en.wikipedia.org/wiki/Stella_Obasanjo",
        "https://www.nonexistent-url-test-12345.com",
    ]

    print("Cache Quality Auditor Test")
    print("=" * 60)

    for url in test_urls:
        result = auditor.audit_url(url)
        print(f"\nURL: {url[:50]}...")
        print(f"Valid: {result.is_valid}")
        print(f"Checks: {result.checks}")
        if result.word_count:
            print(f"Word count: {result.word_count}")
        if result.error:
            print(f"Error: {result.error}")
