"""
URL Fetcher Component

Reliably fetch content from URLs with robust error handling, retries, and proper timeouts.
Handles various content types and provides detailed error information.
"""

import requests
import time
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a URL fetch operation"""
    url: str
    success: bool
    content: Optional[bytes] = None
    content_type: Optional[str] = None
    encoding: Optional[str] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    final_url: Optional[str] = None  # After redirects


class URLFetcher:
    """
    Robust URL fetcher with configurable retries, timeouts, and error handling.
    """

    def __init__(self,
                 timeout: int = 30,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 max_content_size: int = 50 * 1024 * 1024,  # 50MB
                 user_agent: str = None):
        """
        Initialize URL fetcher with configuration.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            max_content_size: Maximum content size to fetch in bytes
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_content_size = max_content_size
        self.user_agent = user_agent or (
            "Mozilla/5.0 (compatible; RAG-Benchmark/1.0; "
            "+https://github.com/adorosario/simple-evals)"
        )

        # Configure session with connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and fetchable"""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ('http', 'https') and parsed.netloc
        except Exception:
            return False

    def _get_content_info(self, response: requests.Response) -> Tuple[str, str]:
        """Extract content type and encoding from response"""
        content_type = response.headers.get('content-type', '').lower()

        # Extract main content type
        main_type = content_type.split(';')[0].strip()

        # Extract encoding
        encoding = response.encoding or 'utf-8'
        if 'charset=' in content_type:
            try:
                encoding = content_type.split('charset=')[1].split(';')[0].strip()
            except (IndexError, AttributeError):
                pass

        return main_type, encoding

    def fetch(self, url: str) -> FetchResult:
        """
        Fetch content from a URL with retries and error handling.

        Args:
            url: URL to fetch

        Returns:
            FetchResult with content and metadata
        """
        if not self._is_valid_url(url):
            return FetchResult(
                url=url,
                success=False,
                error_message="Invalid URL format"
            )

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Fetching {url} (attempt {attempt + 1}/{self.max_retries + 1})")

                # Make request with streaming to check content size
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    stream=True,
                    allow_redirects=True
                )

                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_content_size:
                    return FetchResult(
                        url=url,
                        success=False,
                        status_code=response.status_code,
                        error_message=f"Content too large: {content_length} bytes"
                    )

                # Check if successful
                response.raise_for_status()

                # Download content with size limit
                content = b''
                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                    if len(content) > self.max_content_size:
                        return FetchResult(
                            url=url,
                            success=False,
                            status_code=response.status_code,
                            error_message=f"Content too large: >{self.max_content_size} bytes"
                        )

                # Extract content information
                content_type, encoding = self._get_content_info(response)
                response_time = time.time() - start_time

                logger.info(f"Successfully fetched {url} ({len(content)} bytes, {content_type})")

                return FetchResult(
                    url=url,
                    success=True,
                    content=content,
                    content_type=content_type,
                    encoding=encoding,
                    status_code=response.status_code,
                    response_time=response_time,
                    final_url=response.url
                )

            except requests.exceptions.Timeout as e:
                last_error = f"Timeout after {self.timeout}s: {str(e)}"
                logger.warning(f"Timeout fetching {url}: {e}")

            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {str(e)}"
                logger.warning(f"Connection error fetching {url}: {e}")

            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error {e.response.status_code}: {str(e)}"
                logger.warning(f"HTTP error fetching {url}: {e}")
                # Don't retry HTTP errors like 404, 403
                if e.response.status_code in (400, 401, 403, 404, 410):
                    break

            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
                logger.warning(f"Request error fetching {url}: {e}")

            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error fetching {url}: {e}")

            # Wait before retry (except on last attempt)
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

        response_time = time.time() - start_time
        return FetchResult(
            url=url,
            success=False,
            error_message=last_error,
            response_time=response_time
        )

    def fetch_multiple(self, urls: list[str],
                      delay_between_requests: float = 0.1) -> list[FetchResult]:
        """
        Fetch multiple URLs with optional delay between requests.

        Args:
            urls: List of URLs to fetch
            delay_between_requests: Delay between requests in seconds

        Returns:
            List of FetchResult objects
        """
        results = []
        total_urls = len(urls)

        logger.info(f"Fetching {total_urls} URLs...")

        for i, url in enumerate(urls):
            logger.debug(f"Processing URL {i+1}/{total_urls}: {url}")

            result = self.fetch(url)
            results.append(result)

            # Progress logging
            if (i + 1) % 10 == 0 or i + 1 == total_urls:
                successful = sum(1 for r in results if r.success)
                logger.info(f"Progress: {i+1}/{total_urls} URLs processed "
                           f"({successful} successful)")

            # Rate limiting
            if delay_between_requests > 0 and i < total_urls - 1:
                time.sleep(delay_between_requests)

        return results

    def get_stats(self, results: list[FetchResult]) -> Dict[str, Any]:
        """Get statistics from fetch results"""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful

        # Error breakdown
        error_counts = {}
        response_times = []
        content_sizes = []
        content_types = {}

        for result in results:
            if result.response_time:
                response_times.append(result.response_time)

            if result.success:
                if result.content:
                    content_sizes.append(len(result.content))
                if result.content_type:
                    content_types[result.content_type] = content_types.get(result.content_type, 0) + 1
            else:
                if result.error_message:
                    error_type = result.error_message.split(':')[0]
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1

        stats = {
            'total_urls': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'error_counts': error_counts,
            'content_types': content_types
        }

        if response_times:
            stats['avg_response_time'] = sum(response_times) / len(response_times)
            stats['min_response_time'] = min(response_times)
            stats['max_response_time'] = max(response_times)

        if content_sizes:
            stats['avg_content_size'] = sum(content_sizes) / len(content_sizes)
            stats['min_content_size'] = min(content_sizes)
            stats['max_content_size'] = max(content_sizes)
            stats['total_content_size'] = sum(content_sizes)

        return stats

    def close(self):
        """Close the session"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()