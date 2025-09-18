"""
Unit tests for URL Fetcher component
"""

import pytest
import responses
import requests
from unittest.mock import patch, MagicMock
import time

from src.url_fetcher import URLFetcher, FetchResult


class TestURLFetcher:
    """Test cases for URLFetcher class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.fetcher = URLFetcher(
            timeout=5,
            max_retries=2,
            retry_delay=0.1  # Fast retries for testing
        )

    def teardown_method(self):
        """Clean up after tests"""
        self.fetcher.close()

    def test_init_default_config(self):
        """Test URLFetcher initialization with default config"""
        fetcher = URLFetcher()
        assert fetcher.timeout == 30
        assert fetcher.max_retries == 3
        assert fetcher.retry_delay == 1.0
        assert fetcher.max_content_size == 50 * 1024 * 1024
        assert "RAG-Benchmark" in fetcher.user_agent
        fetcher.close()

    def test_init_custom_config(self):
        """Test URLFetcher initialization with custom config"""
        fetcher = URLFetcher(
            timeout=10,
            max_retries=5,
            retry_delay=2.0,
            max_content_size=1024,
            user_agent="Custom Agent"
        )
        assert fetcher.timeout == 10
        assert fetcher.max_retries == 5
        assert fetcher.retry_delay == 2.0
        assert fetcher.max_content_size == 1024
        assert fetcher.user_agent == "Custom Agent"
        fetcher.close()

    def test_is_valid_url(self):
        """Test URL validation"""
        # Valid URLs
        assert self.fetcher._is_valid_url("http://example.com")
        assert self.fetcher._is_valid_url("https://example.com/path")
        assert self.fetcher._is_valid_url("https://sub.example.com:8080/path?param=value")

        # Invalid URLs
        assert not self.fetcher._is_valid_url("ftp://example.com")
        assert not self.fetcher._is_valid_url("not_a_url")
        assert not self.fetcher._is_valid_url("")
        assert not self.fetcher._is_valid_url("http://")
        assert not self.fetcher._is_valid_url("https://")

    @responses.activate
    def test_fetch_successful_html(self):
        """Test successful fetch of HTML content"""
        url = "http://example.com"
        content = "<html><body>Test content</body></html>"

        responses.add(
            responses.GET,
            url,
            body=content,
            status=200,
            content_type="text/html; charset=utf-8"
        )

        result = self.fetcher.fetch(url)

        assert result.success is True
        assert result.url == url
        assert result.content == content.encode('utf-8')
        assert result.content_type == "text/html"
        assert result.encoding == "utf-8"
        assert result.status_code == 200
        assert result.error_message is None
        assert result.response_time is not None
        assert result.response_time > 0

    @responses.activate
    def test_fetch_successful_json(self):
        """Test successful fetch of JSON content"""
        url = "http://api.example.com/data"
        content = '{"key": "value"}'

        responses.add(
            responses.GET,
            url,
            body=content,
            status=200,
            content_type="application/json"
        )

        result = self.fetcher.fetch(url)

        assert result.success is True
        assert result.content_type == "application/json"
        assert result.content == content.encode('utf-8')

    @responses.activate
    def test_fetch_http_404_error(self):
        """Test fetch with 404 error (no retry)"""
        url = "http://example.com/notfound"

        responses.add(
            responses.GET,
            url,
            status=404
        )

        result = self.fetcher.fetch(url)

        assert result.success is False
        assert result.url == url
        assert result.content is None
        assert "404" in result.error_message
        assert result.response_time is not None

    @responses.activate
    def test_fetch_http_500_error_with_retry(self):
        """Test fetch with 500 error (should retry)"""
        url = "http://example.com/server_error"

        # First two requests fail, third succeeds
        responses.add(responses.GET, url, status=500)
        responses.add(responses.GET, url, status=500)
        responses.add(responses.GET, url, body="Success", status=200)

        result = self.fetcher.fetch(url)

        assert result.success is True
        assert result.content == b"Success"
        assert len(responses.calls) == 3  # Should have retried

    @responses.activate
    def test_fetch_connection_error_with_retry(self):
        """Test fetch with connection error (should retry)"""
        url = "http://example.com"

        # Simulate connection errors
        responses.add(responses.GET, url, body=requests.ConnectionError("Connection failed"))
        responses.add(responses.GET, url, body=requests.ConnectionError("Connection failed"))
        responses.add(responses.GET, url, body="Success", status=200)

        result = self.fetcher.fetch(url)

        assert result.success is True
        assert result.content == b"Success"

    @responses.activate
    def test_fetch_timeout_error(self):
        """Test fetch with timeout error"""
        url = "http://example.com"

        def timeout_callback(request):
            raise requests.Timeout("Request timed out")

        responses.add_callback(
            responses.GET,
            url,
            callback=timeout_callback
        )

        result = self.fetcher.fetch(url)

        assert result.success is False
        assert "Timeout" in result.error_message
        assert result.response_time is not None

    def test_fetch_invalid_url(self):
        """Test fetch with invalid URL"""
        result = self.fetcher.fetch("not_a_url")

        assert result.success is False
        assert result.url == "not_a_url"
        assert result.error_message == "Invalid URL format"
        assert result.content is None

    @responses.activate
    def test_fetch_content_too_large_header(self):
        """Test fetch with content too large (from Content-Length header)"""
        url = "http://example.com/large_file"

        responses.add(
            responses.GET,
            url,
            status=200,
            headers={"Content-Length": str(60 * 1024 * 1024)}  # 60MB > 50MB limit
        )

        result = self.fetcher.fetch(url)

        assert result.success is False
        assert "Content too large" in result.error_message

    @responses.activate
    def test_fetch_content_too_large_streaming(self):
        """Test fetch with content too large (detected during streaming)"""
        url = "http://example.com/large_stream"

        # Create large content
        large_content = "x" * (51 * 1024 * 1024)  # 51MB > 50MB limit

        responses.add(
            responses.GET,
            url,
            body=large_content,
            status=200,
            stream=True
        )

        result = self.fetcher.fetch(url)

        assert result.success is False
        assert "Content too large" in result.error_message

    @responses.activate
    def test_fetch_with_redirects(self):
        """Test fetch with redirects"""
        original_url = "http://example.com/redirect"
        final_url = "http://example.com/final"
        content = "Final content"

        responses.add(
            responses.GET,
            original_url,
            status=302,
            headers={"Location": final_url}
        )
        responses.add(
            responses.GET,
            final_url,
            body=content,
            status=200
        )

        result = self.fetcher.fetch(original_url)

        assert result.success is True
        assert result.url == original_url
        assert result.final_url == final_url
        assert result.content == content.encode()

    def test_get_content_info(self):
        """Test content type and encoding extraction"""
        # Mock response with content type
        mock_response = MagicMock()
        mock_response.headers = {
            'content-type': 'text/html; charset=utf-8'
        }
        mock_response.encoding = 'utf-8'

        content_type, encoding = self.fetcher._get_content_info(mock_response)

        assert content_type == "text/html"
        assert encoding == "utf-8"

        # Test with different content type
        mock_response.headers = {
            'content-type': 'application/json; charset=iso-8859-1'
        }
        mock_response.encoding = 'iso-8859-1'

        content_type, encoding = self.fetcher._get_content_info(mock_response)

        assert content_type == "application/json"
        assert encoding == "iso-8859-1"

    @responses.activate
    def test_fetch_multiple_urls(self):
        """Test fetching multiple URLs"""
        urls = [
            "http://example.com/1",
            "http://example.com/2",
            "http://example.com/3"
        ]

        for i, url in enumerate(urls):
            responses.add(
                responses.GET,
                url,
                body=f"Content {i+1}",
                status=200
            )

        results = self.fetcher.fetch_multiple(urls, delay_between_requests=0)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.success is True
            assert result.url == urls[i]
            assert result.content == f"Content {i+1}".encode()

    @responses.activate
    def test_fetch_multiple_with_failures(self):
        """Test fetching multiple URLs with some failures"""
        urls = [
            "http://example.com/success",
            "http://example.com/fail",
            "http://example.com/success2"
        ]

        responses.add(responses.GET, urls[0], body="Success 1", status=200)
        responses.add(responses.GET, urls[1], status=404)
        responses.add(responses.GET, urls[2], body="Success 2", status=200)

        results = self.fetcher.fetch_multiple(urls, delay_between_requests=0)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

    def test_get_stats_empty(self):
        """Test statistics calculation with empty results"""
        stats = self.fetcher.get_stats([])

        assert stats['total_urls'] == 0
        assert stats['successful'] == 0
        assert stats['failed'] == 0
        assert stats['success_rate'] == 0

    def test_get_stats_with_results(self):
        """Test statistics calculation with mixed results"""
        results = [
            FetchResult(
                url="http://example.com/1",
                success=True,
                content=b"Content 1",
                content_type="text/html",
                response_time=1.0
            ),
            FetchResult(
                url="http://example.com/2",
                success=False,
                error_message="HTTP error 404",
                response_time=0.5
            ),
            FetchResult(
                url="http://example.com/3",
                success=True,
                content=b"Content 3 longer",
                content_type="text/html",
                response_time=2.0
            )
        ]

        stats = self.fetcher.get_stats(results)

        assert stats['total_urls'] == 3
        assert stats['successful'] == 2
        assert stats['failed'] == 1
        assert stats['success_rate'] == 2/3
        assert stats['avg_response_time'] == 1.17  # (1.0 + 0.5 + 2.0) / 3
        assert stats['content_types']['text/html'] == 2
        assert stats['error_counts']['HTTP error'] == 1

    def test_context_manager(self):
        """Test URLFetcher as context manager"""
        with URLFetcher() as fetcher:
            assert fetcher.session is not None
        # Session should be closed after context exit

    @pytest.mark.parametrize("url,expected", [
        ("http://example.com", True),
        ("https://example.com", True),
        ("ftp://example.com", False),
        ("invalid", False),
        ("", False),
    ])
    def test_url_validation_parametrized(self, url, expected):
        """Test URL validation with various inputs"""
        assert self.fetcher._is_valid_url(url) == expected