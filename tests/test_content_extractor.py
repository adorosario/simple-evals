"""
Unit tests for Content Extractor component
"""

import pytest
from unittest.mock import patch, MagicMock

from src.content_extractor import ContentExtractor, ExtractedContent
from src.url_fetcher import FetchResult


class TestContentExtractor:
    """Test cases for ContentExtractor class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.extractor = ContentExtractor()

    def test_init_default_config(self):
        """Test ContentExtractor initialization with defaults"""
        extractor = ContentExtractor()
        assert extractor.max_text_length == 1_000_000
        assert extractor.preserve_structure is True
        assert extractor.extract_links is False
        assert extractor.extract_metadata is True

    def test_init_custom_config(self):
        """Test ContentExtractor initialization with custom config"""
        extractor = ContentExtractor(
            max_text_length=500_000,
            preserve_structure=False,
            extract_links=True,
            extract_metadata=False
        )
        assert extractor.max_text_length == 500_000
        assert extractor.preserve_structure is False
        assert extractor.extract_links is True
        assert extractor.extract_metadata is False

    def test_detect_content_type_from_url(self):
        """Test content type detection from URL"""
        # PDF
        assert 'pdf' in self.extractor._detect_content_type(b'', 'http://example.com/doc.pdf')

        # HTML
        assert 'html' in self.extractor._detect_content_type(b'', 'http://example.com/page.html')

        # Text
        assert 'text' in self.extractor._detect_content_type(b'', 'http://example.com/readme.txt')

        # JSON
        assert 'json' in self.extractor._detect_content_type(b'', 'http://example.com/data.json')

    def test_detect_content_type_from_content(self):
        """Test content type detection from content magic bytes"""
        # PDF magic bytes
        pdf_content = b'%PDF-1.4\n'
        assert 'pdf' in self.extractor._detect_content_type(pdf_content, 'http://example.com/unknown')

        # HTML content
        html_content = b'<html><head><title>Test</title></head></html>'
        assert 'html' in self.extractor._detect_content_type(html_content, 'http://example.com/unknown')

        # DOCTYPE HTML
        doctype_content = b'<!doctype html><html></html>'
        assert 'html' in self.extractor._detect_content_type(doctype_content, 'http://example.com/unknown')

        # XML content
        xml_content = b'<?xml version="1.0"?><root></root>'
        assert 'xml' in self.extractor._detect_content_type(xml_content, 'http://example.com/unknown')

    def test_decode_content_utf8(self):
        """Test content decoding with UTF-8"""
        content = "Hello, 世界! 🌍".encode('utf-8')
        decoded = self.extractor._decode_content(content, 'utf-8')
        assert decoded == "Hello, 世界! 🌍"

    def test_decode_content_latin1(self):
        """Test content decoding with Latin-1"""
        content = "Café résumé".encode('latin-1')
        decoded = self.extractor._decode_content(content, 'latin-1')
        assert decoded == "Café résumé"

    def test_decode_content_auto_detect(self):
        """Test content decoding with auto-detection"""
        content = "Hello, world!".encode('utf-8')
        decoded = self.extractor._decode_content(content, None)
        assert decoded == "Hello, world!"

    def test_decode_content_fallback(self):
        """Test content decoding with fallback to error ignore"""
        # Invalid UTF-8 bytes
        content = b'\xff\xfe\x00\x00Hello'
        decoded = self.extractor._decode_content(content, None)
        assert decoded is not None  # Should not fail
        assert 'Hello' in decoded

    def test_clean_text(self):
        """Test text cleaning functionality"""
        # Multiple spaces
        dirty_text = "Hello    world   with     spaces"
        clean = self.extractor._clean_text(dirty_text)
        assert clean == "Hello world with spaces"

        # Multiple newlines
        dirty_text = "Line 1\n\n\n\nLine 2\n\n\n\nLine 3"
        clean = self.extractor._clean_text(dirty_text)
        assert clean == "Line 1\n\nLine 2\n\nLine 3"

        # Leading/trailing whitespace
        dirty_text = "   \n\nText with whitespace\n\n   "
        clean = self.extractor._clean_text(dirty_text)
        assert clean == "Text with whitespace"

    def test_extract_text_plain(self):
        """Test plain text extraction"""
        content = "This is plain text content.\nWith multiple lines."
        result = self.extractor.extract_from_content(
            content.encode('utf-8'),
            'http://example.com/text.txt',
            'text/plain'
        )

        assert result.success is True
        assert result.text == "This is plain text content. With multiple lines."
        assert result.content_type == 'text/plain'
        assert result.word_count == 8
        assert result.extraction_method == 'direct'

    @patch('src.content_extractor.HAS_BS4', True)
    def test_extract_html_with_beautifulsoup(self):
        """Test HTML extraction with BeautifulSoup available"""
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Heading</h1>
            <p>This is a paragraph with important content.</p>
            <script>console.log('ignored');</script>
            <style>body { color: red; }</style>
            <div>Another paragraph with text.</div>
        </body>
        </html>
        """

        with patch('src.content_extractor.BeautifulSoup') as mock_bs:
            # Mock BeautifulSoup behavior
            mock_soup = MagicMock()
            mock_bs.return_value = mock_soup

            # Mock title
            mock_title = MagicMock()
            mock_title.get_text.return_value = "Test Page"
            mock_soup.find.return_value = mock_title

            # Mock script/style removal
            mock_soup.__call__.return_value = []

            # Mock content selection
            mock_soup.select_one.return_value = None  # No main content areas

            # Mock text extraction
            mock_elem1 = MagicMock()
            mock_elem1.get_text.return_value = "Main Heading"
            mock_elem2 = MagicMock()
            mock_elem2.get_text.return_value = "This is a paragraph with important content."
            mock_soup.find_all.return_value = [mock_elem1, mock_elem2]

            result = self.extractor.extract_from_content(
                html_content.encode('utf-8'),
                'http://example.com/test.html',
                'text/html'
            )

            assert result.success is True
            assert result.title == "Test Page"
            assert result.content_type == 'text/html'
            assert result.extraction_method == 'beautifulsoup'

    @patch('src.content_extractor.HAS_BS4', False)
    def test_extract_html_fallback_without_beautifulsoup(self):
        """Test HTML extraction fallback when BeautifulSoup not available"""
        html_content = "<html><body>Simple text content</body></html>"

        result = self.extractor.extract_from_content(
            html_content.encode('utf-8'),
            'http://example.com/test.html',
            'text/html'
        )

        # Should fall back to text extraction
        assert result.success is True
        assert result.content_type == 'text/plain'
        assert result.extraction_method == 'direct'

    def test_extract_from_fetch_result_success(self):
        """Test extraction from successful fetch result"""
        fetch_result = FetchResult(
            url="http://example.com/test.txt",
            success=True,
            content=b"Test content from fetch",
            content_type="text/plain",
            encoding="utf-8"
        )

        result = self.extractor.extract_from_fetch_result(fetch_result)

        assert result.success is True
        assert result.url == "http://example.com/test.txt"
        assert result.text == "Test content from fetch"

    def test_extract_from_fetch_result_failure(self):
        """Test extraction from failed fetch result"""
        fetch_result = FetchResult(
            url="http://example.com/notfound.txt",
            success=False,
            error_message="404 Not Found"
        )

        result = self.extractor.extract_from_fetch_result(fetch_result)

        assert result.success is False
        assert result.url == "http://example.com/notfound.txt"
        assert result.error_message == "404 Not Found"

    def test_extract_from_fetch_result_no_content(self):
        """Test extraction from fetch result with no content"""
        fetch_result = FetchResult(
            url="http://example.com/empty.txt",
            success=True,
            content=None
        )

        result = self.extractor.extract_from_fetch_result(fetch_result)

        assert result.success is False
        assert "No content to extract" in result.error_message

    def test_extract_empty_content(self):
        """Test extraction from empty content"""
        result = self.extractor.extract_from_content(
            b'',
            'http://example.com/empty.txt'
        )

        assert result.success is False
        assert result.error_message == "Empty content"

    def test_extract_multiple_contents(self):
        """Test extracting from multiple fetch results"""
        fetch_results = [
            FetchResult(
                url="http://example.com/1.txt",
                success=True,
                content=b"Content 1",
                content_type="text/plain"
            ),
            FetchResult(
                url="http://example.com/2.txt",
                success=True,
                content=b"Content 2",
                content_type="text/plain"
            ),
            FetchResult(
                url="http://example.com/3.txt",
                success=False,
                error_message="Failed to fetch"
            )
        ]

        results = self.extractor.extract_multiple(fetch_results)

        assert len(results) == 3
        assert results[0].success is True
        assert results[0].text == "Content 1"
        assert results[1].success is True
        assert results[1].text == "Content 2"
        assert results[2].success is False

    def test_get_stats_empty(self):
        """Test statistics with empty results"""
        stats = self.extractor.get_stats([])

        assert stats['total_extractions'] == 0
        assert stats['successful'] == 0
        assert stats['failed'] == 0
        assert stats['success_rate'] == 0

    def test_get_stats_with_results(self):
        """Test statistics with mixed results"""
        results = [
            ExtractedContent(
                url="http://example.com/1",
                success=True,
                text="Content 1",
                content_type="text/html",
                word_count=10,
                extraction_method="beautifulsoup"
            ),
            ExtractedContent(
                url="http://example.com/2",
                success=False,
                error_message="Failed"
            ),
            ExtractedContent(
                url="http://example.com/3",
                success=True,
                text="Content 3",
                content_type="text/plain",
                word_count=20,
                extraction_method="direct"
            )
        ]

        stats = self.extractor.get_stats(results)

        assert stats['total_extractions'] == 3
        assert stats['successful'] == 2
        assert stats['failed'] == 1
        assert stats['success_rate'] == 2/3
        assert stats['content_types']['text/html'] == 1
        assert stats['content_types']['text/plain'] == 1
        assert stats['extraction_methods']['beautifulsoup'] == 1
        assert stats['extraction_methods']['direct'] == 1
        assert stats['avg_word_count'] == 15
        assert stats['total_words'] == 30

    def test_max_text_length_limit(self):
        """Test that text is truncated to max length"""
        extractor = ContentExtractor(max_text_length=10)
        long_content = "This is a very long text content that exceeds the limit"

        result = extractor.extract_from_content(
            long_content.encode('utf-8'),
            'http://example.com/long.txt',
            'text/plain'
        )

        assert result.success is True
        assert len(result.text) == 10

    def test_extract_with_fallback_html_detection(self):
        """Test fallback extraction with HTML-like content"""
        # Content that looks like HTML but no explicit content type
        html_content = b"<div>Some content</div><p>More text</p>"

        with patch.object(self.extractor, '_extract_html') as mock_html:
            mock_html.return_value = ExtractedContent(
                url="http://example.com",
                success=True,
                text="Some content More text"
            )

            result = self.extractor.extract_from_content(
                html_content,
                'http://example.com/unknown',
                None  # No content type
            )

            # Should try HTML extraction first
            mock_html.assert_called_once()

    def test_extract_with_exception_handling(self):
        """Test that exceptions during extraction are handled gracefully"""
        with patch.object(self.extractor, '_extract_text', side_effect=Exception("Test error")):
            result = self.extractor.extract_from_content(
                b"test content",
                'http://example.com/test.txt',
                'text/plain'
            )

            assert result.success is False
            assert "Extraction error: Test error" in result.error_message

    @pytest.mark.parametrize("content_type,expected_method", [
        ('text/html', '_extract_html'),
        ('application/pdf', '_extract_pdf'),
        ('text/plain', '_extract_text'),
        ('application/json', '_extract_text'),
        ('application/xml', '_extract_text'),
        ('unknown/type', '_extract_with_fallback'),
    ])
    def test_content_type_routing(self, content_type, expected_method):
        """Test that content types are routed to correct extraction methods"""
        with patch.object(self.extractor, expected_method) as mock_method:
            mock_method.return_value = ExtractedContent(
                url="http://example.com",
                success=True
            )

            self.extractor.extract_from_content(
                b"test content",
                'http://example.com/test',
                content_type
            )

            mock_method.assert_called_once()