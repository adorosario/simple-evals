"""
Unit tests for Knowledge Base Builder component
"""

import pytest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from src.knowledge_base_builder import KnowledgeBaseBuilder, BuildResult
from src.url_fetcher import FetchResult
from src.content_extractor import ExtractedContent
from src.content_processor import ProcessedDocument


class TestKnowledgeBaseBuilder:
    """Test cases for KnowledgeBaseBuilder class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.builder = KnowledgeBaseBuilder(
            output_dir=os.path.join(self.temp_dir, 'output'),
            cache_dir=os.path.join(self.temp_dir, 'cache'),
            min_document_words=20,
            use_cache=False  # Disable cache for testing
        )

    def test_init_default_config(self):
        """Test KnowledgeBaseBuilder initialization with defaults"""
        builder = KnowledgeBaseBuilder()
        assert builder.output_dir == "knowledge_base"
        assert builder.cache_dir == "cache/url_cache"
        assert builder.min_document_words == 50
        assert builder.max_documents is None
        assert builder.use_cache is True
        assert builder.force_refresh is False

    def test_init_custom_config(self):
        """Test KnowledgeBaseBuilder initialization with custom config"""
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = KnowledgeBaseBuilder(
                output_dir=os.path.join(temp_dir, 'custom'),
                cache_dir=os.path.join(temp_dir, 'cache'),
                min_document_words=100,
                max_documents=500,
                use_cache=False,
                force_refresh=True
            )
            assert builder.output_dir == os.path.join(temp_dir, 'custom')
            assert builder.min_document_words == 100
            assert builder.max_documents == 500
            assert builder.use_cache is False
            assert builder.force_refresh is True

    def test_load_urls_from_file(self):
        """Test loading URLs from file"""
        # Create test URL file
        url_file = os.path.join(self.temp_dir, 'test_urls.txt')
        test_urls = [
            'https://example.com/1',
            'https://example.com/2',
            '# This is a comment',
            '',  # Empty line
            'https://example.com/3',
            'invalid-url',  # Should be skipped
            'http://example.com/4'
        ]

        with open(url_file, 'w') as f:
            f.write('\n'.join(test_urls))

        urls = self.builder._load_urls_from_file(url_file)

        # Should load only valid URLs (4 total)
        assert len(urls) == 4
        assert 'https://example.com/1' in urls
        assert 'https://example.com/2' in urls
        assert 'https://example.com/3' in urls
        assert 'http://example.com/4' in urls

    def test_load_urls_from_nonexistent_file(self):
        """Test loading URLs from non-existent file"""
        with pytest.raises(FileNotFoundError):
            self.builder._load_urls_from_file('/nonexistent/file.txt')

    def test_load_urls_from_empty_file(self):
        """Test loading URLs from empty file"""
        url_file = os.path.join(self.temp_dir, 'empty.txt')
        with open(url_file, 'w') as f:
            f.write('# Only comments\n\n')

        with pytest.raises(ValueError, match="No valid URLs found"):
            self.builder._load_urls_from_file(url_file)

    @patch('src.knowledge_base_builder.URLFetcher')
    @patch('src.knowledge_base_builder.ContentExtractor')
    @patch('src.knowledge_base_builder.ContentProcessor')
    def test_process_url_batch(self, mock_processor, mock_extractor, mock_fetcher):
        """Test processing a batch of URLs"""
        # Mock URL fetcher
        mock_fetch_result = FetchResult(
            url="https://example.com/test",
            success=True,
            content=b"Test content",
            content_type="text/html"
        )
        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.fetch.return_value = mock_fetch_result

        # Mock content extractor
        mock_extract_result = ExtractedContent(
            url="https://example.com/test",
            success=True,
            text="Test content " * 30,  # Enough words
            title="Test Title",
            word_count=60
        )
        mock_extractor_instance = mock_extractor.return_value
        mock_extractor_instance.extract_from_fetch_result.return_value = mock_extract_result

        # Mock content processor
        mock_document = ProcessedDocument(
            filename="test.txt",
            content="Processed test content " * 30,
            url="https://example.com/test",
            title="Test Title",
            word_count=90
        )
        mock_processor_instance = mock_processor.return_value
        mock_processor_instance.process_content.return_value = mock_document

        # Set up builder with mocked components
        self.builder.url_fetcher = mock_fetcher_instance
        self.builder.content_extractor = mock_extractor_instance
        self.builder.content_processor = mock_processor_instance

        # Test processing
        urls = ["https://example.com/test"]
        documents, stats = self.builder._process_url_batch(urls)

        assert len(documents) == 1
        assert documents[0] == mock_document
        assert stats['fetches'] == 1
        assert stats['extractions'] == 1
        assert stats['processing'] == 1
        assert len(stats['errors']) == 0

    @patch('src.knowledge_base_builder.URLFetcher')
    def test_process_url_batch_with_failures(self, mock_fetcher):
        """Test processing batch with various failures"""
        # Mock different failure scenarios
        def mock_fetch_side_effect(url):
            if 'fail-fetch' in url:
                return FetchResult(url=url, success=False, error_message="Network error")
            elif 'fail-extract' in url:
                return FetchResult(url=url, success=True, content=b"Content", content_type="text/html")
            else:
                return FetchResult(url=url, success=True, content=b"Good content", content_type="text/html")

        mock_fetcher_instance = mock_fetcher.return_value
        mock_fetcher_instance.fetch.side_effect = mock_fetch_side_effect

        # Mock extractor to fail on certain URLs
        def mock_extract_side_effect(fetch_result):
            if 'fail-extract' in fetch_result.url:
                return ExtractedContent(
                    url=fetch_result.url,
                    success=False,
                    error_message="Extraction failed"
                )
            else:
                return ExtractedContent(
                    url=fetch_result.url,
                    success=True,
                    text="Good content " * 30,
                    word_count=60
                )

        self.builder.content_extractor.extract_from_fetch_result.side_effect = mock_extract_side_effect

        # Mock processor to return valid documents
        self.builder.content_processor.process_content.return_value = ProcessedDocument(
            filename="test.txt",
            content="Processed content",
            url="test",
            word_count=30
        )

        # Set up fetcher
        self.builder.url_fetcher = mock_fetcher_instance

        # Test with mixed URLs
        urls = [
            "https://example.com/good",
            "https://example.com/fail-fetch",
            "https://example.com/fail-extract"
        ]

        documents, stats = self.builder._process_url_batch(urls)

        assert len(documents) == 1  # Only one should succeed
        assert stats['fetches'] == 2  # fetch success for 2 URLs
        assert stats['extractions'] == 1  # extract success for 1 URL
        assert stats['processing'] == 1  # process success for 1 URL
        assert len(stats['errors']) == 2  # 2 failures

    def test_build_from_urls_integration(self):
        """Test end-to-end build from URLs (integration test)"""
        # Create a minimal builder for integration testing
        builder = KnowledgeBaseBuilder(
            output_dir=os.path.join(self.temp_dir, 'integration'),
            use_cache=False,
            min_document_words=5  # Low threshold for testing
        )

        # Mock successful responses
        with patch.object(builder.url_fetcher, 'fetch') as mock_fetch, \
             patch.object(builder.content_extractor, 'extract_from_fetch_result') as mock_extract, \
             patch.object(builder.content_processor, 'process_content') as mock_process:

            # Set up mocks
            mock_fetch.return_value = FetchResult(
                url="test", success=True, content=b"content", content_type="text/html"
            )

            mock_extract.return_value = ExtractedContent(
                url="test", success=True, text="Test content " * 20, word_count=40
            )

            mock_process.return_value = ProcessedDocument(
                filename="test.txt", content="Processed", url="test", word_count=20
            )

            # Run build
            urls = ["https://example.com/1", "https://example.com/2"]
            result = builder.build_from_urls(urls)

            # Verify result
            assert isinstance(result, BuildResult)
            assert result.total_urls == 2
            assert result.documents_created == 2
            assert os.path.exists(result.output_directory)

    def test_generate_build_stats(self):
        """Test build statistics generation"""
        documents = [
            ProcessedDocument("doc1.txt", "Content 1 " * 10, "url1", word_count=20),
            ProcessedDocument("doc2.txt", "Content 2 " * 15, "url2", word_count=30),
            ProcessedDocument("doc3.txt", "Content 3 " * 5, "url3", word_count=10)
        ]

        urls = ["https://example.com/1.html", "https://example.com/2.pdf", "https://example.com/3"]

        stats = self.builder._generate_build_stats(documents, urls)

        assert stats['total_documents'] == 3
        assert stats['total_words'] == 60
        assert stats['avg_words_per_doc'] == 20
        assert stats['min_words'] == 10
        assert stats['max_words'] == 30
        assert stats['success_rate'] == 1.0  # 3 docs from 3 URLs
        assert stats['ready_for_openai_upload'] is True

    def test_save_build_metadata(self):
        """Test saving build metadata"""
        build_stats = {'total_documents': 5, 'avg_words': 100}
        errors = ['Error 1', 'Error 2']

        self.builder._save_build_metadata(build_stats, errors)

        # Check metadata file was created
        metadata_file = os.path.join(self.builder.output_dir, 'build_metadata.json')
        assert os.path.exists(metadata_file)

        # Check metadata content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        assert metadata['build_stats'] == build_stats
        assert metadata['errors'] == errors
        assert metadata['total_errors'] == 2
        assert 'build_config' in metadata

    def test_get_document_list_for_openai(self):
        """Test getting document list for OpenAI upload"""
        # Create some test files
        os.makedirs(self.builder.output_dir, exist_ok=True)

        test_files = ['doc1.txt', 'doc2.txt', '.hidden.txt', 'readme.md', 'doc3.txt']
        for filename in test_files:
            filepath = os.path.join(self.builder.output_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test content")

        file_list = self.builder.get_document_list_for_openai()

        # Should only include .txt files that don't start with .
        expected_files = 3  # doc1.txt, doc2.txt, doc3.txt
        assert len(file_list) == expected_files

        for filepath in file_list:
            assert filepath.endswith('.txt')
            assert os.path.basename(filepath) in ['doc1.txt', 'doc2.txt', 'doc3.txt']

    def test_context_manager(self):
        """Test context manager functionality"""
        with KnowledgeBaseBuilder(output_dir=self.temp_dir) as builder:
            assert isinstance(builder, KnowledgeBaseBuilder)

        # Should exit without errors

    def test_build_result_serialization(self):
        """Test BuildResult can be serialized to dict"""
        result = BuildResult(
            total_urls=100,
            successful_fetches=95,
            successful_extractions=90,
            successful_processing=85,
            documents_created=80,
            output_directory="/test/output",
            build_stats={'test': 'stats'},
            errors=['error1', 'error2']
        )

        result_dict = result.to_dict()

        assert result_dict['total_urls'] == 100
        assert result_dict['successful_fetches'] == 95
        assert result_dict['documents_created'] == 80
        assert result_dict['build_stats'] == {'test': 'stats'}
        assert result_dict['errors'] == ['error1', 'error2']