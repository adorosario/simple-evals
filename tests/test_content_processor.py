"""
Unit tests for Content Processor component
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.content_processor import ContentProcessor, ProcessedDocument
from src.content_extractor import ExtractedContent


class TestContentProcessor:
    """Test cases for ContentProcessor class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.processor = ContentProcessor()

    def test_init_default_config(self):
        """Test ContentProcessor initialization with defaults"""
        processor = ContentProcessor()
        assert processor.min_word_count == 50

    def test_init_custom_config(self):
        """Test ContentProcessor initialization with custom config"""
        processor = ContentProcessor(min_word_count=100)
        assert processor.min_word_count == 100

    def test_process_successful_content(self):
        """Test processing successful extracted content"""
        extracted = ExtractedContent(
            url="https://example.com/test",
            success=True,
            text="This is a test document with enough words to pass the minimum word count filter. " * 10,
            title="Test Document",
            word_count=80
        )

        result = self.processor.process_content(extracted)

        assert result is not None
        assert isinstance(result, ProcessedDocument)
        assert result.url == "https://example.com/test"
        assert result.title == "Test Document"
        assert result.word_count >= 50
        assert result.filename.startswith("doc_")
        assert result.filename.endswith(".txt")
        assert "This is a test document" in result.content

    def test_process_failed_extraction(self):
        """Test processing failed extraction"""
        extracted = ExtractedContent(
            url="https://example.com/failed",
            success=False,
            error_message="404 Not Found"
        )

        result = self.processor.process_content(extracted)
        assert result is None

    def test_process_no_text_content(self):
        """Test processing with no text content"""
        extracted = ExtractedContent(
            url="https://example.com/empty",
            success=True,
            text=None
        )

        result = self.processor.process_content(extracted)
        assert result is None

    def test_process_too_short_content(self):
        """Test processing content that's too short"""
        processor = ContentProcessor(min_word_count=100)

        extracted = ExtractedContent(
            url="https://example.com/short",
            success=True,
            text="Too short",
            word_count=2
        )

        result = processor.process_content(extracted)
        assert result is None

    def test_text_cleaning(self):
        """Test text cleaning functionality"""
        dirty_text = "This   has    multiple     spaces\n\n\n\nAnd many newlines\n\n\nAnd excessive punctuation!!!"

        cleaned = self.processor._clean_text(dirty_text)

        # Check whitespace normalization
        assert "   " not in cleaned
        assert "multiple spaces" in cleaned

        # Check newline normalization
        lines = cleaned.split('\n')
        consecutive_empty = 0
        max_consecutive_empty = 0
        for line in lines:
            if line.strip() == "":
                consecutive_empty += 1
                max_consecutive_empty = max(max_consecutive_empty, consecutive_empty)
            else:
                consecutive_empty = 0

        assert max_consecutive_empty <= 2  # Should be at most double newline

    def test_filename_generation(self):
        """Test filename generation from URL"""
        extracted = ExtractedContent(
            url="https://example.com/test-page",
            success=True,
            text="Test content " * 20
        )

        result = self.processor.process_content(extracted)

        assert result is not None
        assert result.filename.startswith("doc_")
        assert result.filename.endswith(".txt")
        assert len(result.filename) == 16  # "doc_" + 12 char hash + ".txt"

        # Same URL should generate same filename
        result2 = self.processor.process_content(extracted)
        assert result.filename == result2.filename

    def test_text_file_generation(self):
        """Test text file content generation"""
        doc = ProcessedDocument(
            filename="test.txt",
            content="This is the main content of the document.",
            url="https://example.com/test",
            title="Test Document",
            word_count=9
        )

        text_file_content = doc.to_text_file()

        # Check header information
        assert "Source: https://example.com/test" in text_file_content
        assert "Title: Test Document" in text_file_content
        assert "Words: 9" in text_file_content

        # Check separator
        assert "=" * 80 in text_file_content

        # Check main content
        assert "This is the main content of the document." in text_file_content

    def test_text_file_without_title(self):
        """Test text file generation without title"""
        doc = ProcessedDocument(
            filename="test.txt",
            content="Content without title.",
            url="https://example.com/notitle",
            word_count=3
        )

        text_file_content = doc.to_text_file()

        assert "Source: https://example.com/notitle" in text_file_content
        assert "Title:" not in text_file_content
        assert "Words: 3" in text_file_content
        assert "Content without title." in text_file_content

    def test_process_multiple(self):
        """Test processing multiple extracted contents"""
        extracted_contents = [
            ExtractedContent(
                url="https://example.com/1",
                success=True,
                text="First document " * 20,
                title="Document 1"
            ),
            ExtractedContent(
                url="https://example.com/2",
                success=True,
                text="Second document " * 25,
                title="Document 2"
            ),
            ExtractedContent(
                url="https://example.com/3",
                success=False,
                error_message="Failed"
            ),
            ExtractedContent(
                url="https://example.com/4",
                success=True,
                text="Too short"  # Only 2 words
            )
        ]

        documents = self.processor.process_multiple(extracted_contents)

        # Should process only the first 2 (successful with enough content)
        assert len(documents) == 2
        assert documents[0].title == "Document 1"
        assert documents[1].title == "Document 2"

    def test_save_documents_to_directory(self):
        """Test saving documents to directory"""
        documents = [
            ProcessedDocument(
                filename="doc1.txt",
                content="Content of first document",
                url="https://example.com/1",
                title="Doc 1",
                word_count=5
            ),
            ProcessedDocument(
                filename="doc2.txt",
                content="Content of second document",
                url="https://example.com/2",
                title="Doc 2",
                word_count=5
            )
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            self.processor.save_documents_to_directory(documents, temp_dir)

            # Check files were created
            files = os.listdir(temp_dir)
            assert "doc1.txt" in files
            assert "doc2.txt" in files

            # Check file contents
            with open(os.path.join(temp_dir, "doc1.txt"), 'r', encoding='utf-8') as f:
                content = f.read()
                assert "Source: https://example.com/1" in content
                assert "Title: Doc 1" in content
                assert "Content of first document" in content

            with open(os.path.join(temp_dir, "doc2.txt"), 'r', encoding='utf-8') as f:
                content = f.read()
                assert "Source: https://example.com/2" in content
                assert "Title: Doc 2" in content
                assert "Content of second document" in content

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text"""
        test_cases = [
            "",
            "   ",
            "\n\n\n",
            "\t\t\t"
        ]

        for empty_text in test_cases:
            extracted = ExtractedContent(
                url="https://example.com/empty",
                success=True,
                text=empty_text
            )

            result = self.processor.process_content(extracted)
            assert result is None

    def test_text_cleaning_edge_cases(self):
        """Test text cleaning with edge cases"""
        # Test with None
        assert self.processor._clean_text(None) == ""

        # Test with empty string
        assert self.processor._clean_text("") == ""

        # Test with only whitespace
        assert self.processor._clean_text("   \n\n\t  ") == ""

        # Test with complex whitespace patterns
        complex_text = "Word1    \n\n\n\n   Word2\t\t\tWord3"
        cleaned = self.processor._clean_text(complex_text)
        assert cleaned == "Word1 Word2 Word3"

    def test_hash_consistency(self):
        """Test that same URLs generate consistent hashes"""
        url = "https://example.com/consistent-test"

        extracted1 = ExtractedContent(url=url, success=True, text="Content " * 20)
        extracted2 = ExtractedContent(url=url, success=True, text="Different content " * 20)

        doc1 = self.processor.process_content(extracted1)
        doc2 = self.processor.process_content(extracted2)

        # Same URL should generate same filename regardless of content
        assert doc1.filename == doc2.filename

    def test_word_count_accuracy(self):
        """Test word count calculation accuracy"""
        test_text = "One two three four five"

        extracted = ExtractedContent(
            url="https://example.com/wordcount",
            success=True,
            text=test_text
        )

        result = self.processor.process_content(extracted)
        assert result.word_count == 5

        # Test with extra whitespace
        test_text_whitespace = "  One   two  three   four    five  "
        extracted.text = test_text_whitespace

        result = self.processor.process_content(extracted)
        assert result.word_count == 5