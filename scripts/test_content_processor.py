#!/usr/bin/env python3
"""
Test script for Content Processor component
Tests processing of extracted content into text files for OpenAI vector store
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.content_processor import ContentProcessor, ProcessedDocument
from src.content_extractor import ExtractedContent, ContentExtractor
from src.url_fetcher import URLFetcher


def test_basic_processing():
    """Test basic content processing"""
    print("\n" + "="*50)
    print("Testing Basic Content Processing")
    print("="*50)

    processor = ContentProcessor(min_word_count=20)

    # Test with sample content
    extracted = ExtractedContent(
        url="https://example.com/test",
        success=True,
        text="This is a test document with sufficient content to pass the word count filter. " * 10,
        title="Sample Test Document",
        content_type="text/html",
        word_count=80,
        extraction_method="beautifulsoup"
    )

    result = processor.process_content(extracted)

    if result:
        print(f"✓ Processing successful")
        print(f"  Filename: {result.filename}")
        print(f"  URL: {result.url}")
        print(f"  Title: {result.title}")
        print(f"  Word count: {result.word_count}")
        print(f"  Content preview: {result.content[:100]}...")
    else:
        print(f"✗ Processing failed")

def test_text_file_generation():
    """Test text file generation for OpenAI"""
    print("\n" + "="*50)
    print("Testing Text File Generation")
    print("="*50)

    doc = ProcessedDocument(
        filename="sample_doc.txt",
        content="This is the main content of a processed document. It contains information that will be used for RAG with OpenAI. The content has been cleaned and is ready for upload to a vector store.",
        url="https://example.com/sample",
        title="Sample Document for RAG",
        word_count=35
    )

    text_file_content = doc.to_text_file()

    print("Generated text file content:")
    print("-" * 40)
    print(text_file_content)
    print("-" * 40)

def test_file_saving():
    """Test saving documents to directory"""
    print("\n" + "="*50)
    print("Testing File Saving")
    print("="*50)

    processor = ContentProcessor()

    # Create sample documents
    documents = [
        ProcessedDocument(
            filename="doc1.txt",
            content="First document content for testing file saving functionality.",
            url="https://example.com/doc1",
            title="Document One",
            word_count=10
        ),
        ProcessedDocument(
            filename="doc2.txt",
            content="Second document content with different information for testing.",
            url="https://example.com/doc2",
            title="Document Two",
            word_count=9
        )
    ]

    # Save to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        processor.save_documents_to_directory(documents, temp_dir)

        # List saved files
        files = os.listdir(temp_dir)
        print(f"✓ Saved {len(files)} files: {files}")

        # Show content of first file
        if "doc1.txt" in files:
            filepath = os.path.join(temp_dir, "doc1.txt")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"✓ File content preview:")
                print("-" * 40)
                print(content[:200] + "..." if len(content) > 200 else content)
                print("-" * 40)

def test_with_real_content():
    """Test processing with real extracted content"""
    print("\n" + "="*50)
    print("Testing with Real Extracted Content")
    print("="*50)

    processor = ContentProcessor(min_word_count=30)

    # Use the content extractor to get real content
    extractor = ContentExtractor()

    # Test with simple HTML content
    html_content = """
    <html>
    <head>
        <title>Real Content Test</title>
    </head>
    <body>
        <h1>Main Article</h1>
        <p>This is a real HTML document that we're testing with the content processor.
        It contains multiple paragraphs and should have enough words to pass our filter.</p>
        <p>The processor should clean this content and prepare it for OpenAI vector store upload.
        This is exactly the kind of content we'll be processing from the 11,000+ URLs in our dataset.</p>
        <p>After processing, this content will be saved as a text file that OpenAI can chunk and embed automatically.</p>
    </body>
    </html>
    """

    # Extract content
    extracted = extractor.extract_from_content(
        html_content.encode('utf-8'),
        'https://example.com/real-content',
        'text/html'
    )

    print(f"Extraction result: {extracted.success}")
    if extracted.success:
        print(f"  Title: {extracted.title}")
        print(f"  Word count: {extracted.word_count}")
        print(f"  Extraction method: {extracted.extraction_method}")

        # Process the extracted content
        document = processor.process_content(extracted)

        if document:
            print(f"✓ Processing successful")
            print(f"  Generated filename: {document.filename}")
            print(f"  Processed word count: {document.word_count}")
            print(f"  Content preview:")
            print("-" * 40)
            print(document.to_text_file()[:300] + "...")
            print("-" * 40)
        else:
            print(f"✗ Processing failed")
    else:
        print(f"✗ Extraction failed: {extracted.error_message}")

def test_filtering():
    """Test content filtering"""
    print("\n" + "="*50)
    print("Testing Content Filtering")
    print("="*50)

    processor = ContentProcessor(min_word_count=50)

    test_cases = [
        {
            'name': 'Sufficient content',
            'content': ExtractedContent(
                url="https://example.com/good",
                success=True,
                text="This document has plenty of words to pass the minimum word count filter. " * 10,
                title="Good Document"
            )
        },
        {
            'name': 'Too short',
            'content': ExtractedContent(
                url="https://example.com/short",
                success=True,
                text="Too short",
                title="Short Document"
            )
        },
        {
            'name': 'Failed extraction',
            'content': ExtractedContent(
                url="https://example.com/failed",
                success=False,
                error_message="404 Not Found"
            )
        },
        {
            'name': 'No text content',
            'content': ExtractedContent(
                url="https://example.com/empty",
                success=True,
                text=None
            )
        }
    ]

    for test_case in test_cases:
        result = processor.process_content(test_case['content'])
        status = "✓ Processed" if result else "✗ Filtered out"
        print(f"  {test_case['name']}: {status}")

def test_multiple_processing():
    """Test processing multiple contents"""
    print("\n" + "="*50)
    print("Testing Multiple Content Processing")
    print("="*50)

    processor = ContentProcessor(min_word_count=25)

    # Create multiple test contents
    extracted_contents = []
    for i in range(5):
        content = ExtractedContent(
            url=f"https://example.com/doc{i}",
            success=True,
            text=f"Document {i} content with sufficient words to pass filtering. " * (5 + i),
            title=f"Document {i}",
            word_count=30 + i*5
        )
        extracted_contents.append(content)

    # Add some that should be filtered out
    extracted_contents.extend([
        ExtractedContent(
            url="https://example.com/short",
            success=True,
            text="Too short",
            title="Short Doc"
        ),
        ExtractedContent(
            url="https://example.com/failed",
            success=False,
            error_message="Failed to extract"
        )
    ])

    documents = processor.process_multiple(extracted_contents)

    print(f"Processed {len(documents)} documents from {len(extracted_contents)} inputs")
    for i, doc in enumerate(documents):
        print(f"  {i+1}. {doc.title} ({doc.word_count} words) -> {doc.filename}")

def main():
    """Run all content processor tests"""
    print("Content Processor Test Suite")
    print("=" * 60)

    try:
        # Run all tests
        test_basic_processing()
        test_text_file_generation()
        test_file_saving()
        test_with_real_content()
        test_filtering()
        test_multiple_processing()

        print("\n" + "="*60)
        print("Content Processor Test Suite Completed Successfully!")
        print("Ready for OpenAI vector store uploads!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())