#!/usr/bin/env python3
"""
Test script for Content Extractor component
Tests extraction from various content types including HTML, PDF, and text
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.content_extractor import ContentExtractor
from src.url_fetcher import URLFetcher
import tempfile

def test_html_extraction():
    """Test HTML content extraction"""
    print("\n" + "="*50)
    print("Testing HTML Content Extraction")
    print("="*50)

    extractor = ContentExtractor()

    # Test with a simple HTML page
    html_content = """
    <html>
    <head>
        <title>Test Document</title>
        <meta name="description" content="A test document for extraction">
    </head>
    <body>
        <h1>Main Heading</h1>
        <p>This is the first paragraph with important content.</p>
        <p>This is the second paragraph with more details.</p>
        <script>console.log('This should be removed');</script>
        <style>body { color: red; }</style>
        <div>
            <h2>Subsection</h2>
            <p>More content in a subsection.</p>
        </div>
    </body>
    </html>
    """

    result = extractor.extract_from_content(
        html_content.encode('utf-8'),
        'http://example.com/test.html',
        'text/html'
    )

    if result.success:
        print(f"✓ HTML extraction successful")
        print(f"  Title: {result.title}")
        print(f"  Content Type: {result.content_type}")
        print(f"  Extraction Method: {result.extraction_method}")
        print(f"  Word Count: {result.word_count}")
        print(f"  Text Preview: {result.text[:200]}...")
        if result.metadata:
            print(f"  Metadata keys: {list(result.metadata.keys())}")
    else:
        print(f"✗ HTML extraction failed: {result.error_message}")

def test_text_extraction():
    """Test plain text extraction"""
    print("\n" + "="*50)
    print("Testing Text Content Extraction")
    print("="*50)

    extractor = ContentExtractor()

    text_content = """
    This is a plain text document with multiple paragraphs.

    It contains various information about different topics.
    The content should be extracted and cleaned properly.

    There might be some    extra   spaces   that need cleaning.


    And multiple line breaks that should be normalized.
    """

    result = extractor.extract_from_content(
        text_content.encode('utf-8'),
        'http://example.com/test.txt',
        'text/plain'
    )

    if result.success:
        print(f"✓ Text extraction successful")
        print(f"  Content Type: {result.content_type}")
        print(f"  Extraction Method: {result.extraction_method}")
        print(f"  Word Count: {result.word_count}")
        print(f"  Text Preview: {result.text[:200]}...")
    else:
        print(f"✗ Text extraction failed: {result.error_message}")

def test_encoding_handling():
    """Test various text encodings"""
    print("\n" + "="*50)
    print("Testing Encoding Handling")
    print("="*50)

    extractor = ContentExtractor()

    # Test UTF-8 with special characters
    utf8_text = "Hello, 世界! Café résumé naïve façade 🌍"

    result = extractor.extract_from_content(
        utf8_text.encode('utf-8'),
        'http://example.com/utf8.txt',
        'text/plain'
    )

    if result.success:
        print(f"✓ UTF-8 encoding handled correctly")
        print(f"  Text: {result.text}")
    else:
        print(f"✗ UTF-8 encoding failed: {result.error_message}")

    # Test Latin-1 encoding
    latin1_text = "Café résumé"
    result = extractor.extract_from_content(
        latin1_text.encode('latin-1'),
        'http://example.com/latin1.txt',
        'text/plain'
    )

    if result.success:
        print(f"✓ Latin-1 encoding handled correctly")
        print(f"  Text: {result.text}")
    else:
        print(f"✗ Latin-1 encoding failed: {result.error_message}")

def test_content_type_detection():
    """Test automatic content type detection"""
    print("\n" + "="*50)
    print("Testing Content Type Detection")
    print("="*50)

    extractor = ContentExtractor()

    # Test HTML detection from content
    html_content = b"<html><body><p>Auto-detected HTML</p></body></html>"
    result = extractor.extract_from_content(
        html_content,
        'http://example.com/unknown',
        None  # No content type provided
    )

    if result.success:
        print(f"✓ HTML auto-detection successful")
        print(f"  Detected content type: {result.content_type}")
        print(f"  Text: {result.text}")
    else:
        print(f"✗ HTML auto-detection failed: {result.error_message}")

def test_with_real_urls():
    """Test extraction with real URLs"""
    print("\n" + "="*50)
    print("Testing with Real URLs")
    print("="*50)

    extractor = ContentExtractor()

    # Test URLs that should work
    test_urls = [
        "https://httpbin.org/html",  # Simple HTML page
        "https://example.com",       # Basic HTML
    ]

    with URLFetcher(timeout=10, max_retries=1) as fetcher:
        for url in test_urls:
            print(f"\nTesting URL: {url}")

            # Fetch content
            fetch_result = fetcher.fetch(url)
            if not fetch_result.success:
                print(f"  ✗ Failed to fetch: {fetch_result.error_message}")
                continue

            # Extract content
            extract_result = extractor.extract_from_fetch_result(fetch_result)

            if extract_result.success:
                print(f"  ✓ Extraction successful")
                print(f"    Content Type: {extract_result.content_type}")
                print(f"    Extraction Method: {extract_result.extraction_method}")
                print(f"    Word Count: {extract_result.word_count}")
                print(f"    Title: {extract_result.title}")
                print(f"    Text Preview: {extract_result.text[:150]}...")
            else:
                print(f"  ✗ Extraction failed: {extract_result.error_message}")

def test_error_handling():
    """Test error handling scenarios"""
    print("\n" + "="*50)
    print("Testing Error Handling")
    print("="*50)

    extractor = ContentExtractor()

    # Test empty content
    result = extractor.extract_from_content(
        b'',
        'http://example.com/empty.txt'
    )

    if not result.success:
        print(f"✓ Empty content handled correctly: {result.error_message}")
    else:
        print(f"✗ Empty content should have failed")

    # Test very large content (should be truncated)
    large_extractor = ContentExtractor(max_text_length=100)
    large_content = "word " * 50  # 200 chars, should be truncated to 100

    result = large_extractor.extract_from_content(
        large_content.encode('utf-8'),
        'http://example.com/large.txt',
        'text/plain'
    )

    if result.success and len(result.text) == 100:
        print(f"✓ Content truncation working correctly")
        print(f"  Original length: {len(large_content)}, Truncated to: {len(result.text)}")
    else:
        print(f"✗ Content truncation not working as expected")

def test_statistics():
    """Test statistics generation"""
    print("\n" + "="*50)
    print("Testing Statistics Generation")
    print("="*50)

    extractor = ContentExtractor()

    # Create some test results
    from src.content_extractor import ExtractedContent

    results = [
        ExtractedContent(
            url="http://example.com/1",
            success=True,
            text="Sample content one",
            content_type="text/html",
            word_count=3,
            extraction_method="beautifulsoup"
        ),
        ExtractedContent(
            url="http://example.com/2",
            success=True,
            text="Sample content two with more words",
            content_type="text/plain",
            word_count=6,
            extraction_method="direct"
        ),
        ExtractedContent(
            url="http://example.com/3",
            success=False,
            error_message="Failed to extract"
        )
    ]

    stats = extractor.get_stats(results)

    print(f"Statistics:")
    print(f"  Total extractions: {stats['total_extractions']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Content types: {stats['content_types']}")
    print(f"  Extraction methods: {stats['extraction_methods']}")
    print(f"  Average word count: {stats.get('avg_word_count', 'N/A')}")
    print(f"  Total words: {stats.get('total_words', 'N/A')}")

def main():
    """Run all content extractor tests"""
    print("Content Extractor Test Suite")
    print("=" * 60)

    try:
        # Run all tests
        test_html_extraction()
        test_text_extraction()
        test_encoding_handling()
        test_content_type_detection()
        test_error_handling()
        test_statistics()
        test_with_real_urls()

        print("\n" + "="*60)
        print("Content Extractor Test Suite Completed Successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())