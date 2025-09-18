"""
Content Extractor Component

Extracts clean text from various content types including HTML, PDF, and plain text.
Handles encoding issues, malformed content, and provides metadata extraction.
"""

import re
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import mimetypes

# Optional imports with fallbacks
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

from .url_fetcher import FetchResult

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Result of content extraction"""
    url: str
    success: bool
    text: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    content_type: Optional[str] = None
    language: Optional[str] = None
    word_count: Optional[int] = None
    error_message: Optional[str] = None
    extraction_method: Optional[str] = None


class ContentExtractor:
    """
    Extracts clean text from various content types with robust error handling.
    """

    def __init__(self,
                 max_text_length: int = 1_000_000,  # 1MB of text
                 preserve_structure: bool = True,
                 extract_links: bool = False,
                 extract_metadata: bool = True):
        """
        Initialize content extractor.

        Args:
            max_text_length: Maximum length of extracted text
            preserve_structure: Whether to preserve paragraph structure
            extract_links: Whether to extract and resolve links
            extract_metadata: Whether to extract metadata
        """
        self.max_text_length = max_text_length
        self.preserve_structure = preserve_structure
        self.extract_links = extract_links
        self.extract_metadata = extract_metadata

        # Check dependencies
        self._check_dependencies()

    def _check_dependencies(self):
        """Check and log available dependencies"""
        missing_deps = []
        if not HAS_BS4:
            missing_deps.append("beautifulsoup4")
        if not HAS_PYPDF2 and not HAS_PDFPLUMBER:
            missing_deps.append("PyPDF2 or pdfplumber")

        if missing_deps:
            logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
            logger.warning("Some content extraction features may be limited")

    def extract_from_fetch_result(self, fetch_result: FetchResult) -> ExtractedContent:
        """
        Extract content from a FetchResult.

        Args:
            fetch_result: Result from URL fetcher

        Returns:
            ExtractedContent with extracted text and metadata
        """
        if not fetch_result.success or not fetch_result.content:
            return ExtractedContent(
                url=fetch_result.url,
                success=False,
                error_message=fetch_result.error_message or "No content to extract"
            )

        return self.extract_from_content(
            content=fetch_result.content,
            url=fetch_result.url,
            content_type=fetch_result.content_type,
            encoding=fetch_result.encoding
        )

    def extract_from_content(self,
                           content: bytes,
                           url: str,
                           content_type: Optional[str] = None,
                           encoding: Optional[str] = None) -> ExtractedContent:
        """
        Extract text from raw content.

        Args:
            content: Raw content bytes
            url: Source URL
            content_type: MIME type of content
            encoding: Text encoding

        Returns:
            ExtractedContent with extracted text and metadata
        """
        if len(content) == 0:
            return ExtractedContent(
                url=url,
                success=False,
                error_message="Empty content"
            )

        # Determine content type if not provided
        if not content_type:
            content_type = self._detect_content_type(content, url)

        try:
            # Route to appropriate extractor
            if content_type and 'html' in content_type.lower():
                return self._extract_html(content, url, encoding)
            elif content_type and 'pdf' in content_type.lower():
                return self._extract_pdf(content, url)
            elif content_type and any(t in content_type.lower() for t in ['text', 'json', 'xml']):
                return self._extract_text(content, url, encoding)
            else:
                # Try to extract as text with fallback
                return self._extract_with_fallback(content, url, encoding)

        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return ExtractedContent(
                url=url,
                success=False,
                error_message=f"Extraction error: {str(e)}",
                content_type=content_type
            )

    def _detect_content_type(self, content: bytes, url: str) -> str:
        """Detect content type from content and URL"""
        # Check file extension
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        if path.endswith('.pdf'):
            return 'application/pdf'
        elif path.endswith(('.html', '.htm')):
            return 'text/html'
        elif path.endswith(('.txt', '.md')):
            return 'text/plain'
        elif path.endswith('.json'):
            return 'application/json'
        elif path.endswith('.xml'):
            return 'application/xml'

        # Check content magic bytes
        if content.startswith(b'%PDF'):
            return 'application/pdf'
        elif b'<html' in content[:1000].lower() or b'<!doctype html' in content[:1000].lower():
            return 'text/html'
        elif content.startswith((b'<?xml', b'<')):
            return 'application/xml'

        # Default to text
        return 'text/plain'

    def _extract_html(self, content: bytes, url: str, encoding: Optional[str] = None) -> ExtractedContent:
        """Extract text from HTML content"""
        if not HAS_BS4:
            return self._extract_text(content, url, encoding)

        try:
            # Decode content
            text_content = self._decode_content(content, encoding)
            if not text_content:
                return ExtractedContent(
                    url=url,
                    success=False,
                    error_message="Failed to decode HTML content"
                )

            # Parse with BeautifulSoup
            soup = BeautifulSoup(text_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            # Extract title
            title = None
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()

            # Extract main content
            # Try to find main content areas first
            main_content = None
            for selector in ['main', 'article', '.content', '#content', '.main', '#main']:
                content_elem = soup.select_one(selector)
                if content_elem:
                    main_content = content_elem
                    break

            if not main_content:
                main_content = soup

            # Extract text
            if self.preserve_structure:
                text_parts = []
                for elem in main_content.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                    text = elem.get_text().strip()
                    if text and len(text) > 10:  # Skip very short elements
                        text_parts.append(text)
                extracted_text = '\n\n'.join(text_parts)
            else:
                extracted_text = main_content.get_text()

            # Clean up text
            extracted_text = self._clean_text(extracted_text)

            # Extract metadata
            metadata = {}
            if self.extract_metadata:
                metadata = self._extract_html_metadata(soup)

            # Extract links if requested
            if self.extract_links:
                links = self._extract_links(soup, url)
                metadata['links'] = links

            # Calculate word count
            word_count = len(extracted_text.split()) if extracted_text else 0

            return ExtractedContent(
                url=url,
                success=True,
                text=extracted_text[:self.max_text_length],
                title=title,
                metadata=metadata,
                content_type='text/html',
                word_count=word_count,
                extraction_method='beautifulsoup'
            )

        except Exception as e:
            logger.error(f"HTML extraction failed for {url}: {e}")
            return ExtractedContent(
                url=url,
                success=False,
                error_message=f"HTML extraction error: {str(e)}",
                content_type='text/html'
            )

    def _extract_pdf(self, content: bytes, url: str) -> ExtractedContent:
        """Extract text from PDF content"""
        if not (HAS_PYPDF2 or HAS_PDFPLUMBER):
            return ExtractedContent(
                url=url,
                success=False,
                error_message="PDF extraction not available (missing PyPDF2/pdfplumber)"
            )

        try:
            import io

            # Try pdfplumber first (usually better text extraction)
            if HAS_PDFPLUMBER:
                try:
                    with pdfplumber.open(io.BytesIO(content)) as pdf:
                        text_parts = []
                        metadata = {'page_count': len(pdf.pages)}

                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text.strip())

                        extracted_text = '\n\n'.join(text_parts)
                        extracted_text = self._clean_text(extracted_text)

                        word_count = len(extracted_text.split()) if extracted_text else 0

                        return ExtractedContent(
                            url=url,
                            success=True,
                            text=extracted_text[:self.max_text_length],
                            metadata=metadata,
                            content_type='application/pdf',
                            word_count=word_count,
                            extraction_method='pdfplumber'
                        )
                except Exception as e:
                    logger.warning(f"pdfplumber extraction failed for {url}: {e}")

            # Fallback to PyPDF2
            if HAS_PYPDF2:
                try:
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                    text_parts = []
                    metadata = {'page_count': len(pdf_reader.pages)}

                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text.strip())

                    extracted_text = '\n\n'.join(text_parts)
                    extracted_text = self._clean_text(extracted_text)

                    word_count = len(extracted_text.split()) if extracted_text else 0

                    return ExtractedContent(
                        url=url,
                        success=True,
                        text=extracted_text[:self.max_text_length],
                        metadata=metadata,
                        content_type='application/pdf',
                        word_count=word_count,
                        extraction_method='pypdf2'
                    )
                except Exception as e:
                    logger.error(f"PyPDF2 extraction failed for {url}: {e}")

            return ExtractedContent(
                url=url,
                success=False,
                error_message="PDF extraction failed with all available methods"
            )

        except Exception as e:
            logger.error(f"PDF extraction failed for {url}: {e}")
            return ExtractedContent(
                url=url,
                success=False,
                error_message=f"PDF extraction error: {str(e)}",
                content_type='application/pdf'
            )

    def _extract_text(self, content: bytes, url: str, encoding: Optional[str] = None) -> ExtractedContent:
        """Extract text from plain text content"""
        try:
            text_content = self._decode_content(content, encoding)
            if not text_content:
                return ExtractedContent(
                    url=url,
                    success=False,
                    error_message="Failed to decode text content"
                )

            # Clean text
            cleaned_text = self._clean_text(text_content)
            word_count = len(cleaned_text.split()) if cleaned_text else 0

            return ExtractedContent(
                url=url,
                success=True,
                text=cleaned_text[:self.max_text_length],
                content_type='text/plain',
                word_count=word_count,
                extraction_method='direct'
            )

        except Exception as e:
            logger.error(f"Text extraction failed for {url}: {e}")
            return ExtractedContent(
                url=url,
                success=False,
                error_message=f"Text extraction error: {str(e)}"
            )

    def _extract_with_fallback(self, content: bytes, url: str, encoding: Optional[str] = None) -> ExtractedContent:
        """Extract with fallback methods"""
        # Try as HTML first
        if b'<' in content and b'>' in content:
            result = self._extract_html(content, url, encoding)
            if result.success:
                return result

        # Fall back to text extraction
        return self._extract_text(content, url, encoding)

    def _decode_content(self, content: bytes, encoding: Optional[str] = None) -> Optional[str]:
        """Decode bytes content to string with encoding detection"""
        encodings_to_try = []

        if encoding:
            encodings_to_try.append(encoding)

        # Add common encodings
        encodings_to_try.extend(['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1'])

        for enc in encodings_to_try:
            try:
                return content.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue

        # Last resort: decode with errors ignored
        try:
            return content.decode('utf-8', errors='ignore')
        except Exception:
            return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _extract_html_metadata(self, soup) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        metadata = {}

        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if name and content:
                metadata[name.lower()] = content

        # OpenGraph and Twitter Card data
        for prop in ['og:title', 'og:description', 'og:type', 'twitter:title', 'twitter:description']:
            meta = soup.find('meta', property=prop) or soup.find('meta', attrs={'name': prop})
            if meta:
                metadata[prop] = meta.get('content')

        # Extract headings
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3']):
            heading_text = h.get_text().strip()
            if heading_text:
                headings.append(heading_text)
        if headings:
            metadata['headings'] = headings[:10]  # Limit to first 10

        return metadata

    def _extract_links(self, soup, base_url: str) -> List[str]:
        """Extract and resolve links from HTML"""
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            if absolute_url not in links:
                links.append(absolute_url)
        return links[:50]  # Limit to first 50 links

    def extract_multiple(self, fetch_results: List[FetchResult]) -> List[ExtractedContent]:
        """Extract content from multiple fetch results"""
        extracted_contents = []
        total_results = len(fetch_results)

        logger.info(f"Extracting content from {total_results} fetch results...")

        for i, fetch_result in enumerate(fetch_results):
            extracted = self.extract_from_fetch_result(fetch_result)
            extracted_contents.append(extracted)

            # Progress logging
            if (i + 1) % 10 == 0 or i + 1 == total_results:
                successful = sum(1 for r in extracted_contents if r.success)
                logger.info(f"Progress: {i+1}/{total_results} content extractions "
                           f"({successful} successful)")

        return extracted_contents

    def get_stats(self, extracted_contents: List[ExtractedContent]) -> Dict[str, Any]:
        """Get statistics from extraction results"""
        total = len(extracted_contents)
        successful = sum(1 for r in extracted_contents if r.success)
        failed = total - successful

        # Content type breakdown
        content_types = {}
        extraction_methods = {}
        word_counts = []
        languages = {}

        for content in extracted_contents:
            if content.success:
                if content.content_type:
                    content_types[content.content_type] = content_types.get(content.content_type, 0) + 1
                if content.extraction_method:
                    extraction_methods[content.extraction_method] = extraction_methods.get(content.extraction_method, 0) + 1
                if content.word_count:
                    word_counts.append(content.word_count)
                if content.language:
                    languages[content.language] = languages.get(content.language, 0) + 1

        stats = {
            'total_extractions': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'content_types': content_types,
            'extraction_methods': extraction_methods,
            'languages': languages
        }

        if word_counts:
            stats['avg_word_count'] = sum(word_counts) / len(word_counts)
            stats['min_word_count'] = min(word_counts)
            stats['max_word_count'] = max(word_counts)
            stats['total_words'] = sum(word_counts)

        return stats