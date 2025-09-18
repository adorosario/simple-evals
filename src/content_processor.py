"""
Content Processor Component

Simple processor that prepares extracted content as text files for OpenAI vector store upload.
OpenAI handles all the chunking, embedding, and RAG logic automatically.
"""

import re
import hashlib
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .content_extractor import ExtractedContent

logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    """A document ready for OpenAI vector store upload"""
    filename: str
    content: str
    url: str
    title: Optional[str] = None
    word_count: int = 0

    def to_text_file(self) -> str:
        """Generate text file content for OpenAI upload"""
        header = f"Source: {self.url}\n"
        if self.title:
            header += f"Title: {self.title}\n"
        header += f"Words: {self.word_count}\n"
        header += "\n" + "="*80 + "\n\n"

        return header + self.content


class ContentProcessor:
    """Prepares extracted content as text files for OpenAI vector store."""

    def __init__(self, min_word_count: int = 50):
        """
        Initialize content processor.

        Args:
            min_word_count: Minimum words to include document
        """
        self.min_word_count = min_word_count

    def process_content(self, extracted_content: ExtractedContent) -> Optional[ProcessedDocument]:
        """
        Process extracted content into a text file ready for OpenAI upload.

        Args:
            extracted_content: Content from ContentExtractor

        Returns:
            ProcessedDocument or None if should be skipped
        """
        if not extracted_content.success or not extracted_content.text:
            logger.debug(f"Skipping {extracted_content.url}: No text content")
            return None

        # Clean the text
        cleaned_text = self._clean_text(extracted_content.text)
        word_count = len(cleaned_text.split())

        # Skip if too short
        if word_count < self.min_word_count:
            logger.debug(f"Skipping {extracted_content.url}: Only {word_count} words")
            return None

        # Generate filename
        url_hash = hashlib.md5(extracted_content.url.encode('utf-8')).hexdigest()[:12]
        filename = f"doc_{url_hash}.txt"

        return ProcessedDocument(
            filename=filename,
            content=cleaned_text,
            url=extracted_content.url,
            title=extracted_content.title,
            word_count=word_count
        )

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning for OpenAI upload"""
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Clean up line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove excessive punctuation
        text = re.sub(r'([.!?]){3,}', r'\1', text)

        return text.strip()

    def process_multiple(self, extracted_contents: List[ExtractedContent]) -> List[ProcessedDocument]:
        """Process multiple extracted contents"""
        documents = []

        for content in extracted_contents:
            doc = self.process_content(content)
            if doc:
                documents.append(doc)

        logger.info(f"Processed {len(documents)} documents from {len(extracted_contents)} extracted contents")
        return documents

    def save_documents_to_directory(self, documents: List[ProcessedDocument], output_dir: str):
        """Save processed documents to directory as text files"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        for doc in documents:
            filepath = os.path.join(output_dir, doc.filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(doc.to_text_file())

        logger.info(f"Saved {len(documents)} text files to {output_dir}")