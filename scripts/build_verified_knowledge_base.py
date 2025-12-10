#!/usr/bin/env python3
"""
Build SimpleQA-Verified Knowledge Base

Creates knowledge_base_verified/ with exactly 1,000 files (one per question)
for academically-rigorous RAG evaluation.

Phases:
1. Load and clean SimpleQA-Verified dataset
2. Audit cache quality for all URLs
3. Fetch missing/invalid URLs
4. Aggregate content per question (1 file per question)
5. Validate that answers appear in content
6. Generate comprehensive audit reports

Usage:
    docker compose run --rm simple-evals python scripts/build_verified_knowledge_base.py
    docker compose run --rm simple-evals python scripts/build_verified_knowledge_base.py --validate-answers
    docker compose run --rm simple-evals python scripts/build_verified_knowledge_base.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.url_cleaner import URLCleaner, extract_urls_from_csv_row, CleaningReport
from src.cache_quality_auditor import CacheQualityAuditor, CacheQualityReport
from src.answer_validator import AnswerValidator, ValidationReport
from src.content_extractor import ContentExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VerifiedKnowledgeBaseBuilder:
    """
    Builds SimpleQA-Verified Knowledge Base with comprehensive validation.
    """

    def __init__(
        self,
        dataset_path: str = "simpleqa-verified/simpleqa_verified.csv",
        cache_dir: str = "cache/url_cache",
        output_dir: str = "knowledge_base_verified",
        min_words: int = 50,
        validate_answers: bool = True,
    ):
        self.dataset_path = Path(dataset_path)
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.min_words = min_words
        self.validate_answers = validate_answers

        # Initialize components
        self.url_cleaner = URLCleaner()
        self.cache_auditor = CacheQualityAuditor(
            cache_dir=str(cache_dir),
            min_words=min_words,
        )
        self.content_extractor = ContentExtractor()
        self.answer_validator = AnswerValidator(use_llm_fallback=validate_answers)

        # Reports
        self.cleaning_report: Optional[CleaningReport] = None
        self.cache_report: Optional[CacheQualityReport] = None
        self.validation_report: Optional[ValidationReport] = None
        self.build_stats: Dict[str, Any] = {}

    def load_dataset(self) -> pd.DataFrame:
        """Load SimpleQA-Verified dataset."""
        logger.info(f"Loading dataset from {self.dataset_path}")

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        df = pd.read_csv(self.dataset_path)
        logger.info(f"Loaded {len(df)} questions")

        return df

    def phase1_clean_urls(self, df: pd.DataFrame) -> Dict[int, List[str]]:
        """
        Phase 1: Clean all URLs from dataset.

        Returns:
            Dict mapping original_index to list of cleaned URLs
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: URL Cleaning")
        logger.info("=" * 60)

        all_urls = []
        question_urls = {}

        for _, row in df.iterrows():
            idx = row['original_index']
            raw_urls = extract_urls_from_csv_row(row['urls'])
            question_urls[idx] = []

            for url in raw_urls:
                result = self.url_cleaner.clean(url)
                if result.success:
                    question_urls[idx].append(result.cleaned_url)
                    all_urls.append(result.cleaned_url)
                else:
                    logger.warning(f"Failed to clean URL for Q{idx}: {url[:50]}... - {result.error}")

        # Generate report
        self.cleaning_report = self.url_cleaner.clean_batch(
            [u for urls in question_urls.values() for u in urls]
        )

        unique_urls = list(set(all_urls))
        logger.info(f"Total URLs: {len(all_urls)}")
        logger.info(f"Unique URLs: {len(unique_urls)}")
        logger.info(f"Questions with URLs: {len([q for q in question_urls.values() if q])}")

        return question_urls

    def phase2_audit_cache(self, question_urls: Dict[int, List[str]]) -> Dict[str, Any]:
        """
        Phase 2: Audit cache quality for all URLs.

        Returns:
            Dict with valid/invalid URL sets
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Cache Quality Audit")
        logger.info("=" * 60)

        # Get unique URLs
        all_urls = list(set(
            url for urls in question_urls.values() for url in urls
        ))
        logger.info(f"Auditing {len(all_urls)} unique URLs...")

        def progress(current, total):
            if current % 100 == 0 or current == total:
                logger.info(f"  Progress: {current}/{total}")

        self.cache_report = self.cache_auditor.audit_urls(all_urls, progress_callback=progress)

        # Categorize URLs
        valid_urls = set()
        invalid_urls = set()

        for result in self.cache_report.results:
            if result.is_valid:
                valid_urls.add(result.url)
            else:
                invalid_urls.add(result.url)

        logger.info(f"Valid URLs: {len(valid_urls)}")
        logger.info(f"Invalid URLs: {len(invalid_urls)}")
        logger.info(f"Validity rate: {len(valid_urls)/len(all_urls)*100:.1f}%")

        # Log check breakdown
        logger.info("\nCheck breakdown:")
        for check, counts in self.cache_report.check_summary.items():
            pass_rate = counts['passed'] / max(counts['passed'] + counts['failed'], 1) * 100
            logger.info(f"  {check}: {pass_rate:.1f}% passed")

        return {
            'valid_urls': valid_urls,
            'invalid_urls': invalid_urls,
            'url_quality': {r.url: r for r in self.cache_report.results},
        }

    def phase3_fetch_missing(self, invalid_urls: set) -> int:
        """
        Phase 3: Fetch missing/invalid URLs.

        Returns:
            Number of URLs successfully fetched
        """
        logger.info("=" * 60)
        logger.info("PHASE 3: Fetch Missing URLs")
        logger.info("=" * 60)

        if not invalid_urls:
            logger.info("No URLs to fetch - all cached content is valid!")
            return 0

        logger.info(f"Would fetch {len(invalid_urls)} URLs...")
        logger.info("(Skipping actual fetch in this run - use --fetch to enable)")

        # TODO: Implement actual fetching with CachedURLFetcher
        # For now, just report what would be fetched
        return 0

    def phase4_build_documents(
        self,
        df: pd.DataFrame,
        question_urls: Dict[int, List[str]],
        cache_audit: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Phase 4: Build knowledge base documents.

        Creates one document per question, aggregating all valid source content.
        IMPORTANT: Documents contain ONLY source content, NO questions/answers.

        Returns:
            List of document metadata
        """
        logger.info("=" * 60)
        logger.info("PHASE 4: Build Documents")
        logger.info("=" * 60)

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        audit_dir = self.output_dir / "audit"
        audit_dir.mkdir(exist_ok=True)

        documents = []
        valid_urls = cache_audit['valid_urls']
        url_quality = cache_audit['url_quality']

        for seq_idx, (_, row) in enumerate(df.iterrows(), 1):
            orig_idx = row['original_index']
            urls = question_urls.get(orig_idx, [])

            # Filter to valid URLs
            valid_question_urls = [u for u in urls if u in valid_urls]

            # Extract content from each valid URL
            content_parts = []
            sources_info = []

            for url in valid_question_urls:
                quality = url_quality.get(url)
                if quality and quality.is_valid:
                    # Get extracted content
                    try:
                        import pickle
                        import hashlib
                        cache_key = hashlib.md5(url.encode()).hexdigest()
                        cache_path = self.cache_dir / f"{cache_key}.pkl"

                        if cache_path.exists():
                            with open(cache_path, 'rb') as f:
                                fetch_result = pickle.load(f)

                            extracted = self.content_extractor.extract_from_fetch_result(fetch_result)

                            if extracted.success and extracted.text:
                                content_parts.append(f"=== SOURCE: {url} ===\n\n{extracted.text}")
                                sources_info.append({
                                    'url': url,
                                    'word_count': extracted.word_count or 0,
                                    'title': extracted.title,
                                })
                    except Exception as e:
                        logger.warning(f"Failed to extract Q{orig_idx} URL {url[:50]}: {e}")

            # Create document content (NO questions or answers!)
            doc_content = "\n\n".join(content_parts)
            word_count = len(doc_content.split()) if doc_content else 0

            # Save document
            doc_filename = f"verified_{seq_idx:04d}.txt"
            doc_path = self.output_dir / doc_filename

            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)

            # Track metadata
            doc_meta = {
                'filename': doc_filename,
                'sequence_index': seq_idx,
                'original_index': orig_idx,
                'topic': row.get('topic', ''),
                'answer_type': row.get('answer_type', ''),
                'urls_attempted': len(urls),
                'urls_valid': len(valid_question_urls),
                'sources_extracted': len(sources_info),
                'word_count': word_count,
                'sources': sources_info,
            }
            documents.append(doc_meta)

            # Progress
            if seq_idx % 100 == 0 or seq_idx == len(df):
                logger.info(f"  Built {seq_idx}/{len(df)} documents")

        logger.info(f"Created {len(documents)} documents")
        logger.info(f"Average word count: {sum(d['word_count'] for d in documents)/len(documents):.0f}")

        return documents

    def phase5_validate_answers(
        self,
        df: pd.DataFrame,
        documents: List[Dict[str, Any]],
    ) -> ValidationReport:
        """
        Phase 5: Validate that answers appear in KB content.

        Returns:
            ValidationReport with coverage statistics
        """
        logger.info("=" * 60)
        logger.info("PHASE 5: Answer Validation")
        logger.info("=" * 60)

        if not self.validate_answers:
            logger.info("Answer validation disabled")
            return None

        # Prepare question/content pairs
        questions = []
        contents = []

        for doc_meta, (_, row) in zip(documents, df.iterrows()):
            questions.append({
                'index': doc_meta['original_index'],
                'question': row['problem'],
                'answer': row['answer'],
            })

            # Load document content
            doc_path = self.output_dir / doc_meta['filename']
            with open(doc_path, 'r', encoding='utf-8') as f:
                contents.append(f.read())

        def progress(current, total):
            if current % 50 == 0 or current == total:
                logger.info(f"  Validated {current}/{total} questions")

        logger.info(f"Validating {len(questions)} question-answer pairs...")
        self.validation_report = self.answer_validator.validate_batch(
            questions, contents, progress_callback=progress
        )

        logger.info(f"Answers found: {self.validation_report.answers_found}/{self.validation_report.total_questions}")
        logger.info(f"Coverage rate: {self.validation_report.coverage_rate*100:.1f}%")
        logger.info(f"By method: {self.validation_report.by_method}")

        return self.validation_report

    def phase6_generate_reports(self, documents: List[Dict[str, Any]]) -> None:
        """
        Phase 6: Generate comprehensive audit reports.
        """
        logger.info("=" * 60)
        logger.info("PHASE 6: Generate Reports")
        logger.info("=" * 60)

        audit_dir = self.output_dir / "audit"

        # Build manifest
        manifest = {
            'version': '1.0.0',
            'build_timestamp': datetime.now().isoformat(),
            'dataset_source': str(self.dataset_path),
            'statistics': {
                'total_documents': len(documents),
                'total_words': sum(d['word_count'] for d in documents),
                'avg_words_per_doc': sum(d['word_count'] for d in documents) / max(len(documents), 1),
                'documents_with_content': sum(1 for d in documents if d['word_count'] > 0),
            },
            'documents': documents,
        }

        with open(self.output_dir / 'build_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        # Save cleaning report
        if self.cleaning_report:
            self.url_cleaner.save_report(
                self.cleaning_report,
                audit_dir / 'url_cleaning_audit.json'
            )

        # Save cache quality report
        if self.cache_report:
            self.cache_auditor.save_report(
                self.cache_report,
                audit_dir / 'cache_quality_audit.json'
            )

        # Save validation report
        if self.validation_report:
            self.answer_validator.save_report(
                self.validation_report,
                audit_dir / 'answer_validation_report.json'
            )

        # Build summary
        summary = {
            'build_timestamp': datetime.now().isoformat(),
            'total_questions': len(documents),
            'total_documents': len(documents),
            'total_words': sum(d['word_count'] for d in documents),
            'cache_quality': {
                'total_urls': self.cache_report.total_urls if self.cache_report else 0,
                'valid_urls': self.cache_report.valid_urls if self.cache_report else 0,
            },
            'answer_coverage': {
                'answers_found': self.validation_report.answers_found if self.validation_report else 0,
                'coverage_rate': self.validation_report.coverage_rate if self.validation_report else 0,
            } if self.validation_report else None,
        }

        with open(self.output_dir / 'build_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Reports saved to {audit_dir}")

    def build(self) -> Dict[str, Any]:
        """
        Execute full build pipeline.

        Returns:
            Build summary statistics
        """
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("SimpleQA-Verified Knowledge Base Builder")
        logger.info("=" * 60)
        logger.info(f"Output directory: {self.output_dir}")

        # Load dataset
        df = self.load_dataset()

        # Phase 1: Clean URLs
        question_urls = self.phase1_clean_urls(df)

        # Phase 2: Audit cache
        cache_audit = self.phase2_audit_cache(question_urls)

        # Phase 3: Fetch missing (placeholder)
        self.phase3_fetch_missing(cache_audit['invalid_urls'])

        # Phase 4: Build documents
        documents = self.phase4_build_documents(df, question_urls, cache_audit)

        # Phase 5: Validate answers
        if self.validate_answers:
            self.phase5_validate_answers(df, documents)

        # Phase 6: Generate reports
        self.phase6_generate_reports(documents)

        # Final summary
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info("BUILD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.1f}s")
        logger.info(f"Documents created: {len(documents)}")
        logger.info(f"Output directory: {self.output_dir}")

        return {
            'success': True,
            'documents': len(documents),
            'duration_seconds': duration,
            'output_dir': str(self.output_dir),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Build SimpleQA-Verified Knowledge Base"
    )
    parser.add_argument(
        '--dataset', '-d',
        default='simpleqa-verified/simpleqa_verified.csv',
        help='Path to SimpleQA-Verified CSV'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='knowledge_base_verified',
        help='Output directory for KB files'
    )
    parser.add_argument(
        '--cache-dir', '-c',
        default='cache/url_cache',
        help='Cache directory for URL content'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=50,
        help='Minimum words per document'
    )
    parser.add_argument(
        '--validate-answers',
        action='store_true',
        default=True,
        help='Enable answer validation (default: True)'
    )
    parser.add_argument(
        '--no-validate-answers',
        action='store_true',
        help='Disable answer validation'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without building'
    )

    args = parser.parse_args()

    # Handle validation flag
    validate = args.validate_answers and not args.no_validate_answers

    if args.dry_run:
        logger.info("DRY RUN - would build with these settings:")
        logger.info(f"  Dataset: {args.dataset}")
        logger.info(f"  Output: {args.output_dir}")
        logger.info(f"  Cache: {args.cache_dir}")
        logger.info(f"  Min words: {args.min_words}")
        logger.info(f"  Validate answers: {validate}")
        return

    builder = VerifiedKnowledgeBaseBuilder(
        dataset_path=args.dataset,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        min_words=args.min_words,
        validate_answers=validate,
    )

    result = builder.build()

    if result['success']:
        logger.info("Build completed successfully!")
        sys.exit(0)
    else:
        logger.error("Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
