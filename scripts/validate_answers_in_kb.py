#!/usr/bin/env python3
"""
Validate Answers in Knowledge Base

Runs hybrid answer validation (string matching + LLM fallback) on existing
knowledge_base_verified/ documents to verify answers are present in content.

Usage:
    docker compose run --rm simple-evals python scripts/validate_answers_in_kb.py
    docker compose run --rm simple-evals python scripts/validate_answers_in_kb.py --no-llm
    docker compose run --rm simple-evals python scripts/validate_answers_in_kb.py --sample 100
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.answer_validator import AnswerValidator, ValidationReport

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_kb_documents(kb_dir: Path) -> Dict[int, str]:
    """
    Load knowledge base documents using manifest for index mapping.

    Returns:
        Dict mapping original_index to document content
    """
    documents = {}

    # Load manifest for index mapping
    manifest_path = kb_dir / "build_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Build mapping from original_index to filename
        idx_to_file = {}
        for doc in manifest.get('documents', []):
            idx_to_file[doc['original_index']] = doc['filename']

        # Load documents using the mapping
        for original_idx, filename in idx_to_file.items():
            doc_path = kb_dir / filename
            if doc_path.exists():
                content = doc_path.read_text(encoding='utf-8')
                documents[original_idx] = content
    else:
        # Fallback: sequential loading (old behavior)
        for doc_file in sorted(kb_dir.glob("verified_*.txt")):
            idx_str = doc_file.stem.replace("verified_", "")
            idx = int(idx_str)
            content = doc_file.read_text(encoding='utf-8')
            documents[idx] = content

    return documents


def validate_answers(
    dataset_path: Path,
    kb_dir: Path,
    use_llm: bool = True,
    sample_size: int = None,
) -> ValidationReport:
    """
    Validate that answers appear in KB content.

    Args:
        dataset_path: Path to SimpleQA-Verified CSV
        kb_dir: Path to knowledge_base_verified directory
        use_llm: Whether to use LLM fallback for string match failures
        sample_size: Optional sample size for testing

    Returns:
        ValidationReport with coverage statistics
    """
    logger.info("=" * 60)
    logger.info("Answer Validation for SimpleQA-Verified KB")
    logger.info("=" * 60)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"KB directory: {kb_dir}")
    logger.info(f"LLM fallback: {use_llm}")
    if sample_size:
        logger.info(f"Sample size: {sample_size}")
    logger.info("=" * 60)

    # Load dataset
    logger.info("Loading dataset...")
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} questions")

    # Load KB documents
    logger.info("Loading KB documents...")
    documents = load_kb_documents(kb_dir)
    logger.info(f"Loaded {len(documents)} documents")

    # Build mapping from original_index to row
    df_by_idx = {row['original_index']: row for _, row in df.iterrows()}

    # Prepare validation data
    questions = []
    contents = []

    for idx in sorted(documents.keys()):
        if idx not in df_by_idx:
            logger.warning(f"No dataset entry for document {idx}")
            continue

        row = df_by_idx[idx]
        # Strip metadata like "(acceptable range: ...)" from answers
        # These annotations break string matching and confuse LLM validation
        answer = row['answer']
        answer = re.sub(r'\s*\(acceptable range:[^)]+\)', '', answer).strip()
        answer = re.sub(r'\s*\(acceptable[^)]*\)', '', answer, flags=re.IGNORECASE).strip()

        questions.append({
            'index': idx,
            'question': row['problem'],
            'answer': answer,
        })
        contents.append(documents[idx])

    # Apply sample size if specified
    if sample_size and sample_size < len(questions):
        questions = questions[:sample_size]
        contents = contents[:sample_size]
        logger.info(f"Sampled {sample_size} questions for validation")

    # Initialize validator
    validator = AnswerValidator(use_llm_fallback=use_llm)

    # Run validation
    logger.info(f"Validating {len(questions)} answers...")

    string_match_only = 0
    llm_verified = 0
    not_found = 0

    def progress_callback(current, total):
        if current % 50 == 0 or current == total:
            logger.info(f"  Progress: {current}/{total}")

    report = validator.validate_batch(questions, contents, progress_callback)

    # Log results
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total questions: {report.total_questions}")
    logger.info(f"Answers found: {report.answers_found}")
    logger.info(f"Answers not found: {report.answers_not_found}")
    logger.info(f"Coverage rate: {report.coverage_rate*100:.1f}%")
    logger.info("")
    logger.info("By method:")
    for method, count in report.by_method.items():
        pct = count / max(report.total_questions, 1) * 100
        logger.info(f"  {method}: {count} ({pct:.1f}%)")

    # Log some not-found examples
    not_found_results = [r for r in report.results if not r.found]
    if not_found_results:
        logger.info("")
        logger.info(f"Examples of answers NOT found ({len(not_found_results)} total):")
        for r in not_found_results[:5]:
            logger.info(f"  Q{r.question_index}: Answer='{r.answer[:50]}...' - {r.llm_explanation or 'No LLM check'}")

    return report


def save_report(report: ValidationReport, output_path: Path) -> None:
    """Save validation report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate answers in SimpleQA-Verified Knowledge Base"
    )
    parser.add_argument(
        '--dataset',
        default='simpleqa-verified/simpleqa_verified.csv',
        help='Path to SimpleQA-Verified CSV'
    )
    parser.add_argument(
        '--kb-dir',
        default='knowledge_base_verified',
        help='Path to knowledge base directory'
    )
    parser.add_argument(
        '--output',
        default='knowledge_base_verified/audit/answer_validation_report.json',
        help='Output path for validation report'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM fallback (string matching only)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample size for testing (default: all)'
    )

    args = parser.parse_args()

    # Validate paths
    dataset_path = Path(args.dataset)
    kb_dir = Path(args.kb_dir)

    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    if not kb_dir.exists():
        logger.error(f"KB directory not found: {kb_dir}")
        sys.exit(1)

    # Run validation
    start_time = datetime.now()

    report = validate_answers(
        dataset_path=dataset_path,
        kb_dir=kb_dir,
        use_llm=not args.no_llm,
        sample_size=args.sample,
    )

    # Save report
    save_report(report, Path(args.output))

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Completed in {duration:.1f}s")

    # Exit with error if coverage is too low
    if report.coverage_rate < 0.90:
        logger.warning(f"Coverage rate {report.coverage_rate*100:.1f}% is below 90% threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
