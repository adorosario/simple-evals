#!/usr/bin/env python3
"""
SimpleQA Knowledge Base Coverage Audit Script

This script audits the SimpleQA dataset to identify rows where all URLs are missing
from the knowledge base, adding a 'rag_should_abstain' column to indicate when
RAG systems should abstain due to lack of supporting evidence.

IMPORTANT: Only documents with ≥50 words are considered valid knowledge base content.
Documents with fewer than 50 words are treated as insufficient content.

Usage:
    python scripts/simpleqa_kb_coverage_audit.py [options]

Options:
    --input TEXT     Input CSV file path [default: ./build-rag/simple_qa_test_set.csv]
    --output TEXT    Output CSV file path [default: ./build-rag/simple_qa_test_set_enhanced.csv]
    --stats          Generate statistical analysis
    --cache-dir TEXT Cache directory [default: ./cache/url_cache]
    --kb-dir TEXT    Knowledge base directory [default: ./knowledge_base_full]
    --verbose        Enable verbose logging
"""

import json
import csv
import hashlib
import os
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import ast
import sys


@dataclass
class URLAuditResult:
    """Results of auditing a single URL"""
    url: str
    in_cache: bool
    cache_success: bool
    in_kb: bool
    failure_reason: Optional[str] = None


@dataclass
class QuestionAuditResult:
    """Results of auditing a single question"""
    row_index: int
    question: str
    answer: str
    topic: str
    answer_type: str
    urls: List[str]
    url_results: List[URLAuditResult]
    total_urls: int
    urls_in_kb: int
    kb_coverage_ratio: float
    rag_should_abstain: int  # 1 if all URLs missing, 0 otherwise


class SimpleQAKBCoverageAuditor:
    """Auditor for SimpleQA dataset knowledge base coverage"""

    def __init__(self, cache_dir: str = "./cache/url_cache", kb_dir: str = "./knowledge_base_full", verbose: bool = False):
        self.cache_dir = Path(cache_dir)
        self.kb_dir = Path(kb_dir)
        self.verbose = verbose

        # Load cache metadata
        self.cache_metadata = self._load_cache_metadata()
        print(f"Loaded cache metadata with {len(self.cache_metadata)} entries")

        # Load KB URL mapping
        self.kb_urls = self._load_kb_urls()
        print(f"Identified {len(self.kb_urls)} URLs in knowledge base")

    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load URL cache metadata"""
        cache_file = self.cache_dir / "cache_metadata.json"
        if not cache_file.exists():
            print(f"Warning: Cache metadata file not found at {cache_file}")
            return {}

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache metadata: {e}")
            return {}

    def _load_kb_urls(self) -> Set[str]:
        """Load URLs that are available in the knowledge base with substantial content (≥50 words)"""
        kb_urls = set()

        # Scan all document files and check word count
        doc_files = list(self.kb_dir.glob("doc_*.txt"))
        if self.verbose:
            print(f"Scanning {len(doc_files)} documents for content validation...")

        valid_docs = 0
        for doc_file in doc_files:
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    source_url = None
                    word_count = 0

                    # Read first few lines to get source URL and word count
                    for i, line in enumerate(f):
                        if i > 10:  # Only check first 10 lines
                            break
                        line = line.strip()
                        if line.startswith('Source:'):
                            source_url = line.split(':', 1)[1].strip()
                        elif line.startswith('Words:'):
                            try:
                                word_count = int(line.split(':', 1)[1].strip())
                            except ValueError:
                                continue

                        # Stop if we have both pieces of info
                        if source_url and word_count:
                            break

                    # Only include URLs with documents that have ≥50 words
                    if source_url and word_count >= 50:
                        kb_urls.add(source_url)
                        valid_docs += 1

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to read {doc_file}: {e}")
                continue

        if self.verbose:
            print(f"Found {valid_docs} documents with ≥50 words")

        return kb_urls

    def _get_url_hash(self, url: str) -> str:
        """Generate cache key (hash) for URL"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()

    def _audit_url(self, url: str) -> URLAuditResult:
        """Audit the status of a single URL"""
        url_hash = self._get_url_hash(url)

        # Check cache status
        in_cache = url_hash in self.cache_metadata
        cache_success = False
        failure_reason = None

        if in_cache:
            cache_entry = self.cache_metadata[url_hash]
            cache_success = cache_entry.get('success', False)
            if not cache_success:
                failure_reason = cache_entry.get('error', 'Unknown cache failure')
        else:
            failure_reason = "URL not found in cache"

        # Check if URL is in knowledge base (with ≥50 words validation)
        # This now checks against our pre-validated set of URLs with substantial content
        in_kb = url in self.kb_urls

        # If not in KB but was in cache successfully, it means document has <50 words
        if not in_kb and cache_success:
            failure_reason = "Document exists but has <50 words (insufficient content)"

        return URLAuditResult(
            url=url,
            in_cache=in_cache,
            cache_success=cache_success,
            in_kb=in_kb,
            failure_reason=failure_reason
        )

    def _parse_metadata(self, metadata_str: str) -> Dict[str, Any]:
        """Parse metadata JSON string safely"""
        try:
            # Try JSON parsing first
            return json.loads(metadata_str)
        except json.JSONDecodeError:
            try:
                # Fallback to ast.literal_eval for Python-style dicts
                return ast.literal_eval(metadata_str)
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Failed to parse metadata: {metadata_str[:100]}... Error: {e}")
                return {}

    def _extract_urls_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract URLs from metadata dictionary"""
        urls = metadata.get('urls', [])
        if not isinstance(urls, list):
            print(f"Warning: URLs field is not a list: {type(urls)}")
            return []
        return [url for url in urls if isinstance(url, str) and url.strip()]

    def audit_question(self, row_index: int, row: Dict[str, str]) -> Optional[QuestionAuditResult]:
        """Audit a single question row"""
        try:
            # Parse metadata
            metadata_str = row.get('metadata', '{}')
            metadata = self._parse_metadata(metadata_str)

            # Extract basic info
            question = row.get('problem', '')
            answer = row.get('answer', '')
            topic = metadata.get('topic', 'Unknown')
            answer_type = metadata.get('answer_type', 'Unknown')

            # Extract URLs
            urls = self._extract_urls_from_metadata(metadata)

            if not urls:
                if self.verbose:
                    print(f"Warning: No URLs found for row {row_index}")
                urls = []

            # Audit each URL
            url_results = []
            urls_in_kb = 0

            for url in urls:
                url_result = self._audit_url(url)
                url_results.append(url_result)
                if url_result.in_kb:
                    urls_in_kb += 1

            # Calculate metrics
            total_urls = len(urls)
            kb_coverage_ratio = urls_in_kb / total_urls if total_urls > 0 else 0.0
            rag_should_abstain = 1 if total_urls > 0 and urls_in_kb == 0 else 0

            return QuestionAuditResult(
                row_index=row_index,
                question=question,
                answer=answer,
                topic=topic,
                answer_type=answer_type,
                urls=urls,
                url_results=url_results,
                total_urls=total_urls,
                urls_in_kb=urls_in_kb,
                kb_coverage_ratio=kb_coverage_ratio,
                rag_should_abstain=rag_should_abstain
            )

        except Exception as e:
            print(f"Error auditing row {row_index}: {e}")
            return None

    def audit_dataset(self, input_file: str) -> List[QuestionAuditResult]:
        """Audit the entire dataset"""
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        results = []
        skipped_rows = 0

        with open(input_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)

            for row_index, row in enumerate(reader, start=1):
                if self.verbose and row_index % 500 == 0:
                    print(f"Processing row {row_index}...")

                result = self.audit_question(row_index, row)
                if result:
                    results.append(result)
                else:
                    skipped_rows += 1

        print(f"Audit complete. Processed {len(results)} rows, skipped {skipped_rows} rows.")
        return results

    def save_enhanced_dataset(self, results: List[QuestionAuditResult], original_file: str, output_file: str):
        """Save enhanced dataset with additional columns"""
        original_path = Path(original_file)
        output_path = Path(output_file)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read original CSV to preserve structure
        with open(original_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            original_fieldnames = reader.fieldnames
            original_rows = list(reader)

        # Prepare enhanced fieldnames
        enhanced_fieldnames = list(original_fieldnames) + [
            'rag_should_abstain',
            'urls_in_kb',
            'total_urls',
            'kb_coverage_ratio'
        ]

        # Create result lookup by row index
        result_lookup = {result.row_index: result for result in results}

        # Write enhanced CSV
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=enhanced_fieldnames)
            writer.writeheader()

            for row_index, original_row in enumerate(original_rows, start=1):
                enhanced_row = original_row.copy()

                if row_index in result_lookup:
                    result = result_lookup[row_index]
                    enhanced_row['rag_should_abstain'] = result.rag_should_abstain
                    enhanced_row['urls_in_kb'] = result.urls_in_kb
                    enhanced_row['total_urls'] = result.total_urls
                    enhanced_row['kb_coverage_ratio'] = f"{result.kb_coverage_ratio:.3f}"
                else:
                    # Fill with default values for skipped rows
                    enhanced_row['rag_should_abstain'] = 0
                    enhanced_row['urls_in_kb'] = 0
                    enhanced_row['total_urls'] = 0
                    enhanced_row['kb_coverage_ratio'] = "0.000"

                writer.writerow(enhanced_row)

        print(f"Enhanced dataset saved to: {output_file}")

    def generate_statistics(self, results: List[QuestionAuditResult], output_dir: str = "."):
        """Generate comprehensive statistical analysis"""
        if not results:
            print("No results to analyze")
            return

        stats = {}

        # Basic statistics
        total_questions = len(results)
        abstain_count = sum(1 for r in results if r.rag_should_abstain == 1)
        abstain_rate = abstain_count / total_questions

        stats['summary'] = {
            'total_questions': total_questions,
            'questions_should_abstain': abstain_count,
            'abstention_rate': abstain_rate,
            'questions_with_kb_coverage': total_questions - abstain_count
        }

        # URL statistics
        total_urls = sum(r.total_urls for r in results)
        urls_in_kb = sum(r.urls_in_kb for r in results)
        url_success_rate = urls_in_kb / total_urls if total_urls > 0 else 0

        stats['url_statistics'] = {
            'total_urls': total_urls,
            'urls_in_kb': urls_in_kb,
            'urls_missing': total_urls - urls_in_kb,
            'url_success_rate': url_success_rate,
            'avg_urls_per_question': total_urls / total_questions if total_questions > 0 else 0
        }

        # Distribution analysis
        abstain_by_topic = defaultdict(int)
        total_by_topic = defaultdict(int)
        abstain_by_answer_type = defaultdict(int)
        total_by_answer_type = defaultdict(int)

        coverage_distribution = []
        urls_per_question_distribution = []

        for result in results:
            abstain_by_topic[result.topic] += result.rag_should_abstain
            total_by_topic[result.topic] += 1
            abstain_by_answer_type[result.answer_type] += result.rag_should_abstain
            total_by_answer_type[result.answer_type] += 1
            coverage_distribution.append(result.kb_coverage_ratio)
            urls_per_question_distribution.append(result.total_urls)

        stats['topic_analysis'] = {
            topic: {
                'total_questions': total_by_topic[topic],
                'should_abstain': abstain_by_topic[topic],
                'abstention_rate': abstain_by_topic[topic] / total_by_topic[topic]
            }
            for topic in total_by_topic.keys()
        }

        stats['answer_type_analysis'] = {
            answer_type: {
                'total_questions': total_by_answer_type[answer_type],
                'should_abstain': abstain_by_answer_type[answer_type],
                'abstention_rate': abstain_by_answer_type[answer_type] / total_by_answer_type[answer_type]
            }
            for answer_type in total_by_answer_type.keys()
        }

        # Coverage distribution
        coverage_counter = Counter([round(ratio, 1) for ratio in coverage_distribution])
        stats['coverage_distribution'] = dict(coverage_counter)

        # URLs per question distribution
        urls_counter = Counter(urls_per_question_distribution)
        stats['urls_per_question_distribution'] = dict(urls_counter)

        # Save statistics
        output_path = Path(output_dir) / "simpleqa_kb_coverage_statistics.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # Print summary
        print(f"\n=== SimpleQA KB Coverage Audit Statistics ===")
        print(f"Total questions: {total_questions:,}")
        print(f"Questions that should abstain: {abstain_count:,} ({abstain_rate:.1%})")
        print(f"Questions with KB coverage: {total_questions - abstain_count:,}")
        print(f"Total URLs: {total_urls:,}")
        print(f"URLs in KB: {urls_in_kb:,} ({url_success_rate:.1%})")
        print(f"URLs missing: {total_urls - urls_in_kb:,}")
        print(f"Average URLs per question: {total_urls / total_questions:.1f}")

        print(f"\nTop topics requiring abstention:")
        topic_abstain_rates = [(topic, data['abstention_rate']) for topic, data in stats['topic_analysis'].items()]
        topic_abstain_rates.sort(key=lambda x: x[1], reverse=True)
        for topic, rate in topic_abstain_rates[:5]:
            count = stats['topic_analysis'][topic]['should_abstain']
            total = stats['topic_analysis'][topic]['total_questions']
            print(f"  {topic}: {count}/{total} ({rate:.1%})")

        print(f"\nStatistics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit SimpleQA dataset for KB coverage")
    parser.add_argument("--input", default="./build-rag/simple_qa_test_set.csv",
                       help="Input CSV file path")
    parser.add_argument("--output", default="./build-rag/simple_qa_test_set_enhanced.csv",
                       help="Output CSV file path")
    parser.add_argument("--stats", action="store_true",
                       help="Generate statistical analysis")
    parser.add_argument("--cache-dir", default="./cache/url_cache",
                       help="Cache directory")
    parser.add_argument("--kb-dir", default="./knowledge_base_full",
                       help="Knowledge base directory")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Initialize auditor
    auditor = SimpleQAKBCoverageAuditor(
        cache_dir=args.cache_dir,
        kb_dir=args.kb_dir,
        verbose=args.verbose
    )

    # Audit dataset
    print(f"Auditing dataset: {args.input}")
    results = auditor.audit_dataset(args.input)

    # Save enhanced dataset
    print(f"Saving enhanced dataset to: {args.output}")
    auditor.save_enhanced_dataset(results, args.input, args.output)

    # Generate statistics if requested
    if args.stats:
        print("Generating statistical analysis...")
        output_dir = Path(args.output).parent
        auditor.generate_statistics(results, str(output_dir))

    print("Audit complete!")


if __name__ == "__main__":
    main()