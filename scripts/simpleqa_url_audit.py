#!/usr/bin/env python3
"""
Simple QA URL Audit Script

This script audits the Simple QA test set to identify records with bad URLs.
A URL is considered "bad" if:
1. It failed to fetch (marked as failure in cache)
2. It's not present in the knowledge base (no corresponding document)

The script generates a comprehensive audit report showing:
- Records where all URLs are bad
- Records where some URLs are bad
- Statistics on URL success/failure rates
- Details on specific failure types
"""

import json
import csv
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import ast


@dataclass
class URLStatus:
    """Status information for a URL"""
    url: str
    in_cache: bool
    cache_success: bool
    in_knowledge_base: bool
    is_bad: bool
    failure_reason: Optional[str] = None


@dataclass
class RecordAudit:
    """Audit results for a single test record"""
    record_id: int
    question: str
    answer: str
    topic: str
    answer_type: str
    urls: List[str]
    url_statuses: List[URLStatus]
    total_urls: int
    bad_urls: int
    good_urls: int
    all_urls_bad: bool
    some_urls_bad: bool


class SimpleQAURLAuditor:
    """Auditor for Simple QA test set URLs"""

    def __init__(self,
                 test_set_path: str = "build-rag/simple_qa_test_set.csv",
                 cache_metadata_path: str = "cache/url_cache/cache_metadata.json",
                 knowledge_base_dir: str = "knowledge_base_full"):
        self.test_set_path = test_set_path
        self.cache_metadata_path = cache_metadata_path
        self.knowledge_base_dir = Path(knowledge_base_dir)

        # Load data
        self.cache_metadata = self._load_cache_metadata()
        self.kb_urls = self._load_knowledge_base_urls()

    def _load_cache_metadata(self) -> Dict:
        """Load URL cache metadata"""
        try:
            with open(self.cache_metadata_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Cache metadata file not found at {self.cache_metadata_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse cache metadata: {e}")
            return {}

    def _load_knowledge_base_urls(self) -> Set[str]:
        """Extract URLs from knowledge base documents"""
        kb_urls = set()

        if not self.knowledge_base_dir.exists():
            print(f"Warning: Knowledge base directory not found at {self.knowledge_base_dir}")
            return kb_urls

        # Look for build metadata first
        build_metadata_path = self.knowledge_base_dir / "build_metadata.json"
        if build_metadata_path.exists():
            try:
                with open(build_metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Extract URLs from document metadata if available
                if 'document_sources' in metadata:
                    for doc_info in metadata['document_sources'].values():
                        if 'source_url' in doc_info:
                            kb_urls.add(doc_info['source_url'])
                    return kb_urls
            except Exception as e:
                print(f"Warning: Failed to load build metadata: {e}")

        # Fallback: scan document files for URL information
        doc_files = list(self.knowledge_base_dir.glob("doc_*.txt"))
        print(f"Scanning {len(doc_files)} knowledge base documents...")

        for doc_file in doc_files:
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    # Read first few lines to look for source URL
                    for i, line in enumerate(f):
                        if i > 10:  # Only check first 10 lines
                            break
                        line = line.strip()
                        if line.startswith('Source:') or line.startswith('URL:'):
                            # Extract URL from source line
                            url = line.split(':', 1)[1].strip()
                            kb_urls.add(url)
                            break
                        elif line.startswith('http'):
                            # Direct URL line
                            kb_urls.add(line)
                            break
            except Exception as e:
                print(f"Warning: Failed to read {doc_file}: {e}")

        return kb_urls

    def _get_url_hash(self, url: str) -> str:
        """Generate cache key (hash) for URL"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()

    def _analyze_url(self, url: str) -> URLStatus:
        """Analyze the status of a single URL"""
        url_hash = self._get_url_hash(url)

        # Check cache status
        in_cache = url_hash in self.cache_metadata
        cache_success = False
        failure_reason = None

        if in_cache:
            cache_entry = self.cache_metadata[url_hash]
            cache_success = cache_entry.get('success', False)
            if not cache_success:
                failure_reason = "Cache indicates fetch failure"
        else:
            failure_reason = "URL not found in cache"

        # Check knowledge base status
        in_knowledge_base = url in self.kb_urls

        # Determine if URL is "bad"
        is_bad = not cache_success or not in_knowledge_base

        if is_bad and not failure_reason:
            if not in_knowledge_base:
                failure_reason = "URL not present in knowledge base"

        return URLStatus(
            url=url,
            in_cache=in_cache,
            cache_success=cache_success,
            in_knowledge_base=in_knowledge_base,
            is_bad=is_bad,
            failure_reason=failure_reason
        )

    def _parse_metadata_field(self, metadata_str: str) -> Dict:
        """Parse the metadata field from CSV (which contains a dict as string)"""
        try:
            # Use ast.literal_eval for safe evaluation of the dict string
            return ast.literal_eval(metadata_str)
        except Exception as e:
            print(f"Warning: Failed to parse metadata: {metadata_str[:100]}... Error: {e}")
            return {}

    def audit_test_set(self) -> List[RecordAudit]:
        """Audit the entire Simple QA test set"""
        audits = []

        try:
            with open(self.test_set_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for i, row in enumerate(reader):
                    try:
                        # Parse metadata field
                        metadata = self._parse_metadata_field(row['metadata'])

                        urls = metadata.get('urls', [])
                        topic = metadata.get('topic', 'Unknown')
                        answer_type = metadata.get('answer_type', 'Unknown')

                        # Analyze each URL
                        url_statuses = [self._analyze_url(url) for url in urls]

                        # Calculate statistics
                        bad_urls = sum(1 for status in url_statuses if status.is_bad)
                        good_urls = len(url_statuses) - bad_urls

                        audit = RecordAudit(
                            record_id=i + 1,
                            question=row['problem'],
                            answer=row['answer'],
                            topic=topic,
                            answer_type=answer_type,
                            urls=urls,
                            url_statuses=url_statuses,
                            total_urls=len(urls),
                            bad_urls=bad_urls,
                            good_urls=good_urls,
                            all_urls_bad=(bad_urls == len(urls) and len(urls) > 0),
                            some_urls_bad=(bad_urls > 0)
                        )

                        audits.append(audit)

                    except Exception as e:
                        print(f"Error processing record {i+1}: {e}")

        except FileNotFoundError:
            print(f"Error: Test set file not found at {self.test_set_path}")
            return []

        return audits

    def generate_report(self, audits: List[RecordAudit], output_file: str = "simpleqa_url_audit_report.txt"):
        """Generate comprehensive audit report"""

        # Calculate overall statistics
        total_records = len(audits)
        records_with_all_bad_urls = sum(1 for audit in audits if audit.all_urls_bad)
        records_with_some_bad_urls = sum(1 for audit in audits if audit.some_urls_bad)
        records_with_all_good_urls = total_records - records_with_some_bad_urls

        total_urls = sum(audit.total_urls for audit in audits)
        total_bad_urls = sum(audit.bad_urls for audit in audits)
        total_good_urls = total_urls - total_bad_urls

        # Group by topic and answer type
        topic_stats = defaultdict(lambda: {'total': 0, 'all_bad': 0, 'some_bad': 0})
        answer_type_stats = defaultdict(lambda: {'total': 0, 'all_bad': 0, 'some_bad': 0})

        for audit in audits:
            topic_stats[audit.topic]['total'] += 1
            answer_type_stats[audit.answer_type]['total'] += 1

            if audit.all_urls_bad:
                topic_stats[audit.topic]['all_bad'] += 1
                answer_type_stats[audit.answer_type]['all_bad'] += 1
            elif audit.some_urls_bad:
                topic_stats[audit.topic]['some_bad'] += 1
                answer_type_stats[audit.answer_type]['some_bad'] += 1

        # Collect failure reasons
        failure_reasons = defaultdict(int)
        for audit in audits:
            for status in audit.url_statuses:
                if status.is_bad and status.failure_reason:
                    failure_reasons[status.failure_reason] += 1

        # Generate report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SIMPLE QA URL AUDIT REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total records: {total_records}\n")
            f.write(f"Records with ALL URLs bad: {records_with_all_bad_urls} ({records_with_all_bad_urls/total_records*100:.1f}%)\n")
            f.write(f"Records with SOME URLs bad: {records_with_some_bad_urls} ({records_with_some_bad_urls/total_records*100:.1f}%)\n")
            f.write(f"Records with ALL URLs good: {records_with_all_good_urls} ({records_with_all_good_urls/total_records*100:.1f}%)\n")
            f.write(f"\nTotal URLs: {total_urls}\n")
            f.write(f"Bad URLs: {total_bad_urls} ({total_bad_urls/total_urls*100:.1f}%)\n")
            f.write(f"Good URLs: {total_good_urls} ({total_good_urls/total_urls*100:.1f}%)\n\n")

            # Cache and KB statistics
            f.write("CACHE AND KNOWLEDGE BASE STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total cache entries: {len(self.cache_metadata)}\n")
            cache_successes = sum(1 for entry in self.cache_metadata.values() if entry.get('success', False))
            f.write(f"Successful cache entries: {cache_successes}\n")
            f.write(f"Failed cache entries: {len(self.cache_metadata) - cache_successes}\n")
            f.write(f"Knowledge base URLs: {len(self.kb_urls)}\n\n")

            # Failure reasons
            f.write("FAILURE REASONS\n")
            f.write("-" * 40 + "\n")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{reason}: {count}\n")
            f.write("\n")

            # Topic breakdown
            f.write("BREAKDOWN BY TOPIC\n")
            f.write("-" * 40 + "\n")
            for topic, stats in sorted(topic_stats.items()):
                total = stats['total']
                all_bad = stats['all_bad']
                some_bad = stats['some_bad']
                f.write(f"{topic}: {total} records\n")
                f.write(f"  All URLs bad: {all_bad} ({all_bad/total*100:.1f}%)\n")
                f.write(f"  Some URLs bad: {some_bad} ({some_bad/total*100:.1f}%)\n")
                f.write(f"  All URLs good: {total-some_bad} ({(total-some_bad)/total*100:.1f}%)\n\n")

            # Answer type breakdown
            f.write("BREAKDOWN BY ANSWER TYPE\n")
            f.write("-" * 40 + "\n")
            for answer_type, stats in sorted(answer_type_stats.items()):
                total = stats['total']
                all_bad = stats['all_bad']
                some_bad = stats['some_bad']
                f.write(f"{answer_type}: {total} records\n")
                f.write(f"  All URLs bad: {all_bad} ({all_bad/total*100:.1f}%)\n")
                f.write(f"  Some URLs bad: {some_bad} ({some_bad/total*100:.1f}%)\n")
                f.write(f"  All URLs good: {total-some_bad} ({(total-some_bad)/total*100:.1f}%)\n\n")

            # Records with all URLs bad
            f.write("RECORDS WITH ALL URLS BAD\n")
            f.write("-" * 80 + "\n")
            all_bad_records = [audit for audit in audits if audit.all_urls_bad]
            for audit in all_bad_records:
                f.write(f"Record {audit.record_id}: {audit.topic} / {audit.answer_type}\n")
                f.write(f"Question: {audit.question}\n")
                f.write(f"Answer: {audit.answer}\n")
                f.write(f"URLs ({len(audit.urls)}):\n")
                for status in audit.url_statuses:
                    f.write(f"  ❌ {status.url}\n")
                    f.write(f"     Reason: {status.failure_reason}\n")
                f.write("\n")

            # Sample records with some URLs bad
            f.write("SAMPLE RECORDS WITH SOME URLS BAD (first 10)\n")
            f.write("-" * 80 + "\n")
            some_bad_records = [audit for audit in audits if audit.some_urls_bad and not audit.all_urls_bad][:10]
            for audit in some_bad_records:
                f.write(f"Record {audit.record_id}: {audit.topic} / {audit.answer_type}\n")
                f.write(f"Question: {audit.question}\n")
                f.write(f"Answer: {audit.answer}\n")
                f.write(f"URLs ({audit.good_urls} good, {audit.bad_urls} bad):\n")
                for status in audit.url_statuses:
                    if status.is_bad:
                        f.write(f"  ❌ {status.url}\n")
                        f.write(f"     Reason: {status.failure_reason}\n")
                    else:
                        f.write(f"  ✅ {status.url}\n")
                f.write("\n")

        print(f"Audit report generated: {output_file}")
        return output_file


def main():
    """Main function to run the audit"""
    print("Starting Simple QA URL audit...")

    # Initialize auditor
    auditor = SimpleQAURLAuditor()

    # Run audit
    print("Analyzing test set records...")
    audits = auditor.audit_test_set()

    if not audits:
        print("No records found to audit!")
        return

    print(f"Analyzed {len(audits)} records")

    # Generate report
    print("Generating audit report...")
    report_file = auditor.generate_report(audits)

    # Print summary to console
    total_records = len(audits)
    records_all_bad = sum(1 for audit in audits if audit.all_urls_bad)
    records_some_bad = sum(1 for audit in audits if audit.some_urls_bad)

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total records analyzed: {total_records}")
    print(f"Records with ALL URLs bad: {records_all_bad}")
    print(f"Records with SOME URLs bad: {records_some_bad}")
    print(f"Records with ALL URLs good: {total_records - records_some_bad}")
    print(f"\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    main()