#!/usr/bin/env python3
"""
Analyze URL overlap between SimpleQA (original) and SimpleQA-Verified datasets.

This script compares URLs between the two datasets to determine:
1. How many URLs are shared vs unique to each dataset
2. Whether existing cached content can be reused
3. What new URLs need to be downloaded for SimpleQA-Verified
"""

import pandas as pd
import ast
import json
import re
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict

# Paths - use /app for Docker container or relative paths for host
SCRIPT_DIR = Path(__file__).parent.parent
ORIGINAL_SIMPLEQA = SCRIPT_DIR / "build-rag/simple_qa_test_set.csv"
VERIFIED_SIMPLEQA = SCRIPT_DIR / "simpleqa-verified/simpleqa_verified.csv"
CACHED_URLS_FILE = SCRIPT_DIR / "build-rag/urls.txt"


def normalize_url(url: str) -> str:
    """Normalize URL for comparison (remove trailing slashes, fragments, etc.)"""
    url = url.strip()
    # Remove trailing parentheses that might be malformed
    url = re.sub(r'\)+$', '', url)
    # Remove trailing slashes
    url = url.rstrip('/')
    # Parse and reconstruct without fragment
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized.lower()


def extract_urls_from_original(df: pd.DataFrame) -> dict:
    """Extract URLs from original SimpleQA dataset metadata field."""
    question_urls = {}
    all_urls = set()

    for idx, row in df.iterrows():
        metadata_str = row['metadata']
        try:
            # Parse the metadata string (it's a Python dict literal)
            metadata = ast.literal_eval(metadata_str)
            urls = metadata.get('urls', [])
            if isinstance(urls, list):
                normalized = [normalize_url(u) for u in urls if u]
                question_urls[idx] = {
                    'question': row['problem'],
                    'answer': row['answer'],
                    'urls': normalized,
                    'original_urls': urls
                }
                all_urls.update(normalized)
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse metadata for row {idx}: {e}")
            question_urls[idx] = {
                'question': row['problem'],
                'answer': row['answer'],
                'urls': [],
                'original_urls': []
            }

    return question_urls, all_urls


def extract_urls_from_verified(df: pd.DataFrame) -> dict:
    """Extract URLs from SimpleQA-Verified dataset urls field."""
    question_urls = {}
    all_urls = set()

    for idx, row in df.iterrows():
        original_index = row['original_index']
        urls_str = str(row['urls']) if pd.notna(row['urls']) else ''

        # Split by comma, handling potential malformed URLs
        raw_urls = [u.strip() for u in urls_str.split(',') if u.strip()]
        normalized = [normalize_url(u) for u in raw_urls if u]

        question_urls[original_index] = {
            'question': row['problem'],
            'answer': row['answer'],
            'topic': row.get('topic', ''),
            'answer_type': row.get('answer_type', ''),
            'multi_step': row.get('multi_step', False),
            'requires_reasoning': row.get('requires_reasoning', False),
            'urls': normalized,
            'original_urls': raw_urls
        }
        all_urls.update(normalized)

    return question_urls, all_urls


def load_cached_urls() -> set:
    """Load URLs that have already been cached."""
    if not CACHED_URLS_FILE.exists():
        return set()

    with open(CACHED_URLS_FILE, 'r') as f:
        urls = [normalize_url(line.strip()) for line in f if line.strip()]
    return set(urls)


def get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return 'unknown'


def analyze_url_overlap():
    """Main analysis function."""
    print("=" * 80)
    print("SimpleQA vs SimpleQA-Verified URL Analysis")
    print("=" * 80)

    # Load datasets
    print("\nLoading datasets...")
    original_df = pd.read_csv(ORIGINAL_SIMPLEQA)
    verified_df = pd.read_csv(VERIFIED_SIMPLEQA)

    print(f"  Original SimpleQA: {len(original_df)} questions")
    print(f"  SimpleQA-Verified: {len(verified_df)} questions")

    # Extract URLs
    print("\nExtracting URLs...")
    original_questions, original_urls = extract_urls_from_original(original_df)
    verified_questions, verified_urls = extract_urls_from_verified(verified_df)

    print(f"  Original SimpleQA unique URLs: {len(original_urls)}")
    print(f"  SimpleQA-Verified unique URLs: {len(verified_urls)}")

    # Load cached URLs
    cached_urls = load_cached_urls()
    print(f"  Currently cached URLs: {len(cached_urls)}")

    # Calculate overlaps
    shared_urls = original_urls & verified_urls
    only_original = original_urls - verified_urls
    only_verified = verified_urls - original_urls

    print("\n" + "=" * 80)
    print("URL OVERLAP ANALYSIS")
    print("=" * 80)
    print(f"\n  URLs in BOTH datasets:           {len(shared_urls):,}")
    print(f"  URLs ONLY in original SimpleQA:  {len(only_original):,}")
    print(f"  URLs ONLY in SimpleQA-Verified:  {len(only_verified):,}")

    # Check coverage of verified URLs by cache
    verified_cached = verified_urls & cached_urls
    verified_not_cached = verified_urls - cached_urls

    print("\n" + "=" * 80)
    print("CACHE COVERAGE FOR SimpleQA-Verified")
    print("=" * 80)
    print(f"\n  Verified URLs already cached:    {len(verified_cached):,} ({100*len(verified_cached)/len(verified_urls):.1f}%)")
    print(f"  Verified URLs NOT cached:        {len(verified_not_cached):,} ({100*len(verified_not_cached)/len(verified_urls):.1f}%)")

    # Analyze questions with matching original_index
    print("\n" + "=" * 80)
    print("QUESTION MAPPING ANALYSIS")
    print("=" * 80)

    verified_original_indices = set(verified_questions.keys())

    # Check how many verified questions map to original questions
    questions_found = 0
    questions_not_found = 0
    url_matches = 0
    url_differences = 0

    for orig_idx, verified_data in verified_questions.items():
        if orig_idx in original_questions:
            questions_found += 1
            orig_data = original_questions[orig_idx]

            # Compare URLs
            orig_url_set = set(orig_data['urls'])
            verified_url_set = set(verified_data['urls'])

            if orig_url_set == verified_url_set:
                url_matches += 1
            else:
                url_differences += 1
        else:
            questions_not_found += 1

    print(f"\n  Questions found in original:     {questions_found}")
    print(f"  Questions NOT found in original: {questions_not_found}")
    print(f"  Questions with SAME URLs:        {url_matches}")
    print(f"  Questions with DIFFERENT URLs:   {url_differences}")

    # Domain analysis for new URLs
    print("\n" + "=" * 80)
    print("DOMAIN ANALYSIS FOR NEW URLs (SimpleQA-Verified only)")
    print("=" * 80)

    domain_counts = defaultdict(int)
    for url in only_verified:
        domain = get_domain(url)
        domain_counts[domain] += 1

    print(f"\n  Top domains for NEW URLs (not in original SimpleQA):")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    {domain}: {count}")

    # Domain analysis for uncached URLs
    print("\n" + "=" * 80)
    print("DOMAIN ANALYSIS FOR UNCACHED URLs")
    print("=" * 80)

    uncached_domain_counts = defaultdict(int)
    for url in verified_not_cached:
        domain = get_domain(url)
        uncached_domain_counts[domain] += 1

    print(f"\n  Top domains for UNCACHED URLs (need to download):")
    for domain, count in sorted(uncached_domain_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    {domain}: {count}")

    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    cache_reuse_pct = 100 * len(verified_cached) / len(verified_urls) if verified_urls else 0

    if cache_reuse_pct >= 90:
        print(f"\n  GOOD NEWS: {cache_reuse_pct:.1f}% of SimpleQA-Verified URLs are already cached!")
        print("  Recommendation: Minor cache update needed.")
    elif cache_reuse_pct >= 70:
        print(f"\n  MODERATE: {cache_reuse_pct:.1f}% of SimpleQA-Verified URLs are already cached.")
        print("  Recommendation: Some new URLs need to be downloaded.")
    else:
        print(f"\n  SIGNIFICANT WORK: Only {cache_reuse_pct:.1f}% of SimpleQA-Verified URLs are cached.")
        print("  Recommendation: Substantial cache rebuild may be needed.")

    print(f"\n  NEW URLs to download: {len(verified_not_cached):,}")
    print(f"  Unique domains to crawl: {len(uncached_domain_counts)}")

    # Export lists for downstream processing
    output_dir = SCRIPT_DIR / "simpleqa-verified"

    # Export verified URLs that need caching
    urls_to_cache_file = output_dir / "urls_to_cache.txt"
    with open(urls_to_cache_file, 'w') as f:
        for url in sorted(verified_not_cached):
            f.write(url + '\n')
    print(f"\n  Exported uncached URLs to: {urls_to_cache_file}")

    # Export all verified URLs
    all_verified_urls_file = output_dir / "all_verified_urls.txt"
    with open(all_verified_urls_file, 'w') as f:
        for url in sorted(verified_urls):
            f.write(url + '\n')
    print(f"  Exported all verified URLs to: {all_verified_urls_file}")

    # Export detailed mapping
    mapping_file = output_dir / "url_analysis_report.json"
    report = {
        'summary': {
            'original_questions': len(original_df),
            'verified_questions': len(verified_df),
            'original_unique_urls': len(original_urls),
            'verified_unique_urls': len(verified_urls),
            'shared_urls': len(shared_urls),
            'only_original_urls': len(only_original),
            'only_verified_urls': len(only_verified),
            'cached_urls': len(cached_urls),
            'verified_cached': len(verified_cached),
            'verified_not_cached': len(verified_not_cached),
            'cache_coverage_pct': cache_reuse_pct,
            'questions_mapped': questions_found,
            'questions_not_mapped': questions_not_found,
            'url_matches': url_matches,
            'url_differences': url_differences,
        },
        'domain_analysis': {
            'new_urls_by_domain': dict(sorted(domain_counts.items(), key=lambda x: -x[1])),
            'uncached_urls_by_domain': dict(sorted(uncached_domain_counts.items(), key=lambda x: -x[1])),
        },
        'urls': {
            'only_verified': list(sorted(only_verified))[:100],  # Sample
            'not_cached': list(sorted(verified_not_cached))[:100],  # Sample
        }
    }

    with open(mapping_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Exported detailed report to: {mapping_file}")

    return report


if __name__ == "__main__":
    report = analyze_url_overlap()
