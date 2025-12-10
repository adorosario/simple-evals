#!/usr/bin/env python3
"""
Refresh Failed KB Files Using JINA Reader

This script:
1. Identifies KB files with empty or insufficient content
2. Uses JINA Reader API to fetch fresh content from source URLs
3. Updates the KB files and manifest

Usage:
    docker compose run --rm simple-evals python scripts/refresh_kb_with_jina.py
    docker compose run --rm simple-evals python scripts/refresh_kb_with_jina.py --dry-run
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class JinaFetcher:
    """Fetch clean content from URLs using JINA Reader API."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        self.api_url = "https://r.jina.ai/"
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("JINA_API_KEY not found in environment")

    def fetch(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch content from URL via JINA.

        Returns:
            (content, title) tuple, or (None, None) on failure
        """
        if not url:
            return None, None

        jina_url = f"{self.api_url}{url}"
        headers = {
            "Accept": "text/plain",
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = httpx.get(jina_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()

            content = response.text.strip()
            if not content:
                return None, None

            # Extract title from first line if present
            lines = content.split("\n")
            title = None
            if lines and lines[0].startswith("Title:"):
                title = lines[0].replace("Title:", "").strip()

            return content, title

        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP {e.response.status_code} fetching {url}")
            return None, None
        except httpx.RequestError as e:
            logger.warning(f"Request error fetching {url}: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None, None


def load_manifest(kb_dir: Path) -> Dict:
    """Load the KB build manifest."""
    manifest_path = kb_dir / "build_manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def save_manifest(kb_dir: Path, manifest: Dict):
    """Save the KB build manifest."""
    manifest_path = kb_dir / "build_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def identify_failed_documents(manifest: Dict) -> List[Dict]:
    """Find documents with no valid URLs or empty content."""
    failed = []
    for doc in manifest["documents"]:
        if doc.get("urls_valid", 0) == 0 or doc.get("word_count", 0) == 0:
            failed.append(doc)
    return failed


def get_urls_for_question(csv_path: Path, original_index: int) -> List[str]:
    """Get source URLs for a question from the dataset."""
    df = pd.read_csv(csv_path)
    row = df[df['original_index'] == original_index]

    if row.empty:
        return []

    urls_str = row.iloc[0].get('urls', '')
    if pd.isna(urls_str) or not urls_str:
        return []

    # URLs are comma-separated
    urls = [u.strip() for u in urls_str.split(',') if u.strip()]
    return urls


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def refresh_failed_documents(
    kb_dir: Path,
    csv_path: Path,
    fetcher: JinaFetcher,
    dry_run: bool = False,
    max_docs: Optional[int] = None,
) -> Dict:
    """
    Refresh failed KB documents using JINA.

    Returns:
        Summary of refresh results
    """
    manifest = load_manifest(kb_dir)
    failed_docs = identify_failed_documents(manifest)

    logger.info(f"Found {len(failed_docs)} failed documents to refresh")

    if max_docs:
        failed_docs = failed_docs[:max_docs]
        logger.info(f"Limited to {max_docs} documents")

    results = {
        "total_attempted": len(failed_docs),
        "successful": 0,
        "failed": 0,
        "details": []
    }

    for i, doc in enumerate(failed_docs):
        original_index = doc["original_index"]
        filename = doc["filename"]
        logger.info(f"[{i+1}/{len(failed_docs)}] Refreshing Q{original_index} ({filename})")

        # Get URLs from dataset
        urls = get_urls_for_question(csv_path, original_index)
        if not urls:
            logger.warning(f"  No URLs found for Q{original_index}")
            results["failed"] += 1
            results["details"].append({
                "original_index": original_index,
                "status": "no_urls",
                "message": "No source URLs in dataset"
            })
            continue

        # Try each URL until one works
        content = None
        title = None
        successful_url = None

        for url in urls:
            logger.info(f"  Trying: {url[:80]}...")
            content, title = fetcher.fetch(url)

            if content and count_words(content) > 100:
                successful_url = url
                logger.info(f"  Success! Got {count_words(content)} words")
                break

            # Rate limit between requests
            time.sleep(0.5)

        if not content:
            logger.warning(f"  All {len(urls)} URLs failed for Q{original_index}")
            results["failed"] += 1
            results["details"].append({
                "original_index": original_index,
                "status": "all_urls_failed",
                "urls_attempted": urls
            })
            continue

        # Update KB file
        file_path = kb_dir / filename
        word_count = count_words(content)

        if not dry_run:
            with open(file_path, "w") as f:
                f.write(content)

            # Update manifest entry
            for manifest_doc in manifest["documents"]:
                if manifest_doc["original_index"] == original_index:
                    manifest_doc["urls_valid"] = 1
                    manifest_doc["sources_extracted"] = 1
                    manifest_doc["word_count"] = word_count
                    manifest_doc["sources"] = [{
                        "url": successful_url,
                        "word_count": word_count,
                        "title": title,
                        "fetched_with": "jina"
                    }]
                    break

        results["successful"] += 1
        results["details"].append({
            "original_index": original_index,
            "status": "success",
            "url": successful_url,
            "word_count": word_count
        })

        logger.info(f"  Updated {filename} with {word_count} words")

    # Save updated manifest
    if not dry_run and results["successful"] > 0:
        save_manifest(kb_dir, manifest)
        logger.info(f"Saved updated manifest")

    return results


def main():
    parser = argparse.ArgumentParser(description="Refresh failed KB files using JINA")
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=Path("knowledge_base_verified"),
        help="Knowledge base directory"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("simpleqa-verified/simpleqa_verified.csv"),
        help="Dataset CSV path"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write files, just show what would be done"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to refresh"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KB Refresh with JINA Reader")
    logger.info("=" * 60)
    logger.info(f"KB directory: {args.kb_dir}")
    logger.info(f"Dataset: {args.csv}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    fetcher = JinaFetcher()

    results = refresh_failed_documents(
        kb_dir=args.kb_dir,
        csv_path=args.csv,
        fetcher=fetcher,
        dry_run=args.dry_run,
        max_docs=args.max_docs,
    )

    logger.info("=" * 60)
    logger.info("REFRESH RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total attempted: {results['total_attempted']}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")

    if results["successful"] > 0:
        logger.info("\nSuccessfully refreshed:")
        for detail in results["details"]:
            if detail["status"] == "success":
                logger.info(f"  Q{detail['original_index']}: {detail['word_count']} words from {detail['url'][:60]}...")

    if results["failed"] > 0:
        logger.info("\nFailed to refresh:")
        for detail in results["details"]:
            if detail["status"] != "success":
                logger.info(f"  Q{detail['original_index']}: {detail['status']}")

    # Save results
    results_path = args.kb_dir / "audit" / "jina_refresh_results.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
