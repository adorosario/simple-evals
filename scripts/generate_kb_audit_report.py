#!/usr/bin/env python3
"""
Generate complete KB audit report for CustomGPT project 88141.
Lists ALL 1000 documents with EVERY available attribute.
"""

import os
import json
import requests
from datetime import datetime
from collections import Counter

API_KEY = os.environ.get("CUSTOMGPT_API_KEY")
PROJECT_ID = os.environ.get("CUSTOMGPT_PROJECT", "88141")
OUTPUT_PATH = "/app/results/run_20260125_201629_042/KB_FULL_AUDIT_REPORT.md"


def fetch_all_pages():
    """Fetch all pages from CustomGPT API.

    Note: For uploads, all pages come in a single API call (no pagination needed).
    """
    url = f"https://app.customgpt.ai/api/v1/projects/{PROJECT_ID}/sources"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()

    # Pages are under data.uploads.pages
    uploads = data.get("data", {}).get("uploads", {})
    pages = uploads.get("pages", [])

    print(f"Fetched {len(pages)} documents from CustomGPT project {PROJECT_ID}")
    return pages


def generate_report(pages):
    """Generate comprehensive markdown report."""

    # Sort by filename for consistent ordering
    pages_sorted = sorted(pages, key=lambda p: p.get("filename", ""))

    # Calculate statistics
    total_size = sum(p.get("filesize", 0) or 0 for p in pages_sorted)
    index_status_counts = Counter(p.get("index_status") for p in pages_sorted)
    crawl_status_counts = Counter(p.get("crawl_status") for p in pages_sorted)

    # Find issues
    issues = []
    for p in pages_sorted:
        if p.get("index_status") != "ok":
            issues.append(f"- {p.get('filename')}: index_status = {p.get('index_status')}")
        if p.get("crawl_status") != "ok":
            issues.append(f"- {p.get('filename')}: crawl_status = {p.get('crawl_status')}")
        if not p.get("filesize") or p.get("filesize", 0) == 0:
            issues.append(f"- {p.get('filename')}: filesize = 0 bytes")

    # Build report
    report = []
    report.append("# CustomGPT Knowledge Base Full Audit Report")
    report.append("")
    report.append(f"**Project ID:** {PROJECT_ID}")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Total Documents:** {len(pages_sorted)}")
    report.append(f"**Total Size:** {total_size:,} bytes ({total_size / (1024*1024):.2f} MB)")
    report.append("")
    report.append("---")
    report.append("")

    # Summary section
    report.append("## Summary Statistics")
    report.append("")
    report.append("### Index Status")
    report.append("")
    for status, count in sorted(index_status_counts.items()):
        report.append(f"- **{status}**: {count}")
    report.append("")

    report.append("### Crawl Status")
    report.append("")
    for status, count in sorted(crawl_status_counts.items()):
        report.append(f"- **{status}**: {count}")
    report.append("")

    # Issues section
    report.append("## Issues Found")
    report.append("")
    if issues:
        for issue in issues:
            report.append(issue)
    else:
        report.append("**No issues found.** All documents have:")
        report.append("- `index_status: ok`")
        report.append("- `crawl_status: ok`")
        report.append("- `filesize > 0`")
    report.append("")
    report.append("---")
    report.append("")

    # Full inventory section
    report.append("## Full Document Inventory")
    report.append("")
    report.append("Below is the complete list of all documents with every available attribute.")
    report.append("")

    # Table header
    report.append("| # | ID | Filename | Index Status | Crawl Status | Filesize | Created | Updated |")
    report.append("|---|-----|----------|--------------|--------------|----------|---------|---------|")

    for i, p in enumerate(pages_sorted, 1):
        page_id = p.get("id", "N/A")
        filename = p.get("filename", "N/A")
        index_status = p.get("index_status", "N/A")
        crawl_status = p.get("crawl_status", "N/A")
        filesize = p.get("filesize", 0) or 0
        created_at = p.get("created_at", "N/A")
        updated_at = p.get("updated_at", "N/A")

        # Format dates more compactly
        if created_at and created_at != "N/A":
            created_at = created_at.split("T")[0] if "T" in created_at else created_at[:10]
        if updated_at and updated_at != "N/A":
            updated_at = updated_at.split("T")[0] if "T" in updated_at else updated_at[:10]

        # Format filesize
        if filesize > 1024*1024:
            size_str = f"{filesize/(1024*1024):.1f} MB"
        elif filesize > 1024:
            size_str = f"{filesize/1024:.1f} KB"
        else:
            size_str = f"{filesize} B"

        report.append(f"| {i} | {page_id} | {filename} | {index_status} | {crawl_status} | {size_str} | {created_at} | {updated_at} |")

    report.append("")
    report.append("---")
    report.append("")

    # Detailed records section (JSON format for each)
    report.append("## Detailed Records (Full JSON)")
    report.append("")
    report.append("For complete attribute reference, here are the first 5 documents in full JSON format:")
    report.append("")

    for i, p in enumerate(pages_sorted[:5], 1):
        report.append(f"### Document {i}: {p.get('filename', 'N/A')}")
        report.append("")
        report.append("```json")
        report.append(json.dumps(p, indent=2, default=str))
        report.append("```")
        report.append("")

    # List all unique fields found
    report.append("## All Fields Available in API Response")
    report.append("")
    all_fields = set()
    for p in pages_sorted:
        all_fields.update(p.keys())
    report.append("The following fields are available for each document:")
    report.append("")
    for field in sorted(all_fields):
        report.append(f"- `{field}`")
    report.append("")

    # Footer
    report.append("---")
    report.append("")
    report.append("*Report generated by `scripts/generate_kb_audit_report.py`*")
    report.append("")

    return "\n".join(report)


def main():
    print(f"Fetching all documents from CustomGPT project {PROJECT_ID}...")
    pages = fetch_all_pages()
    print(f"Total documents fetched: {len(pages)}")

    print("Generating report...")
    report = generate_report(pages)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Writing report to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)

    print(f"Done! Report written to {OUTPUT_PATH}")
    print(f"Total documents: {len(pages)}")


if __name__ == "__main__":
    main()
