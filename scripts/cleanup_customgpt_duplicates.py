#!/usr/bin/env python3
"""
Clean up duplicate files in CustomGPT project.

This script:
1. Fetches all pages from the CustomGPT project
2. Groups by filename
3. Keeps only ONE successfully indexed copy per filename
4. Deletes all duplicates and failed copies

Usage:
    docker compose run --rm simple-evals python scripts/cleanup_customgpt_duplicates.py
    docker compose run --rm simple-evals python scripts/cleanup_customgpt_duplicates.py --dry-run
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import requests
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Clean up duplicate CustomGPT files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    args = parser.parse_args()

    api_key = os.environ["CUSTOMGPT_API_KEY"]
    project_id = os.environ["CUSTOMGPT_PROJECT"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print("=" * 70)
    print(f"CUSTOMGPT DUPLICATE CLEANUP - Project {project_id}")
    print("=" * 70)
    print()

    # Get all pages
    print("Fetching all pages...")
    url = f"https://app.customgpt.ai/api/v1/projects/{project_id}/sources"
    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        print(f"ERROR: Failed to fetch sources: {resp.status_code}")
        print(resp.text)
        return 1

    data = resp.json()
    uploads = data.get("data", {}).get("uploads", {})
    pages = uploads.get("pages", [])

    print(f"Found {len(pages)} total pages")
    print()

    # Group by filename
    files_by_name = defaultdict(list)
    for page in pages:
        filename = page.get("filename")
        if filename:
            files_by_name[filename].append(page)

    print(f"Unique filenames: {len(files_by_name)}")
    print()

    # Determine what to keep and what to delete
    pages_to_keep = []
    pages_to_delete = []

    for filename, file_pages in sorted(files_by_name.items()):
        # Sort by index_status (ok first) then by created_at (oldest first)
        sorted_pages = sorted(
            file_pages,
            key=lambda p: (0 if p.get("index_status") == "ok" else 1, p.get("created_at", ""))
        )

        # Keep the first one with index_status=ok, or the first one if none are ok
        kept = False
        for page in sorted_pages:
            if not kept and page.get("index_status") == "ok":
                pages_to_keep.append(page)
                kept = True
            else:
                pages_to_delete.append(page)

        # If no ok pages, keep the first one anyway and mark others for deletion
        if not kept and sorted_pages:
            pages_to_keep.append(sorted_pages[0])
            pages_to_delete.extend(sorted_pages[1:])

    print(f"Pages to KEEP: {len(pages_to_keep)}")
    print(f"Pages to DELETE: {len(pages_to_delete)}")
    print()

    # Show status breakdown of pages to keep
    keep_status = defaultdict(int)
    for p in pages_to_keep:
        keep_status[p.get("index_status", "unknown")] += 1

    print("Status of pages to KEEP:")
    for status, count in sorted(keep_status.items()):
        print(f"  {status}: {count}")
    print()

    # Show status breakdown of pages to delete
    delete_status = defaultdict(int)
    for p in pages_to_delete:
        delete_status[p.get("index_status", "unknown")] += 1

    print("Status of pages to DELETE:")
    for status, count in sorted(delete_status.items()):
        print(f"  {status}: {count}")
    print()

    if args.dry_run:
        print("=" * 70)
        print("DRY RUN - No changes made")
        print("=" * 70)
        print()
        print("Sample pages that would be DELETED (first 20):")
        for p in pages_to_delete[:20]:
            print(f"  - {p.get('filename')} (ID: {p.get('id')}, status: {p.get('index_status')})")
        return 0

    # Delete the duplicate pages
    print("=" * 70)
    print("DELETING DUPLICATE PAGES...")
    print("=" * 70)

    deleted = 0
    failed = 0

    for i, page in enumerate(pages_to_delete):
        page_id = page.get("id")
        filename = page.get("filename")

        # Delete the page
        delete_url = f"https://app.customgpt.ai/api/v1/projects/{project_id}/pages/{page_id}"

        try:
            resp = requests.delete(delete_url, headers=headers)

            if resp.status_code in [200, 204]:
                deleted += 1
                if deleted % 100 == 0:
                    print(f"  Deleted {deleted}/{len(pages_to_delete)} pages...")
            else:
                failed += 1
                if failed <= 5:  # Only show first 5 failures
                    print(f"  FAILED to delete {filename} (ID: {page_id}): {resp.status_code}")
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  ERROR deleting {filename} (ID: {page_id}): {e}")

        # Rate limiting - small delay between deletes
        if (i + 1) % 50 == 0:
            time.sleep(0.5)

    print()
    print(f"Deleted: {deleted}")
    print(f"Failed: {failed}")
    print()

    # Verify final state
    print("=" * 70)
    print("VERIFYING FINAL STATE...")
    print("=" * 70)

    resp = requests.get(url, headers=headers)
    data = resp.json()
    uploads = data.get("data", {}).get("uploads", {})
    final_pages = uploads.get("pages", [])

    print(f"Total pages remaining: {len(final_pages)}")

    # Count by status
    final_status = defaultdict(int)
    for p in final_pages:
        final_status[p.get("index_status", "unknown")] += 1

    print("Status breakdown:")
    for status, count in sorted(final_status.items()):
        symbol = "✅" if status == "ok" else "❌" if status == "failed" else "⏳"
        print(f"  {symbol} {status}: {count}")

    # Count unique filenames
    unique_files = set(p.get("filename") for p in final_pages)
    print(f"Unique filenames: {len(unique_files)}")

    print()
    print("=" * 70)
    if len(final_pages) == 1000 and final_status.get("ok", 0) == 1000:
        print("✅ CLEANUP COMPLETE: Exactly 1000 files, all indexed successfully!")
    elif len(final_pages) == len(unique_files):
        print(f"✅ CLEANUP COMPLETE: {len(final_pages)} unique files (no duplicates)")
    else:
        print(f"⚠️  Cleanup done but state is: {len(final_pages)} pages, {len(unique_files)} unique files")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
