#!/usr/bin/env python3
"""
Verify 1-to-1 alignment between Knowledge Base files and indexed stores.

This script verifies that all files in knowledge_base_verified/ are properly
indexed in both OpenAI Vector Store and CustomGPT Project, with auto-fix
capability for any mismatches.

Usage:
    docker compose run --rm simple-evals python scripts/verify_kb_alignment.py
    docker compose run --rm simple-evals python scripts/verify_kb_alignment.py --dry-run
    docker compose run --rm simple-evals python scripts/verify_kb_alignment.py --no-fix
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_kb_files(kb_dir: Path) -> Set[str]:
    """Get all verified_*.txt filenames from the knowledge base directory."""
    files = sorted([f.name for f in kb_dir.glob("verified_*.txt")])
    return set(files)


def get_openai_files(client: OpenAI, vector_store_id: str) -> Tuple[Set[str], Dict[str, str]]:
    """
    Get all filenames from OpenAI vector store.

    Returns:
        (set of filenames, dict of file_id -> filename for orphan cleanup)
    """
    all_files = {}  # file_id -> filename
    after = None
    page_num = 0

    print("  Fetching OpenAI vector store files...")
    while True:
        page_num += 1
        params = {"limit": 100}
        if after:
            params["after"] = after

        page = client.vector_stores.files.list(vector_store_id, **params)

        for f in page.data:
            # Get actual filename from file object
            try:
                file_obj = client.files.retrieve(f.id)
                filename = file_obj.filename
                all_files[f.id] = filename
            except Exception as e:
                print(f"    Warning: Could not retrieve file {f.id}: {e}")
                all_files[f.id] = f"unknown_{f.id}"

        print(f"    Page {page_num}: {len(page.data)} files (total: {len(all_files)})")

        if not hasattr(page, 'has_more') or not page.has_more:
            break
        if page.data:
            after = page.data[-1].id
        else:
            break

    return set(all_files.values()), all_files


def get_customgpt_files(api_key: str, project_id: int) -> Tuple[Set[str], Dict[int, str]]:
    """
    Get all filenames from CustomGPT project.

    Returns:
        (set of filenames, dict of page_id -> filename for orphan cleanup)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print("  Fetching CustomGPT project files...")
    url = f"https://app.customgpt.ai/api/v1/projects/{project_id}/sources"
    resp = requests.get(url, headers=headers)

    if resp.status_code != 200:
        raise Exception(f"Failed to fetch CustomGPT sources: {resp.status_code} - {resp.text}")

    data = resp.json()
    uploads = data.get("data", {}).get("uploads", {})
    pages = uploads.get("pages", [])

    all_files = {}  # page_id -> filename
    for p in pages:
        page_id = p.get("id")
        filename = p.get("filename")
        if filename:
            all_files[page_id] = filename

    print(f"    Found {len(all_files)} files")
    return set(all_files.values()), all_files


def fix_openai_missing(
    missing_files: Set[str],
    kb_dir: Path,
    client: OpenAI,
    vector_store_id: str
) -> int:
    """Re-upload missing files to OpenAI vector store."""
    fixed = 0
    for filename in sorted(missing_files):
        filepath = kb_dir / filename
        if not filepath.exists():
            print(f"    ERROR: Source file not found: {filepath}")
            continue

        try:
            with open(filepath, 'rb') as f:
                file_obj = client.files.create(file=f, purpose='assistants')
            client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_obj.id
            )
            print(f"    Uploaded {filename} to OpenAI")
            fixed += 1
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"    ERROR uploading {filename}: {e}")

    return fixed


def fix_customgpt_missing(
    missing_files: Set[str],
    kb_dir: Path,
    api_key: str,
    project_id: int
) -> int:
    """Re-upload missing files to CustomGPT project."""
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    fixed = 0
    for filename in sorted(missing_files):
        filepath = kb_dir / filename
        if not filepath.exists():
            print(f"    ERROR: Source file not found: {filepath}")
            continue

        try:
            with open(filepath, 'rb') as f:
                files = {'file': (filename, f, 'text/plain')}
                resp = requests.post(
                    f"https://app.customgpt.ai/api/v1/projects/{project_id}/sources/upload",
                    headers=headers,
                    files=files
                )

            if resp.status_code in [200, 201]:
                print(f"    Uploaded {filename} to CustomGPT")
                fixed += 1
            else:
                print(f"    ERROR uploading {filename}: {resp.status_code} - {resp.text[:200]}")

            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"    ERROR uploading {filename}: {e}")

    return fixed


def fix_openai_orphans(
    orphan_files: Set[str],
    file_id_map: Dict[str, str],
    client: OpenAI,
    vector_store_id: str
) -> int:
    """Remove orphan files from OpenAI vector store."""
    # Reverse the map to get filename -> file_id
    filename_to_id = {v: k for k, v in file_id_map.items()}

    fixed = 0
    for filename in sorted(orphan_files):
        file_id = filename_to_id.get(filename)
        if not file_id:
            print(f"    ERROR: Could not find file_id for {filename}")
            continue

        try:
            client.vector_stores.files.delete(
                vector_store_id=vector_store_id,
                file_id=file_id
            )
            print(f"    Deleted orphan {filename} from OpenAI")
            fixed += 1
            time.sleep(0.1)
        except Exception as e:
            print(f"    ERROR deleting {filename}: {e}")

    return fixed


def fix_customgpt_orphans(
    orphan_files: Set[str],
    page_id_map: Dict[int, str],
    api_key: str,
    project_id: int
) -> int:
    """Remove orphan pages from CustomGPT project."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Reverse the map to get filename -> page_id
    filename_to_id = {v: k for k, v in page_id_map.items()}

    fixed = 0
    for filename in sorted(orphan_files):
        page_id = filename_to_id.get(filename)
        if not page_id:
            print(f"    ERROR: Could not find page_id for {filename}")
            continue

        try:
            resp = requests.delete(
                f"https://app.customgpt.ai/api/v1/projects/{project_id}/pages/{page_id}",
                headers=headers
            )
            if resp.status_code in [200, 204]:
                print(f"    Deleted orphan {filename} from CustomGPT")
                fixed += 1
            else:
                print(f"    ERROR deleting {filename}: {resp.status_code}")
            time.sleep(0.1)
        except Exception as e:
            print(f"    ERROR deleting {filename}: {e}")

    return fixed


def main():
    parser = argparse.ArgumentParser(description="Verify KB alignment across all sources")
    parser.add_argument("--dry-run", action="store_true", help="Show mismatches without fixing")
    parser.add_argument("--no-fix", action="store_true", help="Same as --dry-run")
    parser.add_argument("--kb-dir", default="knowledge_base_verified", help="KB directory")
    args = parser.parse_args()

    dry_run = args.dry_run or args.no_fix
    kb_dir = Path(args.kb_dir)

    # Configuration
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")
    customgpt_api_key = os.environ.get("CUSTOMGPT_API_KEY")
    customgpt_project = int(os.environ.get("CUSTOMGPT_PROJECT", 0))

    if not all([openai_api_key, vector_store_id, customgpt_api_key, customgpt_project]):
        print("ERROR: Missing required environment variables")
        print("  Required: OPENAI_API_KEY, OPENAI_VECTOR_STORE_ID, CUSTOMGPT_API_KEY, CUSTOMGPT_PROJECT")
        return 1

    client = OpenAI(api_key=openai_api_key)

    print("=" * 70)
    print("KNOWLEDGE BASE ALIGNMENT VERIFICATION")
    print("=" * 70)
    print(f"  KB Directory: {kb_dir}")
    print(f"  OpenAI Vector Store: {vector_store_id}")
    print(f"  CustomGPT Project: {customgpt_project}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'AUTO-FIX ENABLED'}")
    print()

    # Step 1: Collect filenames from all sources
    print("STEP 1: Collecting filenames from all sources")
    print("-" * 50)

    print("\n[A] Local Knowledge Base:")
    kb_files = get_kb_files(kb_dir)
    print(f"    Found {len(kb_files)} files")

    print("\n[B] OpenAI Vector Store:")
    openai_files, openai_file_map = get_openai_files(client, vector_store_id)
    print(f"    Found {len(openai_files)} unique filenames")

    print("\n[C] CustomGPT Project:")
    customgpt_files, customgpt_page_map = get_customgpt_files(customgpt_api_key, customgpt_project)
    print(f"    Found {len(customgpt_files)} unique filenames")

    # Step 2: Compare sets
    print("\n" + "=" * 70)
    print("STEP 2: Comparing filenames across sources")
    print("-" * 50)

    # Find mismatches
    in_kb_not_openai = kb_files - openai_files
    in_kb_not_customgpt = kb_files - customgpt_files
    in_openai_not_kb = openai_files - kb_files
    in_customgpt_not_kb = customgpt_files - kb_files

    # Perfect intersection
    in_all_three = kb_files & openai_files & customgpt_files

    print(f"\n  Files in all 3 sources: {len(in_all_three)}")
    print(f"  Files in KB only (missing from OpenAI): {len(in_kb_not_openai)}")
    print(f"  Files in KB only (missing from CustomGPT): {len(in_kb_not_customgpt)}")
    print(f"  Orphans in OpenAI (not in KB): {len(in_openai_not_kb)}")
    print(f"  Orphans in CustomGPT (not in KB): {len(in_customgpt_not_kb)}")

    # Show specific mismatches
    if in_kb_not_openai:
        print(f"\n  Missing from OpenAI ({len(in_kb_not_openai)}):")
        for f in sorted(in_kb_not_openai)[:10]:
            print(f"    - {f}")
        if len(in_kb_not_openai) > 10:
            print(f"    ... and {len(in_kb_not_openai) - 10} more")

    if in_kb_not_customgpt:
        print(f"\n  Missing from CustomGPT ({len(in_kb_not_customgpt)}):")
        for f in sorted(in_kb_not_customgpt)[:10]:
            print(f"    - {f}")
        if len(in_kb_not_customgpt) > 10:
            print(f"    ... and {len(in_kb_not_customgpt) - 10} more")

    if in_openai_not_kb:
        print(f"\n  Orphans in OpenAI ({len(in_openai_not_kb)}):")
        for f in sorted(in_openai_not_kb)[:10]:
            print(f"    - {f}")
        if len(in_openai_not_kb) > 10:
            print(f"    ... and {len(in_openai_not_kb) - 10} more")

    if in_customgpt_not_kb:
        print(f"\n  Orphans in CustomGPT ({len(in_customgpt_not_kb)}):")
        for f in sorted(in_customgpt_not_kb)[:10]:
            print(f"    - {f}")
        if len(in_customgpt_not_kb) > 10:
            print(f"    ... and {len(in_customgpt_not_kb) - 10} more")

    # Check if aligned
    all_aligned = (
        len(in_kb_not_openai) == 0 and
        len(in_kb_not_customgpt) == 0 and
        len(in_openai_not_kb) == 0 and
        len(in_customgpt_not_kb) == 0
    )

    if all_aligned:
        print("\n" + "=" * 70)
        print("RESULT: PERFECT ALIGNMENT")
        print("=" * 70)
        print(f"  All {len(kb_files)} files are properly indexed in both OpenAI and CustomGPT")

        # Save audit report
        audit = {
            "timestamp": datetime.now().isoformat(),
            "status": "ALIGNED",
            "kb_files": len(kb_files),
            "openai_files": len(openai_files),
            "customgpt_files": len(customgpt_files),
            "aligned_count": len(in_all_three),
            "mismatches": None
        }
        audit_path = kb_dir / "alignment_audit.json"
        with open(audit_path, 'w') as f:
            json.dump(audit, f, indent=2)
        print(f"\n  Audit saved to: {audit_path}")

        return 0

    # Step 3: Auto-fix (if not dry run)
    print("\n" + "=" * 70)
    print("STEP 3: AUTO-FIX" if not dry_run else "STEP 3: MISMATCHES FOUND (DRY RUN)")
    print("-" * 50)

    if dry_run:
        print("\n  Dry run mode - no changes made")
        print(f"  Would fix {len(in_kb_not_openai)} missing from OpenAI")
        print(f"  Would fix {len(in_kb_not_customgpt)} missing from CustomGPT")
        print(f"  Would delete {len(in_openai_not_kb)} orphans from OpenAI")
        print(f"  Would delete {len(in_customgpt_not_kb)} orphans from CustomGPT")
        return 1

    fixes_made = 0

    if in_kb_not_openai:
        print(f"\n  Fixing {len(in_kb_not_openai)} missing from OpenAI...")
        fixes_made += fix_openai_missing(in_kb_not_openai, kb_dir, client, vector_store_id)

    if in_kb_not_customgpt:
        print(f"\n  Fixing {len(in_kb_not_customgpt)} missing from CustomGPT...")
        fixes_made += fix_customgpt_missing(in_kb_not_customgpt, kb_dir, customgpt_api_key, customgpt_project)

    if in_openai_not_kb:
        print(f"\n  Deleting {len(in_openai_not_kb)} orphans from OpenAI...")
        fixes_made += fix_openai_orphans(in_openai_not_kb, openai_file_map, client, vector_store_id)

    if in_customgpt_not_kb:
        print(f"\n  Deleting {len(in_customgpt_not_kb)} orphans from CustomGPT...")
        fixes_made += fix_customgpt_orphans(in_customgpt_not_kb, customgpt_page_map, customgpt_api_key, customgpt_project)

    print(f"\n  Total fixes applied: {fixes_made}")

    # Save audit report
    audit = {
        "timestamp": datetime.now().isoformat(),
        "status": "FIXED",
        "kb_files": len(kb_files),
        "openai_files": len(openai_files),
        "customgpt_files": len(customgpt_files),
        "aligned_count": len(in_all_three),
        "mismatches": {
            "missing_from_openai": sorted(in_kb_not_openai),
            "missing_from_customgpt": sorted(in_kb_not_customgpt),
            "orphans_in_openai": sorted(in_openai_not_kb),
            "orphans_in_customgpt": sorted(in_customgpt_not_kb)
        },
        "fixes_applied": fixes_made
    }
    audit_path = kb_dir / "alignment_audit.json"
    with open(audit_path, 'w') as f:
        json.dump(audit, f, indent=2)
    print(f"\n  Audit saved to: {audit_path}")

    print("\n" + "=" * 70)
    print("RE-VERIFY RECOMMENDED")
    print("=" * 70)
    print("  Run this script again to verify fixes were applied correctly")

    return 0


if __name__ == "__main__":
    sys.exit(main())
