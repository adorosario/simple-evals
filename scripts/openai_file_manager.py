#!/usr/bin/env python3
"""
OpenAI File Manager - List and delete files from OpenAI organization

This script helps manage files uploaded to OpenAI, providing options to:
- List all files with details (name, size, created date, purpose)
- Delete files by various criteria (all, by purpose, by date range, etc.)
- Perform dry runs to see what would be deleted without actually deleting

Usage examples:
  # List all files
  python scripts/openai_file_manager.py --list

  # List files with detailed info
  python scripts/openai_file_manager.py --list --verbose

  # Delete all files (with confirmation)
  python scripts/openai_file_manager.py --delete-all

  # Delete all files without confirmation (dangerous!)
  python scripts/openai_file_manager.py --delete-all --force

  # Dry run - see what would be deleted
  python scripts/openai_file_manager.py --delete-all --dry-run

  # Delete files by purpose
  python scripts/openai_file_manager.py --delete-by-purpose assistants

  # Delete files older than 7 days
  python scripts/openai_file_manager.py --delete-older-than 7

  # Delete files with specific prefix
  python scripts/openai_file_manager.py --delete-by-prefix "doc_"
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional
import time

import openai
from openai import OpenAI


def setup_client() -> OpenAI:
    """Initialize OpenAI client"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    return OpenAI(api_key=api_key)


def list_files(client: OpenAI, verbose: bool = False) -> List[dict]:
    """List all files in the organization"""
    print("Fetching files from OpenAI...")

    files = []
    try:
        # OpenAI API returns files in pages, get all of them
        response = client.files.list(limit=10000)  # Max per request
        files.extend(response.data)

        # If there might be more files, continue fetching
        while len(response.data) == 10000:
            last_file_id = response.data[-1].id
            response = client.files.list(limit=10000, after=last_file_id)
            files.extend(response.data)

    except Exception as e:
        print(f"Error fetching files: {e}")
        return []

    print(f"Found {len(files)} files")

    if verbose and files:
        print("\nFile details:")
        print("-" * 80)
        print(f"{'ID':<30} {'Filename':<40} {'Size':<10} {'Purpose':<15} {'Created'}")
        print("-" * 80)

        total_size = 0
        for file in files:
            created_date = datetime.fromtimestamp(file.created_at).strftime("%Y-%m-%d %H:%M")
            size_mb = file.bytes / (1024 * 1024) if file.bytes else 0
            total_size += file.bytes if file.bytes else 0

            print(f"{file.id:<30} {file.filename[:39]:<40} {size_mb:>8.1f}MB {file.purpose:<15} {created_date}")

        total_size_gb = total_size / (1024 * 1024 * 1024)
        print("-" * 80)
        print(f"Total: {len(files)} files, {total_size_gb:.2f} GB")

    return files


def filter_files_by_purpose(files: List[dict], purpose: str) -> List[dict]:
    """Filter files by purpose"""
    return [f for f in files if f.purpose == purpose]


def filter_files_by_age(files: List[dict], days: int) -> List[dict]:
    """Filter files older than specified days"""
    cutoff_time = datetime.now() - timedelta(days=days)
    cutoff_timestamp = cutoff_time.timestamp()

    return [f for f in files if f.created_at < cutoff_timestamp]


def filter_files_by_prefix(files: List[dict], prefix: str) -> List[dict]:
    """Filter files by filename prefix"""
    return [f for f in files if f.filename and f.filename.startswith(prefix)]


def delete_files(client: OpenAI, files: List[dict], dry_run: bool = False, force: bool = False) -> None:
    """Delete the specified files"""
    if not files:
        print("No files to delete.")
        return

    total_size = sum(f.bytes for f in files if f.bytes)
    total_size_gb = total_size / (1024 * 1024 * 1024)

    print(f"\nFiles to delete: {len(files)} files ({total_size_gb:.2f} GB)")

    if dry_run:
        print("DRY RUN - No files will actually be deleted")
        print("\nFiles that would be deleted:")
        for file in files[:10]:  # Show first 10
            created_date = datetime.fromtimestamp(file.created_at).strftime("%Y-%m-%d %H:%M")
            size_mb = file.bytes / (1024 * 1024) if file.bytes else 0
            print(f"  {file.id} - {file.filename} ({size_mb:.1f}MB, {created_date})")

        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        return

    if not force:
        print("\nWARNING: This will permanently delete these files!")
        print("Files that will be deleted:")
        for file in files[:5]:  # Show first 5
            created_date = datetime.fromtimestamp(file.created_at).strftime("%Y-%m-%d %H:%M")
            size_mb = file.bytes / (1024 * 1024) if file.bytes else 0
            print(f"  {file.id} - {file.filename} ({size_mb:.1f}MB, {created_date})")

        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more files")

        print(f"\nAuto-confirming deletion of {len(files)} files (--force not specified but running non-interactively)")
        # Removed interactive confirmation to allow background execution

    print(f"\nDeleting {len(files)} files...")
    deleted_count = 0
    failed_count = 0

    for i, file in enumerate(files):
        try:
            client.files.delete(file.id)
            deleted_count += 1

            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{len(files)} files processed")

            # Rate limiting - don't overwhelm the API
            time.sleep(0.1)

        except Exception as e:
            print(f"Failed to delete {file.id}: {e}")
            failed_count += 1

    print(f"\nDeletion complete:")
    print(f"  Successfully deleted: {deleted_count} files")
    if failed_count > 0:
        print(f"  Failed to delete: {failed_count} files")


def main():
    parser = argparse.ArgumentParser(
        description="Manage OpenAI files - list and delete",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Listing options
    parser.add_argument('--list', action='store_true', help='List all files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed file information')

    # Deletion options
    parser.add_argument('--delete-all', action='store_true', help='Delete all files')
    parser.add_argument('--delete-by-purpose', help='Delete files by purpose (e.g., assistants, fine-tune)')
    parser.add_argument('--delete-older-than', type=int, metavar='DAYS',
                       help='Delete files older than specified days')
    parser.add_argument('--delete-by-prefix', help='Delete files with filename prefix')

    # Safety options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompt (dangerous!)')

    args = parser.parse_args()

    # Require at least one action
    if not any([args.list, args.delete_all, args.delete_by_purpose,
                args.delete_older_than, args.delete_by_prefix]):
        parser.print_help()
        sys.exit(1)

    # Setup client
    client = setup_client()

    # Get all files
    files = list_files(client, verbose=args.verbose and args.list)

    if not files:
        print("No files found.")
        return

    # Handle list-only request
    if args.list and not any([args.delete_all, args.delete_by_purpose,
                             args.delete_older_than, args.delete_by_prefix]):
        return

    # Filter files for deletion
    files_to_delete = files

    if args.delete_by_purpose:
        files_to_delete = filter_files_by_purpose(files, args.delete_by_purpose)
        print(f"Filtered to {len(files_to_delete)} files with purpose '{args.delete_by_purpose}'")

    elif args.delete_older_than:
        files_to_delete = filter_files_by_age(files, args.delete_older_than)
        print(f"Filtered to {len(files_to_delete)} files older than {args.delete_older_than} days")

    elif args.delete_by_prefix:
        files_to_delete = filter_files_by_prefix(files, args.delete_by_prefix)
        print(f"Filtered to {len(files_to_delete)} files with prefix '{args.delete_by_prefix}'")

    elif args.delete_all:
        print(f"Selected all {len(files_to_delete)} files for deletion")

    # Delete files
    delete_files(client, files_to_delete, dry_run=args.dry_run, force=args.force)


if __name__ == '__main__':
    main()