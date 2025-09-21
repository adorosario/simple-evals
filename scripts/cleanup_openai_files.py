#!/usr/bin/env python3
"""
OpenAI Files Mass Cleanup Script

Deletes ALL files from OpenAI organization to clean up after multiple upload attempts.
This addresses the issue of having tens of thousands of files flooding the Files list.

Usage:
    # Delete all files (with confirmation)
    python scripts/cleanup_openai_files.py --delete-all

    # Delete all files without confirmation (dangerous!)
    python scripts/cleanup_openai_files.py --delete-all --force

    # Dry run - see what would be deleted
    python scripts/cleanup_openai_files.py --delete-all --dry-run

    # Delete files older than 1 day (to keep recent uploads)
    python scripts/cleanup_openai_files.py --delete-older-than 1

    # Delete only files with specific prefix
    python scripts/cleanup_openai_files.py --delete-by-prefix "doc_"
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import List
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import OpenAI


def setup_client() -> OpenAI:
    """Initialize OpenAI client"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    return OpenAI(api_key=api_key)


def get_all_files(client: OpenAI) -> List:
    """Get all files from OpenAI organization"""
    print("🔍 Fetching all files from OpenAI...")

    all_files = []
    try:
        # Use smaller batch size like the working pagination test
        after = None
        batch_count = 0
        batch_size = 100  # Smaller batch size that works reliably

        while True:
            batch_count += 1
            print(f"   Fetching batch {batch_count}...")

            if after:
                response = client.files.list(limit=batch_size, after=after)
            else:
                response = client.files.list(limit=batch_size)

            batch_files = list(response.data)
            all_files.extend(batch_files)

            print(f"   Batch {batch_count}: {len(batch_files)} files (total: {len(all_files)})")

            # If we got fewer than the limit, we're done
            if len(batch_files) < batch_size:
                break

            # Safety check to avoid infinite loops
            if batch_count > 200:  # More than 20k files seems excessive
                print(f"   ⚠️ Safety break at {batch_count} batches")
                break

            # Set up for next batch
            after = batch_files[-1].id

        print(f"📊 Total files found: {len(all_files)}")
        return all_files

    except Exception as e:
        print(f"❌ Error fetching files: {e}")
        return []


def filter_files_by_age(files: List, days: int) -> List:
    """Filter files older than specified days"""
    cutoff_time = datetime.now() - timedelta(days=days)
    cutoff_timestamp = cutoff_time.timestamp()

    filtered = [f for f in files if f.created_at < cutoff_timestamp]
    print(f"🗓️  Filtered to {len(filtered)} files older than {days} days")
    return filtered


def filter_files_by_prefix(files: List, prefix: str) -> List:
    """Filter files by filename prefix"""
    filtered = [f for f in files if f.filename and f.filename.startswith(prefix)]
    print(f"🔤 Filtered to {len(filtered)} files with prefix '{prefix}'")
    return filtered


def delete_single_file(client: OpenAI, file) -> tuple[bool, str]:
    """Delete a single file and verify actual deletion"""
    try:
        # Delete the file
        client.files.delete(file.id)

        # Verify deletion by trying to retrieve it
        time.sleep(0.1)  # Small delay for API consistency
        try:
            client.files.retrieve(file.id)
            # If we can still retrieve it, deletion failed
            return False, f"{file.id}: Still exists after delete call"
        except Exception:
            # If retrieve fails, file was actually deleted
            return True, file.id

    except Exception as e:
        return False, f"{file.id}: Delete failed - {str(e)[:100]}"


def delete_files_batch(client: OpenAI, files: List, dry_run: bool = False, force: bool = False) -> None:
    """Delete files in parallel with progress tracking"""

    if not files:
        print("📭 No files to delete.")
        return

    total_size = sum(f.bytes for f in files if f.bytes)
    total_size_gb = total_size / (1024 * 1024 * 1024)

    print(f"\n📋 DELETION SUMMARY:")
    print(f"   Files to delete: {len(files):,}")
    print(f"   Total size: {total_size_gb:.2f} GB")

    if dry_run:
        print("\n🧪 DRY RUN MODE - No files will actually be deleted")
        print("\n📄 Sample files that would be deleted:")
        for file in files[:10]:
            created_date = datetime.fromtimestamp(file.created_at).strftime("%Y-%m-%d %H:%M")
            size_mb = file.bytes / (1024 * 1024) if file.bytes else 0
            print(f"   {file.id} - {file.filename} ({size_mb:.1f}MB, {created_date})")

        if len(files) > 10:
            print(f"   ... and {len(files) - 10:,} more files")
        return

    if not force:
        print(f"\n⚠️  WARNING: This will permanently delete {len(files):,} files!")
        print("📄 Sample files that will be deleted:")
        for file in files[:5]:
            created_date = datetime.fromtimestamp(file.created_at).strftime("%Y-%m-%d %H:%M")
            size_mb = file.bytes / (1024 * 1024) if file.bytes else 0
            print(f"   {file.id} - {file.filename} ({size_mb:.1f}MB, {created_date})")

        if len(files) > 5:
            print(f"   ... and {len(files) - 5:,} more files")

        print(f"\n🔴 Type 'DELETE ALL {len(files)}' to confirm mass deletion:")
        confirm = input(">>> ").strip()
        expected = f"DELETE ALL {len(files)}"

        if confirm != expected:
            print("❌ Deletion cancelled.")
            return

    print(f"\n🗑️  Deleting {len(files):,} files in parallel...")
    print("📊 Progress tracking:")

    deleted_count = 0
    failed_count = 0
    start_time = time.time()

    # Use smaller batches and fewer workers to avoid rate limiting
    max_workers = 5  # Conservative to avoid overwhelming API
    batch_size = 50  # Smaller batches for better progress tracking

    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(files) + batch_size - 1) // batch_size

        print(f"   Batch {batch_num}/{total_batches}: Processing {len(batch)} files...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all deletion tasks
            future_to_file = {executor.submit(delete_single_file, client, file): file for file in batch}

            # Process completed tasks with better timeout handling
            completed = 0
            try:
                for future in as_completed(future_to_file, timeout=120):  # 2 min timeout
                    try:
                        success, result = future.result(timeout=30)  # 30s per deletion
                        if success:
                            deleted_count += 1
                            if completed % 10 == 0:  # Show progress every 10 deletions
                                print(f"     ✅ Deleted {deleted_count} files...")
                        else:
                            failed_count += 1
                            print(f"     ❌ Failed to delete {result}")
                        completed += 1
                    except Exception as e:
                        failed_count += 1
                        print(f"     ⏰ Timeout/error: {str(e)[:50]}")
            except TimeoutError:
                # Handle batch timeout - cancel remaining futures and continue
                print(f"     ⏰ Batch timeout - {completed}/{len(batch)} completed, continuing...")
                for future in future_to_file:
                    if not future.done():
                        future.cancel()
                        failed_count += 1

        # Progress update every batch
        elapsed = time.time() - start_time
        progress_pct = (deleted_count + failed_count) / len(files) * 100
        remaining = len(files) - (deleted_count + failed_count)
        rate = (deleted_count + failed_count) / elapsed if elapsed > 0 else 0
        eta_minutes = (remaining / rate / 60) if rate > 0 else 0

        print(f"   📈 Progress: {deleted_count + failed_count:,}/{len(files):,} ({progress_pct:.1f}%) - Rate: {rate:.1f}/s - ETA: {eta_minutes:.1f}m")

        # Small delay between batches to be respectful of API
        time.sleep(1)

    elapsed_total = time.time() - start_time

    print(f"\n✅ Deletion process complete!")
    print(f"   🗑️  Successfully deleted: {deleted_count:,} files")
    print(f"   ❌ Failed to delete: {failed_count:,} files")
    print(f"   ⏱️  Total time: {elapsed_total/60:.1f} minutes")
    print(f"   📊 Average rate: {deleted_count/elapsed_total:.1f} files/second")

    # Verify final count by checking OpenAI directly
    print(f"\n🔍 Verifying actual file count with OpenAI...")
    try:
        remaining_files = get_all_files(client)
        print(f"   📊 Files remaining in OpenAI: {len(remaining_files):,}")

        if len(remaining_files) == 0:
            print("   🎉 All files successfully deleted!")
        elif len(remaining_files) < len(files):
            print(f"   ✅ Progress made: {len(files) - len(remaining_files):,} files deleted")
        else:
            print(f"   ⚠️  No progress detected - check API permissions or rate limits")
    except Exception as e:
        print(f"   ❌ Could not verify final count: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Mass cleanup of OpenAI files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Deletion options
    parser.add_argument('--delete-all', action='store_true', help='Delete ALL files')
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
    if not any([args.delete_all, args.delete_older_than, args.delete_by_prefix]):
        parser.print_help()
        sys.exit(1)

    # Setup client
    client = setup_client()

    # Get all files
    files = get_all_files(client)

    if not files:
        print("📭 No files found in organization.")
        return

    # Filter files for deletion
    files_to_delete = files

    if args.delete_older_than:
        files_to_delete = filter_files_by_age(files, args.delete_older_than)

    elif args.delete_by_prefix:
        files_to_delete = filter_files_by_prefix(files, args.delete_by_prefix)

    elif args.delete_all:
        print(f"🎯 Selected ALL {len(files_to_delete):,} files for deletion")

    # Delete files
    delete_files_batch(client, files_to_delete, dry_run=args.dry_run, force=args.force)


if __name__ == '__main__':
    main()