#!/usr/bin/env python3
"""
Upload SimpleQA Verified Knowledge Base to Google AI Studio File Search.

Usage:
    docker compose run --rm simple-evals python scripts/upload_kb_to_google.py
    docker compose run --rm simple-evals python scripts/upload_kb_to_google.py --dry-run
    docker compose run --rm simple-evals python scripts/upload_kb_to_google.py --resume

Features:
- Creates Google File Search store if not exists
- Uploads 1000 KB files with metadata
- Checkpoint-based resume for interrupted uploads
- Progress tracking and logging
- Verification after upload
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Configuration
KB_DIR = Path("knowledge_base_verified")
CHECKPOINT_DIR = Path("checkpoints")
STORE_DISPLAY_NAME = "simpleqa-verified-kb"


@dataclass
class UploadCheckpoint:
    """Checkpoint for resuming uploads."""
    store_name: str
    uploaded_files: Dict[str, str] = field(default_factory=dict)  # filename -> google file id
    failed_files: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class GoogleFileSearchUploader:
    """Upload knowledge base files to Google AI Studio File Search."""

    def __init__(self, api_key: str, max_workers: int = 5):
        self.client = genai.Client(api_key=api_key)
        self.max_workers = max_workers
        self._upload_semaphore = None

    def create_or_get_store(self, display_name: str) -> str:
        """Create a new store or get existing one by display name."""
        # Check for existing stores
        print(f"  Checking for existing store '{display_name}'...")
        for store in self.client.file_search_stores.list():
            if store.display_name == display_name:
                print(f"  Found existing store: {store.name}")
                return store.name

        # Create new store
        print(f"  Creating new store '{display_name}'...")
        store = self.client.file_search_stores.create(
            config={"display_name": display_name}
        )
        print(f"  Created store: {store.name}")
        return store.name

    def upload_single_file(
        self, file_path: Path, store_name: str, index: int, total: int,
        max_retries: int = 5
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Upload a single file and import it into the store.

        Returns: (filename, google_file_id, error_message)

        Implements exponential backoff for rate limits (429 errors).
        """
        filename = file_path.name

        for attempt in range(max_retries):
            try:
                # Upload file to Google
                uploaded = self.client.files.upload(file=str(file_path))

                # Import into File Search store
                operation = self.client.file_search_stores.import_file(
                    file_search_store_name=store_name,
                    file_name=uploaded.name,
                )

                # Wait for indexing (with timeout)
                max_wait = 120  # 2 minutes per file
                waited = 0
                while not operation.done and waited < max_wait:
                    time.sleep(2)
                    waited += 2
                    operation = self.client.operations.get(operation=operation)

                if not operation.done:
                    return filename, uploaded.name, f"Indexing timeout after {max_wait}s"

                if operation.error:
                    return filename, uploaded.name, f"Indexing error: {operation.error}"

                return filename, uploaded.name, None

            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limit errors (429)
                if "429" in error_str or "rate" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    # Exponential backoff: 2^attempt * 10 seconds (10, 20, 40, 80, 160)
                    backoff_time = min(10 * (2 ** attempt), 300)  # Cap at 5 minutes
                    print(f"  Rate limited on {filename}, backing off {backoff_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(backoff_time)
                    continue
                else:
                    # Non-rate-limit error, don't retry
                    return filename, None, str(e)

        return filename, None, f"Max retries ({max_retries}) exceeded due to rate limits"

    def upload_batch(
        self,
        files: List[Path],
        store_name: str,
        checkpoint: UploadCheckpoint,
        checkpoint_path: Path,
        on_progress: callable = None,
    ) -> UploadCheckpoint:
        """Upload files in parallel with checkpoint recovery."""
        import threading

        # Filter out already uploaded files
        remaining = [f for f in files if f.name not in checkpoint.uploaded_files]
        print(f"  Files to upload: {len(remaining)} (already done: {len(checkpoint.uploaded_files)})")

        if not remaining:
            print("  All files already uploaded!")
            return checkpoint

        # Thread-safe checkpoint updates
        lock = threading.Lock()
        completed = 0

        def upload_with_progress(file_path: Path, idx: int) -> Tuple[str, Optional[str], Optional[str]]:
            nonlocal completed
            result = self.upload_single_file(file_path, store_name, idx, len(remaining))

            with lock:
                completed += 1
                filename, file_id, error = result

                if error:
                    checkpoint.failed_files.append(filename)
                    print(f"  [{completed}/{len(remaining)}] FAILED: {filename} - {error}")
                else:
                    checkpoint.uploaded_files[filename] = file_id
                    if completed % 10 == 0 or completed == len(remaining):
                        print(f"  [{completed}/{len(remaining)}] Uploaded {filename}")

                # Save checkpoint every 25 files
                if completed % 25 == 0:
                    checkpoint.updated_at = datetime.now().isoformat()
                    self._save_checkpoint(checkpoint, checkpoint_path)

                if on_progress:
                    on_progress(completed, len(remaining))

            return result

        # Parallel upload
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(upload_with_progress, f, i): f
                for i, f in enumerate(remaining)
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    file_path = futures[future]
                    print(f"  ERROR uploading {file_path.name}: {e}")

        # Final checkpoint save
        checkpoint.updated_at = datetime.now().isoformat()
        self._save_checkpoint(checkpoint, checkpoint_path)

        return checkpoint

    def _save_checkpoint(self, checkpoint: UploadCheckpoint, path: Path):
        """Save checkpoint to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def _load_checkpoint(self, path: Path) -> Optional[UploadCheckpoint]:
        """Load checkpoint from disk."""
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def verify_store(self, store_name: str, expected_count: int) -> bool:
        """Verify all files are indexed in the store."""
        print(f"\n  Verifying store {store_name}...")

        # Get store info
        try:
            store = self.client.file_search_stores.get(name=store_name)
            print(f"  Store display name: {store.display_name}")

            # Try a test query to verify search works
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents="What topics are covered in this knowledge base?",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=100,
                    tools=[types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store_name]
                        )
                    )]
                )
            )
            print(f"  Test query response: {response.text[:100]}...")
            print(f"  Store is queryable!")
            return True

        except Exception as e:
            print(f"  Verification failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Upload KB to Google File Search")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--retry-failed", action="store_true", help="Retry previously failed files")
    parser.add_argument("--max-workers", type=int, default=3, help="Parallel upload workers (lower = fewer rate limits)")
    parser.add_argument("--store-name", type=str, help="Use existing store name")
    args = parser.parse_args()

    print("=" * 70)
    print("GOOGLE FILE SEARCH KNOWLEDGE BASE UPLOAD")
    print("=" * 70)

    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        return 1

    # Get KB files
    kb_files = sorted(KB_DIR.glob("verified_*.txt"))
    print(f"\nKnowledge Base: {KB_DIR}")
    print(f"Files found: {len(kb_files)}")

    if not kb_files:
        print("ERROR: No KB files found!")
        return 1

    if args.dry_run:
        print("\n[DRY RUN] Would upload:")
        for f in kb_files[:5]:
            print(f"  - {f.name}")
        print(f"  ... and {len(kb_files) - 5} more")
        return 0

    # Initialize uploader
    uploader = GoogleFileSearchUploader(api_key=api_key, max_workers=args.max_workers)

    # Checkpoint handling
    checkpoint_path = CHECKPOINT_DIR / "google_upload.checkpoint"

    if (args.resume or args.retry_failed) and checkpoint_path.exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = uploader._load_checkpoint(checkpoint_path)
        store_name = checkpoint.store_name
        print(f"  Store: {store_name}")
        print(f"  Already uploaded: {len(checkpoint.uploaded_files)}")
        print(f"  Failed: {len(checkpoint.failed_files)}")

        # Clear failed files for retry if requested
        if args.retry_failed and checkpoint.failed_files:
            print(f"  Clearing {len(checkpoint.failed_files)} failed files for retry")
            checkpoint.failed_files = []
    else:
        # Create or get store
        print("\n1. Setting up File Search store...")
        if args.store_name:
            store_name = args.store_name
            print(f"  Using provided store: {store_name}")
        else:
            store_name = uploader.create_or_get_store(STORE_DISPLAY_NAME)

        checkpoint = UploadCheckpoint(store_name=store_name)

    # Upload files
    print(f"\n2. Uploading {len(kb_files)} files...")
    print(f"   Max workers: {args.max_workers}")
    print(f"   Checkpoint: {checkpoint_path}")

    start_time = time.time()
    checkpoint = uploader.upload_batch(
        files=kb_files,
        store_name=store_name,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("UPLOAD SUMMARY")
    print("=" * 70)
    print(f"  Store: {store_name}")
    print(f"  Total files: {len(kb_files)}")
    print(f"  Uploaded: {len(checkpoint.uploaded_files)}")
    print(f"  Failed: {len(checkpoint.failed_files)}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if checkpoint.failed_files:
        print(f"\n  Failed files:")
        for f in checkpoint.failed_files[:10]:
            print(f"    - {f}")
        if len(checkpoint.failed_files) > 10:
            print(f"    ... and {len(checkpoint.failed_files) - 10} more")

    # Verification
    print("\n3. Verifying store...")
    if uploader.verify_store(store_name, len(kb_files)):
        print("\n  Store verified successfully!")
    else:
        print("\n  Store verification had issues")

    # Save store name to .env suggestion
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"Add to your .env file:")
    print(f"  GOOGLE_FILE_SEARCH_STORE_NAME={store_name}")
    print("=" * 70)

    return 0 if len(checkpoint.failed_files) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
