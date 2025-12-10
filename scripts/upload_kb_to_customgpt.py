#!/usr/bin/env python3
"""
Upload Knowledge Base Files to CustomGPT

This script:
1. Creates a new CustomGPT project
2. Uploads all files from knowledge_base_verified/ to the project
3. Monitors upload progress and handles errors

Usage:
    docker compose run --rm simple-evals python scripts/upload_kb_to_customgpt.py
    docker compose run --rm simple-evals python scripts/upload_kb_to_customgpt.py --dry-run
    docker compose run --rm simple-evals python scripts/upload_kb_to_customgpt.py --max-files 10  # Test with 10 files
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomGPTUploader:
    """Upload files to CustomGPT via API."""

    BASE_URL = "https://app.customgpt.ai/api/v1"

    def __init__(self, api_key: str, timeout: int = 120):
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }

    def create_project(self, project_name: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Create a new CustomGPT project.

        Returns:
            (project_id, error_message) tuple
        """
        url = f"{self.BASE_URL}/projects"

        # For file-based projects, we create with minimal sitemap or use a placeholder
        # Based on API docs, sitemap_path is required but can be a dummy for file uploads
        payload = {
            "project_name": project_name,
            # Some APIs require a sitemap even for file-based projects
            # We'll try without first, then with a placeholder if needed
        }

        logger.info(f"Creating project: {project_name}")

        try:
            # First try: Create project with just name (for file-upload projects)
            response = requests.post(
                url,
                headers={**self.headers, "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 201 or response.status_code == 200:
                data = response.json()
                project_id = data.get('data', {}).get('id')
                if project_id:
                    logger.info(f"Project created successfully: ID={project_id}")
                    return project_id, None
                else:
                    return None, f"Project created but no ID in response: {data}"

            elif response.status_code == 422:
                # Validation error - might need sitemap_path
                logger.warning(f"Validation error, trying with placeholder sitemap...")

                # Try with a minimal valid sitemap URL
                payload["sitemap_path"] = "https://example.com/sitemap.xml"
                response = requests.post(
                    url,
                    headers={**self.headers, "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code in [200, 201]:
                    data = response.json()
                    project_id = data.get('data', {}).get('id')
                    if project_id:
                        logger.info(f"Project created with placeholder: ID={project_id}")
                        return project_id, None

            error_msg = f"HTTP {response.status_code}: {response.text[:500]}"
            logger.error(f"Failed to create project: {error_msg}")
            return None, error_msg

        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def upload_file(
        self,
        project_id: int,
        file_path: Path,
        retries: int = 3,
        retry_delay: float = 2.0
    ) -> Tuple[bool, Optional[str]]:
        """
        Upload a single file to a CustomGPT project.

        Returns:
            (success, error_message) tuple
        """
        url = f"{self.BASE_URL}/projects/{project_id}/sources"

        for attempt in range(retries):
            try:
                with open(file_path, 'rb') as f:
                    files = {
                        'file': (file_path.name, f, 'text/plain')
                    }

                    response = requests.post(
                        url,
                        headers=self.headers,  # Don't set Content-Type for multipart
                        files=files,
                        timeout=self.timeout
                    )

                if response.status_code in [200, 201]:
                    return True, None

                elif response.status_code == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 30))
                    logger.warning(f"Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                elif response.status_code in [502, 503, 504]:
                    # Gateway errors - retry with backoff
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(f"Gateway error {response.status_code}, retrying in {delay}s...")
                    time.sleep(delay)
                    continue

                else:
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    if attempt < retries - 1:
                        logger.warning(f"Upload failed, retrying... ({error_msg})")
                        time.sleep(retry_delay)
                        continue
                    return False, error_msg

            except requests.exceptions.Timeout:
                if attempt < retries - 1:
                    logger.warning(f"Timeout, retrying...")
                    time.sleep(retry_delay)
                    continue
                return False, "Request timeout after all retries"

            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    logger.warning(f"Request error, retrying... ({str(e)})")
                    time.sleep(retry_delay)
                    continue
                return False, f"Request error: {str(e)}"

        return False, "Max retries exceeded"

    def get_project_info(self, project_id: int) -> Optional[Dict]:
        """Get project information including source count."""
        url = f"{self.BASE_URL}/projects/{project_id}"

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            if response.status_code == 200:
                return response.json().get('data', {})
        except Exception as e:
            logger.warning(f"Failed to get project info: {e}")

        return None


def get_kb_files(kb_dir: Path, max_files: Optional[int] = None) -> List[Path]:
    """Get list of knowledge base files to upload."""
    files = sorted(kb_dir.glob("verified_*.txt"))

    if max_files:
        files = files[:max_files]

    return files


def main():
    parser = argparse.ArgumentParser(description="Upload knowledge base to CustomGPT")
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=Path("knowledge_base_verified"),
        help="Knowledge base directory"
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="SimpleQA-Verified-KB-v1",
        help="Name for the new CustomGPT project"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to upload (for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually upload, just show what would be done"
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help="Existing project ID to upload to (skips project creation)"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from file index (for resuming)"
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("CUSTOMGPT_API_KEY")
    if not api_key:
        logger.error("CUSTOMGPT_API_KEY not found in environment")
        sys.exit(1)

    # Get files to upload
    files = get_kb_files(args.kb_dir, args.max_files)
    if not files:
        logger.error(f"No files found in {args.kb_dir}")
        sys.exit(1)

    # Skip files if resuming
    if args.start_from > 0:
        files = files[args.start_from:]
        logger.info(f"Resuming from file index {args.start_from}")

    logger.info("=" * 60)
    logger.info("CustomGPT Knowledge Base Upload")
    logger.info("=" * 60)
    logger.info(f"KB directory: {args.kb_dir}")
    logger.info(f"Project name: {args.project_name}")
    logger.info(f"Files to upload: {len(files)}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN - No changes will be made")
        for i, f in enumerate(files[:10]):
            logger.info(f"  Would upload: {f.name}")
        if len(files) > 10:
            logger.info(f"  ... and {len(files) - 10} more files")
        return

    # Initialize uploader
    uploader = CustomGPTUploader(api_key)

    # Create or use existing project
    project_id = args.project_id
    if not project_id:
        project_id, error = uploader.create_project(args.project_name)
        if not project_id:
            logger.error(f"Failed to create project: {error}")
            sys.exit(1)

    logger.info(f"Using project ID: {project_id}")

    # Upload files
    results = {
        "project_id": project_id,
        "project_name": args.project_name,
        "total_files": len(files),
        "successful": 0,
        "failed": 0,
        "errors": []
    }

    start_time = time.time()

    for i, file_path in enumerate(files):
        logger.info(f"[{i+1}/{len(files)}] Uploading {file_path.name}...")

        success, error = uploader.upload_file(project_id, file_path)

        if success:
            results["successful"] += 1
            logger.info(f"  ✓ Uploaded successfully")
        else:
            results["failed"] += 1
            results["errors"].append({
                "file": file_path.name,
                "error": error
            })
            logger.error(f"  ✗ Failed: {error}")

        # Rate limiting - be gentle with the API
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            logger.info(f"  Progress: {i+1}/{len(files)} files ({rate:.1f} files/min)")
            time.sleep(1)  # Small pause every 10 files

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("UPLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Project ID: {project_id}")
    logger.info(f"Total files: {results['total_files']}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Rate: {results['successful']/elapsed*60:.1f} files/min")

    if results["failed"] > 0:
        logger.warning(f"\nFailed files:")
        for err in results["errors"][:10]:
            logger.warning(f"  {err['file']}: {err['error']}")
        if len(results["errors"]) > 10:
            logger.warning(f"  ... and {len(results['errors']) - 10} more")

    # Save results
    results_path = args.kb_dir / "customgpt_upload_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Print env var suggestion
    logger.info(f"\n📋 Add to your .env file:")
    logger.info(f"CUSTOMGPT_PROJECT_VERIFIED={project_id}")


if __name__ == "__main__":
    main()
