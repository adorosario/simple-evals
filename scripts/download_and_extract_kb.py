#!/usr/bin/env python3
"""
Download and Extract Knowledge Base Assets

This script downloads pre-built knowledge base and cache assets from GitHub releases
to accelerate SimpleQA evaluation setup, eliminating the need to rebuild from scratch.

Usage:
    python scripts/download_and_extract_kb.py [--cache-only] [--kb-only] [--force]

Examples:
    python scripts/download_and_extract_kb.py                    # Download all assets
    python scripts/download_and_extract_kb.py --cache-only       # Download only cache
    python scripts/download_and_extract_kb.py --kb-only          # Download only knowledge bases
    python scripts/download_and_extract_kb.py --force            # Force re-download existing assets
"""

import os
import sys
import argparse
import requests
import tarfile
import hashlib
from pathlib import Path
from typing import Dict, Optional
import tempfile

# GitHub release configuration
GITHUB_REPO = "adorosario/simple-evals"
RELEASE_TAG = "kb-v1.1"
BASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

# Asset configuration
ASSETS = {
    "cache": {
        "filename": "cache-url-cache-v1.0.tar.gz",
        "extract_to": ".",
        "size_mb": 837,
        "description": "URL cache from 11,000+ web sources"
    },
    "kb_full": {
        "filename": "knowledge-base-full-v1.0.tar.gz",
        "extract_to": ".",
        "size_mb": 135,
        "description": "Complete processed knowledge base documents"
    },
    "kb_merged": {
        "filename": "knowledge-base-merged-v1.0.tar.gz",
        "extract_to": ".",
        "size_mb": 135,
        "description": "Merged knowledge base variant"
    }
}

def check_existing_directories() -> Dict[str, bool]:
    """Check which directories already exist"""
    return {
        "cache": os.path.exists("cache/url_cache") and len(os.listdir("cache/url_cache")) > 0,
        "kb_full": os.path.exists("knowledge_base_full") and len(os.listdir("knowledge_base_full")) > 0,
        "kb_merged": os.path.exists("knowledge_base_merged") and len(os.listdir("knowledge_base_merged")) > 0,
    }

def download_file(url: str, local_path: str, expected_size_mb: int) -> bool:
    """Download a file with progress indication"""
    print(f"📥 Downloading {os.path.basename(local_path)} ({expected_size_mb}MB)...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress indicator every 50MB
                    if downloaded % (50 * 1024 * 1024) == 0:
                        progress_mb = downloaded / (1024 * 1024)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"   Progress: {progress_mb:.0f}MB ({percent:.1f}%)")
                        else:
                            print(f"   Progress: {progress_mb:.0f}MB")

        final_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✅ Downloaded {os.path.basename(local_path)} ({final_size_mb:.0f}MB)")
        return True

    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False

def extract_tar_gz(tar_path: str, extract_to: str) -> bool:
    """Extract a tar.gz file"""
    print(f"📂 Extracting {os.path.basename(tar_path)}...")

    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_to)
        print(f"✅ Extracted {os.path.basename(tar_path)}")
        return True

    except Exception as e:
        print(f"❌ Error extracting {tar_path}: {e}")
        return False

def download_and_extract_asset(asset_key: str, asset_config: Dict, force: bool = False) -> bool:
    """Download and extract a single asset"""
    filename = asset_config["filename"]
    url = f"{BASE_URL}/{filename}"

    # Create temporary file for download
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp_file:
        tmp_path = tmp_file.name

    try:
        # Download
        if not download_file(url, tmp_path, asset_config["size_mb"]):
            return False

        # Extract
        if not extract_tar_gz(tmp_path, asset_config["extract_to"]):
            return False

        return True

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def main():
    parser = argparse.ArgumentParser(
        description="Download and extract knowledge base assets from GitHub releases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--cache-only", action="store_true",
                       help="Download only the cache directory")
    parser.add_argument("--kb-only", action="store_true",
                       help="Download only the knowledge base directories")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if directories exist")
    parser.add_argument("--list", action="store_true",
                       help="List available assets and their status")

    args = parser.parse_args()

    if args.cache_only and args.kb_only:
        print("❌ Error: Cannot specify both --cache-only and --kb-only")
        sys.exit(1)

    # Check existing directories
    existing = check_existing_directories()

    if args.list:
        print("📋 Asset Status:")
        print(f"   Cache directory: {'✅ EXISTS' if existing['cache'] else '❌ MISSING'}")
        print(f"   Knowledge base (full): {'✅ EXISTS' if existing['kb_full'] else '❌ MISSING'}")
        print(f"   Knowledge base (merged): {'✅ EXISTS' if existing['kb_merged'] else '❌ MISSING'}")
        print("\n📦 Available Downloads:")
        for key, config in ASSETS.items():
            print(f"   {config['filename']} ({config['size_mb']}MB) - {config['description']}")
        return

    # Determine which assets to download
    assets_to_download = {}

    if args.cache_only:
        assets_to_download = {"cache": ASSETS["cache"]}
    elif args.kb_only:
        assets_to_download = {k: v for k, v in ASSETS.items() if k.startswith("kb_")}
    else:
        assets_to_download = ASSETS

    # Check if anything needs to be downloaded
    downloads_needed = []
    for key in assets_to_download:
        if args.force or not existing[key]:
            downloads_needed.append(key)
        else:
            print(f"⏭️  Skipping {key} (already exists, use --force to re-download)")

    if not downloads_needed:
        print("🎉 All requested assets already exist! Use --force to re-download.")
        return

    # Calculate total download size
    total_size = sum(ASSETS[key]["size_mb"] for key in downloads_needed)
    print(f"🚀 Starting download of {len(downloads_needed)} assets ({total_size}MB total)")
    print(f"   Release: {GITHUB_REPO}/releases/tag/{RELEASE_TAG}")

    # Download and extract each asset
    success_count = 0
    for key in downloads_needed:
        print(f"\n📦 Processing {key}...")
        if download_and_extract_asset(key, ASSETS[key], args.force):
            success_count += 1
        else:
            print(f"❌ Failed to process {key}")

    # Summary
    if success_count == len(downloads_needed):
        print(f"\n🎉 Success! Downloaded and extracted {success_count}/{len(downloads_needed)} assets")
        print("\n🚀 Ready to run SimpleQA evaluations:")
        print("   docker compose run --rm simple-evals python scripts/confidence_threshold_benchmark.py")
    else:
        print(f"\n⚠️  Partial success: {success_count}/{len(downloads_needed)} assets completed")
        print("   Some assets failed to download. Check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()