#!/usr/bin/env python3
"""
Create OpenAI Vector Store from Checkpoint File

This script loads the checkpoint file from a previous upload session and creates
a vector store with all the uploaded files.

Usage:
    docker compose run --rm simple-evals python scripts/create_vector_store_from_checkpoint.py
"""

import os
import pickle
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main():
    # Load checkpoint
    checkpoint_path = Path("checkpoints/upload_1765294410.checkpoint")

    print(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Extract file IDs
    file_ids = [
        f["openai_file_id"]
        for f in checkpoint.get("completed_files", [])
        if f.get("openai_file_id")
    ]

    print(f"Found {len(file_ids)} file IDs in checkpoint")

    if not file_ids:
        print("ERROR: No file IDs found in checkpoint!")
        return 1

    # Create OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Create vector store
    # Note: OpenAI's vector_stores.create() can only accept up to 500 file_ids at once
    print(f"Creating vector store 'SimpleQA-Verified-KB-v1'...")

    try:
        # Create with first batch of files
        first_batch = file_ids[:500]
        print(f"Adding first batch: {len(first_batch)} files...")

        vector_store = client.vector_stores.create(
            name="SimpleQA-Verified-KB-v1",
            file_ids=first_batch
        )

        print(f"Vector store created: {vector_store.id}")
        print(f"Status: {vector_store.status}")

        # Add remaining files in batches
        if len(file_ids) > 500:
            remaining = file_ids[500:]
            batch_size = 100

            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(remaining) + batch_size - 1) // batch_size

                print(f"Adding batch {batch_num}/{total_batches}: {len(batch)} files...")

                for file_id in batch:
                    try:
                        client.vector_stores.files.create(
                            vector_store_id=vector_store.id,
                            file_id=file_id
                        )
                    except Exception as e:
                        print(f"  Warning: Failed to add {file_id}: {e}")

                # Brief pause between batches to avoid rate limits
                if i + batch_size < len(remaining):
                    time.sleep(1)

        # Get final status
        final_store = client.vector_stores.retrieve(vector_store.id)

        print("\n" + "=" * 60)
        print("VECTOR STORE CREATED SUCCESSFULLY")
        print("=" * 60)
        print(f"Vector Store ID: {final_store.id}")
        print(f"Name: {final_store.name}")
        print(f"Status: {final_store.status}")
        print(f"File Counts: {final_store.file_counts}")
        print("=" * 60)
        print(f"\nAdd to your .env file:")
        print(f"OPENAI_VECTOR_STORE_ID={final_store.id}")

        return 0

    except Exception as e:
        print(f"ERROR creating vector store: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
