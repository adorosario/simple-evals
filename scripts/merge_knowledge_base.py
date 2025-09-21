#!/usr/bin/env python3
"""
Knowledge Base File Merger

Merges multiple small knowledge base files into larger chunks to reduce
the number of files for OpenAI vector store processing.

This addresses OpenAI's infrastructure issues with processing thousands
of individual files by creating ~80 merged files instead of 8,009 individual files.

Usage:
    python scripts/merge_knowledge_base.py knowledge_base_full knowledge_base_merged --files-per-merge 100
    python scripts/merge_knowledge_base.py --help
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
import hashlib
from datetime import datetime


def read_file_content(file_path: Path) -> Dict[str, Any]:
    """Read and parse a knowledge base file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Parse the header if it exists
        lines = content.split('\n')
        metadata = {}
        content_start = 0

        # Look for metadata in first few lines
        for i, line in enumerate(lines[:10]):
            if line.startswith('Source: '):
                metadata['source'] = line.replace('Source: ', '').strip()
            elif line.startswith('Title: '):
                metadata['title'] = line.replace('Title: ', '').strip()
            elif line.startswith('Words: '):
                metadata['words'] = line.replace('Words: ', '').strip()
            elif line.startswith('===='):
                content_start = i + 1
                break

        # Get the actual content after metadata
        if content_start > 0:
            actual_content = '\n'.join(lines[content_start:]).strip()
        else:
            actual_content = content

        return {
            'filename': file_path.name,
            'metadata': metadata,
            'content': actual_content,
            'size': len(content)
        }

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def create_merged_file(files_data: List[Dict[str, Any]], output_path: Path, merge_index: int) -> Dict[str, Any]:
    """Create a merged file from multiple source files"""

    # Generate merged filename
    merged_filename = f"merged_{merge_index:03d}.txt"
    merged_path = output_path / merged_filename

    # Collect metadata
    total_words = 0
    sources = []
    total_size = 0

    # Build merged content
    merged_content_parts = []

    # Add header with merge info
    merged_content_parts.append(f"MERGED KNOWLEDGE BASE FILE {merge_index}")
    merged_content_parts.append(f"Generated: {datetime.now().isoformat()}")
    merged_content_parts.append(f"Contains {len(files_data)} documents")
    merged_content_parts.append("=" * 80)
    merged_content_parts.append("")

    # Add each document with clear separation
    for i, file_data in enumerate(files_data):
        if not file_data:
            continue

        # Document header
        merged_content_parts.append(f"DOCUMENT {i+1}/{len(files_data)}")
        merged_content_parts.append(f"Source File: {file_data['filename']}")

        # Add original metadata if available
        if file_data['metadata']:
            for key, value in file_data['metadata'].items():
                merged_content_parts.append(f"{key.title()}: {value}")
                if key == 'words' and value.isdigit():
                    total_words += int(value)
                elif key == 'source':
                    sources.append(value)

        merged_content_parts.append("-" * 40)
        merged_content_parts.append("")

        # Add content
        merged_content_parts.append(file_data['content'])
        merged_content_parts.append("")
        merged_content_parts.append("=" * 80)
        merged_content_parts.append("")

        total_size += file_data['size']

    # Write merged file
    merged_content = '\n'.join(merged_content_parts)

    with open(merged_path, 'w', encoding='utf-8') as f:
        f.write(merged_content)

    # Return metadata about the merged file
    return {
        'filename': merged_filename,
        'path': str(merged_path),
        'source_files': [f['filename'] for f in files_data if f],
        'source_count': len([f for f in files_data if f]),
        'total_words': total_words,
        'total_size': len(merged_content),
        'sources': sources[:10],  # Limit to first 10 sources
        'created': datetime.now().isoformat()
    }


def create_build_metadata(merged_files: List[Dict[str, Any]], output_dir: Path) -> None:
    """Create build metadata for the merged knowledge base"""

    total_files = len(merged_files)
    total_size = sum(f['total_size'] for f in merged_files)
    total_source_files = sum(f['source_count'] for f in merged_files)
    total_words = sum(f['total_words'] for f in merged_files)

    metadata = {
        'build_stats': {
            'total_documents': total_files,
            'total_words': total_words,
            'total_characters': total_size,
            'avg_words_per_doc': total_words / total_files if total_files > 0 else 0,
            'avg_chars_per_doc': total_size / total_files if total_files > 0 else 0,
            'source_files_merged': total_source_files,
            'merge_ratio': f"{total_source_files}:{total_files}"
        },
        'build_info': {
            'build_date': datetime.now().isoformat(),
            'build_type': 'merged_knowledge_base',
            'total_merged_files': total_files,
            'total_source_files': total_source_files,
            'total_size_bytes': total_size,
            'files': merged_files
        }
    }

    # Write metadata
    metadata_path = output_dir / 'build_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # Write summary
    summary = {
        'total_files': total_files,
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'source_files_merged': total_source_files,
        'compression_ratio': round(total_source_files / total_files, 1),
        'build_date': datetime.now().isoformat()
    }

    summary_path = output_dir / 'build_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Merged {total_source_files} files into {total_files} files")
    print(f"Total size: {summary['total_size_mb']} MB")
    print(f"Compression ratio: {summary['compression_ratio']}:1")


def merge_knowledge_base(input_dir: Path, output_dir: Path, files_per_merge: int = 100) -> None:
    """Main function to merge knowledge base files"""

    # Validate input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all doc files
    doc_files = sorted(list(input_dir.glob("doc_*.txt")))

    if not doc_files:
        raise ValueError(f"No doc_*.txt files found in {input_dir}")

    print(f"Found {len(doc_files)} files to merge")
    print(f"Creating merged files with {files_per_merge} files each")

    merged_files = []
    merge_index = 1

    # Process files in chunks
    for i in range(0, len(doc_files), files_per_merge):
        chunk_files = doc_files[i:i + files_per_merge]

        print(f"Processing merge {merge_index}: {len(chunk_files)} files")

        # Read all files in this chunk
        files_data = []
        for file_path in chunk_files:
            file_data = read_file_content(file_path)
            files_data.append(file_data)

        # Create merged file
        merged_info = create_merged_file(files_data, output_dir, merge_index)
        merged_files.append(merged_info)

        merge_index += 1

    # Create metadata
    create_build_metadata(merged_files, output_dir)

    print(f"Merging complete! Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge knowledge base files for OpenAI vector store upload",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('input_dir', type=Path, help='Input directory containing doc_*.txt files')
    parser.add_argument('output_dir', type=Path, help='Output directory for merged files')
    parser.add_argument('--files-per-merge', type=int, default=100,
                       help='Number of files to merge into each output file (default: 100)')
    parser.add_argument('--clean', action='store_true',
                       help='Clean output directory before starting')

    args = parser.parse_args()

    # Clean output directory if requested
    if args.clean and args.output_dir.exists():
        print(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    try:
        merge_knowledge_base(args.input_dir, args.output_dir, args.files_per_merge)
        print("✅ Knowledge base merging completed successfully!")

    except Exception as e:
        print(f"❌ Error during merging: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())