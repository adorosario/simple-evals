#!/usr/bin/env python3
"""
Robust Knowledge Base Upload Script

Enhanced upload script with:
- Concurrent uploads with intelligent batching
- Comprehensive progress tracking and auditing
- Resume capability from interruptions
- Automatic integrity validation
- Detailed reporting and recommendations

This script replaces the basic upload_knowledge_base_to_openai.py with
a production-ready, scalable solution for large knowledge bases.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.enhanced_openai_vector_store import (
    EnhancedOpenAIVectorStoreManager, UploadConfig
)
from src.upload_audit_logger import UploadAuditLogger
from src.vector_store_validator import VectorStoreValidator, ValidationConfig

def setup_logging():
    """Setup logging configuration"""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('robust_upload.log')
        ]
    )

    return logging.getLogger(__name__)

def validate_prerequisites(args, logger):
    """Validate all prerequisites before starting upload"""
    logger.info("🔍 Validating prerequisites...")

    # Check knowledge base directory
    kb_dir = Path(args.knowledge_base_dir)
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    # Find knowledge base files
    txt_files = list(kb_dir.glob("*.txt"))
    if len(txt_files) == 0:
        raise ValueError(f"No .txt files found in {kb_dir}")

    logger.info(f"   Found {len(txt_files):,} files to upload")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in txt_files)
    logger.info(f"   Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")

    # Check for build metadata
    metadata_file = kb_dir / "build_metadata.json"
    build_metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            build_metadata = json.load(f)
        logger.info(f"   Found build metadata: {build_metadata['build_stats']['total_documents']} documents")

    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    logger.info(f"   API key found: {api_key[:8]}...")

    # Estimate costs
    estimated_tokens = total_size / 750  # Rough estimate
    estimated_monthly_cost = estimated_tokens * 0.10 / 1_000_000
    logger.info(f"   Estimated monthly cost: ${estimated_monthly_cost:.2f}")

    return txt_files, total_size, build_metadata

def create_upload_config(args):
    """Create upload configuration from command line arguments"""
    return UploadConfig(
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        max_requests_per_minute=args.rate_limit,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay,
        exponential_backoff=args.exponential_backoff,
        upload_timeout_seconds=args.timeout,
        enable_checkpoints=args.enable_checkpoints,
        checkpoint_interval=args.checkpoint_interval,
        validate_file_hashes=args.validate_hashes
    )

def create_validation_config(args):
    """Create validation configuration"""
    return ValidationConfig(
        validate_file_count=True,
        validate_file_content=args.validate_content,
        content_sample_size=args.content_sample_size,
        test_search_functionality=args.test_search,
        benchmark_performance=args.benchmark_performance,
        benchmark_query_count=args.benchmark_queries
    )

def progress_callback(completed: int, total: int, current_file: str):
    """Progress callback for upload tracking"""
    percentage = (completed / total * 100) if total > 0 else 0
    print(f"\r📤 Progress: {completed:,}/{total:,} ({percentage:.1f}%) - {Path(current_file).name}", end='', flush=True)

def update_env_file(vector_store_id: str, logger):
    """Update .env file with new vector store ID"""
    env_file = Path(".env")
    if not env_file.exists():
        logger.warning("⚠️  .env file not found, cannot auto-update")
        return False

    try:
        # Read current .env
        with open(env_file, 'r') as f:
            env_content = f.read()

        # Update or add OPENAI_VECTOR_STORE_ID
        lines = env_content.split('\n')
        updated_lines = []
        updated = False

        for line in lines:
            if line.startswith('OPENAI_VECTOR_STORE_ID='):
                updated_lines.append(f'OPENAI_VECTOR_STORE_ID={vector_store_id}')
                updated = True
            else:
                updated_lines.append(line)

        if not updated:
            # Add new line
            if env_content and not env_content.endswith('\n'):
                updated_lines.append('')
            updated_lines.append(f'OPENAI_VECTOR_STORE_ID={vector_store_id}')

        # Write back
        with open(env_file, 'w') as f:
            f.write('\n'.join(updated_lines))

        logger.info(f"   ✅ Updated .env file with vector store ID")
        return True

    except Exception as e:
        logger.warning(f"   ⚠️  Could not update .env file: {e}")
        return False

def save_upload_report(upload_report: dict, validation_report: dict, output_dir: Path, logger):
    """Save comprehensive upload and validation reports"""
    try:
        # Create reports directory
        reports_dir = output_dir / "upload_reports"
        reports_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save upload report
        upload_report_file = reports_dir / f"upload_report_{timestamp}.json"
        with open(upload_report_file, 'w') as f:
            json.dump(upload_report, f, indent=2, default=str)

        # Save validation report
        validation_report_file = reports_dir / f"validation_report_{timestamp}.json"
        with open(validation_report_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)

        # Save combined summary
        summary_file = reports_dir / f"upload_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(generate_summary_report(upload_report, validation_report))

        logger.info(f"   📄 Reports saved to: {reports_dir}")
        return reports_dir

    except Exception as e:
        logger.error(f"Failed to save reports: {e}")
        return None

def generate_summary_report(upload_report: dict, validation_report: dict) -> str:
    """Generate human-readable summary report"""
    lines = []
    lines.append("=" * 80)
    lines.append("KNOWLEDGE BASE UPLOAD SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Upload Summary
    lines.append("UPLOAD RESULTS:")
    lines.append(f"  Vector Store ID: {upload_report.get('vector_store_id', 'N/A')}")
    lines.append(f"  Total Files: {upload_report.get('total_files', 0):,}")
    lines.append(f"  Successful: {upload_report.get('successful_files', 0):,}")
    lines.append(f"  Failed: {upload_report.get('failed_files', 0):,}")
    lines.append(f"  Success Rate: {upload_report.get('success_rate', 0):.1%}")
    lines.append(f"  Duration: {upload_report.get('duration_seconds', 0)/60:.1f} minutes")
    lines.append(f"  Average Speed: {upload_report.get('average_speed_mbps', 0):.2f} MB/s")
    lines.append("")

    # Validation Summary
    if validation_report:
        lines.append("VALIDATION RESULTS:")
        lines.append(f"  Overall Status: {validation_report.get('overall_status', 'unknown').upper()}")
        lines.append(f"  Files Validated: {validation_report.get('valid_files', 0)}/{validation_report.get('total_files_found', 0)}")
        lines.append(f"  Search Tests: {validation_report.get('search_tests_passed', 0)}/{validation_report.get('search_tests_passed', 0) + validation_report.get('search_tests_failed', 0)}")

        if validation_report.get('performance_benchmark'):
            perf = validation_report['performance_benchmark']
            lines.append(f"  Avg Query Time: {perf.get('average_response_time', 0):.3f}s")
            lines.append(f"  Queries/Second: {perf.get('queries_per_second', 0):.2f}")
        lines.append("")

    # Recommendations
    recommendations = []
    if upload_report.get('failed_files', 0) > 0:
        recommendations.append("Review and re-upload failed files")

    if validation_report and validation_report.get('recommendations'):
        recommendations.extend(validation_report['recommendations'])

    if recommendations:
        lines.append("RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

    # Next Steps
    lines.append("NEXT STEPS:")
    if upload_report.get('vector_store_id'):
        lines.append("  1. Vector store is ready for use in RAG applications")
        lines.append("  2. Update your application config with the vector store ID")
        lines.append("  3. Run three-way benchmark to test RAG performance")
    else:
        lines.append("  1. Fix upload issues and retry")
        lines.append("  2. Check logs for detailed error information")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)

def main():
    """Main upload script"""
    parser = argparse.ArgumentParser(description="Robust Knowledge Base Upload to OpenAI")

    # Basic arguments
    parser.add_argument("knowledge_base_dir", help="Directory containing knowledge base files")
    parser.add_argument("--store-name", default=None, help="Name for the vector store")
    parser.add_argument("--session-id", default=None, help="Session ID for resume capability")

    # Upload configuration
    parser.add_argument("--max-workers", type=int, default=10, help="Max concurrent workers")
    parser.add_argument("--batch-size", type=int, default=50, help="Files per batch")
    parser.add_argument("--rate-limit", type=int, default=1000, help="Max requests per minute")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per file")
    parser.add_argument("--retry-delay", type=float, default=1.0, help="Initial retry delay")
    parser.add_argument("--timeout", type=int, default=300, help="Upload timeout per file")

    # Advanced options
    parser.add_argument("--exponential-backoff", action="store_true", help="Use exponential backoff")
    parser.add_argument("--enable-checkpoints", action="store_true", default=True, help="Enable checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint every N files")
    parser.add_argument("--validate-hashes", action="store_true", help="Validate file hashes")

    # Validation options
    parser.add_argument("--skip-validation", action="store_true", help="Skip post-upload validation")
    parser.add_argument("--validate-content", action="store_true", help="Validate file content")
    parser.add_argument("--content-sample-size", type=int, default=10, help="Files to sample for content validation")
    parser.add_argument("--test-search", action="store_true", default=True, help="Test search functionality")
    parser.add_argument("--benchmark-performance", action="store_true", help="Benchmark performance")
    parser.add_argument("--benchmark-queries", type=int, default=20, help="Number of benchmark queries")

    # Resume option
    parser.add_argument("--resume", action="store_true", help="Resume interrupted upload")

    # Dry run
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without uploading")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()

    logger.info("🚀 ROBUST KNOWLEDGE BASE UPLOAD STARTING")
    logger.info("=" * 70)

    try:
        # Validate prerequisites
        txt_files, total_size, build_metadata = validate_prerequisites(args, logger)

        # Create configurations
        upload_config = create_upload_config(args)
        validation_config = create_validation_config(args)

        # Create store name if not provided
        store_name = args.store_name or f"Knowledge Base {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # Show configuration
        logger.info("📋 CONFIGURATION:")
        logger.info(f"   Knowledge Base: {args.knowledge_base_dir}")
        logger.info(f"   Store Name: {store_name}")
        logger.info(f"   Files to Upload: {len(txt_files):,}")
        logger.info(f"   Total Size: {total_size/1024/1024:.1f} MB")
        logger.info(f"   Max Workers: {upload_config.max_workers}")
        logger.info(f"   Batch Size: {upload_config.batch_size}")
        logger.info(f"   Rate Limit: {upload_config.max_requests_per_minute} req/min")
        logger.info(f"   Checkpoints: {'Enabled' if upload_config.enable_checkpoints else 'Disabled'}")
        logger.info(f"   Validation: {'Enabled' if not args.skip_validation else 'Disabled'}")
        logger.info("")

        if args.dry_run:
            logger.info("🔍 DRY RUN MODE - No actual upload will be performed")
            logger.info("✅ All prerequisites validated successfully")
            return 0

        # Confirm before proceeding (for interactive use)
        if sys.stdin.isatty():
            logger.info("⚠️  UPLOAD CONFIRMATION:")
            logger.info(f"   About to upload {len(txt_files):,} files ({total_size/1024/1024:.1f} MB)")
            logger.info(f"   This may take 15-30 minutes and incur OpenAI API costs")

            response = input("   Continue? (y/N): ").strip().lower()
            if response != 'y':
                logger.info("Upload cancelled by user")
                return 0

        # Create audit logger
        session_id = args.session_id or f"upload_{int(time.time())}"
        audit_logger = UploadAuditLogger(
            session_id=session_id,
            audit_dir="audit_logs"
        )

        # Create file list with metadata for validation
        expected_files = []
        for file_path in txt_files:
            expected_files.append({
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_hash': None  # Could add hash calculation here if needed
            })

        upload_start_time = time.time()

        # Perform upload
        logger.info("📤 STARTING UPLOAD...")
        with EnhancedOpenAIVectorStoreManager(
            config=upload_config,
            audit_logger=audit_logger,
            progress_callback=progress_callback
        ) as manager:

            if args.resume and args.session_id:
                # Resume interrupted upload
                logger.info(f"   Resuming session: {args.session_id}")
                vector_store_id, upload_report = manager.resume_upload(args.session_id, store_name)
            else:
                # Start new upload
                file_paths = [str(f) for f in txt_files]
                vector_store_id, upload_report = manager.upload_files_concurrent(
                    file_paths, store_name, session_id
                )

        print()  # New line after progress
        upload_duration = time.time() - upload_start_time

        # Finalize audit logging
        audit_summary = audit_logger.finalize_session()

        logger.info("✅ UPLOAD COMPLETED!")
        logger.info(f"   Vector Store ID: {vector_store_id}")
        logger.info(f"   Duration: {upload_duration/60:.1f} minutes")
        logger.info(f"   Success Rate: {upload_report['success_rate']:.1%}")
        logger.info(f"   Average Speed: {upload_report['average_speed_mbps']:.2f} MB/s")
        logger.info("")

        # Update .env file
        logger.info("🔧 UPDATING CONFIGURATION...")
        update_env_file(vector_store_id, logger)

        # Perform validation if enabled
        validation_report = None
        if not args.skip_validation:
            logger.info("🔍 STARTING VALIDATION...")

            validator = VectorStoreValidator(config=validation_config)
            validation_result = validator.validate_vector_store(vector_store_id, expected_files)
            validation_report = validation_result.to_dict()

            logger.info(f"   Validation Status: {validation_result.overall_status.upper()}")
            logger.info(f"   File Success Rate: {validation_result.success_rate():.1%}")
            logger.info(f"   Search Success Rate: {validation_result.search_success_rate():.1%}")

            if validation_result.performance_benchmark:
                perf = validation_result.performance_benchmark
                logger.info(f"   Avg Query Time: {perf.average_response_time:.3f}s")

            logger.info("")

        # Save comprehensive reports
        logger.info("📊 GENERATING REPORTS...")
        kb_dir = Path(args.knowledge_base_dir)
        reports_dir = save_upload_report(upload_report, validation_report, kb_dir, logger)

        # Print final summary
        logger.info("🎉 UPLOAD PROCESS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info("SUMMARY:")
        logger.info(f"  Vector Store ID: {vector_store_id}")
        logger.info(f"  Files Uploaded: {upload_report['successful_files']:,}/{upload_report['total_files']:,}")
        logger.info(f"  Success Rate: {upload_report['success_rate']:.1%}")
        logger.info(f"  Total Duration: {upload_duration/60:.1f} minutes")

        if validation_report:
            logger.info(f"  Validation Status: {validation_report['overall_status'].upper()}")

        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("  1. Vector store is ready for RAG applications")
        logger.info("  2. Run three-way benchmark: python scripts/three_way_rag_benchmark.py")
        logger.info("  3. Check detailed reports in: upload_reports/")
        logger.info("")

        return 0

    except KeyboardInterrupt:
        print()
        logger.info("⏹️  Upload interrupted by user")
        logger.info("   Session state saved for potential resume")
        return 130

    except Exception as e:
        logger.error(f"💥 Upload failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())