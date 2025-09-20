"""
Upload Audit Logger

Comprehensive logging and auditing system for OpenAI vector store uploads.
Provides detailed tracking, performance metrics, and audit trails.
"""

import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

class UploadStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"

@dataclass
class FileUploadRecord:
    """Record for individual file upload attempt"""
    file_path: str
    file_size: int
    attempt_number: int
    status: UploadStatus
    start_time: float
    end_time: Optional[float] = None
    openai_file_id: Optional[str] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    upload_speed_bytes_per_sec: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    def duration(self) -> Optional[float]:
        """Calculate upload duration in seconds"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None

@dataclass
class BatchUploadRecord:
    """Record for batch upload metrics"""
    batch_id: int
    batch_size: int
    start_time: float
    end_time: Optional[float] = None
    successful_uploads: int = 0
    failed_uploads: int = 0
    total_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def duration(self) -> Optional[float]:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None

    def success_rate(self) -> float:
        total = self.successful_uploads + self.failed_uploads
        return self.successful_uploads / total if total > 0 else 0.0

@dataclass
class UploadSessionSummary:
    """Summary of entire upload session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_bytes: int = 0
    total_batches: int = 0
    vector_store_id: Optional[str] = None
    vector_store_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def duration(self) -> Optional[float]:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None

    def success_rate(self) -> float:
        return self.successful_files / self.total_files if self.total_files > 0 else 0.0

    def average_speed_bytes_per_sec(self) -> Optional[float]:
        duration = self.duration()
        if duration and duration > 0:
            return self.total_bytes / duration
        return None

class UploadAuditLogger:
    """
    Comprehensive audit logger for OpenAI vector store uploads.

    Tracks detailed metrics, performance data, and creates audit trails
    for compliance and troubleshooting.
    """

    def __init__(self,
                 session_id: Optional[str] = None,
                 audit_dir: str = "audit_logs",
                 log_level: int = logging.INFO):
        """
        Initialize audit logger.

        Args:
            session_id: Unique session identifier (auto-generated if None)
            audit_dir: Directory to store audit files
            log_level: Logging level
        """
        self.session_id = session_id or f"upload_{int(time.time())}_{id(self)}"
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # Initialize session data
        self.session_summary = UploadSessionSummary(
            session_id=self.session_id,
            start_time=time.time()
        )

        self.file_records: Dict[str, List[FileUploadRecord]] = {}
        self.batch_records: List[BatchUploadRecord] = []
        self.current_batch: Optional[BatchUploadRecord] = None

        # Audit file paths - define before setting up logger
        self.session_file = self.audit_dir / f"{self.session_id}_session.json"
        self.detailed_log = self.audit_dir / f"{self.session_id}_detailed.log"
        self.summary_report = self.audit_dir / f"{self.session_id}_summary.json"

        # Setup logging
        self.logger = self._setup_logger(log_level)

        self.logger.info(f"Upload audit session started: {self.session_id}")

    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Setup structured logger for the session"""
        logger = logging.getLogger(f"upload_audit_{self.session_id}")
        logger.setLevel(log_level)

        # Clear any existing handlers
        logger.handlers.clear()

        # File handler for detailed logs
        file_handler = logging.FileHandler(self.detailed_log)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler for progress updates
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger

    def start_batch(self, batch_id: int, batch_size: int, total_bytes: int):
        """Start tracking a new batch"""
        self.current_batch = BatchUploadRecord(
            batch_id=batch_id,
            batch_size=batch_size,
            start_time=time.time(),
            total_bytes=total_bytes
        )

        self.logger.info(
            f"Starting batch {batch_id}: {batch_size} files, "
            f"{total_bytes:,} bytes ({total_bytes/1024/1024:.1f} MB)"
        )

    def end_batch(self):
        """Complete current batch tracking"""
        if self.current_batch:
            self.current_batch.end_time = time.time()
            self.batch_records.append(self.current_batch)

            duration = self.current_batch.duration()
            speed = self.current_batch.total_bytes / duration if duration else 0

            self.logger.info(
                f"Batch {self.current_batch.batch_id} completed: "
                f"{self.current_batch.successful_uploads}/{self.current_batch.batch_size} successful, "
                f"duration: {duration:.1f}s, speed: {speed/1024/1024:.1f} MB/s"
            )

            self.session_summary.total_batches += 1
            self.current_batch = None

    def log_file_upload_start(self, file_path: str, file_size: int, attempt_number: int = 1):
        """Log start of file upload"""
        record = FileUploadRecord(
            file_path=file_path,
            file_size=file_size,
            attempt_number=attempt_number,
            status=UploadStatus.IN_PROGRESS,
            start_time=time.time()
        )

        if file_path not in self.file_records:
            self.file_records[file_path] = []
        self.file_records[file_path].append(record)

        self.logger.debug(f"Upload started: {Path(file_path).name} ({file_size:,} bytes)")
        return record

    def log_file_upload_success(self, file_path: str, openai_file_id: str):
        """Log successful file upload"""
        records = self.file_records.get(file_path, [])
        if records:
            record = records[-1]  # Get latest attempt
            record.status = UploadStatus.SUCCESS
            record.end_time = time.time()
            record.openai_file_id = openai_file_id

            duration = record.duration()
            if duration and duration > 0:
                record.upload_speed_bytes_per_sec = record.file_size / duration

            # Only increment counters if this is the first success for this file
            has_previous_success = any(r.status == UploadStatus.SUCCESS for r in records[:-1])
            if not has_previous_success:
                self.session_summary.successful_files += 1
                self.session_summary.total_bytes += record.file_size

                if self.current_batch:
                    self.current_batch.successful_uploads += 1

            self.logger.debug(
                f"Upload success: {Path(file_path).name} -> {openai_file_id} "
                f"({duration:.2f}s, {record.upload_speed_bytes_per_sec/1024/1024:.1f} MB/s)"
            )

    def log_file_upload_failure(self, file_path: str, error: Exception, error_type: str = None):
        """Log failed file upload"""
        records = self.file_records.get(file_path, [])
        if records:
            record = records[-1]  # Get latest attempt
            record.status = UploadStatus.FAILED
            record.end_time = time.time()
            record.error_message = str(error)
            record.error_type = error_type or type(error).__name__

            # Only increment failed counter if this file doesn't already have a success
            has_success = any(r.status == UploadStatus.SUCCESS for r in records)
            if not has_success:
                # Only increment on first failure for this file or if it's the final failure
                has_previous_failure = any(r.status == UploadStatus.FAILED for r in records[:-1])
                if not has_previous_failure:
                    if self.current_batch:
                        self.current_batch.failed_uploads += 1

            self.logger.warning(
                f"Upload failed: {Path(file_path).name} - {record.error_type}: {record.error_message}"
            )

    def log_file_upload_retry(self, file_path: str, attempt_number: int):
        """Log file upload retry"""
        records = self.file_records.get(file_path, [])
        if records:
            records[-1].status = UploadStatus.RETRYING

        # Start new attempt record
        return self.log_file_upload_start(file_path,
                                        records[0].file_size if records else 0,
                                        attempt_number)

    def log_file_skipped(self, file_path: str, reason: str):
        """Log skipped file"""
        self.session_summary.skipped_files += 1
        self.logger.info(f"File skipped: {Path(file_path).name} - {reason}")

    def update_progress(self, completed: int, total: int, current_file: str = None):
        """Log progress update"""
        percentage = (completed / total * 100) if total > 0 else 0

        # Calculate ETA
        elapsed = time.time() - self.session_summary.start_time
        if completed > 0:
            eta_seconds = (elapsed / completed) * (total - completed)
            eta_str = f"ETA: {eta_seconds/60:.1f}m"
        else:
            eta_str = "ETA: calculating..."

        progress_msg = f"Progress: {completed}/{total} ({percentage:.1f}%) - {eta_str}"
        if current_file:
            progress_msg += f" - Current: {Path(current_file).name}"

        self.logger.info(progress_msg)

    def set_vector_store_info(self, vector_store_id: str, vector_store_name: str):
        """Set vector store information"""
        self.session_summary.vector_store_id = vector_store_id
        self.session_summary.vector_store_name = vector_store_name

        self.logger.info(f"Vector store created: {vector_store_name} ({vector_store_id})")

    def finalize_session(self):
        """Finalize the upload session and generate reports"""
        self.session_summary.end_time = time.time()

        # Recalculate final counts based on actual file outcomes
        successful_files = 0
        failed_files = 0
        total_bytes = 0

        for file_path, records in self.file_records.items():
            if records:
                # Get the final status for this file
                final_record = max(records, key=lambda r: r.start_time)

                if final_record.status == UploadStatus.SUCCESS:
                    successful_files += 1
                    total_bytes += final_record.file_size
                elif any(r.status == UploadStatus.SUCCESS for r in records):
                    # File eventually succeeded
                    successful_files += 1
                    success_record = next(r for r in records if r.status == UploadStatus.SUCCESS)
                    total_bytes += success_record.file_size
                else:
                    # File never succeeded
                    failed_files += 1

        # Update session summary with correct counts
        self.session_summary.successful_files = successful_files
        self.session_summary.failed_files = failed_files
        self.session_summary.total_bytes = total_bytes
        self.session_summary.total_files = (
            self.session_summary.successful_files +
            self.session_summary.failed_files +
            self.session_summary.skipped_files
        )

        # Generate comprehensive reports
        self._save_session_data()
        self._generate_summary_report()

        # Log final summary
        duration = self.session_summary.duration()
        avg_speed = self.session_summary.average_speed_bytes_per_sec()

        self.logger.info("=" * 60)
        self.logger.info("UPLOAD SESSION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Duration: {duration/60:.1f} minutes" if duration else "Duration: N/A")
        self.logger.info(f"Total files: {self.session_summary.total_files}")
        self.logger.info(f"Successful: {self.session_summary.successful_files}")
        self.logger.info(f"Failed: {self.session_summary.failed_files}")
        self.logger.info(f"Skipped: {self.session_summary.skipped_files}")
        self.logger.info(f"Success rate: {self.session_summary.success_rate():.1%}")
        self.logger.info(f"Total data: {self.session_summary.total_bytes/1024/1024:.1f} MB")
        if avg_speed:
            self.logger.info(f"Average speed: {avg_speed/1024/1024:.1f} MB/s")
        self.logger.info(f"Vector store: {self.session_summary.vector_store_id}")
        self.logger.info("=" * 60)

        return self.session_summary

    def _save_session_data(self):
        """Save detailed session data to JSON"""
        session_data = {
            'summary': self.session_summary.to_dict(),
            'batches': [batch.to_dict() for batch in self.batch_records],
            'file_records': {
                path: [record.to_dict() for record in records]
                for path, records in self.file_records.items()
            }
        }

        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        self.logger.info(f"Session data saved: {self.session_file}")

    def _generate_summary_report(self):
        """Generate human-readable summary report"""
        duration = self.session_summary.duration()
        avg_speed = self.session_summary.average_speed_bytes_per_sec()

        # Error analysis
        error_analysis = self._analyze_errors()
        performance_stats = self._calculate_performance_stats()

        report = {
            'session_info': {
                'session_id': self.session_id,
                'timestamp': datetime.fromtimestamp(self.session_summary.start_time, tz=timezone.utc).isoformat(),
                'duration_minutes': duration / 60 if duration else None,
                'vector_store_id': self.session_summary.vector_store_id,
                'vector_store_name': self.session_summary.vector_store_name
            },
            'upload_statistics': {
                'total_files': self.session_summary.total_files,
                'successful_files': self.session_summary.successful_files,
                'failed_files': self.session_summary.failed_files,
                'skipped_files': self.session_summary.skipped_files,
                'success_rate_percentage': self.session_summary.success_rate() * 100,
                'total_bytes': self.session_summary.total_bytes,
                'total_mb': self.session_summary.total_bytes / 1024 / 1024,
                'average_speed_mbps': avg_speed / 1024 / 1024 if avg_speed else None
            },
            'batch_statistics': {
                'total_batches': len(self.batch_records),
                'average_batch_duration_seconds': sum(b.duration() or 0 for b in self.batch_records) / len(self.batch_records) if self.batch_records else 0,
                'batch_success_rates': [b.success_rate() for b in self.batch_records]
            },
            'error_analysis': error_analysis,
            'performance_metrics': performance_stats,
            'audit_files': {
                'detailed_log': str(self.detailed_log),
                'session_data': str(self.session_file),
                'summary_report': str(self.summary_report)
            }
        }

        with open(self.summary_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Summary report generated: {self.summary_report}")
        return report

    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze upload errors and categorize them"""
        error_types = {}
        failed_files = []

        for file_path, records in self.file_records.items():
            failed_records = [r for r in records if r.status == UploadStatus.FAILED]
            if failed_records:
                latest_failure = failed_records[-1]
                error_type = latest_failure.error_type or "Unknown"

                if error_type not in error_types:
                    error_types[error_type] = 0
                error_types[error_type] += 1

                failed_files.append({
                    'file_path': file_path,
                    'error_type': error_type,
                    'error_message': latest_failure.error_message,
                    'attempts': len(records)
                })

        return {
            'error_types': error_types,
            'failed_files': failed_files,
            'total_errors': len(failed_files)
        }

    def _calculate_performance_stats(self) -> Dict[str, Any]:
        """Calculate detailed performance statistics"""
        successful_records = []
        for records in self.file_records.values():
            successful_records.extend([r for r in records if r.status == UploadStatus.SUCCESS])

        if not successful_records:
            return {'message': 'No successful uploads for analysis'}

        durations = [r.duration() for r in successful_records if r.duration()]
        speeds = [r.upload_speed_bytes_per_sec for r in successful_records if r.upload_speed_bytes_per_sec]
        file_sizes = [r.file_size for r in successful_records]

        return {
            'upload_durations': {
                'min_seconds': min(durations) if durations else None,
                'max_seconds': max(durations) if durations else None,
                'average_seconds': sum(durations) / len(durations) if durations else None,
            },
            'upload_speeds': {
                'min_mbps': min(speeds) / 1024 / 1024 if speeds else None,
                'max_mbps': max(speeds) / 1024 / 1024 if speeds else None,
                'average_mbps': sum(speeds) / len(speeds) / 1024 / 1024 if speeds else None,
            },
            'file_sizes': {
                'min_bytes': min(file_sizes) if file_sizes else None,
                'max_bytes': max(file_sizes) if file_sizes else None,
                'average_bytes': sum(file_sizes) / len(file_sizes) if file_sizes else None,
                'total_bytes': sum(file_sizes) if file_sizes else None
            }
        }