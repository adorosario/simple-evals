"""
Enhanced OpenAI Vector Store Manager

High-performance, resilient upload manager for OpenAI vector stores with:
- Concurrent file uploads with rate limiting
- Intelligent batching and retry logic
- Resume capability from checkpoints
- Comprehensive progress tracking and auditing
- Built-in integrity validation
"""

import os
import json
import time
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
import pickle

import openai
from openai import OpenAI

from .upload_audit_logger import UploadAuditLogger

logger = logging.getLogger(__name__)

@dataclass
class UploadConfig:
    """Configuration for enhanced upload process"""
    # Concurrency settings
    max_workers: int = 10  # Concurrent upload threads
    batch_size: int = 50   # Files per batch

    # Rate limiting (requests per minute)
    max_requests_per_minute: int = 1000

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True

    # Timeout settings
    upload_timeout_seconds: int = 300  # 5 minutes per file

    # Progress reporting
    progress_callback_interval: int = 10  # Report progress every N files

    # Resume capability
    enable_checkpoints: bool = True
    checkpoint_interval: int = 100  # Save checkpoint every N files

    # File validation
    validate_file_hashes: bool = True
    min_file_size_bytes: int = 1  # Minimum file size
    max_file_size_bytes: int = 50 * 1024 * 1024  # 50MB OpenAI limit

    def __post_init__(self):
        """Validate configuration"""
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

@dataclass
class FileUploadItem:
    """Individual file upload item with metadata"""
    file_path: str
    file_size: int
    file_hash: Optional[str] = None
    upload_attempts: int = 0
    openai_file_id: Optional[str] = None
    upload_status: str = "pending"  # pending, uploading, success, failed, skipped
    last_error: Optional[str] = None
    upload_start_time: Optional[float] = None
    upload_end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileUploadItem':
        return cls(**data)

    def duration(self) -> Optional[float]:
        if self.upload_start_time and self.upload_end_time:
            return self.upload_end_time - self.upload_start_time
        return None

@dataclass
class UploadSession:
    """Upload session state for resume capability"""
    session_id: str
    config: UploadConfig
    total_files: int
    completed_files: List[FileUploadItem]
    pending_files: List[FileUploadItem]
    failed_files: List[FileUploadItem]
    vector_store_id: Optional[str] = None
    session_start_time: float = 0
    last_checkpoint_time: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'config': asdict(self.config),
            'total_files': self.total_files,
            'completed_files': [f.to_dict() for f in self.completed_files],
            'pending_files': [f.to_dict() for f in self.pending_files],
            'failed_files': [f.to_dict() for f in self.failed_files],
            'vector_store_id': self.vector_store_id,
            'session_start_time': self.session_start_time,
            'last_checkpoint_time': self.last_checkpoint_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UploadSession':
        return cls(
            session_id=data['session_id'],
            config=UploadConfig(**data['config']),
            total_files=data['total_files'],
            completed_files=[FileUploadItem.from_dict(f) for f in data['completed_files']],
            pending_files=[FileUploadItem.from_dict(f) for f in data['pending_files']],
            failed_files=[FileUploadItem.from_dict(f) for f in data['failed_files']],
            vector_store_id=data.get('vector_store_id'),
            session_start_time=data.get('session_start_time', 0),
            last_checkpoint_time=data.get('last_checkpoint_time', 0)
        )

class RateLimiter:
    """Rate limiter for API requests"""

    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = Lock()

    def acquire(self) -> bool:
        """Acquire permission to make a request"""
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

    def wait_time(self) -> float:
        """Get time to wait before next request is allowed"""
        with self.lock:
            if not self.requests:
                return 0.0
            oldest_request = min(self.requests)
            return max(0, 60 - (time.time() - oldest_request))

class EnhancedOpenAIVectorStoreManager:
    """
    Enhanced OpenAI vector store manager with concurrent uploads,
    intelligent batching, error recovery, and comprehensive auditing.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 config: Optional[UploadConfig] = None,
                 audit_logger: Optional[UploadAuditLogger] = None,
                 progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """
        Initialize enhanced vector store manager.

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            config: Upload configuration
            audit_logger: Optional audit logger instance
            progress_callback: Optional progress callback function
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        if not self.client.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.config = config or UploadConfig()
        self.audit_logger = audit_logger
        self.progress_callback = progress_callback

        # Rate limiter for API requests
        self.rate_limiter = RateLimiter(self.config.max_requests_per_minute)

        # Thread pool for concurrent uploads
        self.executor = None
        self.upload_semaphore = Semaphore(self.config.max_workers)

        # Session state
        self.current_session: Optional[UploadSession] = None
        self.session_lock = Lock()

        logger.info(f"Enhanced OpenAI Vector Store Manager initialized")
        logger.info(f"  Max workers: {self.config.max_workers}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Rate limit: {self.config.max_requests_per_minute} req/min")

    def __enter__(self):
        """Context manager entry"""
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.executor:
            self.executor.shutdown(wait=True)

    def create_upload_session(self,
                             file_paths: List[str],
                             session_id: Optional[str] = None) -> UploadSession:
        """
        Create a new upload session with file validation and preparation.

        Args:
            file_paths: List of file paths to upload
            session_id: Optional session ID (auto-generated if None)

        Returns:
            UploadSession object
        """
        session_id = session_id or f"upload_{int(time.time())}_{id(self)}"

        logger.info(f"Creating upload session: {session_id}")
        logger.info(f"Validating {len(file_paths)} files...")

        # Validate and prepare files
        upload_items = []
        skipped_files = []

        for file_path in file_paths:
            try:
                item = self._validate_and_prepare_file(file_path)
                if item:
                    upload_items.append(item)
                else:
                    skipped_files.append(file_path)
            except Exception as e:
                logger.warning(f"Skipping invalid file {file_path}: {e}")
                skipped_files.append(file_path)

        if skipped_files:
            logger.warning(f"Skipped {len(skipped_files)} invalid files")

        if not upload_items:
            raise ValueError("No valid files found for upload")

        # Create session
        session = UploadSession(
            session_id=session_id,
            config=self.config,
            total_files=len(upload_items),
            completed_files=[],
            pending_files=upload_items,
            failed_files=[],
            session_start_time=time.time()
        )

        logger.info(f"Upload session created: {len(upload_items)} files ready")
        return session

    def _validate_and_prepare_file(self, file_path: str) -> Optional[FileUploadItem]:
        """Validate and prepare a single file for upload"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        file_size = path.stat().st_size

        if file_size < self.config.min_file_size_bytes:
            raise ValueError(f"File too small: {file_size} bytes")

        if file_size > self.config.max_file_size_bytes:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.config.max_file_size_bytes})")

        # Calculate file hash for validation
        file_hash = None
        if self.config.validate_file_hashes:
            try:
                file_hash = self._calculate_file_hash(file_path)
            except Exception as e:
                logger.warning(f"Could not calculate hash for {file_path}: {e}")

        return FileUploadItem(
            file_path=str(path.absolute()),
            file_size=file_size,
            file_hash=file_hash
        )

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for validation"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def upload_files_concurrent(self,
                               file_paths: List[str],
                               store_name: str,
                               session_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Upload files concurrently and create vector store.

        Args:
            file_paths: List of file paths to upload
            store_name: Name for the vector store
            session_id: Optional session ID for resume capability

        Returns:
            Tuple of (vector_store_id, upload_report)
        """
        # Try to resume existing session or create new one
        session = self._load_or_create_session(file_paths, session_id)

        if not self.executor:
            raise RuntimeError("Manager must be used as context manager")

        try:
            # Upload files in batches
            self._upload_files_in_batches(session)

            # Create vector store if files were uploaded successfully
            if session.completed_files:
                vector_store_id = self._create_vector_store_from_session(session, store_name)
                session.vector_store_id = vector_store_id
            else:
                raise Exception("No files were successfully uploaded")

            # Generate final report
            report = self._generate_upload_report(session)

            # Save final session state
            if self.config.enable_checkpoints:
                self._save_session_checkpoint(session)

            logger.info(f"Upload completed: {vector_store_id}")
            return vector_store_id, report

        except Exception as e:
            logger.error(f"Upload failed: {e}")

            # Save session state for potential resume
            if self.config.enable_checkpoints:
                self._save_session_checkpoint(session)

            raise

    def _load_or_create_session(self,
                               file_paths: List[str],
                               session_id: Optional[str]) -> UploadSession:
        """Load existing session or create new one"""
        if session_id and self.config.enable_checkpoints:
            try:
                session = self._load_session_checkpoint(session_id)
                logger.info(f"Resumed session {session_id}: "
                          f"{len(session.completed_files)} completed, "
                          f"{len(session.pending_files)} pending")
                return session
            except Exception as e:
                logger.warning(f"Could not resume session {session_id}: {e}")

        # Create new session
        return self.create_upload_session(file_paths, session_id)

    def _upload_files_in_batches(self, session: UploadSession):
        """Upload files in concurrent batches"""
        batch_number = 0

        while session.pending_files:
            batch_number += 1

            # Get next batch
            batch_size = min(self.config.batch_size, len(session.pending_files))
            current_batch = session.pending_files[:batch_size]
            session.pending_files = session.pending_files[batch_size:]

            # Calculate batch metrics for audit logging
            batch_total_bytes = sum(item.file_size for item in current_batch)

            if self.audit_logger:
                self.audit_logger.start_batch(batch_number, len(current_batch), batch_total_bytes)

            logger.info(f"Starting batch {batch_number}: {len(current_batch)} files "
                       f"({batch_total_bytes:,} bytes)")

            # Upload batch concurrently
            self._upload_batch_concurrent(session, current_batch, batch_number)

            if self.audit_logger:
                self.audit_logger.end_batch()

            # Save checkpoint periodically
            if (self.config.enable_checkpoints and
                len(session.completed_files) % self.config.checkpoint_interval == 0):
                self._save_session_checkpoint(session)

    def _upload_batch_concurrent(self,
                                session: UploadSession,
                                batch: List[FileUploadItem],
                                batch_number: int):
        """Upload a batch of files concurrently"""
        futures = []

        # Submit upload tasks
        for item in batch:
            future = self.executor.submit(self._upload_single_file_with_retry, session, item)
            futures.append(future)

        # Process completed uploads
        for future in as_completed(futures):
            try:
                item = future.result()

                with self.session_lock:
                    if item.upload_status == "success":
                        session.completed_files.append(item)
                    else:
                        session.failed_files.append(item)

                # Report progress
                completed_count = len(session.completed_files)
                total_count = session.total_files

                if self.progress_callback:
                    self.progress_callback(completed_count, total_count, item.file_path)

                if self.audit_logger:
                    self.audit_logger.update_progress(
                        completed_count,
                        total_count,
                        Path(item.file_path).name
                    )

            except Exception as e:
                logger.error(f"Batch upload error: {e}")

    def _upload_single_file_with_retry(self,
                                      session: UploadSession,
                                      item: FileUploadItem) -> FileUploadItem:
        """Upload a single file with retry logic"""
        for attempt in range(self.config.max_retries + 1):
            try:
                # Wait for rate limiter
                self._wait_for_rate_limit()

                # Acquire upload semaphore
                with self.upload_semaphore:
                    return self._upload_single_file(item, attempt + 1)

            except Exception as e:
                item.upload_attempts = attempt + 1
                item.last_error = str(e)

                logger.warning(
                    f"Upload attempt {attempt + 1} failed for {Path(item.file_path).name}: {e}"
                )

                if self.audit_logger:
                    if attempt == 0:
                        # First attempt
                        self.audit_logger.log_file_upload_failure(item.file_path, e)
                    elif attempt < self.config.max_retries:
                        # Retry attempt
                        self.audit_logger.log_file_upload_retry(item.file_path, attempt + 2)
                    else:
                        # Final failure
                        self.audit_logger.log_file_upload_failure(item.file_path, e)

                if attempt < self.config.max_retries:
                    # Calculate retry delay with exponential backoff
                    delay = self.config.retry_delay_seconds
                    if self.config.exponential_backoff:
                        delay *= (2 ** attempt)

                    logger.debug(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    # Final failure
                    item.upload_status = "failed"
                    logger.error(f"Upload failed permanently: {Path(item.file_path).name}")
                    return item

        return item

    def _upload_single_file(self, item: FileUploadItem, attempt_number: int) -> FileUploadItem:
        """Upload a single file to OpenAI"""
        item.upload_status = "uploading"
        item.upload_start_time = time.time()

        if self.audit_logger:
            if attempt_number == 1:
                self.audit_logger.log_file_upload_start(item.file_path, item.file_size)
            else:
                self.audit_logger.log_file_upload_retry(item.file_path, attempt_number)

        try:
            with open(item.file_path, 'rb') as file:
                response = self.client.files.create(
                    file=file,
                    purpose="assistants"
                )

                item.openai_file_id = response.id
                item.upload_status = "success"
                item.upload_end_time = time.time()

                if self.audit_logger:
                    self.audit_logger.log_file_upload_success(item.file_path, response.id)

                logger.debug(f"Uploaded {Path(item.file_path).name} -> {response.id}")
                return item

        except Exception as e:
            item.upload_end_time = time.time()
            raise e

    def _wait_for_rate_limit(self):
        """Wait if rate limit is exceeded"""
        while not self.rate_limiter.acquire():
            wait_time = self.rate_limiter.wait_time()
            if wait_time > 0:
                logger.debug(f"Rate limit hit, waiting {wait_time:.1f}s...")
                time.sleep(min(wait_time, 1.0))  # Wait max 1 second at a time

    def _create_vector_store_from_session(self, session: UploadSession, store_name: str) -> str:
        """Create vector store from successfully uploaded files"""
        successful_file_ids = [item.openai_file_id for item in session.completed_files
                              if item.openai_file_id]

        if not successful_file_ids:
            raise Exception("No files were successfully uploaded")

        logger.info(f"Creating vector store '{store_name}' with {len(successful_file_ids)} files...")

        try:
            # Create empty vector store first (OpenAI limits initial file_ids to 100)
            vector_store = self.client.vector_stores.create(name=store_name)
            logger.info(f"Vector store created: {vector_store.id}")

            # Add files in very conservative batches due to server errors
            batch_size = min(5, len(successful_file_ids))
            total_batches = (len(successful_file_ids) + batch_size - 1) // batch_size

            for i in range(0, len(successful_file_ids), batch_size):
                batch_num = i // batch_size + 1
                batch_file_ids = successful_file_ids[i:i + batch_size]

                logger.info(f"Adding batch {batch_num}/{total_batches}: {len(batch_file_ids)} files to vector store")

                # Add files to vector store using file_batches endpoint
                file_batch = self.client.vector_stores.file_batches.create(
                    vector_store_id=vector_store.id,
                    file_ids=batch_file_ids
                )

                # Wait for this batch to be processed
                self._wait_for_file_batch_processing(vector_store.id, file_batch.id)

                # Add delay between batches to reduce API strain
                if batch_num < total_batches:  # Don't delay after the last batch
                    delay_seconds = 10
                    logger.info(f"Waiting {delay_seconds}s before next batch...")
                    time.sleep(delay_seconds)

            # Wait for overall processing
            logger.info("Waiting for vector store processing...")
            processed_store = self._wait_for_vector_store_processing(vector_store.id)

            if self.audit_logger:
                self.audit_logger.set_vector_store_info(vector_store.id, store_name)

            return vector_store.id

        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    def _wait_for_file_batch_processing(self, vector_store_id: str, batch_id: str, timeout: int = 600) -> None:
        """Wait for file batch processing to complete with resilience for server errors"""
        start_time = time.time()
        last_status = None
        server_error_count = 0
        max_server_errors = 5

        while time.time() - start_time < timeout:
            try:
                batch = self.client.vector_stores.file_batches.retrieve(
                    vector_store_id=vector_store_id,
                    batch_id=batch_id
                )

                # Reset server error count on successful API call
                server_error_count = 0

                if batch.status != last_status:
                    logger.info(f"File batch status: {batch.status}")
                    last_status = batch.status

                if batch.status == 'completed':
                    duration = time.time() - start_time
                    logger.info(f"File batch processing completed in {duration:.1f}s")
                    return
                elif batch.status == 'failed':
                    raise Exception(f"File batch processing failed")
                elif batch.status in ['in_progress', 'pending']:
                    time.sleep(5)  # Check every 5 seconds for batches
                else:
                    logger.warning(f"Unknown file batch status: {batch.status}")
                    time.sleep(5)

            except Exception as e:
                error_str = str(e).lower()

                # Check if it's a server error (5xx)
                if any(error_type in error_str for error_type in ['500', '502', '503', '504', 'internal server error', 'bad gateway', 'service unavailable', 'gateway timeout']):
                    server_error_count += 1
                    logger.warning(f"Server error #{server_error_count}/{max_server_errors} for batch {batch_id}: {e}")

                    if server_error_count >= max_server_errors:
                        logger.error(f"Too many server errors ({server_error_count}) for batch {batch_id}, giving up")
                        raise Exception(f"File batch processing failed due to persistent server errors")

                    # Exponential backoff for server errors
                    backoff_delay = min(30, 5 * (2 ** server_error_count))
                    logger.info(f"Waiting {backoff_delay}s before retry due to server error...")
                    time.sleep(backoff_delay)
                else:
                    # Non-server error, log and continue with shorter delay
                    logger.error(f"Error checking file batch status: {e}")
                    time.sleep(5)

        raise Exception(f"File batch processing timeout after {timeout}s")

    def _wait_for_vector_store_processing(self, vector_store_id: str, timeout: int = 1800) -> Any:
        """Wait for vector store processing to complete"""
        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout:
            try:
                store = self.client.vector_stores.retrieve(vector_store_id)

                if store.status != last_status:
                    logger.info(f"Vector store status: {store.status}")
                    last_status = store.status

                if store.status == 'completed':
                    duration = time.time() - start_time
                    logger.info(f"Vector store processing completed in {duration:.1f}s")
                    return store
                elif store.status == 'failed':
                    raise Exception(f"Vector store processing failed")
                elif store.status in ['in_progress', 'pending']:
                    time.sleep(10)  # Check every 10 seconds
                else:
                    logger.warning(f"Unknown vector store status: {store.status}")
                    time.sleep(10)

            except Exception as e:
                logger.error(f"Error checking vector store status: {e}")
                time.sleep(10)

        raise TimeoutError(f"Vector store processing timed out after {timeout} seconds")

    def _generate_upload_report(self, session: UploadSession) -> Dict[str, Any]:
        """Generate comprehensive upload report"""
        successful_files = len(session.completed_files)
        failed_files = len(session.failed_files)
        total_files = session.total_files

        total_bytes = sum(item.file_size for item in session.completed_files)
        duration = time.time() - session.session_start_time

        # Calculate speed metrics
        avg_speed_bytes_per_sec = total_bytes / duration if duration > 0 else 0

        # Error analysis
        error_types = {}
        for item in session.failed_files:
            error_type = type(Exception(item.last_error or "Unknown")).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            'session_id': session.session_id,
            'vector_store_id': session.vector_store_id,
            'duration_seconds': duration,
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'success_rate': successful_files / total_files if total_files > 0 else 0,
            'total_bytes': total_bytes,
            'average_speed_mbps': avg_speed_bytes_per_sec / (1024 * 1024),
            'error_types': error_types,
            'failed_file_paths': [item.file_path for item in session.failed_files],
            'config': asdict(session.config)
        }

    def _save_session_checkpoint(self, session: UploadSession):
        """Save session state for resume capability"""
        try:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)

            checkpoint_file = checkpoint_dir / f"{session.session_id}.checkpoint"

            with open(checkpoint_file, 'wb') as f:
                pickle.dump(session.to_dict(), f)

            session.last_checkpoint_time = time.time()
            logger.debug(f"Checkpoint saved: {checkpoint_file}")

        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _load_session_checkpoint(self, session_id: str) -> UploadSession:
        """Load session state from checkpoint"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_file = checkpoint_dir / f"{session_id}.checkpoint"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

        with open(checkpoint_file, 'rb') as f:
            session_data = pickle.load(f)

        return UploadSession.from_dict(session_data)

    def resume_upload(self, session_id: str, store_name: str) -> Tuple[str, Dict[str, Any]]:
        """Resume an interrupted upload session"""
        logger.info(f"Resuming upload session: {session_id}")

        session = self._load_session_checkpoint(session_id)

        if not self.executor:
            raise RuntimeError("Manager must be used as context manager")

        try:
            # Continue uploading remaining files
            if session.pending_files:
                self._upload_files_in_batches(session)

            # Create vector store if not already created
            if not session.vector_store_id and session.completed_files:
                vector_store_id = self._create_vector_store_from_session(session, store_name)
                session.vector_store_id = vector_store_id

            # Generate final report
            report = self._generate_upload_report(session)

            # Save final state
            self._save_session_checkpoint(session)

            logger.info(f"Resume completed: {session.vector_store_id}")
            return session.vector_store_id, report

        except Exception as e:
            logger.error(f"Resume failed: {e}")
            self._save_session_checkpoint(session)
            raise