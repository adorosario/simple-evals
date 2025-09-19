"""
Comprehensive Audit Logging System for RAG Benchmark
Logs all API calls, timing metrics, and responses for full auditability
"""

import os
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TimingMetrics:
    """Detailed timing metrics for API calls"""
    start_time: float
    first_byte_time: Optional[float]  # TTFB - Time to First Byte
    end_time: float
    total_duration: float  # TTLB - Time to Last Byte
    request_preparation_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time_unix': self.start_time,
            'start_time_iso': datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
            'first_byte_time_unix': self.first_byte_time,
            'first_byte_time_iso': datetime.fromtimestamp(self.first_byte_time, tz=timezone.utc).isoformat() if self.first_byte_time else None,
            'end_time_unix': self.end_time,
            'end_time_iso': datetime.fromtimestamp(self.end_time, tz=timezone.utc).isoformat(),
            'total_duration_seconds': self.total_duration,
            'ttfb_seconds': self.first_byte_time - self.start_time if self.first_byte_time else None,
            'ttlb_seconds': self.total_duration,
            'request_preparation_seconds': self.request_preparation_time
        }


@dataclass
class APICallRecord:
    """Complete record of an API call for audit purposes"""
    call_id: str
    run_id: str
    sampler_name: str
    question_id: str
    question_text: str
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]]
    response_text: Optional[str]
    timing_metrics: TimingMetrics
    success: bool
    error_message: Optional[str]
    retry_count: int
    api_endpoint: str
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'call_id': self.call_id,
            'run_id': self.run_id,
            'sampler_name': self.sampler_name,
            'question_id': self.question_id,
            'question_text': self.question_text,
            'request_data': self.request_data,
            'response_data': self.response_data,
            'response_text': self.response_text,
            'timing_metrics': self.timing_metrics.to_dict(),
            'success': self.success,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'api_endpoint': self.api_endpoint,
            'model_used': self.model_used,
            'metadata': {
                'logged_at': datetime.now(tz=timezone.utc).isoformat(),
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                'audit_version': '1.0'
            }
        }


class AuditLogger:
    """Comprehensive audit logger for benchmark runs"""

    def __init__(self, run_id: str, output_dir: str = "audit_logs"):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.run_dir = self.output_dir / f"run_{run_id}"

        # Create directory structure
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "api_calls").mkdir(exist_ok=True)
        (self.run_dir / "responses").mkdir(exist_ok=True)
        (self.run_dir / "errors").mkdir(exist_ok=True)

        self.api_calls: List[APICallRecord] = []
        self.run_metadata = {
            'run_id': run_id,
            'start_time': time.time(),
            'start_time_iso': datetime.now(tz=timezone.utc).isoformat(),
            'environment': self._capture_environment(),
            'samplers_tested': [],
            'total_questions': 0,
            'audit_version': '1.0'
        }

        # Save initial metadata
        self._save_run_metadata()

    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environment details for audit trail"""
        return {
            'openai_version': self._get_package_version('openai'),
            'anthropic_version': self._get_package_version('anthropic'),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'environment_variables': {
                'has_openai_key': bool(os.environ.get('OPENAI_API_KEY')),
                'has_customgpt_key': bool(os.environ.get('CUSTOMGPT_API_KEY')),
                'has_vector_store_id': bool(os.environ.get('OPENAI_VECTOR_STORE_ID')),
                'openai_vector_store_id': os.environ.get('OPENAI_VECTOR_STORE_ID', 'NOT_SET')
            }
        }

    def _get_package_version(self, package_name: str) -> str:
        """Get package version safely"""
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except:
            return "unknown"

    def start_api_call(self, sampler_name: str, question_id: str, question_text: str,
                      request_data: Dict[str, Any], api_endpoint: str, model_used: str) -> str:
        """Start timing an API call and return call_id for tracking"""
        call_id = str(uuid.uuid4())

        # Log the start of API call
        call_start_record = {
            'call_id': call_id,
            'run_id': self.run_id,
            'sampler_name': sampler_name,
            'question_id': question_id,
            'question_text': question_text,
            'request_data': request_data,
            'api_endpoint': api_endpoint,
            'model_used': model_used,
            'started_at': datetime.now(tz=timezone.utc).isoformat(),
            'start_time_unix': time.time()
        }

        # Save individual call start record
        call_file = self.run_dir / "api_calls" / f"call_{call_id}_start.json"
        with open(call_file, 'w') as f:
            json.dump(call_start_record, f, indent=2)

        return call_id

    def log_first_byte(self, call_id: str):
        """Log when first byte is received (TTFB)"""
        ttfb_record = {
            'call_id': call_id,
            'first_byte_time': time.time(),
            'first_byte_time_iso': datetime.now(tz=timezone.utc).isoformat()
        }

        ttfb_file = self.run_dir / "api_calls" / f"call_{call_id}_ttfb.json"
        with open(ttfb_file, 'w') as f:
            json.dump(ttfb_record, f, indent=2)

    def complete_api_call(self, call_id: str, response_data: Optional[Dict[str, Any]],
                         response_text: Optional[str], success: bool,
                         error_message: Optional[str] = None, retry_count: int = 0):
        """Complete an API call record"""

        # Load the start record to get timing info
        start_file = self.run_dir / "api_calls" / f"call_{call_id}_start.json"
        ttfb_file = self.run_dir / "api_calls" / f"call_{call_id}_ttfb.json"

        start_record = {}
        if start_file.exists():
            with open(start_file, 'r') as f:
                start_record = json.load(f)

        ttfb_record = {}
        if ttfb_file.exists():
            with open(ttfb_file, 'r') as f:
                ttfb_record = json.load(f)

        end_time = time.time()
        start_time = start_record.get('start_time_unix', end_time)
        first_byte_time = ttfb_record.get('first_byte_time')

        # Create timing metrics
        timing_metrics = TimingMetrics(
            start_time=start_time,
            first_byte_time=first_byte_time,
            end_time=end_time,
            total_duration=end_time - start_time,
            request_preparation_time=0.1  # Estimate - could be measured more precisely
        )

        # Create complete API call record
        api_call_record = APICallRecord(
            call_id=call_id,
            run_id=self.run_id,
            sampler_name=start_record.get('sampler_name', 'unknown'),
            question_id=start_record.get('question_id', 'unknown'),
            question_text=start_record.get('question_text', ''),
            request_data=start_record.get('request_data', {}),
            response_data=response_data,
            response_text=response_text,
            timing_metrics=timing_metrics,
            success=success,
            error_message=error_message,
            retry_count=retry_count,
            api_endpoint=start_record.get('api_endpoint', 'unknown'),
            model_used=start_record.get('model_used', 'unknown')
        )

        # Save complete record
        complete_file = self.run_dir / "api_calls" / f"call_{call_id}_complete.json"
        with open(complete_file, 'w') as f:
            json.dump(api_call_record.to_dict(), f, indent=2)

        # Save response separately for easy access
        if response_text:
            response_file = self.run_dir / "responses" / f"response_{call_id}.txt"
            with open(response_file, 'w') as f:
                f.write(response_text)

        # Save error separately if there was one
        if error_message:
            error_file = self.run_dir / "errors" / f"error_{call_id}.json"
            error_record = {
                'call_id': call_id,
                'error_message': error_message,
                'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                'sampler_name': api_call_record.sampler_name,
                'question_id': api_call_record.question_id
            }
            with open(error_file, 'w') as f:
                json.dump(error_record, f, indent=2)

        # Add to in-memory list
        self.api_calls.append(api_call_record)

        return api_call_record

    def add_sampler(self, sampler_name: str, sampler_config: Dict[str, Any]):
        """Record sampler configuration"""
        self.run_metadata['samplers_tested'].append({
            'name': sampler_name,
            'config': sampler_config,
            'added_at': datetime.now(tz=timezone.utc).isoformat()
        })
        self._save_run_metadata()

    def set_total_questions(self, count: int):
        """Set total number of questions for the run"""
        self.run_metadata['total_questions'] = count
        self._save_run_metadata()

    def _save_run_metadata(self):
        """Save run metadata"""
        metadata_file = self.run_dir / "run_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.run_metadata, f, indent=2)

    def generate_final_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        end_time = time.time()

        # Calculate aggregate statistics
        successful_calls = [call for call in self.api_calls if call.success]
        failed_calls = [call for call in self.api_calls if not call.success]

        sampler_stats = {}
        for call in self.api_calls:
            if call.sampler_name not in sampler_stats:
                sampler_stats[call.sampler_name] = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'total_duration': 0,
                    'avg_ttfb': [],
                    'avg_ttlb': [],
                    'errors': []
                }

            stats = sampler_stats[call.sampler_name]
            stats['total_calls'] += 1

            if call.success:
                stats['successful_calls'] += 1
                stats['total_duration'] += call.timing_metrics.total_duration
                stats['avg_ttlb'].append(call.timing_metrics.total_duration)
                if call.timing_metrics.first_byte_time:
                    ttfb = call.timing_metrics.first_byte_time - call.timing_metrics.start_time
                    stats['avg_ttfb'].append(ttfb)
            else:
                stats['failed_calls'] += 1
                if call.error_message:
                    stats['errors'].append(call.error_message)

        # Calculate averages
        for sampler_name, stats in sampler_stats.items():
            if stats['avg_ttfb']:
                stats['avg_ttfb_seconds'] = sum(stats['avg_ttfb']) / len(stats['avg_ttfb'])
            else:
                stats['avg_ttfb_seconds'] = None

            if stats['avg_ttlb']:
                stats['avg_ttlb_seconds'] = sum(stats['avg_ttlb']) / len(stats['avg_ttlb'])
            else:
                stats['avg_ttlb_seconds'] = None

            # Remove raw lists to clean up report
            del stats['avg_ttfb']
            del stats['avg_ttlb']

        audit_report = {
            'audit_metadata': {
                'report_version': '1.0',
                'generated_at': datetime.now(tz=timezone.utc).isoformat(),
                'run_id': self.run_id,
                'auditor_instructions': {
                    'description': 'This audit report contains complete logs of all API calls made during the benchmark',
                    'verification_steps': [
                        'Check run_metadata.json for environment and configuration',
                        'Verify all API calls in api_calls/ directory',
                        'Review timing_metrics for each call',
                        'Check error logs in errors/ directory',
                        'Validate response integrity in responses/ directory'
                    ],
                    'integrity_checks': [
                        'Verify call_ids are unique across all calls',
                        'Check timing consistency (end_time > start_time)',
                        'Validate retry counts and error handling',
                        'Confirm response completeness for successful calls'
                    ]
                }
            },
            'run_summary': {
                'run_id': self.run_id,
                'start_time': self.run_metadata['start_time'],
                'end_time': end_time,
                'total_duration_seconds': end_time - self.run_metadata['start_time'],
                'environment': self.run_metadata['environment'],
                'samplers_tested': self.run_metadata['samplers_tested'],
                'total_questions_planned': self.run_metadata['total_questions'],
                'total_api_calls_made': len(self.api_calls)
            },
            'aggregate_statistics': {
                'total_calls': len(self.api_calls),
                'successful_calls': len(successful_calls),
                'failed_calls': len(failed_calls),
                'success_rate': len(successful_calls) / len(self.api_calls) if self.api_calls else 0,
                'sampler_breakdown': sampler_stats
            },
            'data_locations': {
                'run_directory': str(self.run_dir),
                'api_calls_directory': str(self.run_dir / "api_calls"),
                'responses_directory': str(self.run_dir / "responses"),
                'errors_directory': str(self.run_dir / "errors"),
                'run_metadata_file': str(self.run_dir / "run_metadata.json")
            },
            'audit_trail': {
                'all_call_ids': [call.call_id for call in self.api_calls],
                'failed_call_ids': [call.call_id for call in failed_calls],
                'retry_summary': {
                    'calls_with_retries': len([call for call in self.api_calls if call.retry_count > 0]),
                    'max_retries_used': max([call.retry_count for call in self.api_calls]) if self.api_calls else 0
                }
            }
        }

        # Save final audit report
        audit_file = self.run_dir / "final_audit_report.json"
        with open(audit_file, 'w') as f:
            json.dump(audit_report, f, indent=2)

        return audit_report