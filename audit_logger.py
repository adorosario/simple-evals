"""
Comprehensive Audit Logging System for Multi-Provider Benchmark
Logs every provider request/response with full traceability
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class AuditLogger:
    """
    Comprehensive audit logger for tracking all provider interactions
    Provides complete traceability for debugging and analysis
    """

    def __init__(self, run_id: str, output_dir: str = "results"):
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.run_dir = self.output_dir / f"run_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create separate log files for different types of events
        self.provider_log_file = self.run_dir / "provider_requests.jsonl"
        self.judge_log_file = self.run_dir / "judge_evaluations.jsonl"
        self.meta_log_file = self.run_dir / "run_metadata.json"

        # Initialize run metadata
        self.run_metadata = {
            "run_id": run_id,
            "start_time": datetime.utcnow().isoformat(),
            "providers": [],
            "total_questions": 0,
            "completed_questions": 0,
            "errors": []
        }

        self.save_metadata()

    def log_provider_request(
        self,
        provider: str,
        question_id: str,
        question: str,
        request_data: Dict[str, Any],
        response: str,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a provider request with complete context"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            "provider": provider,
            "question_id": question_id,
            "question": question,
            "request": {
                "model": request_data.get("model"),
                "temperature": request_data.get("temperature"),
                "max_tokens": request_data.get("max_tokens"),
                "messages": request_data.get("messages", []),
                "system_message": request_data.get("system_message"),
                "additional_params": {k: v for k, v in request_data.items()
                                   if k not in ["model", "temperature", "max_tokens", "messages", "system_message"]}
            },
            "response": {
                "content": response,
                "latency_ms": latency_ms,
                "char_count": len(response),
                "word_count": len(response.split()) if response else 0
            },
            "metadata": metadata or {}
        }

        # Write to provider log file
        with open(self.provider_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_judge_evaluation(
        self,
        question_id: str,
        question: str,
        target_answer: str,
        provider_responses: Dict[str, str],
        judge_prompt: str,
        judge_response: str,
        grades: Dict[str, str],
        reasoning: str,
        latency_ms: float,
        judge_model_config: Dict[str, any] = None,
        metadata: Dict[str, any] = None
    ):
        """Log LLM-As-A-Judge evaluation with full explanation"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            "question_id": question_id,
            "question": question,
            "target_answer": target_answer,
            "provider_responses": provider_responses,
            "judge": {
                "model_config": judge_model_config or {
                    "model": "UNKNOWN",  # Will be populated with actual config
                    "temperature": "UNKNOWN",
                    "response_format": "UNKNOWN",
                    "service_tier": "UNKNOWN"
                },
                "prompt": judge_prompt,
                "response": judge_response,
                "reasoning": reasoning,
                "latency_ms": latency_ms
            },
            "metadata": metadata or {},
            "grades": grades,
            "analysis": {
                "response_lengths": {provider: len(response) for provider, response in provider_responses.items()},
                "judge_response_length": len(judge_response)
            }
        }

        # Write to judge log file
        with open(self.judge_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_abstention_classification(
        self,
        question_id: str,
        question: str,
        provider_response: str,
        provider_name: str,
        classification_type: str,
        confidence: float,
        reasoning: str,
        classifier_model: str,
        metadata: Dict[str, any] = None
    ):
        """Log abstention classifier decisions for audit trail"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            "question_id": question_id,
            "question": question,
            "provider_response": provider_response,
            "provider_name": provider_name,
            "classifier": {
                "model": classifier_model,
                "classification": classification_type,
                "confidence": confidence,
                "reasoning": reasoning
            },
            "metadata": metadata or {}
        }

        # Write to a separate abstention classification log
        abstention_log_file = self.run_dir / "abstention_classifications.jsonl"
        with open(abstention_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_judge_consistency_validation(
        self,
        consistency_summary: Dict[str, any],
        run_timestamp: str = None
    ):
        """Log judge consistency validation results for audit trail"""
        log_entry = {
            "timestamp": run_timestamp or datetime.utcnow().isoformat(),
            "run_id": self.run_id,
            "validation_type": "judge_consistency",
            "summary": consistency_summary,
            "validation_metadata": {
                "temperature_expected": 0.0,
                "deterministic_expected": True,
                "critical_threshold": 1.0  # 100% consistency required
            }
        }

        # Write to judge consistency validation log
        consistency_log_file = self.run_dir / "judge_consistency_validation.jsonl"
        with open(consistency_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_error(self, component: str, error: str, context: Dict[str, Any] = None):
        """Log errors with context"""
        error_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "component": component,
            "error": error,
            "context": context or {}
        }

        self.run_metadata["errors"].append(error_entry)
        self.save_metadata()

    def update_progress(self, completed_questions: int, total_questions: int = None):
        """Update run progress"""
        if total_questions is not None:
            self.run_metadata["total_questions"] = total_questions
        self.run_metadata["completed_questions"] = completed_questions
        self.save_metadata()

    def add_provider(self, provider: str, config: Dict[str, Any]):
        """Register a provider in the run metadata"""
        provider_info = {
            "name": provider,
            "config": config,
            "registered_at": datetime.utcnow().isoformat()
        }
        self.run_metadata["providers"].append(provider_info)
        self.save_metadata()

    def finalize_run(self, results: Dict[str, Any]):
        """Finalize the run with summary results"""
        self.run_metadata.update({
            "end_time": datetime.utcnow().isoformat(),
            "status": "completed",
            "results": results
        })
        self.save_metadata()

    def save_metadata(self):
        """Save run metadata to file"""
        with open(self.meta_log_file, "w", encoding="utf-8") as f:
            json.dump(self.run_metadata, f, indent=2, ensure_ascii=False)

    def get_run_summary(self) -> Dict[str, Any]:
        """Get a summary of the current run"""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "providers": [p["name"] for p in self.run_metadata["providers"]],
            "progress": f"{self.run_metadata['completed_questions']}/{self.run_metadata['total_questions']}",
            "logs": {
                "provider_requests": str(self.provider_log_file),
                "judge_evaluations": str(self.judge_log_file),
                "metadata": str(self.meta_log_file)
            }
        }


def create_run_id() -> str:
    """Create a unique run ID based on timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds truncated to ms


def load_audit_logs(run_id: str, output_dir: str = "results") -> Dict[str, Any]:
    """Load audit logs for a specific run"""
    run_dir = Path(output_dir) / f"run_{run_id}"

    if not run_dir.exists():
        raise ValueError(f"Run directory not found: {run_dir}")

    logs = {}

    # Load provider requests
    provider_log_file = run_dir / "provider_requests.jsonl"
    if provider_log_file.exists():
        with open(provider_log_file, "r", encoding="utf-8") as f:
            logs["provider_requests"] = [json.loads(line) for line in f]

    # Load judge evaluations
    judge_log_file = run_dir / "judge_evaluations.jsonl"
    if judge_log_file.exists():
        with open(judge_log_file, "r", encoding="utf-8") as f:
            logs["judge_evaluations"] = [json.loads(line) for line in f]

    # Load metadata
    meta_log_file = run_dir / "run_metadata.json"
    if meta_log_file.exists():
        with open(meta_log_file, "r", encoding="utf-8") as f:
            logs["metadata"] = json.load(f)

    return logs