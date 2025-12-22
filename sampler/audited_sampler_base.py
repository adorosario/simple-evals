"""
Base class for samplers with audit logging support
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from custom_types import MessageList, SamplerBase


class AuditedSamplerBase(SamplerBase, ABC):
    """
    Base class for samplers that support audit logging
    Wraps the original sampler functionality with comprehensive logging
    """

    def __init__(self, audit_logger=None):
        self.audit_logger = audit_logger
        self.provider_name = self.__class__.__name__.replace("Sampler", "").replace("Audited", "")
        # Track latency for the most recent request
        self._last_latency_ms: float = 0.0

    def set_audit_logger(self, audit_logger):
        """Set the audit logger for this sampler"""
        self.audit_logger = audit_logger

    def __call__(self, message_list: MessageList, question_id: str = None, return_metrics: bool = False):
        """
        Enhanced call method with audit logging.

        Args:
            message_list: List of messages to send to the model
            question_id: Optional question ID for audit logging
            return_metrics: If True, return (response, metrics) tuple for thread-safe metric capture.
                           If False (default), return just the response string.

        Returns:
            If return_metrics=False: str (response text)
            If return_metrics=True: tuple(str, dict) where dict contains:
                - latency_ms: Request latency in milliseconds
                - token_usage: Dict with prompt_tokens, completion_tokens, total_tokens (or None)
                - estimated_cost_usd: Estimated cost in USD (or None)
        """
        start_time = time.time()

        # Extract the question for logging
        question = ""
        for message in message_list:
            if message.get("role") == "user":
                question = message.get("content", "")
                break

        try:
            # Call the actual implementation, passing question_id if the method supports it
            if hasattr(self, '_make_request') and 'question_id' in self._make_request.__code__.co_varnames:
                response = self._make_request(message_list, question_id=question_id)
            else:
                response = self._make_request(message_list)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            # Store latency for retrieval after call (kept for backward compatibility)
            self._last_latency_ms = latency_ms

            # Get metadata (includes token_usage and cost) - capture immediately for thread safety
            metadata = self._get_metadata()

            # Prepare request data for logging (after request is made to capture actual parameters)
            request_data = self._get_request_data(message_list)

            # Log the successful request
            if self.audit_logger and question_id:
                self.audit_logger.log_provider_request(
                    provider=self.provider_name,
                    question_id=question_id,
                    question=question,
                    request_data=request_data,
                    response=response,
                    latency_ms=latency_ms,
                    metadata=metadata
                )

            # Return metrics atomically if requested (thread-safe)
            if return_metrics:
                performance_metrics = {
                    "latency_ms": latency_ms,
                    "token_usage": metadata.get("token_usage"),
                    "estimated_cost_usd": metadata.get("estimated_cost_usd")
                }
                return response, performance_metrics

            return response

        except Exception as e:
            # Calculate latency even for failed requests
            latency_ms = (time.time() - start_time) * 1000
            # Store latency for retrieval after call (kept for backward compatibility)
            self._last_latency_ms = latency_ms

            # Get metadata for error case
            metadata = self._get_metadata()

            # Prepare request data for logging (best effort, may have incomplete data)
            try:
                request_data = self._get_request_data(message_list)
            except:
                request_data = {"error": "Failed to prepare request data"}

            # Log the failed request
            if self.audit_logger and question_id:
                self.audit_logger.log_provider_request(
                    provider=self.provider_name,
                    question_id=question_id,
                    question=question,
                    request_data=request_data,
                    response="",
                    latency_ms=latency_ms,
                    metadata={"error": str(e), **metadata}
                )

                self.audit_logger.log_error(
                    component=f"{self.provider_name}_sampler",
                    error=str(e),
                    context={"question_id": question_id, "question": question}
                )

            # Re-raise the exception
            raise

    @abstractmethod
    def _make_request(self, message_list: MessageList) -> str:
        """
        Actual implementation of the request - to be implemented by subclasses
        """
        pass

    @abstractmethod
    def _get_request_data(self, message_list: MessageList) -> Dict[str, Any]:
        """
        Get request data for logging - to be implemented by subclasses
        """
        pass

    def _get_metadata(self) -> Dict[str, Any]:
        """
        Get additional metadata for logging - can be overridden by subclasses
        """
        return {}

    def get_last_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the last request.
        Returns latency, token usage, and cost if available.
        """
        metadata = self._get_metadata()
        return {
            "latency_ms": self._last_latency_ms,
            "token_usage": metadata.get("token_usage"),
            "estimated_cost_usd": metadata.get("estimated_cost_usd")
        }