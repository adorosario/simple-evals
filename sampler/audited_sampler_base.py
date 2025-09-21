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

    def set_audit_logger(self, audit_logger):
        """Set the audit logger for this sampler"""
        self.audit_logger = audit_logger

    def __call__(self, message_list: MessageList, question_id: str = None) -> str:
        """
        Enhanced call method with audit logging
        """
        start_time = time.time()

        # Extract the question for logging
        question = ""
        for message in message_list:
            if message.get("role") == "user":
                question = message.get("content", "")
                break

        # Prepare request data for logging
        request_data = self._get_request_data(message_list)

        try:
            # Call the actual implementation
            response = self._make_request(message_list)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Log the successful request
            if self.audit_logger and question_id:
                self.audit_logger.log_provider_request(
                    provider=self.provider_name,
                    question_id=question_id,
                    question=question,
                    request_data=request_data,
                    response=response,
                    latency_ms=latency_ms,
                    metadata=self._get_metadata()
                )

            return response

        except Exception as e:
            # Calculate latency even for failed requests
            latency_ms = (time.time() - start_time) * 1000

            # Log the failed request
            if self.audit_logger and question_id:
                self.audit_logger.log_provider_request(
                    provider=self.provider_name,
                    question_id=question_id,
                    question=question,
                    request_data=request_data,
                    response="",
                    latency_ms=latency_ms,
                    metadata={"error": str(e), **self._get_metadata()}
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