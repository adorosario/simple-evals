"""
Audited OpenAI Vanilla Sampler with comprehensive logging
"""

import os
import time
import threading
from typing import Any, Dict, Optional

from openai import OpenAI
import openai

from custom_types import MessageList
from sampler.audited_sampler_base import AuditedSamplerBase
from pricing_config import calculate_cost

# Global rate limiter for OpenAI API for fair comparison with other providers
_openai_vanilla_semaphore = threading.Semaphore(5)  # Max 5 concurrent requests


class AuditedOpenAIVanillaSampler(AuditedSamplerBase):
    """
    OpenAI Vanilla sampler with audit logging capabilities
    """

    def __init__(
        self,
        model: str = "gpt-5.1",  # GPT-5.1 (SOTA December 2025)
        system_message: str | None = None,
        temperature: float = 0,
        max_tokens: int = 1024,
        audit_logger=None
    ):
        super().__init__(audit_logger)
        self.client = OpenAI()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_name = "OpenAI_Vanilla"
        # Token usage tracking
        self._last_usage: Optional[Dict[str, int]] = None

    def _pack_message(self, role: str, content: Any):
        """Pack message for OpenAI API format"""
        return {"role": str(role), "content": content}

    def _make_request(self, message_list: MessageList) -> str:
        """
        Make the actual OpenAI API request
        """
        # Use global semaphore for fair comparison with other providers
        with _openai_vanilla_semaphore:
            # Prepare messages for the API
            messages = []

            # Add system message if provided
            if self.system_message:
                messages.append(self._pack_message("system", self.system_message))

            # Add conversation history
            messages.extend(message_list)

            # Reset usage tracking
            self._last_usage = None

            trial = 0
            while True:
                try:
                    # Standard chat completion - no RAG
                    # Use max_completion_tokens for GPT-5.x models
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_tokens,
                    )

                    # Capture token usage from response
                    if hasattr(response, 'usage') and response.usage:
                        self._last_usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }

                    return response.choices[0].message.content or ""

                except openai.BadRequestError as e:
                    print(f"Bad Request Error: {e}")
                    return ""

                except Exception as e:
                    exception_backoff = 2**trial
                    print(
                        f"Vanilla OpenAI exception, retrying {trial} after {exception_backoff} sec: {e}"
                    )
                    time.sleep(exception_backoff)
                    trial += 1

                    if trial > 5:  # Max retries
                        print(f"Max retries exceeded, returning empty response")
                        return ""

    def _get_request_data(self, message_list: MessageList) -> Dict[str, Any]:
        """Get request data for audit logging"""
        # Prepare messages for logging
        messages = []
        if self.system_message:
            messages.append(self._pack_message("system", self.system_message))
        messages.extend(message_list)

        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "system_message": self.system_message,
            "provider_type": "vanilla_llm"
        }

    def _get_metadata(self) -> Dict[str, Any]:
        """Get additional metadata for logging including token usage and cost"""
        metadata = {
            "provider_type": "vanilla_llm",
            "uses_rag": False,
            "api_endpoint": "chat.completions.create"
        }

        # Add token usage if available
        if self._last_usage:
            metadata["token_usage"] = self._last_usage
            # Calculate cost
            cost = calculate_cost(
                self.model,
                self._last_usage.get("prompt_tokens", 0),
                self._last_usage.get("completion_tokens", 0)
            )
            metadata["estimated_cost_usd"] = cost

        return metadata