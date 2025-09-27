"""
Audited OpenAI RAG Sampler with comprehensive logging
"""

import os
import time
import threading
from typing import Any, Dict

from openai import OpenAI
import openai

from custom_types import MessageList
from sampler.audited_sampler_base import AuditedSamplerBase

# Global rate limiter for OpenAI RAG API for fair comparison with other providers
_openai_rag_semaphore = threading.Semaphore(5)  # Max 5 concurrent requests


class AuditedOpenAIRAGSampler(AuditedSamplerBase):
    """
    OpenAI RAG sampler with audit logging capabilities
    """

    def __init__(
        self,
        model: str = "gpt-4.1",  # GPT-4.1
        vector_store_id: str | None = None,
        system_message: str | None = None,
        temperature: float = 0,
        max_tokens: int = 1024,
        audit_logger=None
    ):
        super().__init__(audit_logger)
        self.client = OpenAI()
        self.model = model
        self.vector_store_id = vector_store_id or os.environ.get("OPENAI_VECTOR_STORE_ID")
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_name = "OpenAI_RAG"

        if not self.vector_store_id:
            raise ValueError(
                "Vector store ID required. Set OPENAI_VECTOR_STORE_ID environment variable "
                "or provide vector_store_id parameter."
            )

    def _pack_message(self, role: str, content: Any):
        """Pack message for OpenAI API format"""
        return {"role": str(role), "content": content}

    def _make_request(self, message_list: MessageList) -> str:
        """
        Make the actual OpenAI RAG API request using Responses API
        """
        # Use global semaphore for fair comparison with other providers
        with _openai_rag_semaphore:
            # Extract the user query - usually the last user message
            user_query = None
            for message in reversed(message_list):
                if message.get("role") == "user":
                    user_query = message.get("content", "")
                    break

            if not user_query:
                raise ValueError("No user message found in message list")

            trial = 0
            while True:
                try:
                    # Use the modern Responses API with file search
                    response = self.client.responses.create(
                        input=user_query,
                        model=self.model,
                        tools=[{
                            "type": "file_search",
                            "vector_store_ids": [self.vector_store_id],
                        }],
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens
                    )

                    # Extract the response content
                    if response.output:
                        # Find the message output in the response
                        for output_item in response.output:
                            if hasattr(output_item, 'type') and output_item.type == 'message':
                                if hasattr(output_item, 'content'):
                                    response_text = ""
                                    for content_block in output_item.content:
                                        if hasattr(content_block, 'text'):
                                            response_text += content_block.text
                                    return response_text.strip()

                    # Fallback - try to get any text from the response
                    if hasattr(response, 'text'):
                        return response.text.strip()

                    return ""

                except openai.BadRequestError as e:
                    print(f"Bad Request Error: {e}")
                    return ""

                except Exception as e:
                    exception_backoff = min(2**trial, 4)  # Cap at 4 seconds
                    print(f"RAG exception, retrying {trial} after {exception_backoff} sec: {e}")
                    time.sleep(exception_backoff)
                    trial += 1

                    if trial > 2:  # Only 2 retries
                        print(f"Max retries exceeded, returning empty response")
                        return ""

    def _get_request_data(self, message_list: MessageList) -> Dict[str, Any]:
        """Get request data for audit logging"""
        # Extract user query for logging
        user_query = None
        for message in reversed(message_list):
            if message.get("role") == "user":
                user_query = message.get("content", "")
                break

        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "vector_store_id": self.vector_store_id,
            "user_query": user_query,
            "messages": message_list,
            "system_message": self.system_message,
            "provider_type": "rag",
            "tools": [{
                "type": "file_search",
                "vector_store_ids": [self.vector_store_id],
            }]
        }

    def _get_metadata(self) -> Dict[str, Any]:
        """Get additional metadata for logging"""
        return {
            "provider_type": "rag",
            "uses_rag": True,
            "vector_store_id": self.vector_store_id,
            "api_endpoint": "responses.create"
        }