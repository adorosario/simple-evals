"""
Audited CustomGPT Sampler with comprehensive logging
"""

import os
import time
import uuid
import threading
from typing import Any, Dict

import requests

from custom_types import MessageList
from sampler.audited_sampler_base import AuditedSamplerBase

# Global rate limiter for CustomGPT API to prevent overwhelming their infrastructure
# Limits concurrent requests across all CustomGPT sampler instances
_customgpt_semaphore = threading.Semaphore(5)  # Max 5 concurrent requests (CustomGPT's recommended limit)


class AuditedCustomGPTSampler(AuditedSamplerBase):
    """
    CustomGPT sampler with audit logging capabilities
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: str = None,
        audit_logger=None
    ):
        super().__init__(audit_logger)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.api_key = os.environ.get("CUSTOMGPT_API_KEY")
        self.project_id = os.environ.get("CUSTOMGPT_PROJECT")
        self.provider_name = "CustomGPT_RAG"

        # Store the actual session ID used for API calls for proper logging
        self._current_session_id = None
        self._current_url = None
        self._current_external_id = None

    def _pack_message(self, role: str, content: str):
        return {"role": str(role), "content": content}

    def _make_request(self, message_list: MessageList, question_id: str = None) -> str:
        """
        Make the actual CustomGPT API request with proper error handling and timeouts
        """
        # Extract the user query - usually the last user message
        prompt = None
        for message in reversed(message_list):
            if message.get("role") == "user":
                prompt = message.get("content", "")
                break

        if not prompt:
            raise ValueError("No user message found in message list")

        # Generate a new session ID for each request to avoid context contamination
        session_id = uuid.uuid4()

        # Build query parameters
        query_params = {
            "stream": "false",  # Use string for query param
            "lang": "en"
        }

        # Add external_id if available
        if question_id and self.audit_logger:
            run_id = getattr(self.audit_logger, 'run_id', 'unknown')
            external_id = f"{run_id}_{question_id}"
            query_params["external_id"] = external_id
            self._current_external_id = external_id

        # Build URL with query parameters
        base_url = f"https://app.customgpt.ai/api/v1/projects/{self.project_id}/conversations/{session_id}/messages"
        query_string = "&".join([f"{k}={v}" for k, v in query_params.items()])
        url = f"{base_url}?{query_string}"

        # Store for logging purposes
        self._current_session_id = str(session_id)
        self._current_url = url

        max_retries = 3
        base_delay = 2.0

        # Use global semaphore to limit concurrent requests to CustomGPT
        with _customgpt_semaphore:
            for attempt in range(max_retries):
                try:
                    start_time = time.time()
                    print(f"CustomGPT API call starting (attempt {attempt + 1}/{max_retries})")

                    # Correct CustomGPT API request format based on official documentation
                    # Use multipart/form-data with query parameters
                    form_data = {
                        "response_source": "default",
                        "prompt": prompt
                    }

                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "application/json"
                        # Don't set Content-Type - let requests handle multipart boundary
                    }

                    response = requests.post(
                        url,
                        data=form_data,  # Use data for form-encoded data
                        headers=headers,
                        timeout=60.0  # 60 second timeout per request
                    )
                    api_time = time.time() - start_time
                    print(f"CustomGPT API call completed in {api_time:.2f}s, status: {response.status_code}")

                    if response.status_code == 200:
                        try:
                            return response.json()['data']["openai_response"]
                        except (KeyError, ValueError) as e:
                            print(f"Invalid JSON response structure: {e}")
                            if attempt == max_retries - 1:
                                return ""
                            continue

                    elif response.status_code == 429:
                        # Rate limit - respect Retry-After header
                        retry_after = int(response.headers.get("Retry-After", 30))
                        print(f"Rate limit exceeded, waiting {retry_after}s")
                        time.sleep(retry_after)
                        continue

                    elif response.status_code in [502, 503, 504]:
                        # Gateway errors - exponential backoff
                        delay = base_delay * (2 ** attempt)
                        print(f"Gateway error {response.status_code}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            time.sleep(delay)
                            continue
                        else:
                            print(f"Max retries exceeded for gateway error {response.status_code}")
                            return ""

                    else:
                        # Other HTTP errors
                        print(f"HTTP Error {response.status_code}: {response.text[:200]}")
                        if attempt == max_retries - 1:
                            return ""
                        time.sleep(base_delay)
                        continue

                except requests.exceptions.Timeout:
                    print(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        print("Max retries exceeded due to timeouts")
                        return ""
                    time.sleep(base_delay * (attempt + 1))
                    continue

                except requests.exceptions.RequestException as e:
                    print(f"Request exception: {e}")
                    if attempt == max_retries - 1:
                        print("Max retries exceeded due to request exceptions")
                        return ""
                    time.sleep(base_delay)
                    continue

            return ""  # Fallback if all retries failed

    def _get_request_data(self, message_list: MessageList) -> Dict[str, Any]:
        """Get request data for audit logging"""
        # Extract user query for logging
        prompt = None
        for message in reversed(message_list):
            if message.get("role") == "user":
                prompt = message.get("content", "")
                break

        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": message_list,
            "system_message": None,  # CustomGPT doesn't use system messages
            "additional_params": {
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
                "stop": self.stop,
                "project_id": self.project_id,
                "session_id": self._current_session_id,
                "prompt": prompt,
                "stream": "false",  # Query parameter format
                "external_id": self._get_external_id(),
                "lang": "en",
                "response_source": "default",
                "provider_type": "customgpt_rag",
                "request_method": "multipart/form-data"
            }
        }

    def _get_external_id(self) -> str:
        """Get the current external_id for logging"""
        return self._current_external_id

    def _get_metadata(self) -> Dict[str, Any]:
        """Get additional metadata for logging"""
        return {
            "provider_type": "customgpt_rag",
            "uses_rag": True,
            "project_id": self.project_id,
            "session_id": self._current_session_id,
            "api_endpoint": "customgpt.ai/api/v1/conversations/messages",
            "actual_url": self._current_url,
            "external_id": self._current_external_id
        }

    @classmethod
    def from_env(cls, audit_logger=None):
        """Create sampler from environment variables with audit logging"""
        model_name = os.environ.get("CUSTOMGPT_MODEL_NAME", "gpt-4.1")
        return cls(model_name=model_name, audit_logger=audit_logger)