"""
Gemini File Search Sampler for Google AI Studio RAG

Uses Google's File Search tool to retrieve relevant content from a
knowledge base store before generating responses.
"""

import os
import time
from typing import Any, Optional

from google import genai
from google.genai import types

from custom_types import MessageList, SamplerBase


class GeminiFileSearchSampler(SamplerBase):
    """
    Sampler that uses Google Gemini with File Search RAG.

    Implements the SamplerBase interface for consistency with other samplers.
    Uses the Google GenAI SDK's File Search tool to retrieve relevant
    documents before generating responses.
    """

    def __init__(
        self,
        store_name: str,
        api_key: Optional[str] = None,
        model: str = "gemini-3-pro-preview",  # Gemini 3 Pro Preview (SOTA December 2025)
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        system_message: Optional[str] = None,
    ):
        """
        Initialize the Gemini File Search sampler.

        Args:
            store_name: The File Search store name (required)
            api_key: Google AI API key (defaults to GEMINI_API_KEY env var)
            model: Model to use (default: gemini-2.5-pro)
            temperature: Temperature for generation (default: 0.0)
            max_output_tokens: Maximum output tokens (default: 1024)
            system_message: Optional system message
        """
        if not store_name:
            raise ValueError("store_name is required")

        self.store_name = store_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_message = system_message

        # Initialize the client
        self.client = genai.Client(api_key=self.api_key)

        # Store last grounding metadata for auditing
        self._last_grounding = None

        # Token usage tracking
        self._last_usage = None

    def _convert_messages(self, message_list: MessageList) -> list:
        """
        Convert OpenAI-style message list to Gemini Content format.

        Gemini uses 'user' and 'model' roles (not 'assistant').
        System messages are handled separately.
        """
        contents = []

        for message in message_list:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Skip system messages - they're handled in system_instruction
            if role == "system":
                continue

            # Map assistant -> model for Gemini
            gemini_role = "model" if role == "assistant" else "user"

            contents.append(types.Content(
                role=gemini_role,
                parts=[types.Part(text=content)]
            ))

        return contents

    def __call__(self, message_list: MessageList) -> str:
        """
        Generate a response using Gemini with File Search RAG.

        Args:
            message_list: List of messages in OpenAI format

        Returns:
            Generated response string
        """
        # Reset grounding and usage metadata
        self._last_grounding = None
        self._last_usage = None

        # Convert messages to Gemini format
        contents = self._convert_messages(message_list)

        # Configure File Search tool
        file_search_tool = types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[self.store_name]
            )
        )

        # Build config
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            tools=[file_search_tool]
        )

        # Add system instruction if provided
        if self.system_message:
            config.system_instruction = self.system_message

        # Retry logic
        max_retries = 3
        trial = 0

        while trial < max_retries:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config
                )

                # Extract grounding metadata if available
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "grounding_metadata"):
                        self._last_grounding = candidate.grounding_metadata

                # Capture token usage from usage_metadata
                # Note: Gemini 3 Pro has thinking tokens enabled by default
                # These are billed at output rate and must be tracked separately
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    um = response.usage_metadata
                    self._last_usage = {
                        "prompt_tokens": getattr(um, 'prompt_token_count', 0) or 0,
                        "completion_tokens": getattr(um, 'candidates_token_count', 0) or 0,
                        "thoughts_tokens": getattr(um, 'thoughts_token_count', 0) or 0,  # Thinking tokens (billed as output)
                        "total_tokens": getattr(um, 'total_token_count', 0) or 0,
                        "cached_tokens": getattr(um, 'cached_content_token_count', 0) or 0  # Cached context tokens
                    }
                    # Debug: Log full usage_metadata to understand token breakdown
                    print(f"DEBUG Gemini usage_metadata: prompt={self._last_usage['prompt_tokens']}, "
                          f"completion={self._last_usage['completion_tokens']}, "
                          f"thoughts={self._last_usage['thoughts_tokens']}, "
                          f"total={self._last_usage['total_tokens']}, "
                          f"cached={self._last_usage['cached_tokens']}")

                # Return response text
                if response.text:
                    return response.text.strip()
                return ""

            except Exception as e:
                trial += 1
                if trial >= max_retries:
                    print(f"Max retries ({max_retries}) exceeded for Gemini: {e}")
                    return ""

                # Exponential backoff
                backoff = min(2 ** trial, 4)
                print(f"Gemini exception, retrying {trial}/{max_retries} after {backoff}s: {e}")
                time.sleep(backoff)

        return ""

    def get_grounding_metadata(self) -> Optional[Any]:
        """
        Get the grounding metadata from the last request.

        Returns grounding chunks and sources from the File Search
        retrieval for audit/debugging purposes.
        """
        return self._last_grounding
