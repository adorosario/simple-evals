"""
OpenAI Vanilla Sampler - No RAG
Implements SamplerBase interface for baseline comparison without knowledge retrieval
"""

import os
import time
from typing import Any
from dotenv import load_dotenv

from openai import OpenAI
import openai

load_dotenv()
from custom_types import MessageList, SamplerBase


class OpenAIVanillaSampler(SamplerBase):
    """
    Vanilla OpenAI sampler without RAG capabilities
    Uses standard chat completions for baseline comparison
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        system_message: str | None = None,
        temperature: float = 0,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pack_message(self, role: str, content: Any):
        """Pack message for OpenAI API format"""
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        """
        Generate response using standard OpenAI Chat Completions API
        No knowledge retrieval - baseline vanilla performance
        """
        # Prepare messages for the API
        messages = []

        # Add system message if provided
        if self.system_message:
            messages.append(self._pack_message("system", self.system_message))

        # Add conversation history
        messages.extend(message_list)

        trial = 0
        while True:
            try:
                # Standard chat completion - no RAG
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

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

    @classmethod
    def from_config(cls, config: dict):
        """Create sampler from configuration dictionary"""
        return cls(
            model=config.get("model", "gpt-4o"),
            system_message=config.get("system_message"),
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", 1024),
        )

    def to_config(self):
        """Export sampler configuration"""
        return {
            "model": self.model,
            "system_message": self.system_message,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_env(cls):
        """Create sampler from environment variables"""
        model = os.environ.get("OPENAI_VANILLA_MODEL", "gpt-4o")
        system_message = os.environ.get("OPENAI_VANILLA_SYSTEM_MESSAGE")

        return cls(
            model=model,
            system_message=system_message
        )