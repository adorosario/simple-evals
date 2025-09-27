"""
OpenAI RAG Sampler using Responses API with File Search
Implements SamplerBase interface for use with SimpleQA evaluation framework
"""

import os
import time
from typing import Any
from dotenv import load_dotenv

from openai import OpenAI
import openai

load_dotenv()
from custom_types import MessageList, SamplerBase


class OpenAIRAGSampler(SamplerBase):
    """
    Sample from OpenAI's Responses API with file search (RAG)
    Uses vector store for document retrieval and enhanced responses
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        vector_store_id: str | None = None,
        system_message: str | None = None,
        temperature: float = 0,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        self.model = model
        self.vector_store_id = vector_store_id or os.environ.get("OPENAI_VECTOR_STORE_ID")
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.vector_store_id:
            raise ValueError(
                "Vector store ID required. Set OPENAI_VECTOR_STORE_ID environment variable "
                "or provide vector_store_id parameter."
            )

        # Verify vector store exists (skip validation for now as API paths changed)
        # The vector store will be validated when we make the actual request

    def _pack_message(self, role: str, content: Any):
        """Pack message for OpenAI API format"""
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        """
        Generate response using OpenAI Responses API with file search
        Modern approach using the new Responses API
        """
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

    @classmethod
    def from_config(cls, config: dict):
        """Create sampler from configuration dictionary"""
        return cls(
            model=config.get("model", "gpt-4o"),
            vector_store_id=config.get("vector_store_id"),
            system_message=config.get("system_message"),
            temperature=config.get("temperature", 0),
            max_tokens=config.get("max_tokens", 1024),
        )

    def to_config(self):
        """Export sampler configuration"""
        return {
            "model": self.model,
            "vector_store_id": self.vector_store_id,
            "system_message": self.system_message,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_env(cls):
        """Create sampler from environment variables"""
        model = os.environ.get("OPENAI_RAG_MODEL", "gpt-4o")
        vector_store_id = os.environ.get("OPENAI_VECTOR_STORE_ID")

        if not vector_store_id:
            raise ValueError("OPENAI_VECTOR_STORE_ID environment variable is not set")

        return cls(
            model=model,
            vector_store_id=vector_store_id
        )