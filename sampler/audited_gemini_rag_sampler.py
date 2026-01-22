"""
Audited Gemini RAG Sampler with comprehensive logging

Wraps the GeminiFileSearchSampler with audit logging and
concurrency control for fair benchmark comparisons.
"""

import os
import threading
from typing import Any, Dict, Optional

from custom_types import MessageList
from sampler.audited_sampler_base import AuditedSamplerBase
from sampler.gemini_file_search_sampler import GeminiFileSearchSampler
from pricing_config import calculate_cost


# Global rate limiter for Gemini RAG API for fair comparison with other providers
_gemini_rag_semaphore = threading.Semaphore(5)  # Max 5 concurrent requests


class AuditedGeminiRAGSampler(AuditedSamplerBase):
    """
    Google Gemini RAG sampler with audit logging capabilities.

    Uses the File Search tool to retrieve relevant documents from
    a Google AI Studio File Search store before generating responses.
    """

    def __init__(
        self,
        store_name: str,
        api_key: Optional[str] = None,
        model: str = "gemini-3-pro-preview",  # Gemini 3 Pro (API still uses -preview)
        temperature: float = 0.0,
        system_message: Optional[str] = None,
        thinking_level: Optional[str] = None,  # "LOW", "HIGH", "MINIMAL" (Flash only)
        audit_logger=None
    ):
        """
        Initialize the audited Gemini RAG sampler.

        Args:
            store_name: The File Search store name (required)
            api_key: Google AI API key (defaults to GEMINI_API_KEY env var)
            model: Model to use (default: gemini-3-pro)
            temperature: Temperature for generation (default: 0.0)
            system_message: Optional system message for RAG prompt
            thinking_level: Thinking level - "LOW", "HIGH", "MINIMAL" (Flash only), or None for default
            audit_logger: Optional audit logger for request tracking
        """
        super().__init__(audit_logger)

        if not store_name:
            raise ValueError("store_name is required")

        # Initialize base sampler
        self.base_sampler = GeminiFileSearchSampler(
            store_name=store_name,
            api_key=api_key,
            model=model,
            temperature=temperature,
            system_message=system_message,
            thinking_level=thinking_level
        )
        self._thinking_level = thinking_level

        # Set provider name for audit logging
        self.provider_name = "Google_Gemini_RAG"

        # Store config for request data
        self._store_name = store_name
        self._model = model
        self._temperature = temperature

    def _make_request(self, message_list: MessageList, question_id: str = None) -> str:
        """
        Make the actual Gemini RAG API request using the base sampler.

        Uses a semaphore to limit concurrent requests for fair comparison.
        """
        with _gemini_rag_semaphore:
            return self.base_sampler(message_list)

    def _get_request_data(self, message_list: MessageList) -> Dict[str, Any]:
        """
        Get request data for audit logging.
        """
        return {
            "model": self.base_sampler.model,
            "temperature": self.base_sampler.temperature,
            "store_name": self.base_sampler.store_name,
            "messages": message_list,
            "system_message": self.base_sampler.system_message,
            "provider_type": "gemini_rag"
        }

    def _pack_message(self, role: str, content: Any) -> Dict[str, Any]:
        """
        Pack a message in the standard format.

        Required by the eval framework for creating message lists.
        """
        return {"role": role, "content": content}

    def _get_metadata(self) -> Dict[str, Any]:
        """
        Get additional metadata for logging.

        Includes grounding information, token usage, and cost when available.
        """
        metadata = {
            "provider_type": "gemini_rag",
            "uses_rag": True,
            "store_name": self.base_sampler.store_name,
            "api_endpoint": "models.generate_content"
        }

        # Include grounding metadata if available
        try:
            grounding = self.base_sampler._last_grounding
            if grounding and hasattr(grounding, "grounding_chunks"):
                chunks = grounding.grounding_chunks
                if chunks and isinstance(chunks, (list, tuple)):
                    metadata["grounding_metadata"] = {
                        "has_grounding": True,
                        "num_chunks": len(chunks)
                    }
                else:
                    metadata["grounding_metadata"] = {
                        "has_grounding": True,
                        "num_chunks": 0
                    }
            else:
                metadata["grounding_metadata"] = {
                    "has_grounding": False,
                    "num_chunks": 0
                }
        except Exception:
            metadata["grounding_metadata"] = {
                "has_grounding": False,
                "num_chunks": 0
            }

        # Include token usage and cost if available
        try:
            usage = self.base_sampler._last_usage
            if usage:
                metadata["token_usage"] = usage

                # Calculate cost - CRITICAL: Include ALL token types!
                # Google Gemini with File Search has multiple token categories:
                #
                # 1. prompt_tokens: User's question only (~100-200 tokens)
                # 2. completion_tokens: Visible response text
                # 3. thoughts_tokens: Thinking/reasoning tokens (billed at output rate)
                # 4. total_tokens: EVERYTHING including RAG retrieved context
                #
                # The "hidden" tokens (total - prompt - comp - thoughts) are RAG context
                # Google says: "Retrieved document tokens are charged as regular tokens"
                # These should be billed at INPUT rate.

                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                thoughts_tokens = usage.get("thoughts_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                # Calculate hidden RAG context tokens
                tracked_tokens = prompt_tokens + completion_tokens + thoughts_tokens
                rag_context_tokens = max(0, total_tokens - tracked_tokens)

                # Input cost = prompt tokens + RAG context tokens (both at input rate)
                total_input_tokens = prompt_tokens + rag_context_tokens

                # Output cost = completion + thinking tokens (both at output rate)
                total_output_tokens = completion_tokens + thoughts_tokens

                cost = calculate_cost(
                    self.base_sampler.model,
                    total_input_tokens,   # Include RAG context in input cost
                    total_output_tokens   # Include thinking tokens in output cost
                )
                metadata["estimated_cost_usd"] = cost

                # Add detailed breakdown for audit clarity
                metadata["cost_breakdown"] = {
                    "user_prompt_tokens": prompt_tokens,
                    "rag_context_tokens": rag_context_tokens,
                    "total_input_billed": total_input_tokens,
                    "completion_tokens": completion_tokens,
                    "thoughts_tokens": thoughts_tokens,
                    "total_output_billed": total_output_tokens,
                    "total_tokens_from_api": total_tokens,
                    "pricing_note": f"See pricing_config.py for {self._model} rates"
                }
        except Exception:
            pass

        return metadata
