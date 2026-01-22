"""
Model pricing configuration for cost calculation.

This module provides pricing tables for various AI models and utility functions
to calculate costs based on token usage. Prices are in USD per million tokens.

Note: CustomGPT uses a subscription model rather than per-token pricing,
so cost calculation is not applicable (returns None).

Pricing sources (as of December 2025):
- OpenAI: https://openai.com/pricing
- Google: https://ai.google.dev/pricing
"""

from typing import Dict, Optional, Any
import numpy as np

# Model pricing per million tokens (USD)
# Format: {"input": price_per_million_input_tokens, "output": price_per_million_output_tokens}
MODEL_PRICING: Dict[str, Optional[Dict[str, float]]] = {
    # OpenAI Models (December 2025)
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # Google Gemini Models (January 2026)
    # Source: https://ai.google.dev/gemini-api/docs/pricing
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},  # Gemini 3 Flash - 75% cheaper than Pro
    "gemini-2.5-pro": {"input": 2.00, "output": 12.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},

    # CustomGPT - per-query pricing for addon queries
    # Uses fixed pricing model rather than per-token
    "customgpt": {"per_query": 0.10},  # $0.10 per query
}


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int
) -> Optional[float]:
    """
    Calculate cost in USD for a single request.

    Args:
        model: Model identifier (e.g., "gpt-5.1", "gemini-3-pro-preview", "customgpt")
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens

    Returns:
        Cost in USD, or None if model pricing is not available

    Examples:
        >>> calculate_cost("gpt-5.1", 1000, 500)
        0.00625  # ($1.25/M * 1000) + ($10/M * 500)

        >>> calculate_cost("customgpt", 0, 0)
        0.10  # Fixed $0.10 per query
    """
    if model not in MODEL_PRICING:
        return None

    pricing = MODEL_PRICING[model]
    if pricing is None:
        return None

    # Handle per-query pricing models (like CustomGPT)
    if "per_query" in pricing:
        return pricing["per_query"]

    # Token-based pricing for OpenAI, Gemini, etc.
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def calculate_latency_stats(latencies: list) -> Dict[str, Optional[float]]:
    """
    Calculate latency statistics from a list of latency values.

    Args:
        latencies: List of latency values in milliseconds

    Returns:
        Dictionary with statistical measures:
        - avg_ms: Mean latency
        - median_ms: Median (p50) latency
        - p95_ms: 95th percentile latency
        - min_ms: Minimum latency
        - max_ms: Maximum latency
    """
    if not latencies:
        return {
            "avg_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    return {
        "avg_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "min_ms": float(min(latencies)),
        "max_ms": float(max(latencies)),
    }


def calculate_cost_stats(
    costs: list,
    tokens: list,
    prompt_tokens_list: Optional[list] = None,
    completion_tokens_list: Optional[list] = None
) -> Dict[str, Optional[float]]:
    """
    Calculate cost statistics from lists of costs and token counts.

    Args:
        costs: List of cost values in USD (may contain None values)
        tokens: List of total token counts
        prompt_tokens_list: Optional list of prompt/input token counts
        completion_tokens_list: Optional list of completion/output token counts

    Returns:
        Dictionary with cost metrics:
        - total_usd: Total cost across all requests
        - avg_per_request_usd: Average cost per request
        - total_tokens: Total tokens used
        - avg_tokens_per_request: Average tokens per request
        - avg_prompt_tokens: Average input tokens per request
        - avg_completion_tokens: Average output tokens per request
    """
    # Filter out None values for cost calculations
    valid_costs = [c for c in costs if c is not None]
    valid_tokens = [t for t in tokens if t is not None and t > 0]

    result = {
        "total_usd": sum(valid_costs) if valid_costs else None,
        "avg_per_request_usd": float(np.mean(valid_costs)) if valid_costs else None,
        "total_tokens": sum(valid_tokens) if valid_tokens else None,
        "avg_tokens_per_request": float(np.mean(valid_tokens)) if valid_tokens else None,
        "avg_prompt_tokens": None,
        "avg_completion_tokens": None,
    }

    if prompt_tokens_list:
        valid_prompt = [t for t in prompt_tokens_list if t is not None and t > 0]
        if valid_prompt:
            result["avg_prompt_tokens"] = float(np.mean(valid_prompt))

    if completion_tokens_list:
        valid_completion = [t for t in completion_tokens_list if t is not None and t > 0]
        if valid_completion:
            result["avg_completion_tokens"] = float(np.mean(valid_completion))

    return result


def format_cost(cost: Optional[float]) -> str:
    """
    Format a cost value for display.

    Args:
        cost: Cost in USD, or None

    Returns:
        Formatted string like "$0.0042" or "N/A"
    """
    if cost is None:
        return "N/A"
    if cost < 0.01:
        return f"${cost:.6f}"
    elif cost < 1:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def format_latency(latency_ms: Optional[float]) -> str:
    """
    Format a latency value for display.

    Args:
        latency_ms: Latency in milliseconds, or None

    Returns:
        Formatted string like "1,234ms" or "N/A"
    """
    if latency_ms is None:
        return "N/A"
    return f"{latency_ms:,.0f}ms"


def format_tokens(tokens: Optional[float]) -> str:
    """
    Format a token count for display.

    Args:
        tokens: Token count, or None

    Returns:
        Formatted string like "1,234" or "N/A"
    """
    if tokens is None:
        return "N/A"
    return f"{tokens:,.0f}"
