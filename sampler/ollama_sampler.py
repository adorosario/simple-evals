"""
Ollama Sampler for SimpleQA Benchmarks

This sampler integrates local Ollama models with the simple-evals framework.
Used for benchmarking LLMs in the Fuji Edge RAG project.

Usage:
    from sampler.ollama_sampler import OllamaSampler

    sampler = OllamaSampler(model="qwen2.5:7b")
    result = sampler([{"role": "user", "content": "What is 2+2?"}])
"""

import time
from typing import Any

import requests

from custom_types import MessageList, SamplerBase


class OllamaSampler(SamplerBase):
    """
    Sample from Ollama's local API.

    Ollama provides an OpenAI-compatible API at /v1/chat/completions.
    This sampler connects to a local Ollama instance for inference.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        system_message: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 120,  # Ollama can be slow on first inference
    ):
        """
        Initialize the Ollama sampler.

        Args:
            model: Ollama model name (e.g., "qwen2.5:7b", "mistral:7b")
            base_url: Ollama API base URL
            system_message: Optional system prompt
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.image_format = "base64"  # Ollama supports base64 images

        # API endpoints
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.tags_url = f"{self.base_url}/api/tags"

    def _check_model_available(self) -> bool:
        """Check if the model is available in Ollama."""
        try:
            response = requests.get(self.tags_url, timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check both exact match and without tag
                model_base = self.model.split(":")[0]
                return any(
                    self.model in name or model_base in name
                    for name in model_names
                )
        except Exception:
            pass
        return False

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        """Handle image content for multimodal models."""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }

    def _handle_text(self, text: str):
        """Handle text content."""
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        """Pack a message in the expected format."""
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        """
        Generate a response from the Ollama model.

        Args:
            message_list: List of messages in OpenAI format

        Returns:
            Generated text response
        """
        # Prepend system message if provided
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list

        # Build request payload
        payload = {
            "model": self.model,
            "messages": message_list,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        trial = 0
        max_retries = 5

        while trial < max_retries:
            try:
                response = requests.post(
                    self.chat_url,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status_code == 404:
                    # Model not found - try to provide helpful error
                    raise ValueError(
                        f"Model '{self.model}' not found. "
                        f"Run 'ollama pull {self.model}' first."
                    )
                else:
                    raise requests.RequestException(
                        f"Ollama API error: {response.status_code} - {response.text}"
                    )

            except requests.exceptions.Timeout:
                exception_backoff = 2 ** trial
                print(
                    f"Timeout on trial {trial}, waiting {exception_backoff}s before retry"
                )
                time.sleep(exception_backoff)
                trial += 1

            except requests.exceptions.ConnectionError as e:
                # Ollama might not be running
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Make sure Ollama is running: 'ollama serve' or 'docker compose up'"
                ) from e

            except Exception as e:
                exception_backoff = 2 ** trial
                print(
                    f"Exception on trial {trial}, waiting {exception_backoff}s: {e}"
                )
                time.sleep(exception_backoff)
                trial += 1

        # If we've exhausted retries, raise an error
        raise RuntimeError(
            f"Failed to get response from Ollama after {max_retries} retries"
        )


class OllamaEmbeddingSampler:
    """
    Generate embeddings using Ollama's embedding models.

    Used for RAGAS evaluation which requires embeddings.
    """

    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.embed_url = f"{self.base_url}/api/embeddings"

    def embed(self, text: str) -> list[float]:
        """
        Generate embeddings for a text string.

        Args:
            text: Input text to embed

        Returns:
            List of embedding floats
        """
        payload = {
            "model": self.model,
            "prompt": text,
        }

        response = requests.post(
            self.embed_url,
            json=payload,
            timeout=60,
        )

        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            raise requests.RequestException(
                f"Embedding API error: {response.status_code} - {response.text}"
            )

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]


# Convenience function for quick testing
def test_ollama_connection(base_url: str = "http://localhost:11434") -> dict:
    """
    Test connection to Ollama and list available models.

    Returns:
        Dict with connection status and available models
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            return {
                "connected": True,
                "models": models,
                "error": None,
            }
        else:
            return {
                "connected": False,
                "models": [],
                "error": f"HTTP {response.status_code}",
            }
    except Exception as e:
        return {
            "connected": False,
            "models": [],
            "error": str(e),
        }


if __name__ == "__main__":
    # Quick test when run directly
    print("Testing Ollama connection...")
    status = test_ollama_connection()
    print(f"Connected: {status['connected']}")
    print(f"Available models: {status['models']}")

    if status["connected"] and status["models"]:
        model = status["models"][0]
        print(f"\nTesting inference with {model}...")
        sampler = OllamaSampler(model=model)
        response = sampler([{"role": "user", "content": "Say hello in one word."}])
        print(f"Response: {response}")
