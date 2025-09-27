import base64
import time
from typing import Any

import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
from custom_types import MessageList, SamplerBase

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0,
        max_tokens: int = 1024,
        service_tier: str | None = None,
        response_format: dict | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.service_tier = service_tier
        self.response_format = response_format
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                # Prepare API parameters
                api_params = {
                    "model": self.model,
                    "messages": message_list,
                }

                # Use max_completion_tokens and default temperature for GPT-5 models
                if self.model.startswith('gpt-5'):
                    api_params["max_completion_tokens"] = self.max_tokens

                    # GPT-5 specific parameters for determinism
                    if self.seed is not None:
                        api_params["seed"] = self.seed
                    if self.reasoning_effort is not None:
                        api_params["reasoning_effort"] = self.reasoning_effort
                else:
                    api_params["temperature"] = self.temperature
                    api_params["max_tokens"] = self.max_tokens

                    # Non-GPT-5 models can still use seed for determinism
                    if self.seed is not None:
                        api_params["seed"] = self.seed

                # Add service_tier if specified and model supports it (only GPT-5 supports flex)
                if self.service_tier and self.model.startswith('gpt-5'):
                    api_params["service_tier"] = self.service_tier

                # Add response_format if specified (structured outputs)
                if self.response_format:
                    api_params["response_format"] = self.response_format

                response = self.client.chat.completions.create(**api_params)
                return response.choices[0].message.content
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
