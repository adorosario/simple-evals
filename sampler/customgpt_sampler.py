import os
import uuid
from dotenv import load_dotenv
import time
import requests
load_dotenv()

from custom_types import SamplerBase

class CustomGPTSampler:
    def __init__(self, model_name: str, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.95, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, stop: str = None):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.api_key = os.environ.get("CUSTOMGPT_API_KEY")
        self.project_id = os.environ.get("CUSTOMGPT_PROJECT")

    def __call__(self, prompt_messages: list[dict]) -> str:
        prompt = prompt_messages[0]["content"]
        # Generate a new session ID for each call to avoid context contamination
        session_id = uuid.uuid4()
        while True:
            response = None
            base_url = f"https://app.customgpt.ai/api/v1/projects/{self.project_id}/conversations/{session_id}/messages"
            url = f"{base_url}?stream=false&lang=en"

            form_data = {
                "response_source": "default",
                "prompt": prompt
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }

            response = requests.post(url, data=form_data, headers=headers)
            if response.status_code == 200:
                return response.json()['data']["openai_response"]
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After", 30)
                print(f"Rate limit exceeded, retrying in {retry_after} seconds")
                time.sleep(int(retry_after))
                continue
            else:
                print(f"Error: {response.status_code}")
                try: 
                    print(response.json())
                except:
                    print(response.content)
                time.sleep(5)
                continue
            

    def _pack_message(self, role: str, content: str):
        return {"role": str(role), "content": content}

    @classmethod
    def from_config(cls, config: dict):
        return cls(
            model_name=config["model_name"],
            max_tokens=config.get("max_tokens", 1024),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.95),
            frequency_penalty=config.get("frequency_penalty", 0.0),
            presence_penalty=config.get("presence_penalty", 0.0),
            stop=config.get("stop", None),
        )

    def to_config(self):
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
        }

    @classmethod
    def from_env(cls):
        model_name = os.environ.get("CUSTOMGPT_MODEL_NAME")
        if model_name is None:
            raise ValueError("CUSTOMGPT_MODEL_NAME environment variable is not set")
        return cls(model_name)
    
