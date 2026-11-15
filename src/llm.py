"""
llm.py — Wrapper client for LLM response generation on cache misses.

Talks to OpenAI's API by default, or to Hugging Face's Inference Providers
router or Groq's API (see LLM_PROVIDER in src/config.py) — all three are
reached through the same `openai` SDK, since each speaks the identical
OpenAI-compatible chat-completions wire format. Only the
api_key/model/base_url differ.
"""

import base64
import io
import time
from pathlib import Path
from typing import Optional, Union
from openai import OpenAI, OpenAIError
from PIL import Image

from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    LLM_MAX_RETRIES,
    LLM_PROVIDER,
    HF_API_KEY,
    HF_MODEL,
    HF_BASE_URL,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_BASE_URL,
)
from src.logger import get_logger

log = get_logger(__name__)

# Anything generate_vision() will accept for its `image` argument. A plain
# http(s) URL string is passed straight through to the API; anything else
# (a local file path, raw bytes, or a PIL.Image) is base64-encoded into a
# data URL, since the API has no way to fetch it itself.
VisionInput = Union[str, Path, bytes, Image.Image]

# Placeholder values shipped in .env.example — treated as "not configured".
_PLACEHOLDER_KEYS = {
    "sk-...your-key-here...",
    "hf_...your-token-here...",
    "gsk_...your-key-here...",
}

# Per-provider (api_key, model, base_url) resolvers. base_url=None means
# "use the openai SDK's own default (api.openai.com)".
_PROVIDERS = {
    "openai": lambda: (OPENAI_API_KEY, OPENAI_MODEL, None),
    "huggingface": lambda: (HF_API_KEY, HF_MODEL, HF_BASE_URL),
    "groq": lambda: (GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL),
}


def _provider_defaults(provider: str):
    """Resolve (api_key, model, base_url) for a given LLM_PROVIDER value."""
    resolver = _PROVIDERS.get(provider, _PROVIDERS["openai"])
    return resolver()


class LLMClient:
    """
    Client for generating responses using an OpenAI-compatible chat API
    (OpenAI itself, or Hugging Face's Inference Providers router), with
    automated retry handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = LLM_MAX_RETRIES,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: Optional API key. Defaults to OPENAI_API_KEY/HF_API_KEY
                from config, chosen by `provider`.
            model: Model identifier. Defaults to OPENAI_MODEL/HF_MODEL from
                config, chosen by `provider`.
            max_retries: Maximum retry attempts for transient API failures.
            provider: "openai" or "huggingface". Defaults to LLM_PROVIDER
                from config.
            base_url: Optional API base URL override. Defaults to the
                provider's own endpoint (OpenAI's default, or HF_BASE_URL).
        """
        self.provider = (provider or LLM_PROVIDER).strip().lower()
        default_api_key, default_model, default_base_url = _provider_defaults(self.provider)
        self.api_key = api_key or default_api_key
        self.model = model or default_model
        self.base_url = base_url or default_base_url
        self.max_retries = max_retries
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of the API client."""
        if self._client is None:
            if not self.api_key or self.api_key in _PLACEHOLDER_KEYS:
                env_var = {
                    "huggingface": "HF_API_KEY",
                    "groq": "GROQ_API_KEY",
                }.get(self.provider, "OPENAI_API_KEY")
                raise ValueError(
                    f"No API key configured for provider '{self.provider}'. "
                    f"Please set {env_var} in your .env file."
                )
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate(self, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a text response for a given query via OpenAI API.

        Args:
            query: Input user prompt string.
            system_prompt: Optional system instruction prompt.

        Returns:
            Generated response string from the LLM.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful, clear, and concise AI assistant."
            })

        messages.append({"role": "user", "content": query})

        for attempt in range(1, self.max_retries + 1):
            try:
                log.info(f"Sending query to {self.provider} ({self.model}), attempt {attempt}/{self.max_retries}...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                )
                answer = response.choices[0].message.content or ""
                log.info(f"{self.provider} response received successfully ({len(answer)} chars).")
                return answer.strip()

            except OpenAIError as e:
                log.warning(f"{self.provider} API error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    log.error(f"Exhausted all retries for {self.provider} API call.")
                    raise RuntimeError(f"{self.provider} API call failed after {self.max_retries} attempts: {e}") from e

                # Exponential backoff: 1s, 2s, 4s...
                sleep_time = 2 ** (attempt - 1)
                time.sleep(sleep_time)
            except Exception as e:
                log.error(f"Unexpected error calling {self.provider} API: {e}")
                raise

    @staticmethod
    def _as_image_url(image: VisionInput) -> str:
        """
        Normalize any supported image input into the string the API's
        `image_url` content part expects: a plain http(s) URL is passed
        through untouched, everything else (a local path, raw bytes, or a
        PIL.Image) is base64-encoded into a `data:` URL, since only the
        caller's machine — not OpenAI's — can read a local file.
        """
        if isinstance(image, str) and image.startswith(("http://", "https://")):
            return image

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        elif isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, (bytes, bytearray)):
            pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image).__name__}. "
                "Expected a URL string, file path, raw bytes, or a PIL.Image."
            )

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def generate_vision(
        self,
        image: VisionInput,
        query: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a text response describing/answering about an image, via a
        vision-capable chat model. Requires whichever model is configured
        (OPENAI_MODEL's default, "gpt-4o-mini", supports image input; if
        using LLM_PROVIDER=huggingface, HF_MODEL must also be a
        vision-capable model — not every open model on the router is.

        Args:
            image: A URL string, local file path, raw image bytes, or a
                PIL.Image.
            query: Optional accompanying text prompt (e.g. "What's in this
                image?"). Defaults to a generic description request.
            system_prompt: Optional system instruction prompt.

        Returns:
            Generated response string from the LLM.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful, clear, and concise AI assistant."
            })

        image_url = self._as_image_url(image)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": query or "Describe this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        })

        for attempt in range(1, self.max_retries + 1):
            try:
                log.info(
                    f"Sending image query to {self.provider} ({self.model}), attempt {attempt}/{self.max_retries}..."
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                )
                answer = response.choices[0].message.content or ""
                log.info(f"{self.provider} vision response received successfully ({len(answer)} chars).")
                return answer.strip()

            except OpenAIError as e:
                log.warning(f"{self.provider} vision API error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    log.error(f"Exhausted all retries for {self.provider} vision API call.")
                    raise RuntimeError(
                        f"{self.provider} vision API call failed after {self.max_retries} attempts: {e}"
                    ) from e

                sleep_time = 2 ** (attempt - 1)
                time.sleep(sleep_time)
            except Exception as e:
                log.error(f"Unexpected error calling {self.provider} vision API: {e}")
                raise
