"""
llm.py — Wrapper client for OpenAI API response generation on cache misses.
"""

import base64
import io
import time
from pathlib import Path
from typing import Optional, Union
from openai import OpenAI, OpenAIError
from PIL import Image

from src.config import OPENAI_API_KEY, OPENAI_MODEL, LLM_MAX_RETRIES
from src.logger import get_logger

log = get_logger(__name__)

# Anything generate_vision() will accept for its `image` argument. A plain
# http(s) URL string is passed straight through to the API; anything else
# (a local file path, raw bytes, or a PIL.Image) is base64-encoded into a
# data URL, since the API has no way to fetch it itself.
VisionInput = Union[str, Path, bytes, Image.Image]


class LLMClient:
    """
    Client for generating responses using OpenAI API with automated retry handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = OPENAI_MODEL,
        max_retries: int = LLM_MAX_RETRIES,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: Optional OpenAI API key. Defaults to OPENAI_API_KEY from config.
            model: OpenAI model identifier (defaults to OPENAI_MODEL).
            max_retries: Maximum retry attempts for transient API failures.
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.max_retries = max_retries
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of the OpenAI client."""
        if self._client is None:
            if not self.api_key or self.api_key == "sk-...your-key-here...":
                raise ValueError(
                    "OpenAI API key is missing or unconfigured. "
                    "Please set OPENAI_API_KEY in your .env file."
                )
            self._client = OpenAI(api_key=self.api_key)
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
                log.info(f"Sending query to OpenAI ({self.model}), attempt {attempt}/{self.max_retries}...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                )
                answer = response.choices[0].message.content or ""
                log.info(f"OpenAI response received successfully ({len(answer)} chars).")
                return answer.strip()

            except OpenAIError as e:
                log.warning(f"OpenAI API error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    log.error("Exhausted all retries for OpenAI API call.")
                    raise RuntimeError(f"OpenAI API call failed after {self.max_retries} attempts: {e}") from e
                
                # Exponential backoff: 1s, 2s, 4s...
                sleep_time = 2 ** (attempt - 1)
                time.sleep(sleep_time)
            except Exception as e:
                log.error(f"Unexpected error calling OpenAI API: {e}")
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
        vision-capable OpenAI chat model (the default OPENAI_MODEL,
        "gpt-4o-mini", supports image input).

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
                    f"Sending image query to OpenAI ({self.model}), attempt {attempt}/{self.max_retries}..."
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                )
                answer = response.choices[0].message.content or ""
                log.info(f"OpenAI vision response received successfully ({len(answer)} chars).")
                return answer.strip()

            except OpenAIError as e:
                log.warning(f"OpenAI vision API error on attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    log.error("Exhausted all retries for OpenAI vision API call.")
                    raise RuntimeError(
                        f"OpenAI vision API call failed after {self.max_retries} attempts: {e}"
                    ) from e

                sleep_time = 2 ** (attempt - 1)
                time.sleep(sleep_time)
            except Exception as e:
                log.error(f"Unexpected error calling OpenAI vision API: {e}")
                raise
