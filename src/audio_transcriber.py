"""
audio_transcriber.py — Speech-to-text front end for audio queries.

Multi-modal caching supports audio not by embedding audio directly, but by
transcribing it to text first and then running that transcript through the
existing text SemanticCache unchanged (see CachePipeline.process_audio_query
in src/pipeline.py). Two spoken queries that say the same thing become the
same cache lookup problem the project already solves well — no separate
audio vector space, index, or similarity threshold to tune.

Uses OpenAI's audio transcription API (Whisper), mirroring src/llm.py's
lazy-client-plus-retry pattern so both OpenAI-backed clients behave the same
way operationally.
"""

import io
import time
from pathlib import Path
from typing import BinaryIO, Optional, Union

from openai import OpenAI, OpenAIError

from src.config import AUDIO_TRANSCRIPTION_MODEL, LLM_MAX_RETRIES, OPENAI_API_KEY
from src.logger import get_logger

log = get_logger(__name__)

# Anything transcribe() will accept as the audio source.
AudioInput = Union[str, Path, bytes, BinaryIO]


class AudioTranscriber:
    """
    Client for transcribing audio to text via OpenAI's Whisper API, with
    automated retry handling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = AUDIO_TRANSCRIPTION_MODEL,
        max_retries: int = LLM_MAX_RETRIES,
    ):
        """
        Initialize the AudioTranscriber.

        Args:
            api_key: Optional OpenAI API key. Defaults to OPENAI_API_KEY from config.
            model: OpenAI transcription model identifier (defaults to
                AUDIO_TRANSCRIPTION_MODEL, e.g. "whisper-1").
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

    @staticmethod
    def _as_file_object(audio: AudioInput):
        """
        Normalize any supported input into what the OpenAI SDK expects for
        the `file` parameter: a binary file-like object, or a
        (filename, data) tuple for raw bytes (a filename is required so the
        API can infer the audio format from its extension).
        """
        if isinstance(audio, (str, Path)):
            return open(audio, "rb")  # caller's `with` isn't possible here; closed by transcribe()
        if isinstance(audio, (bytes, bytearray)):
            return ("audio.wav", io.BytesIO(audio))
        return audio  # assume it's already a file-like object (has .read())

    def transcribe(self, audio: AudioInput) -> str:
        """
        Transcribe an audio query to text.

        Args:
            audio: A file path, raw audio bytes, or an open binary file-like
                object (e.g. `open("query.mp3", "rb")`).

        Returns:
            The transcribed text.
        """
        opened_here = isinstance(audio, (str, Path))
        file_obj = self._as_file_object(audio)

        try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    log.info(
                        f"Sending audio to OpenAI ({self.model}), attempt {attempt}/{self.max_retries}..."
                    )
                    response = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=file_obj,
                    )
                    transcript = (response.text or "").strip()
                    log.info(f"Audio transcribed successfully ({len(transcript)} chars).")
                    return transcript

                except OpenAIError as e:
                    log.warning(f"OpenAI transcription error on attempt {attempt}/{self.max_retries}: {e}")
                    if attempt == self.max_retries:
                        log.error("Exhausted all retries for OpenAI transcription call.")
                        raise RuntimeError(
                            f"OpenAI audio transcription failed after {self.max_retries} attempts: {e}"
                        ) from e

                    # Exponential backoff: 1s, 2s, 4s...
                    time.sleep(2 ** (attempt - 1))
                except Exception as e:
                    log.error(f"Unexpected error calling OpenAI transcription API: {e}")
                    raise
        finally:
            if opened_here and hasattr(file_obj, "close"):
                file_obj.close()
