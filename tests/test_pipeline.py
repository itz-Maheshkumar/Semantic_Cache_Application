"""
test_pipeline.py — Unit and integration tests for Module 4 (LLM Interface, Request Logger & Pipeline).
"""

import base64
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from openai import OpenAIError

from src.audio_transcriber import AudioTranscriber
from src.cache import SemanticCache
from src.embedder import Embedder
from src.llm import LLMClient
from src.models import CacheResult, RequestLog
from src.pipeline import CachePipeline
from src.request_logger import RequestLogger

# A tiny (1x1, red) valid PNG, used to test generate_vision()'s raw-bytes
# encoding path without shipping a real image fixture file.
_TINY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.fixture(scope="module")
def embedder():
    """Shared Embedder instance for unit tests."""
    return Embedder()


@pytest.fixture
def temp_env(tmp_path: Path):
    """Fixture providing isolated temporary paths for SQLite DB, index, and metadata."""
    db_path = tmp_path / "test_logs.db"
    index_path = tmp_path / "test_cache.faiss"
    metadata_path = tmp_path / "test_metadata.json"
    return db_path, index_path, metadata_path


@pytest.fixture
def mock_llm():
    """Mock LLMClient that returns predictable responses without calling real APIs."""
    client = MagicMock(spec=LLMClient)
    client.generate.side_effect = lambda query, system_prompt=None: f"Mock response to: '{query}'"
    return client


def test_request_logger_basic(temp_env):
    """Test RequestLogger database insertion, retrieval, stats, and clear."""
    db_path, _, _ = temp_env
    logger = RequestLogger(db_path=db_path)

    assert len(logger.get_all()) == 0

    log1 = RequestLog(
        query="What is AI?",
        response="Artificial Intelligence.",
        is_hit=False,
        similarity_score=0.0,
        latency_ms=1200.5,
    )
    log2 = RequestLog(
        query="Tell me about AI?",
        response="Artificial Intelligence.",
        is_hit=True,
        similarity_score=0.91,
        latency_ms=4.2,
        matched_query="What is AI?",
    )

    logger.log(log1)
    logger.log(log2)

    logs = logger.get_all()
    assert len(logs) == 2
    assert logs[0].query == "Tell me about AI?"  # Most recent first
    assert logs[0].is_hit is True
    assert logs[1].is_hit is False
    # Neither log above set `modality` — it should default/round-trip as
    # None rather than raising or coercing to some other value.
    assert logs[0].modality is None

    stats = logger.get_stats()
    assert stats["total_requests"] == 2
    assert stats["total_hits"] == 1
    assert stats["total_misses"] == 1
    assert stats["hit_rate_pct"] == 50.0

    logger.clear()
    assert len(logger.get_all()) == 0


def test_request_logger_persists_and_retrieves_modality(temp_env):
    """The `modality` label (added for multi-modal caching) should
    round-trip through SQLite like any other RequestLog field."""
    db_path, _, _ = temp_env
    logger = RequestLogger(db_path=db_path)

    logger.log(RequestLog(
        query="a photo", response="A cat.", is_hit=False,
        similarity_score=0.0, latency_ms=800.0, modality="image",
    ))
    logger.log(RequestLog(
        query="How do I reset my password?", response="Go to settings.",
        is_hit=False, similarity_score=0.0, latency_ms=650.0, modality="audio",
    ))

    logs = logger.get_all()
    assert {entry.modality for entry in logs} == {"image", "audio"}


def test_request_logger_migrates_legacy_db_without_modality_column(tmp_path: Path):
    """A DB file created before multi-modal support existed won't have the
    `modality` column. RequestLogger must migrate it in place on open
    (ALTER TABLE), not crash, and existing rows should come back with
    modality=None rather than losing data."""
    import sqlite3

    db_path = tmp_path / "legacy_logs.db"

    # Build a pre-migration schema by hand — exactly what request_logs
    # looked like before this feature added the `modality` column.
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE request_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            is_hit INTEGER NOT NULL,
            similarity_score REAL NOT NULL,
            latency_ms REAL NOT NULL,
            timestamp TEXT NOT NULL,
            matched_query TEXT
        )
    """)
    conn.execute(
        "INSERT INTO request_logs (query, response, is_hit, similarity_score, latency_ms, timestamp, matched_query) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("legacy query", "legacy response", 0, 0.0, 42.0, "2023-01-01T00:00:00+00:00", None),
    )
    conn.commit()
    conn.close()

    # Opening a RequestLogger against this file should migrate it silently.
    logger = RequestLogger(db_path=db_path)

    logs = logger.get_all()
    assert len(logs) == 1
    assert logs[0].query == "legacy query"
    assert logs[0].modality is None  # backfilled column, no data to have had

    # And the migrated DB should accept new writes with modality set.
    logger.log(RequestLog(
        query="a new photo", response="A dog.", is_hit=False,
        similarity_score=0.0, latency_ms=10.0, modality="image",
    ))
    logs = logger.get_all()
    assert len(logs) == 2
    assert logs[0].modality == "image"  # most recent first
    assert logs[1].modality is None


def test_pipeline_hit_miss_flow(embedder: Embedder, mock_llm: MagicMock, temp_env):
    """Test full CachePipeline flow: Miss -> Cache -> Hit -> Stats."""
    db_path, index_path, metadata_path = temp_env

    cache = SemanticCache(
        embedder=embedder,
        threshold=0.85,
        index_path=index_path,
        metadata_path=metadata_path,
        auto_load=False,
    )
    request_logger = RequestLogger(db_path=db_path)

    pipeline = CachePipeline(
        cache=cache,
        llm_client=mock_llm,
        request_logger=request_logger,
    )

    # 1. First query — should be a MISS
    q1 = "How do I reset my user password?"
    res1 = pipeline.process_query(q1)

    assert res1["is_hit"] is False
    assert res1["query"] == q1
    assert "Mock response to:" in res1["response"]
    assert mock_llm.generate.call_count == 1

    # 2. Second query (paraphrased) — should be a HIT
    q2 = "How can I reset my password?"
    res2 = pipeline.process_query(q2)

    assert res2["is_hit"] is True
    assert res2["similarity_score"] >= 0.85
    assert res2["response"] == res1["response"]
    # LLM should NOT have been called a second time
    assert mock_llm.generate.call_count == 1
    assert res2["latency_ms"] < 200.0  # Fast cached response

    # 3. Third query (unrelated) — should be a MISS
    q3 = "What is the capital of France?"
    res3 = pipeline.process_query(q3)

    assert res3["is_hit"] is False
    assert mock_llm.generate.call_count == 2

    # 4. Verify request logger metrics
    stats = request_logger.get_stats()
    assert stats["total_requests"] == 3
    assert stats["total_hits"] == 1
    assert stats["total_misses"] == 2
    assert round(stats["hit_rate_pct"], 1) == 33.3


def test_pipeline_empty_query_raises(temp_env):
    """Test that submitting empty or whitespace queries raises ValueError."""
    db_path, index_path, metadata_path = temp_env
    pipeline = CachePipeline(
        cache=MagicMock(spec=SemanticCache),
        llm_client=MagicMock(spec=LLMClient),
        request_logger=RequestLogger(db_path=db_path),
    )

    with pytest.raises(ValueError):
        pipeline.process_query("")

    with pytest.raises(ValueError):
        pipeline.process_query("   ")


# ── Multi-Modal Caching: process_image_query / process_audio_query ─────────────
#
# LLM/vision/transcription calls are always mocked here — this project's CI
# has no OPENAI_API_KEY secret, so no test may depend on a real OpenAI call
# succeeding. MagicMock(spec=...) is used throughout (as with mock_llm above)
# so a typo'd method name or wrong signature fails the test immediately
# rather than silently no-op'ing.

def test_process_image_query_without_image_cache_raises(temp_env):
    """A pipeline built without an image_cache can't serve image queries —
    this should fail loudly, not silently fall back to the text cache."""
    db_path, _, _ = temp_env
    pipeline = CachePipeline(
        cache=MagicMock(spec=SemanticCache),
        llm_client=MagicMock(spec=LLMClient),
        request_logger=RequestLogger(db_path=db_path),
    )
    with pytest.raises(RuntimeError):
        pipeline.process_image_query(b"some-photo-bytes")


def test_process_audio_query_without_audio_transcriber_raises(temp_env):
    db_path, _, _ = temp_env
    pipeline = CachePipeline(
        cache=MagicMock(spec=SemanticCache),
        llm_client=MagicMock(spec=LLMClient),
        request_logger=RequestLogger(db_path=db_path),
    )
    with pytest.raises(RuntimeError):
        pipeline.process_audio_query(b"some-audio-bytes")


def test_process_image_query_hit_flow(temp_env):
    """An image cache HIT should return the cached response without ever
    invoking the vision LLM."""
    db_path, _, _ = temp_env
    request_logger = RequestLogger(db_path=db_path)

    mock_image_cache = MagicMock(spec=SemanticCache)
    mock_image_cache.get.return_value = CacheResult(
        query="cat.png",
        response="A photo of a cat.",
        similarity_score=0.97,
        cached_at="2024-01-01T00:00:00+00:00",
        matched_query="cat.png",
    )
    mock_llm = MagicMock(spec=LLMClient)

    pipeline = CachePipeline(
        cache=MagicMock(spec=SemanticCache),
        llm_client=mock_llm,
        request_logger=request_logger,
        image_cache=mock_image_cache,
    )

    result = pipeline.process_image_query(b"cat-photo-bytes", image_label="cat.png")

    assert result["is_hit"] is True
    assert result["response"] == "A photo of a cat."
    assert result["modality"] == "image"
    mock_image_cache.get.assert_called_once_with(b"cat-photo-bytes", query_label="cat.png")
    mock_llm.generate_vision.assert_not_called()

    logs = request_logger.get_all()
    assert len(logs) == 1
    assert logs[0].modality == "image"
    assert logs[0].is_hit is True


def test_process_image_query_miss_flow(temp_env):
    """An image cache MISS should fall back to LLMClient.generate_vision()
    and store the fresh pair back into image_cache."""
    db_path, _, _ = temp_env
    request_logger = RequestLogger(db_path=db_path)

    mock_image_cache = MagicMock(spec=SemanticCache)
    mock_image_cache.get.return_value = None
    mock_image_cache.display_query.return_value = "image:deadbeefcafe"
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.generate_vision.return_value = "A dog running in a park."

    pipeline = CachePipeline(
        cache=MagicMock(spec=SemanticCache),
        llm_client=mock_llm,
        request_logger=request_logger,
        image_cache=mock_image_cache,
    )

    result = pipeline.process_image_query(b"dog-photo-bytes", prompt="What is this?")

    assert result["is_hit"] is False
    assert result["response"] == "A dog running in a park."
    assert result["modality"] == "image"
    assert result["query"] == "image:deadbeefcafe"
    mock_llm.generate_vision.assert_called_once_with(b"dog-photo-bytes", query="What is this?")
    mock_image_cache.put.assert_called_once_with(
        b"dog-photo-bytes", "A dog running in a park.", auto_save=True, query_label=None
    )

    logs = request_logger.get_all()
    assert len(logs) == 1
    assert logs[0].modality == "image"
    assert logs[0].is_hit is False


def test_process_audio_query_transcribes_then_reuses_text_pipeline(
    embedder: Embedder, mock_llm: MagicMock, temp_env
):
    """Audio isn't embedded/cached directly — it's transcribed, then run
    through the exact same text cache flow as process_query(). A second,
    paraphrased audio query should HIT the text cache, same as two
    paraphrased typed queries would."""
    db_path, index_path, metadata_path = temp_env

    cache = SemanticCache(
        embedder=embedder,
        threshold=0.85,
        index_path=index_path,
        metadata_path=metadata_path,
        auto_load=False,
    )
    request_logger = RequestLogger(db_path=db_path)
    mock_transcriber = MagicMock(spec=AudioTranscriber)
    mock_transcriber.transcribe.return_value = "How do I reset my password?"

    pipeline = CachePipeline(
        cache=cache,
        llm_client=mock_llm,
        request_logger=request_logger,
        audio_transcriber=mock_transcriber,
    )

    result1 = pipeline.process_audio_query(b"fake-wav-bytes-1")
    assert result1["is_hit"] is False
    assert result1["modality"] == "audio"
    assert result1["transcript"] == "How do I reset my password?"
    assert result1["query"] == "How do I reset my password?"
    mock_transcriber.transcribe.assert_called_once_with(b"fake-wav-bytes-1")

    mock_transcriber.transcribe.return_value = "How can I reset my password?"
    result2 = pipeline.process_audio_query(b"fake-wav-bytes-2")
    assert result2["is_hit"] is True
    assert result2["modality"] == "audio"
    assert result2["response"] == result1["response"]
    # The vision/text LLM should not have been called a second time.
    assert mock_llm.generate.call_count == 1

    logs = request_logger.get_all()
    assert len(logs) == 2
    assert all(entry.modality == "audio" for entry in logs)


def test_process_audio_query_raises_on_empty_transcript(temp_env):
    """A transcript that comes back empty/whitespace-only (e.g. silence)
    can't be cached or looked up — this should fail loudly rather than
    silently caching an empty string."""
    db_path, _, _ = temp_env
    mock_transcriber = MagicMock(spec=AudioTranscriber)
    mock_transcriber.transcribe.return_value = "   "

    pipeline = CachePipeline(
        cache=MagicMock(spec=SemanticCache),
        llm_client=MagicMock(spec=LLMClient),
        request_logger=RequestLogger(db_path=db_path),
        audio_transcriber=mock_transcriber,
    )

    with pytest.raises(ValueError):
        pipeline.process_audio_query(b"silence.wav")


# ── Multi-Modal Caching: LLMClient.generate_vision() ────────────────────────────
#
# Never calls the real OpenAI API (no OPENAI_API_KEY in CI) — the client's
# lazily-initialized `_client` is replaced directly with a MagicMock, the
# same trick used to skip LLMClient's API-key validation entirely.

def _llm_client_with_mock_openai(response_text: str, max_retries: int = 3):
    """An LLMClient wired to a mock OpenAI client that always returns
    `response_text` from chat.completions.create()."""
    client = LLMClient(api_key="sk-test-not-a-real-key", max_retries=max_retries)
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=response_text))]
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.return_value = fake_response
    client._client = mock_openai_client
    return client, mock_openai_client


def test_generate_vision_passes_through_a_url_unchanged():
    """A plain http(s) URL should be sent to the API as-is — no need to
    download-and-reencode an image the API can already fetch itself."""
    client, mock_openai_client = _llm_client_with_mock_openai("It's a red bicycle.")

    result = client.generate_vision("https://example.com/bike.jpg", query="What is this?")

    assert result == "It's a red bicycle."
    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
    content_parts = messages[-1]["content"]
    assert content_parts[0] == {"type": "text", "text": "What is this?"}
    assert content_parts[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/bike.jpg"},
    }


def test_generate_vision_encodes_raw_bytes_as_a_data_url():
    """Raw image bytes (e.g. an uploaded file) can't be fetched by the API,
    so they must be base64-encoded into a data: URL instead."""
    client, mock_openai_client = _llm_client_with_mock_openai("A tiny red square.")

    result = client.generate_vision(_TINY_PNG_BYTES, query="Describe this.")

    assert result == "A tiny red square."
    content_parts = mock_openai_client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    image_url = content_parts[1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")


def test_generate_vision_defaults_to_a_generic_prompt_when_query_omitted():
    client, mock_openai_client = _llm_client_with_mock_openai("A cat sitting on a windowsill.")

    client.generate_vision(_TINY_PNG_BYTES)

    content_parts = mock_openai_client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert content_parts[0]["text"] == "Describe this image."


def test_generate_vision_rejects_an_unsupported_input_type():
    client = LLMClient(api_key="sk-test-not-a-real-key")
    with pytest.raises(TypeError):
        client.generate_vision(12345, query="What is this?")


def test_generate_vision_retries_then_raises_runtime_error(monkeypatch):
    """Mirrors LLMClient.generate()'s retry contract: transient OpenAI
    errors are retried up to max_retries times, then surfaced as a
    RuntimeError rather than the raw OpenAIError."""
    client = LLMClient(api_key="sk-test-not-a-real-key", max_retries=2)
    mock_openai_client = MagicMock()
    mock_openai_client.chat.completions.create.side_effect = OpenAIError("transient failure")
    client._client = mock_openai_client
    monkeypatch.setattr("src.llm.time.sleep", lambda _seconds: None)  # skip real backoff delay

    with pytest.raises(RuntimeError):
        client.generate_vision(_TINY_PNG_BYTES, query="Describe this.")

    assert mock_openai_client.chat.completions.create.call_count == 2
