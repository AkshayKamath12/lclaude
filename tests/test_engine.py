"""Unit tests for the InferenceEngine and its error handling."""

import unittest
from unittest.mock import patch

import httpx
import ollama

from lclaude.engine import (
    InferenceEngine,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaEngineError,
)


class TestInferenceEngine(unittest.TestCase):
    """Test suite for InferenceEngine transport and error handling."""

    def setUp(self) -> None:
        """Runs before each test: sets up an isolated engine instance."""
        self.engine = InferenceEngine(
            model="qwen2.5:7b-instruct",
            host="http://localhost:11434",
            timeout=5.0,
        )

    # --------------------------------------------------------------------------
    # 1. READINESS & MODEL CHECKS
    # --------------------------------------------------------------------------

    def test_verify_ready_success_exact_match(self) -> None:
        """Passes when the model exists exactly as named in the daemon registry."""
        mock_response = {
            "models": [{"model": "qwen2.5:7b-instruct"}]
        }
        with patch.object(self.engine.ollama_client, "list", return_value=mock_response):
            # Should not raise any exception
            self.engine.verify_ready()

    def test_verify_readiness_success_latest_tag(self) -> None:
        """Passes when the daemon appends ':latest' to the model name."""
        mock_response = {
            "models": [{"name": "qwen2.5:7b-instruct:latest"}]
        }
        with patch.object(self.engine.ollama_client, "list", return_value=mock_response):
            self.engine.verify_ready()

    def test_verify_readiness_missing_model_raises_domain_error(self) -> None:
        """Raises ModelNotFoundError when the requested model is not installed."""
        mock_response = {
            "models": [{"model": "llama3:8b"}]
        }
        with patch.object(self.engine.ollama_client, "list", return_value=mock_response):
            with self.assertRaises(ModelNotFoundError) as ctx:
                self.engine.verify_ready()
            self.assertIn("not available locally", str(ctx.exception))

    def test_verify_readiness_daemon_unreachable_raises_connection_error(self) -> None:
        """Translates httpx.ConnectError into OllamaConnectionError."""
        with patch.object(
            self.engine.ollama_client,
            "list",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with self.assertRaises(OllamaConnectionError) as ctx:
                self.engine.verify_ready()
            self.assertIn("Could not connect to Ollama daemon", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 2. TOKEN STREAMING & TRANSLATION
    # --------------------------------------------------------------------------

    def test_stream_chat_yields_tokens_sequentially(self) -> None:
        """Verifies that stream_chat unwraps and yields text content chunks."""
        mock_chunks = [
            {"message": {"role": "assistant", "content": "Hello"}},
            {"message": {"role": "assistant", "content": " world"}},
            {"message": {"role": "assistant", "content": "!"}},
            {"message": {"role": "assistant", "content": ""}},  # empty done token
        ]
        with patch.object(self.engine.ollama_client, "chat", return_value=iter(mock_chunks)):
            messages = [{"role": "user", "content": "Hi"}]
            tokens = list(self.engine.stream_chat(messages))

            self.assertEqual(tokens, ["Hello", " world", "!"])

    def test_stream_chat_midstream_disconnect_raises_connection_error(self) -> None:
        """Verifies that the _error_boundary context manager protects active streaming."""
        def broken_generator():
            yield {"message": {"content": "First token"}}
            raise httpx.ConnectError("Socket closed abruptly")

        with patch.object(self.engine.ollama_client, "chat", return_value=broken_generator()):
            generator = self.engine.stream_chat([{"role": "user", "content": "Test"}])
            
            # The first token yields successfully
            self.assertEqual(next(generator), "First token")
            
            # The drop on token 2 should be caught and translated
            with self.assertRaises(OllamaConnectionError):
                next(generator)

    def test_stream_chat_server_error_mapping(self) -> None:
        """Maps 500/internal errors to OllamaEngineError."""
        with patch.object(
            self.engine.ollama_client,
            "chat",
            side_effect=ollama.ResponseError("Model crashed in VRAM", status_code=500),
        ):
            with self.assertRaises(OllamaEngineError) as ctx:
                list(self.engine.stream_chat([{"role": "user", "content": "Crash"}]))
            self.assertIn("Ollama server error [500]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()