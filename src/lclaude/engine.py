"""Inference engine module for communicating with the local Ollama daemon."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import ollama


class OllamaEngineError(Exception):
    """Base exception for all failures originating within inference engine"""

class OllamaConnectionError(OllamaEngineError):
    """Base exception for network connection issues with Ollama"""

class ModelNotFoundError(OllamaEngineError):
    """Raised when specified model not in local registry"""

@dataclass(frozen=True)
class Usage:
    """Counts reported by Ollama for a completed request, never UI state."""

    prompt_eval_count: int | None = None
    eval_count: int | None = None


class InferenceEngine:
    """manages connections with Ollama and token streaming"""

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
        num_ctx: int | None = None,
        num_predict: int = 2048,
    ):
        if num_ctx is not None and num_ctx <= 0:
            raise ValueError("num_ctx must be positive")
        if num_predict <= 0:
            raise ValueError("num_predict must be positive")
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.last_usage: Usage | None = None
        self.model = model
        self.host = host.rstrip("/")
        self.ollama_client = ollama.Client(host=host, timeout=timeout)

    @contextmanager
    def _error_boundary(
        self, action: str = "inference"
    ) -> Generator[None, None, None]:
        """Centralized error boundary translating low-level network/Ollama errors."""
        try:
            yield
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama daemon at '{self.host}' during"
                f" {action}. Ensure 'ollama serve' is running."
            ) from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404 and action == "token streaming":
                raise ModelNotFoundError(
                    f"Model '{self.model}' not found by Ollama."
                ) from exc
            raise OllamaEngineError(
                f"Ollama server error [{exc.status_code}] during {action}:"
                f" {exc.error}"
            ) from exc
        except OllamaEngineError:
            # Let our own domain exceptions pass through without double-wrapping
            raise
        except Exception as exc:
            raise OllamaEngineError(
                f"Unexpected error during {action}: {exc}"
            ) from exc

    def list_models(self) -> list[str]:
        """List model names on the configured daemon without changing state."""
        with self._error_boundary("listing models"):
            response = self.ollama_client.list()

            models = response.get("models", [])
            model_names: list[str] = []
            for model in models:
                model_name = model.get("model") or model.get("name") or ""
                if model_name:
                    model_names.append(model_name)

            return sorted(set(model_names))

    def _resolve_model(self, name: str, models: list[str]) -> str:
        if name in models:
            return name
        for available in models:
            if available == f"{name}:latest" or name == f"{available}:latest":
                return available
        raise ModelNotFoundError(
            f"Model '{name}' is not available locally. "
            f"Run 'ollama pull {name}' in your terminal to download it."
        )

    def verify_ready(self, *, allow_fallback: bool = False) -> list[str]:
        """Validate startup selection and return the process model catalog."""
        models = self.list_models()
        try:
            self._resolve_model(self.model, models)
        except ModelNotFoundError:
            if not allow_fallback or not models:
                raise
            self.model = models[0]
        return models

    def set_model(self, name: str, models: list[str]) -> None:
        """Validate first so failed or interrupted selection preserves the model."""
        self.model = self._resolve_model(name, models)
        self.last_usage = None

    def context_limit(self, *, load: bool = False) -> int | None:
        """Query runtime allocation afresh; optionally load without sending a prompt."""
        if self.num_ctx is not None:
            return self.num_ctx
        with self._error_boundary("context discovery"):
            for attempt in range(2 if load else 1):
                response = self.ollama_client.ps()
                for model in response.get("models", []):
                    name = model.get("model") or model.get("name")
                    if name not in (self.model, f"{self.model}:latest") and (
                        not isinstance(name, str) or f"{name}:latest" != self.model
                    ):
                        continue
                    limit = model.get("context_length")
                    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
                        return limit
                if load and attempt == 0:
                    self.ollama_client.generate(model=self.model, prompt="", stream=False)
        return None

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        """Streams text tokens generated by model for ongoing conversation."""

        self.last_usage = None
        generation_options = dict(options or {})
        generation_options["num_predict"] = self.num_predict
        if self.num_ctx is not None:
            generation_options["num_ctx"] = self.num_ctx
        usage = None
        with self._error_boundary("token streaming"):
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options=generation_options
            )

            for chunk in response:
                if chunk.get("done"):
                    usage = Usage(chunk.get("prompt_eval_count"), chunk.get("eval_count"))
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content

            self.last_usage = usage
