"""Inference engine module for communicating with the local Ollama daemon."""

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

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


class OllamaTransport:
    """SDK model discovery with lossless JSON chat streaming.

    SDK message models discard unrecognized fields, including optional call IDs.
    Read chat chunks directly through HTTPX so validation sees the original arguments
    and identifiers. The public SDK still handles model discovery and loading.
    """

    def __init__(self, host: str, timeout: float) -> None:
        self._models = ollama.Client(host=host, timeout=timeout)
        self.host = host.rstrip("/")
        self.timeout = timeout

    def list(self) -> ollama.ListResponse:
        return self._models.list()

    def ps(self) -> ollama.ProcessResponse:
        return self._models.ps()

    def generate(self, model: str, prompt: str,
                 stream: Literal[False] = False) -> ollama.GenerateResponse:
        return self._models.generate(model=model, prompt=prompt, stream=stream)

    def chat(self, *, model: str, messages: Sequence[dict[str, Any]],
             tools: Sequence[dict[str, Any]] | None, stream: bool,
             options: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
        with httpx.stream(
            "POST", f"{self.host}/api/chat",
            json={"model": model, "messages": messages, "tools": tools,
                  "stream": stream, "options": options}, timeout=self.timeout,
        ) as response:
            if response.is_error:
                response.read()
                raise ollama.ResponseError(response.text, response.status_code)
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if not isinstance(chunk, dict):
                    raise OllamaEngineError("Ollama returned a non-object chat chunk")
                if chunk.get("error"):
                    raise ollama.ResponseError(str(chunk["error"]))
                yield chunk


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
        self.last_tool_calls: list[dict[str, Any]] = []
        self.model = model
        self.host = host.rstrip("/")
        self.ollama_client = OllamaTransport(host=host, timeout=timeout)

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
        tools: list[dict[str, Any]] | None = None,
    ) -> Generator[str, None, None]:
        """Stream text; publish structured calls and usage only after a completed response."""
        self.last_usage = None
        self.last_tool_calls = []
        generation_options = dict(options or {})
        generation_options["num_predict"] = self.num_predict
        if self.num_ctx is not None:
            generation_options["num_ctx"] = self.num_ctx
        calls: list[dict[str, Any]] = []
        usage = None
        done = False
        with self._error_boundary("token streaming"):
            response = self.ollama_client.chat(
                model=self.model, messages=messages, tools=tools,
                stream=True, options=generation_options,
            )
            try:
                for chunk in response:
                    message = chunk.get("message", {})
                    for call in message.get("tool_calls") or []:
                        item = call.model_dump(exclude_none=True) if hasattr(
                            call, "model_dump"
                        ) else deepcopy(call)
                        if not isinstance(item, dict):
                            raise OllamaEngineError("Malformed structured tool call")
                        calls.append(item)
                    content = message.get("content") or ""
                    if content:
                        yield content
                    if chunk.get("done"):
                        usage = Usage(chunk.get("prompt_eval_count"), chunk.get("eval_count"))
                        done = True
                if not done:
                    raise OllamaEngineError("Ollama stream ended before its completion marker")
            finally:
                close = getattr(response, "close", None)
                if close is not None:
                    close()
            self.last_tool_calls = calls
            self.last_usage = usage
