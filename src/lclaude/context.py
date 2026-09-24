"""Complete prompt accounting, independent of transport, state, and rendering."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Protocol


@dataclass(frozen=True)
class PromptCount:
    instructions: int
    conversation: int
    overhead: int
    estimated: bool = True

    @property
    def total(self) -> int:
        return self.instructions + self.conversation + self.overhead


class TokenCounter(Protocol):
    """Adapters must count the full chat template, not just tokenize content."""

    def count(self, messages: Sequence[Mapping[str, str]]) -> PromptCount: ...


class ConservativeTokenCounter:
    """Approximate content at three UTF-8 bytes per token, plus chat framing.

    This is a display heuristic, not a fit guarantee. It leaves more headroom
    than the common four-character heuristic without treating every byte as a
    token. Actual tokenization and templates vary; only Ollama's completed
    prompt_eval_count is reported as exact.
    """

    def count(self, messages: Sequence[Mapping[str, str]]) -> PromptCount:
        instructions = conversation = 0
        for message in messages:
            size = ceil(len(message["content"].encode("utf-8")) / 3)
            if message["role"] == "system":
                instructions += size
            else:
                conversation += size
        return PromptCount(instructions, conversation, 16 * (len(messages) + 1))


@dataclass(frozen=True)
class ContextBudget:
    response_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.response_tokens <= 0:
            raise ValueError("Reply tokens must be positive.")
