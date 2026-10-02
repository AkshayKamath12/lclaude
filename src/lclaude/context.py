"""Complete prompt accounting, independent of transport, state, and rendering."""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from typing import Any, Protocol


@dataclass(frozen=True)
class PromptCount:
    instructions: int
    conversation: int
    overhead: int
    estimated: bool = True
    tools: int = 0

    @property
    def total(self) -> int:
        return self.instructions + self.conversation + self.overhead + self.tools


class TokenCounter(Protocol):
    """Adapters must count the full chat template, not just tokenize content."""

    def count(self, messages: Sequence[Mapping[str, Any]],
              tools: Sequence[Mapping[str, Any]] = ()) -> PromptCount: ...


class ConservativeTokenCounter:
    """Approximate content at three UTF-8 bytes per token, plus chat framing.

    This is a display heuristic, not a fit guarantee. It leaves more headroom
    than the common four-character heuristic without treating every byte as a
    token. Actual tokenization and templates vary; only Ollama's completed
    prompt_eval_count is reported as exact.
    """

    def count(self, messages: Sequence[Mapping[str, Any]],
              tools: Sequence[Mapping[str, Any]] = ()) -> PromptCount:
        instructions = conversation = 0
        for message in messages:
            text = message["content"]
            metadata = {k: v for k, v in message.items() if k not in ("role", "content")}
            if metadata:
                text += json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
            size = ceil(len(text.encode("utf-8")) / 3)
            if message["role"] == "system":
                instructions += size
            else:
                conversation += size
        tool_bytes = json.dumps(tools, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        tool_count = ceil(len(tool_bytes) / 3) if tools else 0
        return PromptCount(instructions, conversation, 16 * (len(messages) + 1), tools=tool_count)


@dataclass(frozen=True)
class ContextBudget:
    response_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.response_tokens <= 0:
            raise ValueError("Reply tokens must be positive.")


class ContextBudgetError(Exception):
    """The next inference request cannot fit its context allocation."""


def assemble_messages(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    limit: int | None, budget: ContextBudget,
) -> tuple[list[dict[str, Any]], PromptCount]:
    """Count the complete request and enforce its budget without modifying history."""
    payload = deepcopy(messages)
    count = ConservativeTokenCounter().count(payload, tools)
    if limit is None:
        raise ContextBudgetError(
            "Cannot determine the model's context allocation. Set --num-ctx explicitly."
        )
    if count.total + budget.response_tokens > limit:
        raise ContextBudgetError(
            f"Estimated prompt ({count.total:,}) plus reply reserve "
            f"({budget.response_tokens:,}) exceeds the {limit:,}-token context. "
            "Increase --num-ctx, reduce --max-response-tokens, or start a new chat. "
            "Recorded command activity is preserved."
        )
    return payload, count
