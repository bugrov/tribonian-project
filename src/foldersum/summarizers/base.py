"""Language model backend interface."""

from typing import Protocol


class LlmClient(Protocol):
    """Anything that can answer a prompt: a local Ollama model, the Claude API, ..."""

    def generate(self, system: str, prompt: str, max_tokens: int) -> str:
        """Return the model's answer; raises LlmError on failure."""
        ...
