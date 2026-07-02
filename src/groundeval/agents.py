"""Agents under test.

An agent takes a prompt and returns raw text. v0.1 ships one deliberately
minimal agent — a single Claude call — because the harness is what's being
tested, not the agent. Adding a model later means adding a class with a
`name` and a `run` method.
"""

import anthropic
from langsmith import traceable

DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeAgent:
    """Single non-streaming Claude message, no tools, no system prompt."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
        self.model = model
        self.max_tokens = max_tokens
        # rate-limit 429s on consecutive large-context calls are routine;
        # the SDK default of 2 retries is not enough to ride them out
        self._client = anthropic.Anthropic(max_retries=5)

    @property
    def name(self) -> str:
        return self.model

    @traceable(run_type="llm", name="claude-agent")
    def run(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
