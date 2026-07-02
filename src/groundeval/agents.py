"""Agents under test.

An agent takes a prompt and returns raw text. v0.1 shipped one deliberately
minimal agent — a single Claude call. v0.2 adds the first tool-using agent:
a LangGraph loop whose tools wrap HealthChain's FHIR operations. Adding an
agent means adding a class with a `name` and a `run` method.
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


def _message_text(content) -> str:
    """LangChain message content is a str or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


class HealthChainToolAgent:
    """LangGraph ReAct agent whose tools wrap HealthChain FHIR operations.

    The agent decides when to look up codes, build resources, and validate —
    the eval measures whether that loop actually lands correct, safe FHIR.
    LangGraph/LangChain pick up LangSmith tracing from the same env vars
    `tracing.init_tracing` sets, so tool calls appear in traces with no
    extra wiring. Imports are lazy so the extraction path doesn't pay for
    the LangChain stack.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
        from langchain_anthropic import ChatAnthropic
        from langgraph.prebuilt import create_react_agent

        from groundeval.hc_tools import get_langchain_tools

        self.model = model
        self._graph = create_react_agent(
            ChatAnthropic(model=model, max_tokens=max_tokens, max_retries=5),
            get_langchain_tools(),
        )

    @property
    def name(self) -> str:
        return f"langgraph+{self.model}"

    def run(self, prompt: str) -> str:
        state = self._graph.invoke(
            {"messages": [("user", prompt)]},
            config={"recursion_limit": 25},
        )
        return _message_text(state["messages"][-1].content)
