"""Agents under test.

An agent takes a prompt and returns raw text. v0.1 shipped one deliberately
minimal agent — a single Claude call. v0.2 adds the first tool-using agent:
a LangGraph loop whose tools wrap HealthChain's FHIR operations. Adding an
agent means adding a class with a `name` and a `run` method.
"""

import json

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


# The artifact under evaluation is what the agent BUILT through the tools, not
# what it typed. Steering the model onto the tools (rather than trusting it to
# narrate then hand-write JSON) is what makes the coding dimension measure the
# model's tool use instead of its parametric recall.
_TOOL_AGENT_SYSTEM = (
    "You are a clinical data agent that writes FHIR by USING TOOLS, never by "
    "hand. For every task you MUST: (1) call lookup_medication_code to get the "
    "RxNorm code from the site catalog — never guess or recall a code; (2) call "
    "build_medication_statement with a looked-up code to construct the resource; "
    "(3) call validate_medication_statement and fix any reported issues. Do not "
    "write resource JSON yourself. Once the built resource validates, reply with "
    "only that resource JSON."
)

_BUILD_TOOL = "build_medication_statement"


def _built_resource(messages) -> str | None:
    """The resource from the agent's last successful build-tool call, as JSON.

    Reading the prediction off the tool trail (not the final assistant message)
    means a run where the model narrates or hand-writes a resource without
    calling the build tool scores as a no-build miss, rather than a lucky regex
    parse of prose. Returns None if no build tool call succeeded.
    """
    for msg in reversed(messages):
        if getattr(msg, "name", None) != _BUILD_TOOL:
            continue
        content = msg.content
        try:
            payload = json.loads(content if isinstance(content, str) else _message_text(content))
        except (json.JSONDecodeError, TypeError):
            continue
        if payload.get("ok") and payload.get("resource") is not None:
            return json.dumps(payload["resource"])
    return None


class HealthChainToolAgent:
    """LangGraph ReAct agent whose tools wrap HealthChain FHIR operations.

    The agent decides when to look up codes, build resources, and validate —
    the eval measures whether that loop actually lands correct, safe FHIR.
    LangGraph/LangChain pick up LangSmith tracing from the same env vars
    `tracing.init_tracing` sets, so tool calls appear in traces with no
    extra wiring. Imports are lazy so the extraction path doesn't pay for
    the LangChain stack.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 4096):
        from langchain_anthropic import ChatAnthropic
        from langgraph.prebuilt import create_react_agent

        from groundeval.hc_tools import get_langchain_tools

        self.model = model
        self._system = _TOOL_AGENT_SYSTEM
        self._graph = create_react_agent(
            ChatAnthropic(model=model, max_tokens=max_tokens, max_retries=5),
            get_langchain_tools(),
        )

    @property
    def name(self) -> str:
        return f"langgraph+{self.model}"

    def run(self, prompt: str) -> str:
        state = self._graph.invoke(
            {"messages": [("system", self._system), ("user", prompt)]},
            config={"recursion_limit": 25},
        )
        # Evaluate what the agent built via the tools; fall back to its final
        # message only when it never produced a valid build (so the failure is
        # visible as a parse miss, not hidden behind a hand-written resource).
        built = _built_resource(state["messages"])
        if built is not None:
            return built
        return _message_text(state["messages"][-1].content)
