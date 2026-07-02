"""Run tracing: LangSmith when configured, local JSON always.

If LANGSMITH_API_KEY is set, the `@traceable` decorators throughout the
harness send runs to LangSmith. Without a key they are inert passthroughs, so
the harness degrades gracefully to local-only logging — every run writes a
JSON record under runs/ either way.
"""

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "groundeval"


def init_tracing() -> bool:
    """Enable LangSmith tracing if an API key is available.

    Returns True when LangSmith tracing is active.
    """
    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)
        logger.info(
            "LangSmith tracing enabled (project=%s)", os.environ["LANGSMITH_PROJECT"]
        )
        return True
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    logger.info("No LANGSMITH_API_KEY found — logging locally only")
    return False


def _to_jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, set):
        return sorted(obj, key=str)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def write_local_trace(record: dict, out_dir: Path | str = "runs") -> Path:
    """Write the run record to runs/<timestamp>.json and return the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}.json"
    path.write_text(json.dumps(record, indent=2, default=_to_jsonable))
    return path
