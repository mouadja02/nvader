from __future__ import annotations

import json
from pathlib import Path

from nvidia_agentic_research_engineer.agents.base import AgentRun

def write_agent_trace(run: AgentRun, output_path: Path) -> None:
    """Serialize an AgentRun to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(run.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
