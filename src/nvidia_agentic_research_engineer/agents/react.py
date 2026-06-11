"""
ReAct (Reason + Act) agent implementation.

Uses an OpenAI-compatible LLM to run a Thought → Action → Observation loop,
executing tools from the registry until a Final Answer is produced or
max_steps is exceeded.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List

from nvidia_agentic_research_engineer.agents.base import (
    AgentRun,
    AgentState,
    AgentStep,
    BaseAgent,
)
from nvidia_agentic_research_engineer.tools.registry import ToolRegistry

_SYSTEM_TEMPLATE = """\
You are a helpful research assistant. Answer the user's question using the tools provided.

{tools_prompt}

To use a tool, respond EXACTLY in this format:

Thought: <your reasoning>
Action: <tool_name>
Action Input: {{"param": "value"}}

After receiving the tool's observation, continue reasoning.

When you have enough information to answer, respond EXACTLY:

Thought: <final reasoning>
Final Answer: <your complete answer>

IMPORTANT:
- Always start with a Thought.
- Use only the tools listed above.
- Action Input must be valid JSON.
- Do NOT combine Action and Final Answer in the same response.
"""


def _parse_llm_response(text: str) -> Dict[str, Any]:
    """Parse Thought / Action / Action Input / Final Answer from LLM output."""
    result: Dict[str, Any] = {}

    thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:)|$)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    final_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_match:
        result["final_answer"] = final_match.group(1).strip()
        return result

    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
    if action_match:
        result["action"] = action_match.group(1).strip()

    input_match = re.search(r"Action Input:\s*(.+?)(?:\n(?![\s{])|$)", text, re.DOTALL)
    if input_match:
        raw = input_match.group(1).strip()
        try:
            result["action_input"] = json.loads(raw)
        except json.JSONDecodeError:
            result["action_input"] = {"query": raw}

    return result


class ReActAgent(BaseAgent):
    """Concrete ReAct agent backed by an OpenAI-compatible LLM."""

    def __init__(
        self,
        name: str,
        registry: ToolRegistry,
        *,
        llm_client: Any = None,
        model: str = "nvidia/nemotron-3-ultra-550b-a55b",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        super().__init__(name, registry)
        self.model = model
        self.temperature = temperature

        if llm_client is not None:
            self._client = llm_client
        else:
            import openai
            import os

            self._client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key or os.environ.get("NVIDIA_API_KEY", ""),
            )

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def execute(self, task: str, max_steps: int = 10) -> AgentRun:
        run = AgentRun(
            agent_name=self.name,
            task=task,
            started_at=datetime.now(),
        )
        self._current_run = run

        system_prompt = _SYSTEM_TEMPLATE.format(
            tools_prompt=self.registry.generate_tools_prompt(),
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        for step_num in range(1, max_steps + 1):
            # ---------- Think ----------
            step = self._think(task, run.steps, messages=messages)
            step.step_number = step_num

            parsed = _parse_llm_response(step.thought or "")
            step.thought = parsed.get("thought", step.thought)

            # ---------- Final Answer? ----------
            if "final_answer" in parsed:
                step.state = AgentState.FINISHED
                step.observation = parsed["final_answer"]
                run.steps.append(step)
                run.final_answer = parsed["final_answer"]
                run.success = True
                break

            # ---------- Action ----------
            action_name = parsed.get("action")
            action_input = parsed.get("action_input", {})
            if action_name:
                step.action = action_name
                step.action_input = action_input if isinstance(action_input, dict) else {}

                step = self._execute_tool_with_retry(step)
            else:
                step.state = AgentState.ERROR
                step.error = "LLM response contained no Action or Final Answer."

            run.steps.append(step)

            # Append assistant + observation to message history
            messages.append({"role": "assistant", "content": step.thought or ""})
            observation_text = step.observation if step.observation else (step.error or "No output.")
            messages.append({"role": "user", "content": f"Observation: {observation_text}"})

            if step.state == AgentState.FINISHED:
                run.final_answer = step.observation
                run.success = True
                break
        else:
            # max_steps exceeded without Final Answer
            run.final_answer = "Max steps exceeded without reaching a final answer."
            run.success = False

        run.finished_at = datetime.now()
        self._current_run = None
        return run

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _think(
        self,
        task: str,
        history: List[AgentStep],
        *,
        messages: List[Dict[str, str]] | None = None,
    ) -> AgentStep:
        """Call the LLM and return a step with the raw response as thought."""
        step = AgentStep(
            step_number=len(history) + 1,
            state=AgentState.THINKING,
            timestamp=datetime.now().isoformat(),
        )

        if messages is None:
            messages = [{"role": "user", "content": task}]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        step.thought = response.choices[0].message.content
        return step

    def _execute_tool_with_retry(self, step: AgentStep, max_retries: int = 1) -> AgentStep:
        """Execute a tool; on failure retry once, then report error as observation."""
        for attempt in range(max_retries + 1):
            try:
                result = self.registry.execute_tool(
                    step.action, **(step.action_input or {})
                )
                step.state = AgentState.OBSERVING
                step.observation = str(result)
                step.error = None
                return step
            except (ValueError, RuntimeError) as exc:
                step.error = str(exc)

        # All retries exhausted — report error as observation so loop continues
        step.state = AgentState.OBSERVING
        step.observation = f"Tool error: {step.error}"
        return step
