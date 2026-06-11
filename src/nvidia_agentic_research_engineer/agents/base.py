"""
Base class for agents in the NVIDIA Agentic Research Engineer framework.
This module defines the abstract base class `BaseAgent`,
which outlines the essential methods and properties that all agents must implement.
It also includes a simple implementation of a tool registry to manage the tools that agents can use to perform tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from nvidia_agentic_research_engineer.tools.registry import ToolRegistry
from pydantic import BaseModel

class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    FINISHED = "finished"
    ERROR = "error"

class AgentStep(BaseModel):
    step_number: int
    state: AgentState
    thought: str | None = None # reasoning trace
    action: str | None = None # tool name called
    action_input: Dict[str, Any] | None = None # parameters for the tool
    observation: str | None = None # result from the tool
    error: str | None = None # error message if any
    timestamp: str | None = None # ISO format timestamp

class AgentRun(BaseModel):
    agent_name: str
    task: str
    steps: List[AgentStep] = []
    final_answer: str | None = None
    success: bool = False
    started_at: datetime
    finished_at: datetime | None = None

class BaseAgent(ABC):
    FINAL_ANSWER_ACTION = "final_answer"

    def __init__(self, name: str, registry: ToolRegistry, llm_config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.registry = registry
        self.llm_config = llm_config
        self._current_run: Optional[AgentRun] = None

    @abstractmethod
    def execute(self, task: str, max_steps: int = 10) -> AgentRun:
        """
        Execute the agent on a given task.
        This method should implement the main loop of the agent, including thinking, acting, and observing.
        It should return an AgentRun object containing the trace of the execution.
        """
        pass

    def _think(self, task: str, history: List[AgentStep]) -> AgentStep:
        """
        Generate the agent's thought process based on the task and execution history.
        This is a helper method that can be used within the run loop.
        Subclasses should override this to integrate with an LLM.
        """
        step_number = len(history) + 1
        return AgentStep(
            step_number=step_number,
            state=AgentState.THINKING,
            thought=f"Analyzing task: {task}",
            timestamp=datetime.now().isoformat(),
        )

    def _act(self, step: AgentStep) -> AgentStep:
        """
        Execute the tool specified in the step's action field using the registry.
        Returns the step updated with the observation or error.
        """
        step.state = AgentState.ACTING

        if not step.action:
            step.state = AgentState.ERROR
            step.error = "No action specified in the step."
            return step

        if step.action == self.FINAL_ANSWER_ACTION:
            step.state = AgentState.FINISHED
            step.observation = (step.action_input or {}).get("answer", "")
            return step

        try:
            result = self.registry.execute_tool(
                step.action, **(step.action_input or {})
            )
            step.state = AgentState.OBSERVING
            step.observation = str(result)
        except (ValueError, RuntimeError) as e:
            step.state = AgentState.ERROR
            step.error = str(e)

        step.timestamp = datetime.now().isoformat()
        return step

    def _should_stop(self, step: AgentStep) -> bool:
        """
        Determine whether the agent should stop execution based on the current step.
        Stops if the step reached a final answer or encountered an error.
        """
        return step.state in (AgentState.FINISHED, AgentState.ERROR)