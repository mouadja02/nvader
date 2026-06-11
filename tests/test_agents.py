"""
Tests for the agent framework: ToolRegistry, BaseAgent, and ReActAgent.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

from nvidia_agentic_research_engineer.agents.base import (
    AgentRun,
    AgentState,
    BaseAgent,
)
from nvidia_agentic_research_engineer.agents.react import ReActAgent, _parse_llm_response
from nvidia_agentic_research_engineer.tools.registry import Tool, ToolParameter, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_echo_tool() -> Tool:
    """A simple tool that echoes its input."""
    return Tool(
        name="echo",
        description="Echoes the input text back.",
        parameters=[ToolParameter(name="text", type="str", description="Text to echo", required=True)],
        func=lambda text: f"Echo: {text}",
    )


def _make_failing_tool() -> Tool:
    """A tool that always raises."""
    def _fail(**kwargs):
        raise RuntimeError("boom")

    return Tool(
        name="fail_tool",
        description="Always fails.",
        parameters=[],
        func=_fail,
    )


def _make_fake_llm_client(responses: List[str]):
    """Return a fake OpenAI-like client that yields pre-canned responses in order."""
    call_idx = {"i": 0}

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    idx = call_idx["i"]
                    call_idx["i"] += 1
                    text = responses[idx] if idx < len(responses) else responses[-1]
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
                    )

    return FakeClient()


# ===========================================================================
# ToolRegistry tests
# ===========================================================================


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = _make_echo_tool()
        registry.register_tool(tool)
        assert registry.get_tool("echo") is tool

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register_tool(_make_echo_tool())
        with pytest.raises(ValueError, match="already registered"):
            registry.register_tool(_make_echo_tool())

    def test_get_missing_returns_none(self):
        registry = ToolRegistry()
        assert registry.get_tool("nonexistent") is None

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register_tool(_make_echo_tool())
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"

    def test_execute_tool(self):
        registry = ToolRegistry()
        registry.register_tool(_make_echo_tool())
        result = registry.execute_tool("echo", text="hello")
        assert result == "Echo: hello"

    def test_execute_missing_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="not found"):
            registry.execute_tool("missing")

    def test_execute_tool_error_raises_runtime(self):
        registry = ToolRegistry()
        registry.register_tool(_make_failing_tool())
        with pytest.raises(RuntimeError, match="boom"):
            registry.execute_tool("fail_tool")

    def test_generate_tools_prompt(self):
        registry = ToolRegistry()
        registry.register_tool(_make_echo_tool())
        prompt = registry.generate_tools_prompt()
        assert "echo" in prompt
        assert "text" in prompt


# ===========================================================================
# ToolResult tests
# ===========================================================================

class TestToolResult:
    def test_success_result(self):
        r = ToolResult(tool_name="echo", success=True, output="ok")
        assert r.success is True
        assert r.output == "ok"
        assert r.error is None

    def test_error_result(self):
        r = ToolResult(tool_name="echo", success=False, error="fail")
        assert r.success is False
        assert r.error == "fail"


# ===========================================================================
# BaseAgent._think() test (mock LLM via subclass)
# ===========================================================================

class TestBaseAgentThink:
    def test_think_returns_step(self):
        """The default _think on BaseAgent returns a step without calling an LLM."""
        class DummyAgent(BaseAgent):
            def execute(self, task, max_steps=10):
                return AgentRun(agent_name=self.name, task=task, started_at=__import__("datetime").datetime.now())

        registry = ToolRegistry()
        agent = DummyAgent(name="dummy", registry=registry)
        step = agent._think("test task", [])
        assert step.state == AgentState.THINKING
        assert step.step_number == 1
        assert "test task" in (step.thought or "")


# ===========================================================================
# _parse_llm_response tests
# ===========================================================================

class TestParseLLMResponse:
    def test_parse_final_answer(self):
        text = "Thought: I know the answer.\nFinal Answer: 42"
        parsed = _parse_llm_response(text)
        assert parsed["thought"] == "I know the answer."
        assert parsed["final_answer"] == "42"
        assert "action" not in parsed

    def test_parse_action(self):
        text = 'Thought: I need to search.\nAction: search_kb\nAction Input: {"query": "test"}'
        parsed = _parse_llm_response(text)
        assert parsed["thought"] == "I need to search."
        assert parsed["action"] == "search_kb"
        assert parsed["action_input"] == {"query": "test"}

    def test_parse_no_markers(self):
        parsed = _parse_llm_response("Just some random text")
        assert "final_answer" not in parsed
        assert "action" not in parsed


# ===========================================================================
# ReActAgent tests
# ===========================================================================

class TestReActAgent:
    def test_final_answer_on_first_step(self):
        """Agent immediately returns Final Answer without tool use."""
        client = _make_fake_llm_client([
            "Thought: I already know.\nFinal Answer: The answer is 42."
        ])
        registry = ToolRegistry()
        agent = ReActAgent(name="test", registry=registry, llm_client=client)
        run = agent.execute("What is 42?", max_steps=5)

        assert run.success is True
        assert "42" in (run.final_answer or "")
        assert len(run.steps) == 1
        assert run.steps[0].state == AgentState.FINISHED

    def test_tool_call_then_final_answer(self):
        """Agent calls a tool, gets observation, then gives Final Answer."""
        client = _make_fake_llm_client([
            'Thought: I need to search.\nAction: echo\nAction Input: {"text": "hello"}',
            "Thought: I got the echo.\nFinal Answer: The echo said hello.",
        ])
        registry = ToolRegistry()
        registry.register_tool(_make_echo_tool())
        agent = ReActAgent(name="test", registry=registry, llm_client=client)
        run = agent.execute("Echo hello", max_steps=5)

        assert run.success is True
        assert len(run.steps) == 2
        assert run.steps[0].action == "echo"
        assert "Echo: hello" in (run.steps[0].observation or "")
        assert "hello" in (run.final_answer or "")

    def test_tool_execution_error_recovery(self):
        """Tool raises → agent observes error → continues reasoning."""
        client = _make_fake_llm_client([
            'Thought: Let me try the tool.\nAction: fail_tool\nAction Input: {}',
            "Thought: The tool failed, I'll answer directly.\nFinal Answer: Tool was broken.",
        ])
        registry = ToolRegistry()
        registry.register_tool(_make_failing_tool())
        agent = ReActAgent(name="test", registry=registry, llm_client=client)
        run = agent.execute("Try the failing tool", max_steps=5)

        assert run.success is True
        assert run.steps[0].observation is not None
        assert "error" in run.steps[0].observation.lower() or "boom" in run.steps[0].observation.lower()
        assert len(run.steps) == 2

    def test_max_steps_enforcement(self):
        """Agent stops after max_steps even without a Final Answer."""
        # LLM always tries an action, never gives Final Answer
        never_ending = 'Thought: Still thinking.\nAction: echo\nAction Input: {"text": "loop"}'
        client = _make_fake_llm_client([never_ending] * 20)
        registry = ToolRegistry()
        registry.register_tool(_make_echo_tool())
        agent = ReActAgent(name="test", registry=registry, llm_client=client)
        run = agent.execute("Loop forever", max_steps=3)

        assert run.success is False
        assert len(run.steps) == 3
        assert "max steps" in (run.final_answer or "").lower()

    def test_missing_tool_reports_error_observation(self):
        """Calling a tool that doesn't exist results in an error observation."""
        client = _make_fake_llm_client([
            'Thought: Use nonexistent.\nAction: no_such_tool\nAction Input: {}',
            "Thought: That failed.\nFinal Answer: Could not find tool.",
        ])
        registry = ToolRegistry()
        agent = ReActAgent(name="test", registry=registry, llm_client=client)
        run = agent.execute("Use missing tool", max_steps=5)

        assert run.success is True
        assert "error" in (run.steps[0].observation or "").lower() or "not found" in (run.steps[0].observation or "").lower()
