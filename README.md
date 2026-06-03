# NVIDIA Agentic Research Engineer

A production-style Agentic AI Research and Engineering Assistant built while preparing for the NVIDIA-Certified Professional: Agentic AI certification.

## Purpose

This project demonstrates practical agentic AI engineering:

- Resource ingestion for papers, docs, GitHub repos, and certification material
- RAG over technical knowledge
- Tool-using ReAct-style agents
- Short-term and long-term memory
- Multi-agent workflows
- Evaluation and benchmarking
- Guardrails, safety, and source attribution
- Logs, traces, monitoring, and reliability metrics
- NVIDIA-oriented integration paths: NIM, NeMo Guardrails, NeMo Agent Toolkit, Triton, TensorRT-LLM

## Project Identity

This is not a chatbot. It is an engineering assistant for AI builders.

It helps turn technical resources into grounded answers, implementation plans, code insights, benchmark reports, and documentation.

## Certification Coverage

1. Agent Architecture and Design
2. Agent Development
3. Evaluation and Tuning
4. Deployment and Scaling
5. Cognition, Planning, and Memory
6. Knowledge Integration and Data Handling
7. NVIDIA Platform Implementation
8. Run, Monitor, and Maintain
9. Safety, Ethics, and Compliance
10. Human-AI Interaction and Oversight

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

nvader info
nvader roadmap
pytest
```

## Current Status

Day 1: Project foundation and architecture skeleton.