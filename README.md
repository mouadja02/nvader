# NVIDIA Agentic Research Engineer
![Status](https://img.shields.io/badge/status-active_build-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Focus](https://img.shields.io/badge/focus-agentic_AI-76B900)

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
make install
nvader info
nvader roadmap
make test
make lint
```

## Developer Workflow

```bash
make install   # install package in editable mode with dev dependencies
make test      # run tests
make lint      # run Ruff checks
make check     # run lint + tests
make clean     # remove local cache artifacts
```

## Current Status

Day 1: Project foundation and architecture skeleton.