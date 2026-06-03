# NVIDIA Agentic AI Certification Map

This document maps project modules to certification domains.

## 1. Agent Architecture and Design

Project modules:
- `agents/`
- `tools/`
- `memory/`
- `core/state.py`
- `docs/architecture.md`

## 2. Agent Development

Project modules:
- `agents/base.py`
- `tools/registry.py`
- `tools/*`
- `cli.py`

## 3. Evaluation and Tuning

Project modules:
- `evaluation/`
- `evals/`
- `tests/`

## 4. Deployment and Scaling

Project modules:
- `Dockerfile`
- `docker-compose.yml`
- `api/`
- `docs/deployment.md`

## 5. Cognition, Planning, and Memory

Project modules:
- `memory/`
- `agents/orchestrator.py`
- `core/state.py`

## 6. Knowledge Integration and Data Handling

Project modules:
- `ingestion/`
- `retrieval/`
- `data/`

## 7. NVIDIA Platform Implementation

Project modules:
- `nvidia/`
- `guardrails/nemo_guardrails_adapter.py`

## 8. Run, Monitor, and Maintain

Project modules:
- `core/tracing.py`
- `data/traces/`
- `evaluation/reports.py`

## 9. Safety, Ethics, and Compliance

Project modules:
- `guardrails/`
- `docs/safety-policy.md`

## 10. Human-AI Interaction and Oversight

Project modules:
- `cli.py`
- future API/UI
- human approval workflow