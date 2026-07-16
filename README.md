# planner-coder-reviewer-agent

Greenfield code-generation system that turns a free-text software ticket into a human-approved ZIP of generated project files. A LangGraph workflow sequences planner, coder, deterministic verification, automated review, and a human approval gate; a FastAPI adapter exposes the run over HTTP and a simple web UI.

## Features

- Submit a natural-language ticket and start a bounded, checkpointed workflow
- Planner → coder → verify → reviewer → human gate → packaging pipeline
- Human decisions: `approve`, `request_changes`, `replan`, or `abort`
- Isolated per-run workspaces under `workspaces/` and ZIP artifacts under `artifacts/`
- HTTP API for status, redacted traces, candidate file preview, and artifact download
- One-page chat UI at `/` for submitting tickets and reviewing gates

## Requirements

- Python 3.12 (project venv uses CPython 3.12.3)
- Azure OpenAI credentials for live planner / coder / reviewer LLM calls
- Packages listed in [`requirements.txt`](requirements.txt)

## Quick start

```bash
cd planner-coder-reviewer-agent

python3 -m venv code-generator-env
source code-generator-env/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then replace placeholder secrets with your values

export PYTHONPATH=src
uvicorn codegen_workflow.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) for the UI, or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for OpenAPI.

## Installation

1. Create and activate a virtual environment (the repo already includes `code-generator-env/` for local use).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put `src/` on `PYTHONPATH` when running the API or tests (there is no installable package metadata yet).

## Configuration

Copy [`.env.example`](.env.example) to `.env` and set values for your environment. Do not commit real secrets.

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_API_KEY` | yes* | Azure OpenAI API key (`OPENAI_API_KEY` is accepted as fallback) |
| `AZURE_OPENAI_ENDPOINT` | yes* | Azure resource endpoint (`OPENAI_BASE_URL` is accepted as fallback) |
| `OPENAI_API_VERSION` | yes* | API version (`AZURE_OPENAI_API_VERSION` is accepted as fallback) |
| `PLANNER_MODEL` | no | Azure deployment name for the planner (default `gpt-4.1-mini`) |
| `CODER_MODEL` | no | Azure deployment name for the coder |
| `REVIEWER_MODEL` | no | Azure deployment name for the reviewer |
| `WORKSPACE_BASE_DIR` | no | Parent directory for run workspaces (default `./workspaces`) |
| `ARTIFACT_BASE_DIR` | no | Directory for packaged artifacts (default `./artifacts`) |
| `LOG_LEVEL` | no | Logging level (default `INFO`) |
| `APP_ENV` | no | `development` or `production` |
| `ALLOWED_ORIGINS` | no | Comma-separated CORS origins (omit or empty to disable CORS) |
| `WORKFLOW_TIMEOUT_SECONDS` | no | Soft bound for synchronous graph invocations (default `600`) |

\*Required for live agent LLM calls. API tests use mocked graph nodes and do not need Azure credentials.

## Usage

### Web UI

With the server running, open `/` and submit a ticket. When the workflow pauses at the reviewer human gate, submit a decision from the UI.

### HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness checks (no LLM call) |
| `POST` | `/run-ticket` | Start a workflow from `{ "ticket": "...", "max_iterations": 1-4 }` |
| `POST` | `/runs/{workflow_id}/decision` | Resume with `{ "decision": "approve\|request_changes\|replan\|abort", "feedback": "..." }` |
| `GET` | `/runs/{workflow_id}` | Current run status |
| `GET` | `/runs/{workflow_id}/trace` | Redacted intermediate trace |
| `GET` | `/runs/{workflow_id}/files` | List files under the run `candidate/` tree |
| `GET` | `/runs/{workflow_id}/files/content?path=...` | Read one candidate file |
| `GET` | `/runs/{workflow_id}/artifact` | Download the approved ZIP when complete |

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/run-ticket \
  -H 'Content-Type: application/json' \
  -d '{"ticket":"Build a small Python CLI that prints hello"}'
```

`request_changes` and `replan` require non-empty `feedback`. Responses use `201`/`200` when a terminal step finishes and `202` when the run is paused for human review.

### Library entry points

```python
from codegen_workflow import build_graph, create_workflow, run_config_for_thread, create_app
```

`thread_id` in the LangGraph runnable config aligns with `workflow_id`, workspace directory name, and artifact naming.

## Architecture

```text
POST /run-ticket
  → initialize_workspace
  → planner
  → coder
  → verify
  → reviewer
  → reviewer_human_gate  ─┬─ approve → package_project → ZIP artifact
                          ├─ request_changes / replan → planner (revision)
                          └─ abort → end
```

The graph owns orchestration and checkpointing (`InMemorySaver` by default). Agent prompts and tools live under `src/codegen_workflow/nodes/` and `tools/`. The FastAPI layer in `src/codegen_workflow/api/` adapts HTTP to graph invoke/resume without duplicating agent logic.

## Project structure

```text
src/codegen_workflow/
  api/           # FastAPI app, routes, schemas, workflow service
  nodes/         # planner, coder, verification, reviewer, human gates
  schemas/       # plan / coder / review / verification / decision models
  tools/         # workspace mutation and read-only helpers
  graph.py       # LangGraph assembly and routing
  state.py       # shared workflow state
  packaging.py   # ZIP packaging node
  workspace.py   # workspace initialization
  llm.py         # Azure OpenAI client construction
tests/           # unit and API tests (mocked LLM nodes in API suite)
requirements.txt
.env.example     # environment variable template
workspaces/      # per-run workspaces (gitignored)
artifacts/       # packaged ZIP outputs
```

## Testing

From the repo root, with dependencies installed:

```bash
source code-generator-env/bin/activate
PYTHONPATH=src pytest tests -q
```

Scope collection to `tests/` so generated files under `workspaces/` are not picked up as tests.

## Development

- Run the API with reload:

```bash
PYTHONPATH=src uvicorn codegen_workflow.api.app:create_app --factory --reload --host 127.0.0.1 --port 8000
```

- Point your IDE interpreter at `code-generator-env/bin/python` and add `src` to the analysis path.
- New agent behavior belongs in `nodes/`; orchestration changes in `graph.py` / `routing.py`; HTTP surface in `api/`.
