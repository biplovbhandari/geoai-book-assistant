# API Design

FastAPI with auto-generated OpenAPI spec at `/docs`. \
Multiple client and admin endpoints - overview below. \
Exact request/response shapes live in the code (Pydantic models); this document captures architectural decisions.

---

## Endpoint Overview

### Client (public, unauthenticated, rate-limited)

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Service readiness (retrieval + generation status) |
| `POST /search` | Retrieval only - ranked chunks, no generation |
| `POST /ask` | Full RAG - JSON response, no recording |
| `POST /chat` | Full RAG - SSE streaming, conversation tracking, DB recording |
| `POST /feedback` | Thumbs up/down on an interaction (upsert on conflict) |
| `POST /share` | Create shareable link for an interaction |
| `GET /share/{uuid}` | View shared Q&A - anyone with the link can view, no login required |

### Admin (`Authorization: Bearer`, validated against `ADMIN_TOKEN` env var)

| Group | Endpoints |
|-------|-----------|
| Conversations | List, detail, update metadata (title/summary/topic) |
| Interactions | List, detail, replay through current pipeline, update feedback tags |
| Pipeline Inspector | Run retrieval with overrides, build context from chunks (stateless, no recording) |
| Analytics | Citation heatmap, retrieval gaps, daily stats |
| Deployments | List, create, activate (immutable snapshots, atomic switch) |
| Prompts | List, detail, create (content-addressed SHA-256 dedup) |
| Evaluation | Eval sets CRUD, eval questions CRUD, run evaluation, view results |
| Debug / Pipeline Trace | `POST /debug/query` - retrieval with per-stage scores (Retrieval Inspector); `GET /debug/interactions` - paginated interaction history; `GET /debug/traces/{id}` - full trace for one interaction; `POST /debug/trace` - live full-pipeline trace (no recording); `POST /debug/groundedness` - claim-level groundedness check |
| Index Runs | List history (read-only, indexing triggered via CLI) |
| Metrics | `GET /metrics` — Prometheus exposition format (`text/plain`). Not in OpenAPI schema. |

---

## Design Decisions

**Three tiers of client endpoints: `/search`, `/ask`, `/chat`.**
`/search` is retrieval only (no LLM cost).
`/ask` runs the full pipeline but returns JSON without recording (for scripts, notebooks, testing).
`/chat` is the widget endpoint - SSE streaming, conversation tracking, and DB recording.
Each serves a different integration pattern and cost profile.

**All resource IDs are server-generated.**
`visitor_id`, `conversation_id`, and `interaction_id` are created by the server and returned in the SSE `meta` event.
The widget stores `visitor_id` and `conversation_id` in `localStorage` and sends them on subsequent requests.
No client-side UUID generation.

**SSE `meta` event fires before tokens.**
The first SSE event contains `interaction_id`, `conversation_id` (when new), `visitor_id` (when new), and the full citations array.
The widget can render citations and wire up feedback/sharing while tokens are still streaming.

**Cursor-based pagination.**
All list endpoints use keyset pagination on `(created_at, id)`.
Stable under concurrent inserts - no duplicates when new data arrives between pages.
Cursor is an opaque base64 token; the client passes it back verbatim.

**Pipeline inspector endpoints are stateless.**
`POST /admin/retrieve` and `POST /admin/context` return results without recording to the database.
`POST /admin/interactions/{id}/replay` re-runs a question through the current pipeline for side-by-side comparison, also without recording.
These are diagnostic tools for deployment preparation.

**Pipeline inspector allows parameter overrides.**
The admin can override `top_k`, `strategy` (dense/hybrid), and `reranker` (on/off) to experiment before creating a deployment.
The response includes `config_used` so the admin sees exactly which settings produced the results.

**Feedback uses upsert semantics.**
`POST /feedback` returns 201 for first submission, 200 for update.
The widget doesn't need to detect duplicates or switch HTTP methods.
If the user changes their mind (hits the wrong button), the widget sends POST again and the server updates the existing row.

**Immutable deployments with atomic activation.**
`POST /admin/deployments` creates inactive.
`PATCH /admin/deployments/{id}/activate` atomically switches active deployment in one transaction.
Partial unique index enforces at most one active deployment.

**Content-addressed prompt versioning.**
`POST /admin/prompts` computes SHA-256 of the content.
Duplicate content returns the existing row (200) instead of creating a duplicate (201).

**Eval runs are long-running (202 Accepted).**
`POST /admin/eval-sets/{id}/run` returns immediately with the `eval_run_id` and `status: "running"`.
Results are populated in the background as questions are processed.
`eval_runs.status` tracks progress: `pending`, `running`, `completed`, `failed`.
Poll `GET /admin/eval-runs/{id}` to check status.

**DB recording is atomic.**
Interactions, citations, and traces are written in one transaction after streaming completes.
The admin never sees partial interactions (response without traces, interaction without citations).

**`X-Request-ID` on all responses.**
Every API response includes an `X-Request-ID` header.
If the incoming request carries one, it is forwarded; otherwise the server generates a UUID4.
The same ID propagates through structured logs for end-to-end request correlation.

**Admin auth is a single Bearer token.**
Validated against `ADMIN_TOKEN` environment variable.
No per-user identity - single admin for now.
Upgrade path: `admin_users` table with per-user tokens when multiple admins are needed.
