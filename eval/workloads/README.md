# Phase 2.4 — Eval workloads

Workload YAML schema is documented in
`src/gateway/eval/workloads.py` (top of file + dataclasses). A workload is
*one* of:

- `prompts:` — explicit per-prompt definitions (single-turn `messages` or
  multi-turn `turns`).
- `generator:` — programmatic synthesis (e.g. needle-in-haystack).
- `mix:` — weighted sampling from other workloads (`mixed_realistic.yaml`).
- `source_router_eval:` — bridge that pulls rows from
  `eval/router_eval.yaml` (so the Phase 1.3 corpus runs through this
  framework without duplication).

## Running

```bash
# inside the gateway container (host Postgres collides with Docker's 5432 — see
# history.md Phase 1.2)
docker exec llm-router-gateway-1 python -m gateway.eval \
    --workload simple_qa \
    --target gateway:auto \
    --api-key <key from tenants.yaml>

# Multi-target comparison
docker exec llm-router-gateway-1 python -m gateway.eval \
    --workload mixed_realistic \
    --target gateway:auto \
    --target gateway:claude-haiku-4-5 \
    --target direct:openai:gpt-4o-mini \
    --api-key <key>
```

Outputs JSON to `eval/results/{date}_{workload}_{target_slug}.json` and a
Markdown summary on stdout.

## Cross-gateway eval (LiteLLM)

```bash
docker compose -f docker-compose.eval.yml up -d litellm
LITELLM_URL=http://localhost:4000 \
LITELLM_API_KEY=$LITELLM_PROXY_API_KEY \
docker exec llm-router-gateway-1 python -m gateway.eval \
    --workload simple_qa \
    --target gateway:auto \
    --target litellm:gpt-4o-mini
```

## Workloads shipped at v1

| File | What |
|---|---|
| `simple_qa.yaml` | 12 short factual single-turn prompts (fast-qa tier) |
| `code_generation.yaml` | 10 HumanEval-style coding prompts |
| `long_context.yaml` | 6 needle-in-haystack prompts, ~8k chars haystack |
| `multi_turn.yaml` | 4 conversations (3–4 turns each) exercising sticky routing + compaction |
| `mixed_realistic.yaml` | Weighted blend, 20 sampled prompts (seed=7) |
| `router_correctness.yaml` | Phase 1.3 router corpus replayed through the framework |

## Targets

| Spec | What |
|---|---|
| `gateway:auto` | This gateway with `model: "auto"` (cascade router) |
| `gateway:<vm>` | This gateway with a fixed virtual model — sanity baseline |
| `direct:<provider>:<model>` | In-process provider call, no proxy overhead |
| `litellm:<model>` | LiteLLM proxy (see `docker-compose.eval.yml`) |

Repeat `--target` to compare side-by-side; the reporter prints a comparison
table when more than one target ran.
