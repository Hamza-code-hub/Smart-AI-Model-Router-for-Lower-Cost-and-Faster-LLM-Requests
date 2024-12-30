# LLM Gateway

An OpenAI-compatible HTTP gateway that routes chat completions across Anthropic, OpenAI, and local Ollama models — with a 3-tier cascade router that picks the cheapest model that can handle each request, cross-provider failover, two-layer caching, per-tenant rate limits and budgets, and a benchmarked eval framework.

**Headline result:** on a mixed realistic workload, auto-routing matched always-use-Opus quality (LLM-judge score 1.00 vs 0.99) at **88% lower cost** and **67% lower p50 latency**. Full methodology and caveats below — including the tests that failed.

```
FastAPI · Postgres + pgvector · Redis · Prometheus + Grafana + OpenTelemetry · Python 3.12
```

---

## How it works

```
                        POST /v1/chat/completions  (model: "auto")
                                      │
                              ┌───────▼────────┐
                              │  Auth + rate   │  per-tenant token bucket (Redis Lua),
                              │  limit + budget│  monthly USD budgets
                              └───────┬────────┘
                                      │
                       ┌──────────────▼──────────────┐
                       │       Cascade router        │
                       │  T1 regex heuristics ~0.2ms │
                       │  T2 pgvector NN      ~18ms  │
                       │  T3 LLM classifier   ~740ms │
                       └──────────────┬──────────────┘
                    fast-qa │ code │ deep-reasoning │ long-context
                                      │
              ┌───────────────────────▼───────────────────────┐
              │   Exact cache (Redis) → semantic cache        │
              │   (pgvector HNSW) → provider dispatch with    │
              │   circuit breaker + cross-provider fallback   │
              └───────┬───────────────┬───────────────┬───────┘
                      ▼               ▼               ▼
                  Anthropic         OpenAI       Ollama (local)
```

- **Cascade router** — three tiers, each a fallthrough: sub-millisecond regex heuristics, a pgvector nearest-neighbor lookup against a hand-curated exemplar corpus (cosine threshold 0.45, chosen by sweep), and finally a Haiku-tier LLM classifier. Only ~55% of decisions pay the LLM-tier cost.
- **Sticky-upward escalation** — multi-turn conversations can escalate to a stronger model mid-conversation but never silently downgrade.
- **Failover + circuit breaker** — every virtual model has a cross-provider fallback route; retryable statuses (408/425/429/5xx) trigger failover, deterministic 4xx does not. Per-(provider, model) breaker state is inspectable at `/admin/breakers` and graphed in Grafana.
- **Conversation compaction** — long conversations are summarized by a cheap model once they cross a token threshold, keeping the last N turns verbatim.
- **LLM-as-judge** — versioned judge prompt, single or ensemble judging in the eval framework, plus an optional 1% production sampling backstop.
- **Observability** — Prometheus metrics, OTel traces, Grafana dashboards, structured logs, per-request cost accounting in Postgres.

## Quick start

```bash
cp .env.example .env          # add your ANTHROPIC_API_KEY / OPENAI_API_KEY
docker compose up -d          # gateway :8000, Postgres, Redis, Prometheus :9090, Grafana :3000
python scripts/generate_key.py   # mint a gateway key; add its hash to config/tenants.yaml
python scripts/seed_exemplars.py # embed the router exemplar corpus

curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-gateway-..." \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "What is the capital of France?"}]}'
```

Any OpenAI client SDK works — point `base_url` at the gateway and pass a virtual model name (`auto`, `fast-qa`, `code`, `deep-reasoning`, `long-context`) or a concrete model. A minimal chat UI ships at `/ui`.

---

## Benchmarks & evals

All numbers below come from checked-in result files under [`eval/results/`](eval/results/) — every figure is traceable to a JSON or log artifact. The eval framework itself lives at `src/gateway/eval/` (`python -m gateway.eval --workload <name> --target gateway:auto --target direct:openai:gpt-4o-mini --judge claude-opus-4-7`), with workloads defined declaratively in [`eval/workloads/`](eval/workloads/) and an auto-generated leaderboard at [`docs/benchmarks.md`](docs/benchmarks.md).

**Design decisions worth calling out:**

- The router eval set ([`eval/router_eval.yaml`](eval/router_eval.yaml), 67 hand-labeled queries) is deliberately **disjoint from the runtime exemplar corpus** — if eval queries appeared in the tier-2 lookup table, the router would trivially self-match and the accuracy number would be data leakage, not measurement. The runner enforces this with a hash-collision check.
- Workloads are **fingerprinted** (SHA over prompts + version), so a leaderboard row is only comparable to rows from the identical workload.
- The leaderboard has an explicit honesty rule: when a direct-provider baseline beats the gateway on a workload, the row stays.

### Test 1 — Cost & quality vs. baselines

Mixed realistic workload (20 prompts, seeded: 50% short Q&A, 30% code, 15% multi-turn, 5% long-context; 23 calls per target), judged by `claude-opus-4-7`:

| Target | Cost (23 calls) | p50 latency | Judge score | Judge pass |
|---|---:|---:|---:|---:|
| `gateway:auto` (cascade router) | **$0.0275** | **1,011 ms** | **1.000** | 100% |
| `gateway:claude-opus-4-8` (always-best) | $0.2347 | 3,076 ms | 0.992 | 100% |
| `direct:openai:gpt-4o-mini` (cost floor) | $0.0022 | 1,276 ms | 0.994 | 100% |

Auto-routing matched the always-Opus ceiling on quality at **88% lower cost** and **67% lower p50 latency**. The honest framing of the gpt-4o-mini row: a single cheap model is ~12× cheaper than auto — the gateway's value is matching best-model quality *when a request needs it* while paying the cheap-model price when it doesn't. On a 23-call sample the judge cannot distinguish 0.994 from 1.000.

*Artifacts: `eval/results/resume_run/2026-06-19_mixed_realistic_*.json` · Caveat: single judge, not an ensemble; judge model (Opus 4-7) is deliberately a different snapshot than the deep-reasoning route (Opus 4-8) to reduce self-preference.*

### Test 2 — Router accuracy

67-query hand-labeled eval set at the production embedding threshold (0.45):

| Metric | Result |
|---|---|
| Overall accuracy | **98.5%** (66/67) |
| Tier 1 (regex heuristic) | 20/20 correct, p50 **0.18 ms** |
| Tier 2 (pgvector NN) | 10/10 correct, p50 **18 ms** |
| Tier 3 (LLM classifier) | 36/37 correct, p50 **741 ms** |
| Routing decision latency | p50 646 ms · p95 1,445 ms · p99 1,663 ms |

45% of routing decisions resolve in under 20 ms; only the residual 55% pay the LLM-classifier cost. Per-category accuracy is 100% for fast-qa, deep-reasoning, and long-context; code is 94.1% — the single misroute is a polarity-edge query ("explain this — no code edits") that is genuinely arguable even for a human labeler. The 0.45 threshold came from a 10-point sweep (0.30 → 0.85) checked in at `eval/results/2026-06-17_router_threshold_sweep.json`.

*Artifacts: `eval/results/resume_run/test2_router_baseline_t045.json`*

### Test 3 — Load & rate limiting

Async load test against the cache-hit path (one warm call primes the exact cache, then 200 requests per concurrency level):

| Concurrency | RPS | p50 | p95 | p99 | 200s | 429s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 50 ms | 53 ms | 60 ms | 200 | 0 |
| 4 | 76 | 50 ms | 61 ms | 67 ms | 200 | 0 |
| 16 | **205** | 64 ms | 170 ms | 182 ms | 200 | 0 |
| 32 | 153 | 149 ms | 446 ms | 607 ms | 180 | 20 |
| 64 | 257 | 163 ms | 306 ms | 368 ms | 18 | 182 |

Sustained **205 RPS with p95 of 170 ms** at 16 concurrent connections, all 200s. Past that, the per-tenant token-bucket rate limiter (600 RPM, Redis Lua script) responds with 429 + `Retry-After: 1` — the 429s at high concurrency are the rate limiter working as designed, not a failure mode.

*Caveat: this measures gateway overhead (auth → rate limit → Redis cache lookup → response) on the cache-hit path. Cold-path latency is dominated by the provider and is reported in Tests 1–2. Artifacts: `eval/results/resume_run/test3_load.json`*

### Test 4 — Cache effectiveness (including a negative result)

Two-pass paraphrase workload (5 prime questions + 5 paraphrases, routed to `fast-qa`):

- **Repeat traffic (pass 2):** 10/10 cache hits, p50 **3.1 ms** vs 961 ms cold — a 99.7% latency reduction, with cached calls costing $0 in provider spend.
- **Paraphrase traffic (pass 1): the semantic cache did not fire.** All 5 paraphrases missed (full latency, full cost) at the 0.95 cosine threshold with MiniLM-L6-v2 embeddings. That threshold is evidently too strict for real paraphrase similarity — this is documented as an open item rather than quietly dropped from the results.

*Artifacts: `eval/results/resume_run/2026-06-19_paraphrase_cache_*.json` (pass 1), `test4_paraphrase_cache_pass2.json` (pass 2)*

### Test 5 — Chaos & failure modes (one pass, one fail)

Infrastructure outages injected with `docker pause`, 5 requests sent mid-outage:

| Scenario | Result |
|---|---|
| **Redis outage** | **Fail-open works.** All 5 mid-outage requests returned 200 (cache and rate limiting degrade silently); latency rose from 1.9 s to ~6.5 s from connection-timeout overhead; full recovery on unpause. |
| **Postgres outage** | **Fail-open does not work.** All 5 requests hung past a 15 s client timeout. Request logging was supposed to degrade silently; in practice a request-path dependency blocks on the dead connection. Recovery on unpause was clean, but this is a real defect the chaos test caught. |
| **Cross-provider failover** | Wired and configured (retryable-status policy, per-route fallbacks), but the production request log shows **0 failover events in 1,081 requests** — the mechanism has never fired against a real provider outage. Unproven ≠ proven. |
| **Circuit breakers** | All closed under healthy conditions; per-(provider, model) state observable via `/admin/breakers` and Grafana. Open/half-open/close cycle not yet exercised end-to-end. |

*Artifacts: `eval/results/resume_run/test5*.log`*

---

## Repository layout

```
src/gateway/          FastAPI app — routes/, routing/ (cascade), providers/,
                      compaction/, cost/, judge/, eval/, observability/
config/               routes.yaml (virtual models + fallbacks), pricing.yaml,
                      tenants.yaml, router_exemplars.yaml
eval/                 workloads/ (declarative YAML), router_eval.yaml (67-query
                      labeled set), results/ (every run's JSON + logs)
docs/benchmarks.md    auto-generated leaderboard (scripts/build_leaderboard.py)
scripts/              eval runners, threshold sweep, load test, key generation
migrations/           Alembic (additive-only policy)
observability/        Prometheus + Grafana + OTel collector config
```

## License

MIT
