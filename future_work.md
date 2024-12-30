# Future Work

Deferred improvements that are not bugs in the current behavior, but are
better-than-today designs worth picking up when the surrounding code is
touched. Each entry captures: what's wrong today, what to change, what to
watch out for, and where the relevant code lives.

---

## 1. Classify after partial compaction (eliminate the one-turn escalation lag)

### Current behavior

`src/gateway/routes/chat.py` runs in this order per `model: "auto"` request:

1. `_resolve_auto()` — classifies the **raw uncompacted** request, then calls
   `apply_sticky()` with `history_token_estimate` derived from the full
   on-the-wire transcript.
2. `resolver.resolve(final_category)` — picks the route config.
3. `maybe_compact()` — compaction runs *after* routing is locked in.

Because `apply_sticky` sees the pre-compaction history size, the strict
two-step gap requirement (see `_escalation_allowed` in
`src/gateway/routing/conversation_state.py`) fires the moment a conversation
crosses `HISTORY_BIG_THRESHOLD = 20_000` tokens, even though we are *about to*
compact that history down to ~3k tokens for this very turn.

Outcome: the first turn that crosses 20k can have a legitimate one-step
escalation blocked. The next turn recovers — `compaction_just_fired = True`
relaxes the gap back to one step — but the bad-tier answer on the trigger
turn is already shipped.

### Proposed change

Split the work into two phases:

1. **Pre-classify structural split** — call `_split_keep_last(messages, 4)`
   (already exists in `src/gateway/compaction/strategy.py`) on the incoming
   messages *before* classification. This costs nothing — it's a list slice,
   no Haiku call.
2. **Classify on the kept tail** (or just the latest user message — even
   cleaner). Feed the post-split estimated history size into `apply_sticky`.
3. **Resolve the route**, then call `maybe_compact` as today — this is the
   real summarization phase that hits Haiku.

Effect: on the first turn over 20k, `apply_sticky` sees ~3k of effective
history, falls into the cheap-regime branch (`gap >= 1` allowed), and the
escalation fires same-turn instead of one turn late.

### Trade-offs to verify before merging

- `LONG_CONTEXT_CHAR_THRESHOLD` in `src/gateway/routing/classifier.py:55-56`
  fires off total raw chars across messages. Classifying on the kept tail
  also disables long-context routing — which is **correct** if we believe
  the compacted payload genuinely doesn't need a long-context model, but
  it's worth confirming that on real eval workloads before flipping.
- The hardcoded `keep_last_n = 4` for the pre-classify split should match
  the route's eventual `keep_last_n_turns`. They are 4 everywhere in
  `config/routes.yaml` today, but if a route ever sets it differently the
  classifier slice and the compaction slice will disagree.
- This adds one structural slice on every auto-routed request. Measure
  before/after p50 to confirm it's negligible (expected — it's a list
  slice, no I/O).

### Files to touch

- `src/gateway/routes/chat.py` — `_resolve_auto` (chat.py:92-131)
- `src/gateway/routing/conversation_state.py` — no API change, but the
  `history_token_estimate` argument's meaning shifts
- `src/gateway/compaction/strategy.py` — expose `_split_keep_last` (or a
  thin wrapper) so chat.py can call it without re-implementing

---

## 2. Per-transition escalation threshold (replace the single 20k constant)

### Current behavior

`src/gateway/routing/conversation_state.py:36` defines a single
`HISTORY_BIG_THRESHOLD = 20_000` that gates **all gap-1 escalations**:

- `fast-qa → code` (gap=1)
- `code → deep-reasoning` (gap=1)

`fast-qa → deep-reasoning` (gap=2) is unaffected — it's always allowed
subject only to `MAX_ESCALATIONS`. The threshold is **only** about gap-1
decisions.

### Why one number is wrong

The switch tax is driven by the destination model's uncached input price, not
by the source. Rough numbers:

| Transition | Destination | Input price | 20k re-ingest cost |
|---|---|---|---|
| `fast-qa → code` | code-tier (Sonnet-ish) | ~$3/1M | ~$0.06 |
| `code → deep-reasoning` | deep-reasoning (Opus) | ~$15/1M | ~$0.30 |

`code → deep-reasoning` costs ~5× more per re-ingested token than
`fast-qa → code`. If 20k is the right "evidence requirement" point for the
cheap transition, the cost-equivalent point for the expensive transition is
closer to ~4k tokens. Today both transitions share the 20k bar, so we're
over-eager about `code → deep-reasoning` on mid-sized conversations.

### Proposed change

Replace the single constant with a per-pair table. Two flavors:

**(a) Static table in code** — simplest, ships with sensible defaults:

```python
ESCALATION_GAP1_THRESHOLDS: dict[tuple[str, str], int] = {
    ("fast-qa", "code"):        20_000,
    ("code", "deep-reasoning"):  4_000,
}
DEFAULT_GAP1_THRESHOLD = 20_000
```

`_escalation_allowed` looks up `(from_cat, to_cat)`, falls back to the
default for unknown pairs. Adding a category later still works without a
matrix update.

**(b) YAML-configured table** — promote it to `config/routes.yaml` or a new
`config/routing.yaml` so the numbers can be tuned without code changes. More
boilerplate, more flexibility.

### Trade-offs to verify before merging

- Hardcoded thresholds are coupled to current model pricing. If we swap the
  code-tier or deep-reasoning model to a cheaper/pricier variant, the
  numbers go stale. A comment pointing at the pricing assumption beside
  each entry is the minimum.
- Eval needs to confirm the 4k number actually moves the cost/quality curve
  in the right direction on real workloads. The math is a starting point,
  not a final answer.
- `_escalation_allowed`'s signature already takes the gap and history size
  but does not take the (from, to) pair. Adding it is a 1-line change in
  the function plus a 1-line change in `apply_sticky` which already has
  both categories in scope.

### Files to touch

- `src/gateway/routing/conversation_state.py` — replace
  `HISTORY_BIG_THRESHOLD`, update `_escalation_allowed` and `apply_sticky`
- `src/gateway/routing/conversation_state.py` docstring at top — the
  "Sticky upward rules" section references HISTORY_BIG_THRESHOLD by name
- Anywhere in `history.md` / `research/phase_1_5_sticky_routing.md` that
  cites the 20k number as the universal threshold (documentation drift)

---

## 3. LRU eviction sweep for `router_exemplars`

### Current behavior

`router_exemplars` accepts inserts from the Haiku auto-seeder until the table
reaches the hard cap of 100,000 rows, after which new high-confidence Haiku
verdicts are silently dropped. Hand-curated rows (`source = 'curated'`) and
machine-seeded rows (`source = 'haiku-seeded'`) share the cap.

The schema (`migrations/versions/a1d2f3b4c5e6_create_router_exemplars_table.py`)
already carries the columns needed to evict intelligently:

- `last_matched_at TIMESTAMPTZ` — bumped whenever `_embedding_route` in
  `src/gateway/routing/classifier.py` returns the row as a tier-2 hit
- `match_count INTEGER` — incremented at the same time
- `source TEXT` — `'curated'` vs `'haiku-seeded'`
- `idx_router_exemplars_last_matched` index makes the eviction query O(log N)

Once the cap is hit, table quality degrades over time: stale seeded rows from
old prompt patterns keep taking up slots while new prompt patterns can't get
in. The first hour of "table at cap" is not the problem; the long tail is.

### Proposed change

Add a nightly cron (or whatever scheduler is in play by then) that runs a
single DELETE keeping the table at or under the cap, evicting only
machine-seeded rows by oldest `last_matched_at` first. Curated rows are
exempt — they're the trusted seed set and should never be auto-evicted.

```sql
DELETE FROM router_exemplars
 WHERE id IN (
    SELECT id
      FROM router_exemplars
     WHERE source = 'haiku-seeded'
     ORDER BY last_matched_at ASC
     LIMIT GREATEST(0,
        (SELECT COUNT(*) FROM router_exemplars) - 100000)
 );
```

Considered and rejected: **opportunistic inline eviction** (run the DELETE on
every seeder insert). Mixes hot-path work into the live request's
background-task queue and makes per-request behaviour harder to reason about.

### Trade-offs to verify before merging

- Eviction frequency: nightly is fine while the cap is 100k and prod traffic
  is moderate, but if seeder throughput ever exceeds (cap / 24 hours) the
  table will spend most of the day above cap. Bump to hourly if seeder
  output is steady at >4k inserts/hour.
- Confirm `match_count` distribution before relying on `last_matched_at`
  alone. If 95% of tier-2 hits go to 5% of rows (likely), a low
  `last_matched_at` strongly correlates with "never used" — fine for
  eviction. If the distribution is flatter than expected,
  `ORDER BY last_matched_at ASC, match_count ASC` is a safer tiebreaker.
- The DELETE holds row locks briefly. At 100k cap and small per-run delete
  sizes (typically <1k) this is invisible; flag if cap or churn grow.

### Files to touch

- New cron entry / scheduler hook — location depends on the deployment
  pattern in place at the time
- No code change to the seeder itself — it just keeps inserting under the cap
- No schema change — `last_matched_at` and `match_count` already exist per
  `migrations/versions/a1d2f3b4c5e6_create_router_exemplars_table.py`

---

## 4. Local-LLM tertiary fallback for cross-provider outages

### Current behavior

`RouteConfig` in `src/gateway/routing/resolver.py:18-22` allows exactly one
`fallback: PrimaryRoute | None`. `attempts()` in `src/gateway/fallback.py:22-27`
returns at most two targets: `[primary]` and `[fallback]`. Every route in
`config/routes.yaml` follows the same shape — Anthropic primary fails over to
OpenAI (or vice versa), and that's the end of the line.

Consequences when both hyperscalers degrade simultaneously:

- An Anthropic + OpenAI joint incident (rare but it happens — shared CDN, BGP
  events, or correlated capacity crunches) turns into a 100% outage for every
  virtual model in `routes.yaml`. There is no third option to try.
- After Task 1 from `tasks_missing_model_resilience.md` shipped, a primary
  that returns 404 model_not_found now fails over — but if the fallback target
  *also* 404s (both providers retired the same model on the same week), the
  client gets a clean 404. The route's safety net is still one-deep.
- The circuit breaker can open both providers' entries for the same virtual
  model at the same time during a wide incident, after which
  `breaker_blocked_all` returns 503 to every caller.

There is no current path to "degraded but alive" — the gateway is binary
(both providers up → serve, both providers down → 503).

### Proposed change

Promote `fallback` from a single target to an ordered list, and use that list
to wire a **local-LLM tertiary** as a final safety net. The Ollama provider
is already plumbed end-to-end (see the existing `local-llm` route — Ollama
ships requests through `gateway.providers.ollama` today as the tier-3
classifier). Reusing that infra is mostly config work, not new code.

Two schema flavors:

**(a) Add `fallbacks: list[PrimaryRoute]`** alongside the existing
`fallback: PrimaryRoute | None` — backwards compatible, the resolver merges
the singular into the list:

```python
class RouteConfig(BaseModel):
    virtual_model: str
    primary: PrimaryRoute
    fallback: PrimaryRoute | None = None       # legacy single
    fallbacks: list[PrimaryRoute] = []         # new ordered list
    compaction: CompactionRouteConfig | None = None
```

`attempts(route)` returns `[primary] + ([fallback] if fallback else []) + fallbacks`.

**(b) Replace `fallback` with `fallbacks: list[PrimaryRoute]`** outright —
cleaner schema, but every route in `config/routes.yaml` needs a YAML edit.
Acceptable since the file is short and hand-maintained.

Either way, the new third entry in (say) `fast-qa` looks like:

```yaml
- virtual_model: "fast-qa"
  primary:
    provider: anthropic
    model: claude-haiku-4-5
  fallbacks:
    - provider: openai
      model: gpt-4o-mini
    - provider: ollama
      model: qwen2:7b           # or whichever local model wins the deferred swap eval
```

The intent is **"keep the gateway answering, accept the quality drop"** —
this is a graceful-degradation tier, not a parity tier. Phase 2.2 already
established that qwen2:7b on CPU loses to Haiku on both accuracy and latency
(see `memory/project_router_phase22_deferred.md`), which is fine for a
tertiary safety net but should be called out in the response somehow (see
trade-offs below).

### Trade-offs to verify before merging

- **Local model latency floor.** `qwen2:7b` on CPU runs in tens of seconds,
  not hundreds of milliseconds. A user-facing fallback that takes 30s to
  reply may be worse UX than a fast 503 — depends on the caller. Consider a
  per-route opt-in flag so streaming-heavy endpoints can skip it and the
  rest can take the slow-but-alive path. The deferred-swap unblock
  conditions in `memory/project_router_phase22_deferred.md` (GPU host, or a
  smaller/faster local model) apply here too — without one of them, the
  tertiary is only useful as an "answer anything > answer nothing" floor.
- **Response should advertise degraded mode.** When the tertiary fires,
  callers should be able to tell. Easiest path: a response header
  (`X-Gateway-Fallback-Tier: local`) plus a structured log event
  `chat.fallback_local_llm`. Existing `record.fallback_used` is boolean and
  insufficient — extend it to `fallback_index: int` (0 = primary, 1 = first
  fallback, …) so the cost tracker and judge can segment results by tier.
- **Cost tracker / pricing.** `config/pricing.yaml` needs an entry for the
  Ollama models. Local inference cost is not zero (electricity + amortized
  hardware) but for accounting purposes $0/token is the honest answer —
  just make sure the cost path doesn't `KeyError` on a missing model.
- **Circuit breaker behavior on three targets.** The breaker per-target
  state already works for two; verify it doesn't accidentally trip the
  Ollama entry on cold-start latency. Today's failure threshold is tuned
  for hyperscaler RPS, not single-host local inference. Consider per-target
  breaker config or a longer half-open window for the local tier.
- **Routes that should *not* get a local tertiary.** `deep-reasoning` and
  `code` have specific capability requirements (long context, tool use,
  high-quality code synthesis) that a 7B local model will fail at badly
  enough to be misleading. Either omit the tertiary on those routes, or
  pick a different local model per route (the schema already supports it —
  it's per-route config).
- **A/B / eval scoring.** The judge (`gateway.judge`) sampling currently
  doesn't know about fallback index. If the local tier serves N% of traffic
  during incidents, the quality dashboard will silently regress until
  `fallback_index` is added as a judge label.

### Files to touch

- `src/gateway/routing/resolver.py` — `RouteConfig`, add `fallbacks` field
  (and the merge logic if going with flavor (a))
- `src/gateway/fallback.py` — `attempts()` returns the full ordered chain
- `src/gateway/routes/chat.py` — `RequestRecord.fallback_used` becomes
  `fallback_index`; the failover log events already include
  `from_model`/`to_model` after Task 2 of the missing-model-resilience
  work, so they'll naturally pick up the new third hop
- `src/gateway/cost/tracker.py` — pricing lookup must tolerate Ollama
  models (or `pricing.yaml` gets an explicit `$0` entry)
- `src/gateway/observability/metrics.py` — add `fallback_index` label
  somewhere visible, e.g. extend `REQUEST_COUNT` or add a dedicated
  `LOCAL_FALLBACK_SERVED` counter so alerting can fire on it
- `config/routes.yaml` — add the tertiary entry to whichever routes opt in
- `config/pricing.yaml` — entries for local model(s)
- `memory/project_router_phase22_deferred.md` — this entry materially
  changes the unblock criteria (it stops being purely about classifier
  swap, gains a user-facing dimension); update when picking this up

---

## How to use this file

- An entry here is a **candidate**, not a commitment. Move it into
  `tasks.md` when promoting it to active work.
- When picking up an entry, re-verify the "current behavior" section — the
  surrounding code may have moved on since the entry was written.
- When an entry is shipped, delete it from this file and add a `history.md`
  line describing what changed. This file is for *pending* improvements
  only.
