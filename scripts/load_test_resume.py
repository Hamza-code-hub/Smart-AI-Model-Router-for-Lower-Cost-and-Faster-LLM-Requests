"""Test 3 — Load test (sustained RPS + latency envelope on cache-hit path)."""
import asyncio
import json
import os
import pathlib
import statistics
import time

import httpx

URL = "http://localhost:8000/v1/chat/completions"
KEY = os.environ["EVAL_GATEWAY_API_KEY"]
BODY = {
    "model": "claude-haiku-4-5",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 0,
    "max_tokens": 50,
}
CONCURRENCY_LEVELS = [1, 4, 16, 32, 64]
N_PER_LEVEL = 200

OUT_DIR = pathlib.Path("eval/results/resume_run")
OUT_DIR.mkdir(parents=True, exist_ok=True)


async def one(client):
    t0 = time.perf_counter()
    r = await client.post(URL, json=BODY, headers={"Authorization": f"Bearer {KEY}"})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    status = r.status_code
    cached = False
    retry_after = None
    if status == 200:
        try:
            data = r.json()
            cached = bool(data.get("cached", False))
        except Exception:
            cached = False
    elif status == 429:
        retry_after = r.headers.get("Retry-After")
    return elapsed_ms, status, cached, retry_after


async def warm(client):
    for _ in range(2):
        await one(client)


async def level(client, conc, n):
    sem = asyncio.Semaphore(conc)

    async def gated():
        async with sem:
            return await one(client)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(gated() for _ in range(n)))
    elapsed = time.perf_counter() - t0
    lats_ok = [x[0] for x in results if x[1] == 200]
    cached_count = sum(1 for x in results if x[2])
    status_counts = {}
    for _, s, _, _ in results:
        status_counts[str(s)] = status_counts.get(str(s), 0) + 1
    retry_after_samples = [x[3] for x in results if x[3] is not None][:5]

    def pct(p):
        if not lats_ok:
            return None
        if len(lats_ok) < 2:
            return lats_ok[0]
        return statistics.quantiles(lats_ok, n=100)[p - 1]

    return {
        "concurrency": conc,
        "n": n,
        "elapsed_s": round(elapsed, 3),
        "rps": round(n / elapsed, 2),
        "ok": len(lats_ok),
        "cached_hits": cached_count,
        "status_counts": status_counts,
        "retry_after_samples": retry_after_samples,
        "p50_ms": round(pct(50), 2) if lats_ok else None,
        "p95_ms": round(pct(95), 2) if lats_ok else None,
        "p99_ms": round(pct(99), 2) if lats_ok else None,
        "mean_ms": round(statistics.mean(lats_ok), 2) if lats_ok else None,
        "max_ms": round(max(lats_ok), 2) if lats_ok else None,
    }


async def main():
    async with httpx.AsyncClient(timeout=30) as c:
        print("[warm] priming exact-match cache...")
        await warm(c)
        await asyncio.sleep(0.5)
        out = []
        for conc in CONCURRENCY_LEVELS:
            print(f"[level] concurrency={conc}, n={N_PER_LEVEL}")
            r = await level(c, conc, N_PER_LEVEL)
            out.append(r)
            print(json.dumps(r, indent=2))
            await asyncio.sleep(1.0)
        (OUT_DIR / "test3_load.json").write_text(json.dumps(out, indent=2))
        print(f"\n[done] wrote {OUT_DIR / 'test3_load.json'}")


if __name__ == "__main__":
    asyncio.run(main())
