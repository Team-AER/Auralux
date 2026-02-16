#!/usr/bin/env python3
"""Smoke-test: hit the Auralux /generate endpoint and poll until done.

Usage:
    python test_generate.py               # single run
    python test_generate.py --loop 5      # repeat 5 times
    python test_generate.py --loop 0      # loop forever until failure

The server must already be running on http://127.0.0.1:8765.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8765"
TIMEOUT = 300  # seconds to wait for a single job


def post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get_json(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as resp:
        return json.loads(resp.read().decode())


def wait_for_server(max_wait: int = 60) -> bool:
    """Wait until the server responds to /health."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            get_json("/health")
            return True
        except Exception:
            time.sleep(1)
    return False


def run_one(run_num: int) -> bool:
    """Submit a generation, poll to completion. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"  Run #{run_num}")
    print(f"{'='*60}")

    resp = post_json("/generate", {
        "prompt": "chill lofi piano",
        "lyrics": (
            "[verse]\n"
            "Sunlight through the window pane\n"
            "Coffee steam and soft refrain\n"
            "Pages turn without a sound\n"
            "Peace is what I finally found\n\n"
            "[chorus]\n"
            "Drifting slow through golden haze\n"
            "Lost inside these quiet days"
        ),
        "tags": ["lofi", "piano", "chill"],
        "duration": 10,
    })

    job_id = resp.get("jobID")
    if not job_id:
        print(f"  ERROR: no jobID in response: {resp}")
        return False

    print(f"  Job: {job_id}")

    start = time.time()
    last_msg = ""
    while time.time() - start < TIMEOUT:
        try:
            status = get_json(f"/jobs/{job_id}")
        except Exception as exc:
            elapsed = time.time() - start
            print(f"  [{elapsed:.0f}s] POLL ERROR: {exc}")
            # Server may have crashed; wait a bit and retry
            time.sleep(2)
            continue

        s = status.get("status", "?")
        msg = status.get("message", "")
        pct = status.get("progress", 0)

        if msg != last_msg:
            elapsed = time.time() - start
            print(f"  [{elapsed:.0f}s] {s} {pct:.0%} — {msg}")
            last_msg = msg

        if s == "completed":
            audio = status.get("audioPath")
            elapsed = time.time() - start
            print(f"  SUCCESS in {elapsed:.1f}s — {audio}")
            return True

        if s == "failed":
            elapsed = time.time() - start
            print(f"  FAILED in {elapsed:.1f}s — {msg}")
            return False

        time.sleep(2)

    print(f"  TIMEOUT after {TIMEOUT}s")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=1,
                        help="Number of runs (0=infinite)")
    args = parser.parse_args()

    print("Waiting for server …")
    if not wait_for_server():
        print("Server not reachable. Exiting.")
        sys.exit(1)
    print("Server ready.")

    total = args.loop if args.loop > 0 else float("inf")
    passed = 0
    failed = 0
    run = 0

    try:
        while run < total:
            run += 1
            ok = run_one(run)
            if ok:
                passed += 1
            else:
                failed += 1
                # After a crash the server restarts; wait for it
                print("  Waiting for server to recover …")
                if not wait_for_server(max_wait=120):
                    print("  Server did not recover. Stopping.")
                    break
    except KeyboardInterrupt:
        print("\nInterrupted.")

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed out of {run} runs")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
