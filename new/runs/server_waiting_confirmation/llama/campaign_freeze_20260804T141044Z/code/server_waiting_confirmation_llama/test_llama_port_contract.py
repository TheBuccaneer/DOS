#!/usr/bin/env python3
"""Focused offline checks that the Llama port is model-bound and Qwen-isolated."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate_server_waiting_schedule as gen  # noqa: E402
import run_server_waiting_confirmation as runner  # noqa: E402


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        ok = bool(condition)
        checks.append((name, ok, detail))
        print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))

    schedule_dir = runner.default_schedule_dir("llama")
    bundle, errors = runner.load_and_validate_bundle(schedule_dir, "llama")
    check("official Llama bundle validates", bundle is not None and not errors, str(errors))
    check("runner registry is exactly Llama-only", set(runner.MODEL_REGISTRY) == {"llama"})
    check("runner model ID is Llama-3.1-8B-Instruct", runner.MODEL_ID == "meta-llama/Llama-3.1-8B-Instruct")
    check("diagnostic block is Llama-bound", runner.DIAGNOSTIC_BLOCK_ID == "llama_off12_k8_rep01")
    check("schedule fingerprint matches runner", bundle is not None and bundle.fingerprint == runner.OFFICIAL_FINGERPRINT)

    episodes = gen.build_episodes(gen.OFFICIAL_SEED)
    qwen_mutated = copy.deepcopy(episodes)
    for ep in qwen_mutated:
        ep.model_key = "qwen"
        ep.model_id = "Qwen/Qwen2.5-7B-Instruct"
        ep.episode_id = ep.episode_id.replace("llama_", "qwen_", 1)
        ep.block_id = ep.block_id.replace("llama_", "qwen_", 1)
    structural_errors = runner.check_structural_schedule(qwen_mutated, gen.OFFICIAL_SEED, "llama")
    check("Qwen-mutated schedule is rejected even before fingerprint acceptance", bool(structural_errors), str(structural_errors[:3]))

    server_text = (HERE / "run_server_waiting_server.sh").read_text(encoding="utf-8")
    check("server launcher contains Llama case", 'llama)  MODEL="meta-llama/Llama-3.1-8B-Instruct"' in server_text)
    check("server launcher contains no accepted Qwen case", "qwen)" not in server_text)

    schedule_json = json.loads((schedule_dir / "server_waiting_confirmation_schedule.json").read_text(encoding="utf-8"))
    check("all 32 schedule rows are Llama-only", len(schedule_json.get("episodes", [])) == 32 and all(
        row.get("model_key") == "llama" and row.get("model_id") == runner.MODEL_ID
        for row in schedule_json.get("episodes", [])
    ))

    passed = sum(ok for _, ok, _ in checks)
    print(f"{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
