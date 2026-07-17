#!/usr/bin/env python3
"""Offline contract tests for run_prefill_confirmation.py.

No GPU, network, tokenizer, or vLLM server is used. The test validates the
actual frozen Llama/Qwen schedule bundles already present under
new/runs/prefill_confirmation/<model>/ and the sibling shell scripts.
"""
from __future__ import annotations

import asyncio
import copy
import py_compile
import tempfile
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_prefill_confirmation as runner  # noqa: E402

checks: list[tuple[str, bool, str]] = []

def check(name: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    checks.append((name, ok, detail))
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))


def main() -> int:
    print("test_run_prefill_confirmation.py")
    print("=" * 78)

    runner_path = SCRIPT_DIR / "run_prefill_confirmation.py"
    timing_test_path = SCRIPT_DIR / "test_prefill_confirmation_timing.py"
    wrapper_path = SCRIPT_DIR / "run_prefill_confirmation.sh"
    server_path = SCRIPT_DIR / "run_server.sh"

    try:
        py_compile.compile(str(runner_path), doraise=True)
        py_compile.compile(str(Path(__file__)), doraise=True)
        py_compile.compile(str(timing_test_path), doraise=True)
        compile_ok = True
    except py_compile.PyCompileError as exc:
        compile_ok = False
        compile_detail = str(exc)
    else:
        compile_detail = ""
    check("py_compile runner and both test files", compile_ok, compile_detail)

    for shell_path in (wrapper_path, server_path):
        proc = subprocess.run(["bash", "-n", str(shell_path)], capture_output=True, text=True)
        check(f"bash -n {shell_path.name}", proc.returncode == 0, proc.stderr)

    check("runner self-test", runner.run_self_test() == 0)
    check("result schema remains timing Schema 5", runner.RESULT_SCHEMA_VERSION == 5)
    check("timing instrumentation version/name frozen",
          runner.TIMING_INSTRUMENTATION_VERSION == 2 and
          runner.TIMING_INSTRUMENTATION_NAME == "task_semaphore_dispatch_wave_v2")
    check("official counts frozen at 96 episodes / 48 blocks",
          runner.OFFICIAL_EPISODE_COUNT_PER_MODEL == 96 and
          runner.OFFICIAL_BLOCK_COUNT_PER_MODEL == 48)
    check("official grid frozen",
          runner.OFFICIAL_OFFLOAD_VALUES == [0, 8, 12] and
          runner.OFFICIAL_CONCURRENCY_VALUES == [4, 8] and
          runner.OFFICIAL_TRIGGER_POSITIONS == [16] and
          runner.OFFICIAL_REPEATS == 8)

    runner.check_run_server_script(server_path)
    check("run_server.sh passes frozen capability check", True)
    server_text = server_path.read_text(encoding="utf-8")
    check("run_server.sh accepts offload 0/8/12", "^(0|8|12)$" in server_text)

    expected_hash_names = {
        "run_prefill_confirmation.py",
        "run_prefill_confirmation.sh",
        "run_server.sh",
        "prefill_confirmation_schedule.json",
        "prefill_confirmation_schedule.csv",
        "prefill_confirmation_schedule_audit.txt",
    }
    check("environment fingerprint tracks exactly confirmation artifacts",
          runner.EXPECTED_ENVIRONMENT_FILE_HASH_NAMES == expected_hash_names)

    loaded = {}
    for model_key in ("llama", "qwen"):
        schedule_dir = runner.default_schedule_dir(model_key)
        bundle, errors = runner.load_and_validate_bundle(schedule_dir, model_key)
        check(f"{model_key}: actual official bundle validates", not errors, str(errors))
        check(f"{model_key}: bundle object returned", bundle is not None)
        if bundle is None:
            continue
        loaded[model_key] = bundle
        plan = runner.build_execution_plan(bundle)
        check(f"{model_key}: 96 regular episodes", plan["regular_episodes"] == 96)
        check(f"{model_key}: 48 blocks/server starts", plan["planned_server_starts"] == 48)
        check(f"{model_key}: 48 stabilization runs", plan["planned_stabilization_runs"] == 48)
        check(f"{model_key}: balanced conditions", plan["no_burst_count"] == 48 and plan["burst_condition_count"] == 48)
        check(f"{model_key}: frozen fingerprint",
              bundle.fingerprint == runner.OFFICIAL_FINGERPRINTS[model_key])
        check(f"{model_key}: smoke target off12/conc8/rep01 exists",
              len(runner.find_block(bundle, f"{model_key}_off12_conc8_rep01")) == 2)
        check(f"{model_key}: all result IDs zero-pad repeat",
              all(f"rep{ep.repeat:02d}" in ep.episode_id for ep in bundle.episodes))

        # Independent mutation checks against structural validation.
        mutated = list(bundle.episodes)
        mutated[0] = replace(mutated[0], concurrency=16)
        check(f"{model_key}: forbidden concurrency mutation rejected",
              bool(runner.check_structural_schedule(mutated, bundle.json_obj["seed"], model_key)))
        mutated = list(bundle.episodes)
        mutated[0] = replace(mutated[0], trigger_after_decode_tokens=1)
        check(f"{model_key}: forbidden trigger mutation rejected",
              bool(runner.check_structural_schedule(mutated, bundle.json_obj["seed"], model_key)))

    # Full offline fake-server integration of the actual hardest smoke block
    # for both models: stabilization + paired no_burst/prefill_burst episodes
    # at offload12/concurrency8/trigger16.
    for port_offset, model_key in enumerate(("llama", "qwen"), start=0):
        bundle = loaded.get(model_key)
        if bundle is None:
            continue
        transport = runner.FakeTransport()
        transport.set_get_response("/health", 200, {})
        transport.set_get_response(
            "/v1/models", 200,
            {"data": [{"id": runner.MODEL_REGISTRY[model_key]["model_id"]}]},
        )
        transport.set_get_response(
            "/openapi.json", 200, {"paths": {"/v1/completions": {}}}
        )

        def factory(payload: dict) -> "runner.FakeStreamScript":
            return runner.FakeStreamScript(
                prompt_token_ids_echo=list(payload["prompt"]),
                token_events=[[1000 + i] for i in range(payload["max_tokens"])],
                usage={
                    "prompt_tokens": len(payload["prompt"]),
                    "completion_tokens": payload["max_tokens"],
                },
            )

        transport.default_script_factory = factory
        with tempfile.TemporaryDirectory() as td:
            smoke_dir = Path(td) / "smoke"
            summary = asyncio.run(
                runner.run_smoke_block(
                    bundle=bundle,
                    block_id=f"{model_key}_off12_conc8_rep01",
                    output_dir=smoke_dir,
                    host="127.0.0.1",
                    port=37991 + port_offset,
                    resume=False,
                    api_key="fake-key",
                    transport=transport,
                    tokenizer=runner.FakeTokenizerAdapter(),
                    server_adapter=runner.FakeServerProcessAdapter(),
                    sleeper=runner.FakeSleeper(),
                    clock=runner.FakeClock(),
                    run_server_path=server_path,
                )
            )
            check(f"{model_key}: fake conc8 smoke block completes",
                  summary.get("overall_status") == "block_complete", str(summary))
            check(f"{model_key}: both fake smoke episodes are valid_complete",
                  set(summary.get("episode_statuses", {}).values()) == {runner.CLASSIFICATION_VALID_COMPLETE})
            check(f"{model_key}: fake burst reaches 8 victims + 4 burst streams",
                  transport.max_active_stream_count == 12, str(transport.max_active_stream_count))
            check(f"{model_key}: fake smoke writes exactly two episode files",
                  len(list((smoke_dir / "episodes").glob("*.json"))) == 2)
            check(f"{model_key}: fake smoke writes exactly one stabilization file",
                  len(list((smoke_dir / "stabilization").glob("*.json"))) == 1)

    if len(loaded) == 2:
        llama_under_qwen, cross_errors = runner.load_and_validate_bundle(
            loaded["llama"].schedule_dir, "qwen"
        )
        check("Llama bundle rejected under --model-key qwen",
              llama_under_qwen is None and bool(cross_errors), str(cross_errors))
        check("Llama/Qwen fingerprints differ",
              loaded["llama"].fingerprint != loaded["qwen"].fingerprint)

    print("=" * 78)
    passed = sum(ok for _, ok, _ in checks)
    print(f"{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
