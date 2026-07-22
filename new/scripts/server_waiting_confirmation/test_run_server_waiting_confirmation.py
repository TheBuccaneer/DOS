#!/usr/bin/env python3
"""Offline contract tests for run_server_waiting_confirmation.py.

No GPU, network, tokenizer, or vLLM server is used. Exercises the
hardest paired smoke block (qwen, offload 12GB, server_max_num_seqs=8)
end-to-end through a FakeTransport / FakeServerProcessAdapter /
FakeSleeper / FakeClock / FakeTokenizerAdapter / FakeEnvironmentProbe --
no I/O beyond a temporary output directory.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "prefill_confirmation"
for p in (str(SCRIPT_DIR), str(BASE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_server_waiting_confirmation as swc  # noqa: E402
import run_prefill_confirmation as base  # noqa: E402
import generate_server_waiting_schedule as gen  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    checks.append((name, ok, detail))
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))


def _build_fake_transport(bundle, block_id: str, k: int, clock) -> "swc.FakeCompletionTransport":
    """Builds a FakeCompletionTransport for one paired block, wrapping
    an inner base.FakeTransport carrying the scripted responses.
    Exactly the first `k` victim indices of EACH episode in the block
    produce real output immediately; the rest produce genuinely ZERO
    output until long after the expected trigger -- modelling true
    server-side non-admission (server_max_num_seqs=k), not merely
    "slower" requests. (Which k indices are "fast" is not the point of
    this integration test -- see test_server_waiting_trigger_timing.py
    for the dedicated, scrambled-index proof that cohort selection is
    genuinely arrival-order-driven, not request_index-driven.)

    `clock` must be the SAME Clock instance passed to the block
    protocol / RunContext, so the transport's stream-open timestamps
    are comparable to task_created_ns/first_token_ns/etc."""
    inner = base.FakeTransport()
    model_id = swc.MODEL_REGISTRY[swc.MODEL_KEY]["model_id"]
    inner.set_get_response("/health", 200, {})
    inner.set_get_response("/v1/models", 200, {"data": [{"id": model_id}]})
    inner.set_get_response("/openapi.json", 200, {"paths": {"/v1/completions": {}}})
    valid_ids = swc.compute_valid_token_ids(base.FakeTokenizerAdapter())
    fast_indices = set(range(k))
    for ep in swc.find_block(bundle, block_id):
        for i in range(ep.victim_request_count):
            req_id = f"{ep.episode_id}:victim:{i}"
            p_seed = base.victim_prompt_seed(ep, i)
            prompt_ids = base.generate_token_id_prompt(p_seed, valid_ids, ep.victim_input_len)
            delay_ticks = 0 if i in fast_indices else 500
            inner.queue_script(req_id, base.FakeStreamScript(
                prompt_token_ids_echo=prompt_ids,
                extra_raw_events_before_finish=[""] * delay_ticks,
                token_events=[[1]] * ep.victim_output_len,
                usage={"prompt_tokens": ep.victim_input_len, "completion_tokens": ep.victim_output_len},
            ))
        for j in range(ep.burst_parallel_requests):
            req_id = f"{ep.episode_id}:burst:{j}"
            b_seed = base.burst_prompt_seed(ep, j)
            b_prompt = base.generate_token_id_prompt(b_seed, valid_ids, ep.burst_input_len)
            inner.queue_script(req_id, base.FakeStreamScript(
                prompt_token_ids_echo=b_prompt,
                token_events=[[1]] * ep.burst_output_len,
                usage={"prompt_tokens": ep.burst_input_len, "completion_tokens": ep.burst_output_len},
            ))

    # Stabilization requests (block_id:stabilization:i) are not part of
    # the trigger/cohort mechanism at all -- a uniform default factory
    # (echoing whatever was actually sent) is sufficient and realistic.
    def factory(payload: dict) -> "base.FakeStreamScript":
        return base.FakeStreamScript(
            prompt_token_ids_echo=list(payload["prompt"]),
            token_events=[[1]] * payload["max_tokens"],
            usage={"prompt_tokens": len(payload["prompt"]), "completion_tokens": payload["max_tokens"]},
        )

    inner.default_script_factory = factory
    return swc.FakeCompletionTransport(inner, clock=clock)


def _build_fake_metrics_transport(k: int) -> "base.FakeTransport":
    """A SEPARATE fake transport dedicated to /metrics -- never the
    same object as the completion transport (requirement 4)."""
    metrics_transport = base.FakeTransport()
    metrics_transport.set_get_response(
        "/metrics", 200, f"vllm:num_requests_running{{}} {k}.0\nvllm:num_requests_waiting{{}} {20 - k}.0\n",
    )
    return metrics_transport


def main() -> int:
    print("test_run_server_waiting_confirmation.py")
    print("=" * 78)

    runner_path = SCRIPT_DIR / "run_server_waiting_confirmation.py"
    gen_path = SCRIPT_DIR / "generate_server_waiting_schedule.py"
    cohort_path = SCRIPT_DIR / "_active_cohort.py"
    wrapper_path = SCRIPT_DIR / "run_server_waiting_confirmation.sh"
    server_path = SCRIPT_DIR / "run_server_waiting_server.sh"
    timing_test_path = SCRIPT_DIR / "test_server_waiting_trigger_timing.py"
    gen_test_path = SCRIPT_DIR / "test_generate_server_waiting_schedule.py"

    import py_compile
    try:
        for p in (runner_path, gen_path, cohort_path, timing_test_path, gen_test_path, Path(__file__)):
            py_compile.compile(str(p), doraise=True)
        compile_ok = True
        compile_detail = ""
    except py_compile.PyCompileError as exc:
        compile_ok = False
        compile_detail = str(exc)
    check("py_compile runner, generator, cohort module, and all test files", compile_ok, compile_detail)

    for shell_path in (wrapper_path, server_path):
        if not shell_path.exists():
            check(f"bash -n {shell_path.name}", False, f"{shell_path} does not exist")
            continue
        proc = subprocess.run(["bash", "-n", str(shell_path)], capture_output=True, text=True)
        check(f"bash -n {shell_path.name}", proc.returncode == 0, proc.stderr)

    check("runner self-test (--self-test path)", swc.run_self_test() == 0)
    check("result schema is independent of the audited Schema 5", swc.RESULT_SCHEMA_VERSION != base.RESULT_SCHEMA_VERSION)
    check(
        "timing instrumentation name/version are new and independent",
        swc.TIMING_INSTRUMENTATION_NAME != base.TIMING_INSTRUMENTATION_NAME
        or swc.TIMING_INSTRUMENTATION_VERSION != base.TIMING_INSTRUMENTATION_VERSION,
    )
    check(
        "official grid frozen: offload{0,12} x server_max_num_seqs{4,8} x repeats 1..4",
        swc.OFFLOAD_VALUES == [0, 12] and swc.SERVER_MAX_NUM_SEQS_VALUES == [4, 8]
        and swc.INITIAL_REPEATS == 4 and swc.TRIGGER_POSITIONS == [16],
    )
    check(
        "official counts frozen at 32 episodes / 16 blocks",
        swc.OFFICIAL_EPISODE_COUNT == 32 and swc.OFFICIAL_BLOCK_COUNT == 16,
    )

    swc.check_run_server_script(server_path)
    check("run_server_waiting_server.sh passes the frozen capability check", True)
    server_text = server_path.read_text(encoding="utf-8")
    check("run_server_waiting_server.sh accepts offload 0/12 only", "^(0|12)$" in server_text)
    check("run_server_waiting_server.sh accepts server_max_num_seqs 4/8 only", "^(4|8)$" in server_text)
    check("run_server_waiting_server.sh passes --max-num-seqs explicitly", "--max-num-seqs" in server_text)

    # --- Audit2 B1 / N2 fix: the wrapper must treat --diagnostic-pair-only
    # as a real run (requiring vLLM + VLLM_API_KEY preflight), not only
    # --smoke-test/--official-run. -----------------------------------------
    wrapper_text = wrapper_path.read_text(encoding="utf-8")
    check(
        "run_server_waiting_confirmation.sh's REAL_RUN case includes --diagnostic-pair-only",
        "--diagnostic-pair-only" in wrapper_text
        and any(
            "--diagnostic-pair-only" in line and "REAL_RUN=1" in line
            for line in wrapper_text.splitlines()
        ),
        wrapper_text,
    )
    # Behavioral confirmation: with a fake venv (no vllm on PATH) and no
    # VLLM_API_KEY, the wrapper must refuse --diagnostic-pair-only for the
    # SAME reason it already refuses --smoke-test (proving the flag is
    # actually inside the REAL_RUN branch at runtime, not just textually
    # present elsewhere in the file).
    with tempfile.TemporaryDirectory() as td:
        fake_venv = Path(td) / "fakevenv"
        (fake_venv / "bin").mkdir(parents=True)
        (fake_venv / "bin" / "activate").write_text("# fake activate\n", encoding="utf-8")
        # A minimal python3 shim so the wrapper's own python3/venv checks pass.
        python_shim = fake_venv / "bin" / "python3"
        python_shim.write_text("#!/usr/bin/env bash\nexec /usr/bin/env python3 \"$@\"\n", encoding="utf-8")
        python_shim.chmod(0o755)
        env = dict(os.environ)
        env["VENV_PATH"] = str(fake_venv)
        env.pop("VLLM_API_KEY", None)
        # Deliberately hide any real 'vllm' by using a minimal PATH that
        # only contains the fake venv's bin plus /usr/bin (for bash/cd/etc.)
        env["PATH"] = f"{fake_venv / 'bin'}:/usr/bin:/bin"
        proc = subprocess.run(
            ["bash", str(wrapper_path), "--diagnostic-pair-only", "--output-dir", "/tmp/unused"],
            capture_output=True, text=True, env=env, cwd=str(SCRIPT_DIR),
        )
        check(
            "wrapper refuses --diagnostic-pair-only without vllm on PATH, exactly as it already does for --smoke-test",
            proc.returncode != 0 and ("vllm" in (proc.stdout + proc.stderr).lower()),
            f"returncode={proc.returncode!r} stdout={proc.stdout!r} stderr={proc.stderr!r}",
        )

    expected_hash_names = {
        "run_server_waiting_confirmation.py", "run_server_waiting_confirmation.sh",
        "run_server_waiting_server.sh", "server_waiting_confirmation_schedule.json",
        "server_waiting_confirmation_schedule.csv", "server_waiting_confirmation_schedule_audit.txt",
        "_active_cohort.py", "run_prefill_confirmation.py",
    }
    check(
        "environment fingerprint tracks exactly this campaign's own artifacts",
        swc.EXPECTED_ENVIRONMENT_FILE_HASH_NAMES == expected_hash_names,
    )

    # --- Bundle load against the actual generated schedule ------------------
    schedule_dir = swc.default_schedule_dir()
    schedule_ready = schedule_dir.exists()
    if not schedule_ready:
        rc = gen.main(["--output-dir", str(schedule_dir)])
        schedule_ready = rc == 0
    check("official schedule is available (pre-existing or generated for this test run)", schedule_ready)

    bundle, errors = swc.load_and_validate_bundle(schedule_dir)
    check("actual official bundle validates", not errors, str(errors))
    check("bundle object returned", bundle is not None)
    if bundle is None:
        print("Cannot continue without a valid bundle.")
        print(f"{sum(1 for _, ok, _ in checks if ok)}/{len(checks)} checks passed")
        return 1

    plan = swc.build_execution_plan(bundle)
    check("32 regular episodes", plan["regular_episodes"] == 32)
    check("16 blocks/server starts", plan["planned_server_starts"] == 16)
    check("16 stabilization runs", plan["planned_stabilization_runs"] == 16)
    check("balanced conditions (16/16)", plan["no_burst_count"] == 16 and plan["burst_condition_count"] == 16)
    check("frozen fingerprint matches OFFICIAL_FINGERPRINT", bundle.fingerprint == swc.OFFICIAL_FINGERPRINT)
    check(
        "hardest smoke block (off12/K8/rep01) exists with exactly 2 episodes",
        len(swc.find_block(bundle, "qwen_off12_k8_rep01")) == 2,
    )
    check(
        "all result IDs use zero-padded repeat",
        all(f"rep{ep.repeat:02d}" in ep.episode_id for ep in bundle.episodes),
    )

    # Independent mutation checks against structural validation.
    from dataclasses import replace
    mutated = list(bundle.episodes)
    mutated[0] = replace(mutated[0], server_max_num_seqs=16)
    check(
        "forbidden server_max_num_seqs mutation rejected",
        bool(swc.check_structural_schedule(mutated, bundle.json_obj["seed"], swc.MODEL_KEY)),
    )
    mutated = list(bundle.episodes)
    mutated[0] = replace(mutated[0], trigger_after_decode_tokens=1)
    check(
        "forbidden trigger mutation rejected",
        bool(swc.check_structural_schedule(mutated, bundle.json_obj["seed"], swc.MODEL_KEY)),
    )

    # --- Dry-run: no process, no network, no output files -------------------
    with tempfile.TemporaryDirectory():
        # run_dry_run's signature takes only a schedule_dir/model_key --
        # there is no transport/server_adapter parameter at all, so it is
        # structurally impossible for it to open a network connection or
        # start a process. We additionally confirm it writes nothing.
        before = sorted(schedule_dir.rglob("*"))
        rc = swc.run_dry_run(schedule_dir, swc.MODEL_KEY)
        after = sorted(schedule_dir.rglob("*"))
    check("dry-run exits 0 against the valid bundle", rc == 0)
    check("dry-run writes no files into the schedule directory", before == after)

    # --- Full offline fake-server integration of the hardest smoke block ---
    smoke_clock = base.FakeClock()
    transport = _build_fake_transport(bundle, "qwen_off12_k8_rep01", k=8, clock=smoke_clock)
    metrics_transport = _build_fake_metrics_transport(k=8)
    with tempfile.TemporaryDirectory() as td:
        smoke_dir = Path(td) / "smoke"
        summary = asyncio.run(
            swc.run_server_waiting_smoke_block(
                bundle=bundle, block_id="qwen_off12_k8_rep01", output_dir=smoke_dir,
                host="127.0.0.1", port=37995, resume=False, api_key="fake-key",
                transport=transport, metrics_transport=metrics_transport, tokenizer=base.FakeTokenizerAdapter(),
                server_adapter=base.FakeServerProcessAdapter(), sleeper=base.FakeSleeper(),
                clock=smoke_clock, run_server_path=server_path,
            )
        )
        check("fake off12/K8 smoke block completes", summary.get("overall_status") == "block_complete", str(summary))
        check(
            "both fake smoke episodes are valid_complete",
            set(summary.get("episode_statuses", {}).values()) == {swc.CLASSIFICATION_VALID_COMPLETE},
            str(summary.get("episode_statuses")),
        )
        check(
            # No client admission semaphore: all 20 victim streams open
            # concurrently regardless of server-side cohort membership;
            # the prefill_burst episode additionally overlaps its 4 burst
            # streams while victims are still open -> 20 + 4 = 24.
            "block reaches 20 concurrent victim streams, 24 once the prefill_burst episode overlaps its burst",
            transport.peak_open_stream_count == 24, str(transport.peak_open_stream_count),
        )
        check(
            "completion transport reports the configured 32/32 pool limits",
            transport.get_diagnostics()["max_connections"] == 32
            and transport.get_diagnostics()["max_keepalive_connections"] == 32,
        )
        check(
            "metrics transport is a genuinely separate object from the completion transport",
            metrics_transport is not transport,
        )
        check("fake smoke writes exactly two episode files", len(list((smoke_dir / "episodes").glob("*.json"))) == 2)
        check("fake smoke writes exactly one stabilization file", len(list((smoke_dir / "stabilization").glob("*.json"))) == 1)

        # Inspect one written episode result file directly.
        ep_files = sorted((smoke_dir / "episodes").glob("*.json"))
        written = json.loads(ep_files[0].read_text(encoding="utf-8"))
        check("written episode result has this module's RESULT_SCHEMA_VERSION", written["result_schema_version"] == swc.RESULT_SCHEMA_VERSION)
        check("written episode result records no_client_admission_semaphore_used=True", written.get("no_client_admission_semaphore_used") is True)
        check(
            "written episode result's trigger has an active_cohort_size of 8",
            written["trigger"]["active_cohort_size"] == 8, str(written["trigger"].get("active_cohort_size")),
        )
        check(
            "written episode result exposes metrics_quality_status",
            written["trigger"]["metrics_quality"]["metrics_quality_status"]
            in ("corroborated", "unavailable", "stale", "contradictory", "unparsable"),
        )
        tce = written.get("transport_concurrency_evidence") or {}
        check(
            "written episode result reports all_20_streams_open_before_first_token == True",
            tce.get("all_20_streams_open_before_first_token") is True, str(tce),
        )
        check("written episode result reports victim_stream_open_count == 20", tce.get("victim_stream_open_count") == 20)
        check(
            "written episode result reports completion_pool_limits 32/32",
            tce.get("completion_pool_limits") == {"max_connections": 32, "max_keepalive_connections": 32},
        )
        first_victim = written["victim_requests"][0]
        check(
            "every victim record has a non-null stream_open_or_response_headers_perf_ns",
            all(type(r.get("stream_open_or_response_headers_perf_ns")) is int for r in written["victim_requests"]),
        )
        check(
            "every victim record has a non-null token_16_perf_ns for cohort members",
            all(
                type(r.get("token_16_perf_ns")) is int
                for r in written["victim_requests"] if r.get("server_exposure_group") == "running_at_trigger_observed"
            ),
        )
        check("every victim record's timestamp_monotonicity_errors is empty", all(
            r.get("timestamp_monotonicity_errors") == [] for r in written["victim_requests"]
        ))
        if written["burst_requests"]:
            check("every burst record's timestamp_monotonicity_errors is empty", all(
                r.get("timestamp_monotonicity_errors") == [] for r in written["burst_requests"]
            ))
            check("every burst record has burst_first_token_perf_ns", all(
                type(r.get("burst_first_token_perf_ns")) is int for r in written["burst_requests"]
            ))

        # --- Resume: re-running the same (already complete) block is a
        # clean no-op. ---------------------------------------------------
        resume_clock = base.FakeClock()
        transport2 = _build_fake_transport(bundle, "qwen_off12_k8_rep01", k=8, clock=resume_clock)
        metrics_transport2 = _build_fake_metrics_transport(k=8)
        summary2 = asyncio.run(
            swc.run_server_waiting_smoke_block(
                bundle=bundle, block_id="qwen_off12_k8_rep01", output_dir=smoke_dir,
                host="127.0.0.1", port=37996, resume=True, api_key="fake-key",
                transport=transport2, metrics_transport=metrics_transport2, tokenizer=base.FakeTokenizerAdapter(),
                server_adapter=base.FakeServerProcessAdapter(), sleeper=base.FakeSleeper(),
                clock=resume_clock, run_server_path=server_path,
            )
        )
        check("resume of an already-complete block is a no-op", summary2.get("overall_status") == "already_complete", str(summary2))
        check("resume no-op never dispatches any fake HTTP stream", transport2.peak_open_stream_count == 0)

        # --- Item 16: resume/integrity rejects malformed or semantically
        # inconsistent results. ---------------------------------------------
        ep0 = swc.find_block(bundle, "qwen_off12_k8_rep01")[0]
        result_path = swc.episode_result_path(smoke_dir, ep0.episode_id)
        original_text = result_path.read_text(encoding="utf-8")

        corrupted = json.loads(original_text)
        corrupted["schedule_row"]["server_max_num_seqs"] = 999  # semantically inconsistent
        result_path.write_text(json.dumps(corrupted), encoding="utf-8")
        cls, notes = swc.classify_result_file(result_path, ep0, bundle.fingerprint, swc.RUN_MODE_SMOKE)
        check(
            "classify_result_file rejects a schedule_row that no longer matches the schedule",
            cls == swc.CLASSIFICATION_INVALID, str((cls, notes)),
        )

        result_path.write_text("{not valid json", encoding="utf-8")
        cls2, notes2 = swc.classify_result_file(result_path, ep0, bundle.fingerprint, swc.RUN_MODE_SMOKE)
        check("classify_result_file rejects malformed JSON as corrupted", cls2 == swc.CLASSIFICATION_CORRUPTED, str((cls2, notes2)))

        result_path.write_text(original_text, encoding="utf-8")  # restore
        cls3, notes3 = swc.classify_result_file(result_path, ep0, bundle.fingerprint, swc.RUN_MODE_SMOKE)
        check("classify_result_file accepts the restored, valid result as valid_complete", cls3 == swc.CLASSIFICATION_VALID_COMPLETE, str(notes3))

        # --- Audit2 N2 fix: the four original B3 adversarial mutations,
        # as EXPLICIT regression tests against classify_result_file()
        # (previously fixed in code and confirmed manually during audit,
        # but not present in the test file itself). ---------------------
        ep_all = swc.find_block(bundle, "qwen_off12_k8_rep01")
        ep_burst = next(e for e in ep_all if e.condition == base.BURST_CONDITION)
        burst_result_path = swc.episode_result_path(smoke_dir, ep_burst.episode_id)
        burst_original_text = burst_result_path.read_text(encoding="utf-8")

        # 1. burst request status changed to "failed".
        mutated1 = json.loads(burst_original_text)
        assert mutated1["burst_requests"], "expected the prefill_burst episode file to have burst_requests"
        mutated1["burst_requests"][0]["status"] = "failed"
        burst_result_path.write_text(json.dumps(mutated1), encoding="utf-8")
        cls_b1, notes_b1 = swc.classify_result_file(burst_result_path, ep_burst, bundle.fingerprint, swc.RUN_MODE_SMOKE)
        check(
            "B3 adversarial #1: classify_result_file rejects a burst request with status changed to 'failed'",
            cls_b1 == swc.CLASSIFICATION_INVALID, str((cls_b1, notes_b1)),
        )
        burst_result_path.write_text(burst_original_text, encoding="utf-8")

        # 2. validation_errors changed to non-empty.
        mutated2 = json.loads(original_text)
        mutated2["validation_errors"] = ["synthetic adversarial error"]
        result_path.write_text(json.dumps(mutated2), encoding="utf-8")
        cls_b2, notes_b2 = swc.classify_result_file(result_path, ep0, bundle.fingerprint, swc.RUN_MODE_SMOKE)
        check(
            "B3 adversarial #2: classify_result_file rejects a result with non-empty validation_errors",
            cls_b2 == swc.CLASSIFICATION_INVALID, str((cls_b2, notes_b2)),
        )
        result_path.write_text(original_text, encoding="utf-8")

        # 3. no_client_admission_semaphore_used removed entirely.
        mutated3 = json.loads(original_text)
        del mutated3["no_client_admission_semaphore_used"]
        result_path.write_text(json.dumps(mutated3), encoding="utf-8")
        cls_b3, notes_b3 = swc.classify_result_file(result_path, ep0, bundle.fingerprint, swc.RUN_MODE_SMOKE)
        check(
            "B3 adversarial #3: classify_result_file rejects a result missing no_client_admission_semaphore_used",
            cls_b3 == swc.CLASSIFICATION_INVALID, str((cls_b3, notes_b3)),
        )
        result_path.write_text(original_text, encoding="utf-8")

        # 4. non-cohort decode_tokens_received_at_trigger changed to None.
        mutated4 = json.loads(original_text)
        active_idx_set = set(mutated4["trigger"].get("active_cohort_request_indices") or [])
        noncohort_victim = next(v for v in mutated4["victim_requests"] if v["request_index"] not in active_idx_set)
        noncohort_victim["decode_tokens_received_at_trigger"] = None
        result_path.write_text(json.dumps(mutated4), encoding="utf-8")
        cls_b4, notes_b4 = swc.classify_result_file(result_path, ep0, bundle.fingerprint, swc.RUN_MODE_SMOKE)
        check(
            "B3 adversarial #4: classify_result_file rejects a non-cohort decode_tokens_received_at_trigger changed to None",
            cls_b4 == swc.CLASSIFICATION_INVALID, str((cls_b4, notes_b4)),
        )
        result_path.write_text(original_text, encoding="utf-8")

        # Integrity manifest build/verify round-trip, then a tampered file
        # must be caught.
        env = swc.FakeEnvironmentProbe().gather(schedule_dir)
        environment_fingerprint = swc.compute_environment_fingerprint(env)
        manifest = swc.build_integrity_manifest(
            smoke_dir, schedule_fingerprint=bundle.fingerprint,
            environment_fingerprint=environment_fingerprint, clock=base.FakeClock(),
        )
        ok, verify_errors = swc.verify_integrity_manifest(
            smoke_dir, manifest, expected_schedule_fingerprint=bundle.fingerprint,
            expected_environment_fingerprint=environment_fingerprint,
        )
        check("integrity manifest round-trips cleanly against disk", ok, str(verify_errors))

        # Tamper with a file after the manifest was built.
        ep_files[0].write_text(ep_files[0].read_text(encoding="utf-8") + "\n// tampered\n", encoding="utf-8")
        ok2, verify_errors2 = swc.verify_integrity_manifest(
            smoke_dir, manifest, expected_schedule_fingerprint=bundle.fingerprint,
            expected_environment_fingerprint=environment_fingerprint,
        )
        check("integrity manifest verification detects a tampered file", not ok2 and bool(verify_errors2))

    # ========================================================================
    # 2026-07-20 hardening pass: B1 fingerprint expansion
    # ========================================================================
    real_env = swc.RealEnvironmentProbe().gather(schedule_dir)
    actual_cohort_hash = base._sha256_file(SCRIPT_DIR / "_active_cohort.py")
    actual_base_hash = base._sha256_file(BASE_DIR / "run_prefill_confirmation.py")
    check(
        "RealEnvironmentProbe hashes _active_cohort.py matching its actual on-disk content",
        real_env["file_hashes"].get("_active_cohort.py") == actual_cohort_hash,
    )
    check(
        "RealEnvironmentProbe hashes the imported base runner matching its actual on-disk content",
        real_env["file_hashes"].get("run_prefill_confirmation.py") == actual_base_hash,
    )
    base_env = swc.FakeEnvironmentProbe().gather(schedule_dir)
    fp_base = swc.compute_environment_fingerprint(base_env)
    for mutated_key in ("_active_cohort.py", "run_prefill_confirmation.py"):
        mutated_env = json.loads(json.dumps(base_env))
        mutated_env["file_hashes"][mutated_key] = "f" * 64 if mutated_env["file_hashes"][mutated_key] != "f" * 64 else "0" * 64
        fp_mutated = swc.compute_environment_fingerprint(mutated_env)
        check(f"mutating {mutated_key}'s hash changes environment_fingerprint", fp_mutated != fp_base)

    # ========================================================================
    # 2026-07-20 hardening pass: persistent transport lifecycle (requirements 1, 4)
    # ========================================================================
    async def _transport_lifecycle_probe():
        ct = swc.PersistentCompletionTransport()
        results = {"before": ct.is_started()}
        await ct.start()
        results["after_start"] = ct.is_started()
        await ct.start()  # idempotent
        results["after_second_start"] = ct.is_started()
        results["diagnostics"] = ct.get_diagnostics()
        await ct.aclose()
        results["after_close"] = ct.is_started()
        await ct.aclose()  # idempotent
        results["after_second_close"] = ct.is_started()

        mt = swc.PersistentMetricsTransport()
        await mt.start()
        results["metrics_started"] = mt.is_started()
        await mt.aclose()
        results["metrics_closed"] = not mt.is_started()
        return results

    try:
        lifecycle = asyncio.run(_transport_lifecycle_probe())
        check("PersistentCompletionTransport starts not-started", lifecycle["before"] is False)
        check("PersistentCompletionTransport.start() is effective", lifecycle["after_start"] is True)
        check("PersistentCompletionTransport.start() is idempotent", lifecycle["after_second_start"] is True)
        check(
            "PersistentCompletionTransport reports exact 32/32 connection limits",
            lifecycle["diagnostics"]["max_connections"] == 32
            and lifecycle["diagnostics"]["max_keepalive_connections"] == 32,
            str(lifecycle["diagnostics"]),
        )
        check("PersistentCompletionTransport.aclose() is effective", lifecycle["after_close"] is False)
        check("PersistentCompletionTransport.aclose() is idempotent", lifecycle["after_second_close"] is False)
        check("PersistentMetricsTransport starts/closes independently", lifecycle["metrics_started"] and lifecycle["metrics_closed"])
    except ImportError as exc:
        check("persistent transport lifecycle probe (httpx required)", False, str(exc))

    # ========================================================================
    # 2026-07-20 hardening pass: --diagnostic-pair-only mode
    # ========================================================================
    diag_episodes, diag_errors = swc.validate_diagnostic_block_selection(bundle)
    check(
        "validate_diagnostic_block_selection accepts the real qwen_off12_k8_rep01 block",
        diag_episodes is not None and not diag_errors, str(diag_errors),
    )
    check(
        "validate_diagnostic_block_selection preserves schedule-defined order_in_block order",
        diag_episodes is not None and [e.order_in_block for e in diag_episodes] == [1, 2],
    )

    # A non-existent/malformed block must be rejected with reasons, not
    # silently accepted.
    class _FakeEpisodeForSelection:
        pass
    empty_bundle_episodes, empty_errors = swc.validate_diagnostic_block_selection(
        swc.LoadedBundle(
            schedule_dir=schedule_dir, json_obj=bundle.json_obj, csv_fieldnames=bundle.csv_fieldnames,
            csv_rows=bundle.csv_rows, audit_text=bundle.audit_text, episodes=[], fingerprint=bundle.fingerprint,
        )
    )
    check("validate_diagnostic_block_selection rejects a bundle with no matching block", empty_bundle_episodes is None and bool(empty_errors))

    fake_diag_env = swc.FakeEnvironmentProbe().gather(schedule_dir)

    with tempfile.TemporaryDirectory() as td:
        nonempty_dir = Path(td) / "diag"
        nonempty_dir.mkdir()
        (nonempty_dir / "leftover.txt").write_text("x", encoding="utf-8")
        diag_transport = _build_fake_transport(bundle, "qwen_off12_k8_rep01", k=8, clock=base.FakeClock())
        diag_metrics_transport = _build_fake_metrics_transport(k=8)
        refused = False
        try:
            asyncio.run(swc.run_diagnostic_pair(
                bundle=bundle, output_dir=nonempty_dir, host="127.0.0.1", port=37997, api_key="fake-key",
                transport=diag_transport, metrics_transport=diag_metrics_transport, tokenizer=base.FakeTokenizerAdapter(),
                server_adapter=base.FakeServerProcessAdapter(), sleeper=base.FakeSleeper(), clock=base.FakeClock(),
                run_server_path=server_path, env=fake_diag_env,
            ))
        except base.ServerLifecycleError:
            refused = True
        check("run_diagnostic_pair refuses a non-empty output directory", refused)

    # --resume is rejected at the CLI layer before any bundle/network work.
    with tempfile.TemporaryDirectory() as td:
        resume_rc = swc.main(["--diagnostic-pair-only", "--resume", "--output-dir", str(Path(td) / "d"), "--schedule-dir", str(schedule_dir)])
        check("--diagnostic-pair-only --resume is rejected by the CLI with exit code 1", resume_rc == 1)
        no_output_dir_rc = swc.main(["--diagnostic-pair-only", "--schedule-dir", str(schedule_dir)])
        check("--diagnostic-pair-only without --output-dir is rejected by the CLI with exit code 1", no_output_dir_rc == 1)

    # Structural check: run_diagnostic_pair never references
    # run_official_campaign and never iterates all_block_ids_in_schedule_order.
    check(
        "run_diagnostic_pair never references run_official_campaign or all-block iteration",
        "run_official_campaign" not in swc.run_diagnostic_pair.__code__.co_names
        and "all_block_ids_in_schedule_order" not in swc.run_diagnostic_pair.__code__.co_names,
    )

    # Fake integration: exactly two regular episodes, one server start, one
    # server stop, and no other block touched.
    with tempfile.TemporaryDirectory() as td:
        diag_dir = Path(td) / "diagnostic_pair"
        diag_clock = base.FakeClock()
        diag_transport = _build_fake_transport(bundle, "qwen_off12_k8_rep01", k=8, clock=diag_clock)
        diag_metrics_transport = _build_fake_metrics_transport(k=8)
        diag_server_adapter = base.FakeServerProcessAdapter()
        diag_summary = asyncio.run(swc.run_diagnostic_pair(
            bundle=bundle, output_dir=diag_dir, host="127.0.0.1", port=37998, api_key="fake-key",
            transport=diag_transport, metrics_transport=diag_metrics_transport, tokenizer=base.FakeTokenizerAdapter(),
            server_adapter=diag_server_adapter, sleeper=base.FakeSleeper(), clock=diag_clock,
            run_server_path=server_path, env=fake_diag_env,
        ))
        check("fake diagnostic pair completes", diag_summary.get("overall_status") == "block_complete", str(diag_summary))
        check("fake diagnostic pair started exactly one server", len(diag_server_adapter.started) == 1, str(len(diag_server_adapter.started)))
        check("fake diagnostic pair wrote exactly two episode files", len(list((diag_dir / "episodes").glob("*.json"))) == 2)
        check("fake diagnostic pair wrote exactly one stabilization file", len(list((diag_dir / "stabilization").glob("*.json"))) == 1)
        check("fake diagnostic pair wrote a JSON summary", (diag_dir / "diagnostic_pair_summary.json").is_file())
        check("fake diagnostic pair wrote a human-readable text summary", (diag_dir / "diagnostic_pair_summary.txt").is_file())
        check(
            "fake diagnostic pair's classification is one of A/B/C/D",
            diag_summary["classification"]["classification"] in (
                swc.DIAGNOSTIC_CLASSIFICATION_A, swc.DIAGNOSTIC_CLASSIFICATION_B,
                swc.DIAGNOSTIC_CLASSIFICATION_C, swc.DIAGNOSTIC_CLASSIFICATION_D,
            ),
            str(diag_summary["classification"]),
        )
        check(
            "fake diagnostic pair's classification is genuinely A/B/C (never silently D) for this well-formed run",
            diag_summary["classification"]["classification"] != swc.DIAGNOSTIC_CLASSIFICATION_D
            and diag_summary["classification"].get("reasons") == [],
            str(diag_summary["classification"]),
        )
        check("fake diagnostic pair's diagnostic_valid is True for this well-formed run", diag_summary.get("diagnostic_valid") is True, str(diag_summary))
        check("fake diagnostic pair's paired_effect_summary is available", diag_summary["paired_effect_summary"].get("available") is True)

        # --- B1: diagnostic provenance must be bound to the runtime
        # environment, not merely computed in memory and discarded. -----
        expected_env_fingerprint = swc.compute_environment_fingerprint(fake_diag_env)
        check(
            "fake diagnostic pair's summary carries environment_fingerprint matching the gathered env",
            diag_summary.get("environment_fingerprint") == expected_env_fingerprint,
            str(diag_summary.get("environment_fingerprint")),
        )
        diag_manifest_path = diag_dir / swc.DIAGNOSTIC_RUN_MANIFEST_FILENAME
        check("fake diagnostic pair wrote a diagnostic_run_manifest.json before/around server start", diag_manifest_path.is_file())
        if diag_manifest_path.is_file():
            diag_manifest_obj = json.loads(diag_manifest_path.read_text(encoding="utf-8"))
            check(
                "diagnostic_run_manifest.json carries the correct environment_fingerprint",
                diag_manifest_obj.get("environment_fingerprint") == expected_env_fingerprint,
            )
            check(
                "diagnostic_run_manifest.json carries the full file_hashes snapshot",
                diag_manifest_obj.get("file_hashes") == fake_diag_env.get("file_hashes"),
            )
            check("diagnostic_run_manifest.json records the diagnostic_block_id", diag_manifest_obj.get("diagnostic_block_id") == swc.DIAGNOSTIC_BLOCK_ID)

            # Semantic provenance validation: hash coverage alone must not
            # legitimize an internally wrong manifest or stabilization file.
            expected_manifest_obj = swc.build_official_run_manifest(
                env=fake_diag_env, bundle=bundle, run_mode=swc.RUN_MODE_DIAGNOSTIC_PAIR,
                output_dir=diag_dir, host="127.0.0.1", port=37998,
                clock=base.FakeClock(),
            )
            # created_utc is captured at the real pre-start call; reuse the
            # stored value when checking exact semantic equality here.
            expected_manifest_obj["created_utc"] = diag_manifest_obj.get("created_utc")
            expected_manifest_obj["diagnostic_block_id"] = swc.DIAGNOSTIC_BLOCK_ID
            manifest_semantic_errors = swc._validate_diagnostic_run_manifest_artifact(
                diag_manifest_obj, expected_manifest_obj,
            )
            check(
                "diagnostic_run_manifest semantic validator accepts the exact stored manifest",
                manifest_semantic_errors == [], str(manifest_semantic_errors),
            )
            bad_manifest_obj = copy.deepcopy(diag_manifest_obj)
            bad_manifest_obj["host"] = "wrong"
            check(
                "diagnostic_run_manifest semantic validator rejects a wrong host",
                bool(swc._validate_diagnostic_run_manifest_artifact(bad_manifest_obj, expected_manifest_obj)),
            )

            expected_diag_episode = next(e for e in diag_episodes if e.condition == "no_burst")
            expected_diag_command = swc.build_server_command(
                server_path, swc.DIAGNOSTIC_MODEL_KEY, swc.DIAGNOSTIC_OFFLOAD_GB,
                swc.DIAGNOSTIC_SERVER_MAX_NUM_SEQS, "127.0.0.1", 37998,
            )
            stab_path = diag_dir / "stabilization" / f"{swc.DIAGNOSTIC_BLOCK_ID}.json"
            stab_obj = json.loads(stab_path.read_text(encoding="utf-8"))
            stab_semantic_errors = swc._validate_diagnostic_stabilization_artifact(
                obj=stab_obj, bundle=bundle, expected_episode=expected_diag_episode,
                expected_environment_fingerprint=expected_env_fingerprint,
                expected_server_command=expected_diag_command,
            )
            check(
                "stabilization semantic validator accepts the exact functional artifact",
                stab_semantic_errors == [], str(stab_semantic_errors),
            )
            bad_stab_obj = copy.deepcopy(stab_obj)
            bad_stab_obj["functional_passed"] = False
            check(
                "stabilization semantic validator rejects functional_passed=False",
                bool(swc._validate_diagnostic_stabilization_artifact(
                    obj=bad_stab_obj, bundle=bundle, expected_episode=expected_diag_episode,
                    expected_environment_fingerprint=expected_env_fingerprint,
                    expected_server_command=expected_diag_command,
                )),
            )

        # environment_fingerprint must also have flowed into each written
        # episode result's own server_metadata (not just the top-level
        # summary/manifest).
        for ep_file in sorted((diag_dir / "episodes").glob("*.json")):
            ep_obj = json.loads(ep_file.read_text(encoding="utf-8"))
            check(
                f"episode result {ep_file.name} carries environment_fingerprint in server_metadata",
                ep_obj.get("server_metadata", {}).get("environment_fingerprint") == expected_env_fingerprint,
            )

        # A mutated file hash must change the fingerprint actually
        # recorded in the diagnostic manifest (not just in an abstract,
        # never-persisted in-memory computation).
        mutated_env = json.loads(json.dumps(fake_diag_env))
        mutated_env["file_hashes"]["_active_cohort.py"] = "0" * 64 if mutated_env["file_hashes"]["_active_cohort.py"] != "0" * 64 else "1" * 64
        check(
            "a one-byte-equivalent mutation of _active_cohort.py's recorded hash changes the fingerprint that would be stored",
            swc.compute_environment_fingerprint(mutated_env) != expected_env_fingerprint,
        )

        # --- Blocker 1 (2026-07-20 fourth hardening pass): final
        # integrity manifest is built and verified ONLY AFTER both
        # summary files are written, so it covers the COMPLETE, final
        # result set -- not a mid-way snapshot. -------------------------
        check("fake diagnostic pair reports integrity_verified is True (returned value)", diag_summary.get("integrity_verified") is True, str(diag_summary.get("integrity_errors")))
        check("fake diagnostic pair reports diagnostic_valid consistent with its classification", isinstance(diag_summary.get("diagnostic_valid"), bool))

        disk_summary_path = diag_dir / "diagnostic_pair_summary.json"
        disk_summary_obj = json.loads(disk_summary_path.read_text(encoding="utf-8"))
        check(
            "on-disk diagnostic_pair_summary.json makes NO self-referential integrity_verified claim (non-circular)",
            "integrity_verified" not in disk_summary_obj and "diagnostic_valid" not in disk_summary_obj,
            str(disk_summary_obj.keys()),
        )
        check(
            "on-disk diagnostic_pair_summary.json instead carries non-circular integrity metadata",
            disk_summary_obj.get("integrity_manifest_filename") == swc.INTEGRITY_MANIFEST_FILENAME
            and disk_summary_obj.get("integrity_finalization_required") is True
            and isinstance(disk_summary_obj.get("integrity_scope"), str),
            str(disk_summary_obj),
        )

        integrity_manifest_path = diag_dir / swc.INTEGRITY_MANIFEST_FILENAME
        check("fake diagnostic pair wrote a final integrity manifest", integrity_manifest_path.is_file())
        if integrity_manifest_path.is_file():
            integrity_obj = json.loads(integrity_manifest_path.read_text(encoding="utf-8"))

            manifest_file_names = {
                (entry.get("relative_path") if isinstance(entry, dict) else entry)
                for entry in (integrity_obj.get("files") or [])
            }
            check(
                "final integrity manifest lists diagnostic_pair_summary.json among its covered files",
                any("diagnostic_pair_summary.json" in str(n) for n in manifest_file_names), str(manifest_file_names),
            )
            check(
                "final integrity manifest lists diagnostic_pair_summary.txt among its covered files",
                any("diagnostic_pair_summary.txt" in str(n) for n in manifest_file_names), str(manifest_file_names),
            )
            check(
                "final integrity manifest does NOT list itself (no recursive self-hash)",
                not any(swc.INTEGRITY_MANIFEST_FILENAME in str(n) for n in manifest_file_names), str(manifest_file_names),
            )

            ok_before_tamper, errors_before_tamper = swc.verify_diagnostic_integrity_manifest(
                diag_dir, integrity_obj, expected_schedule_fingerprint=bundle.fingerprint,
                expected_environment_fingerprint=expected_env_fingerprint,
            )
            check(
                "final integrity manifest verifies with STRICTLY ZERO errors against the complete, finished output tree",
                ok_before_tamper and not errors_before_tamper, str(errors_before_tamper),
            )

            # Proof that verification genuinely happens AFTER the summary
            # files are written: the manifest's recorded hash for
            # diagnostic_pair_summary.json exactly matches the hash of
            # what is ACTUALLY on disk -- which could only be true if the
            # manifest was built from the real, final file content, not a
            # placeholder computed beforehand.
            summary_hash_entries = [
                entry for entry in (integrity_obj.get("files") or [])
                if isinstance(entry, dict) and "diagnostic_pair_summary.json" in str(entry.get("relative_path"))
            ]
            check("integrity manifest has exactly one entry for diagnostic_pair_summary.json", len(summary_hash_entries) == 1, str(summary_hash_entries))
            if summary_hash_entries:
                actual_summary_hash = base._sha256_file(disk_summary_path)
                check(
                    "integrity manifest's recorded hash for diagnostic_pair_summary.json matches its actual on-disk content "
                    "(proves the manifest was built AFTER the summary was written, not before)",
                    summary_hash_entries[0].get("sha256") == actual_summary_hash,
                    str((summary_hash_entries[0].get("sha256"), actual_summary_hash)),
                )

            # Tamper with diagnostic_pair_summary.json specifically.
            original_summary_text = disk_summary_path.read_text(encoding="utf-8")
            disk_summary_path.write_text(original_summary_text + "\n// tampered\n", encoding="utf-8")
            ok_after_json_tamper, errors_after_json_tamper = swc.verify_diagnostic_integrity_manifest(
                diag_dir, integrity_obj, expected_schedule_fingerprint=bundle.fingerprint,
                expected_environment_fingerprint=expected_env_fingerprint,
            )
            check(
                "tampering diagnostic_pair_summary.json is detected by the final integrity manifest",
                not ok_after_json_tamper and bool(errors_after_json_tamper),
            )
            disk_summary_path.write_text(original_summary_text, encoding="utf-8")  # restore

            # Tamper with diagnostic_pair_summary.txt specifically.
            disk_summary_txt_path = diag_dir / "diagnostic_pair_summary.txt"
            original_summary_txt_text = disk_summary_txt_path.read_text(encoding="utf-8")
            disk_summary_txt_path.write_text(original_summary_txt_text + "\n// tampered\n", encoding="utf-8")
            ok_after_txt_tamper, errors_after_txt_tamper = swc.verify_diagnostic_integrity_manifest(
                diag_dir, integrity_obj, expected_schedule_fingerprint=bundle.fingerprint,
                expected_environment_fingerprint=expected_env_fingerprint,
            )
            check(
                "tampering diagnostic_pair_summary.txt is detected by the final integrity manifest",
                not ok_after_txt_tamper and bool(errors_after_txt_tamper),
            )
            disk_summary_txt_path.write_text(original_summary_txt_text, encoding="utf-8")  # restore

            # An extra, unlisted file dropped into the output tree must
            # also be detected.
            rogue_file = diag_dir / "unexpected_extra_file.txt"
            rogue_file.write_text("should not be here", encoding="utf-8")
            ok_after_extra_file, errors_after_extra_file = swc.verify_diagnostic_integrity_manifest(
                diag_dir, integrity_obj, expected_schedule_fingerprint=bundle.fingerprint,
                expected_environment_fingerprint=expected_env_fingerprint,
            )
            check(
                "an extra, unlisted file in the output tree is detected by the final integrity manifest",
                not ok_after_extra_file and bool(errors_after_extra_file),
            )
            rogue_file.unlink()

            # --- Blocker A (2026-07-20 fifth hardening pass): a same-named
            # file anywhere OTHER than the true root must be treated as an
            # ordinary extra file -- never silently excluded by basename
            # alone. ------------------------------------------------------
            nested_dir = diag_dir / "nested"
            nested_dir.mkdir(exist_ok=True)
            nested_manifest = nested_dir / swc.INTEGRITY_MANIFEST_FILENAME
            nested_manifest.write_text("{}", encoding="utf-8")
            ok_nested, errors_nested = swc.verify_diagnostic_integrity_manifest(
                diag_dir, integrity_obj, expected_schedule_fingerprint=bundle.fingerprint,
                expected_environment_fingerprint=expected_env_fingerprint,
            )
            check(
                f"a nested {swc.INTEGRITY_MANIFEST_FILENAME} (nested/{swc.INTEGRITY_MANIFEST_FILENAME}) is "
                f"detected as an unexpected file, not silently excluded by basename",
                not ok_nested and any("nested/" in e and swc.INTEGRITY_MANIFEST_FILENAME in e for e in errors_nested),
                str(errors_nested),
            )
            nested_manifest.unlink()
            nested_dir.rmdir()

            deep_nested_dir = diag_dir / "a" / "b" / "c"
            deep_nested_dir.mkdir(parents=True, exist_ok=True)
            deep_nested_manifest = deep_nested_dir / swc.INTEGRITY_MANIFEST_FILENAME
            deep_nested_manifest.write_text("{}", encoding="utf-8")
            ok_deep_nested, errors_deep_nested = swc.verify_diagnostic_integrity_manifest(
                diag_dir, integrity_obj, expected_schedule_fingerprint=bundle.fingerprint,
                expected_environment_fingerprint=expected_env_fingerprint,
            )
            check(
                f"a deeply nested a/b/c/{swc.INTEGRITY_MANIFEST_FILENAME} is also detected as unexpected",
                not ok_deep_nested and any(
                    f"a/b/c/{swc.INTEGRITY_MANIFEST_FILENAME}" in e for e in errors_deep_nested
                ),
                str(errors_deep_nested),
            )
            deep_nested_manifest.unlink()
            deep_nested_dir.rmdir()
            (diag_dir / "a" / "b").rmdir()
            (diag_dir / "a").rmdir()

            ok_after_nested_cleanup, errors_after_nested_cleanup = swc.verify_diagnostic_integrity_manifest(
                diag_dir, integrity_obj, expected_schedule_fingerprint=bundle.fingerprint,
                expected_environment_fingerprint=expected_env_fingerprint,
            )
            check(
                "after removing the nested test files, the manifest verifies cleanly again",
                ok_after_nested_cleanup and not errors_after_nested_cleanup, str(errors_after_nested_cleanup),
            )

            # Final coding pass: files already present BEFORE manifest
            # construction must not be blessed into a newly generated
            # manifest. The builder hashes only the exact whitelist.
            root_manifest_path = diag_dir / swc.INTEGRITY_MANIFEST_FILENAME
            root_manifest_text = root_manifest_path.read_text(encoding="utf-8")
            root_manifest_path.unlink()
            prebuild_rogue = diag_dir / "unexpected_prebuild.txt"
            prebuild_rogue.write_text("rogue", encoding="utf-8")
            prebuild_errors = swc._validate_diagnostic_artifact_whitelist(
                diag_dir, include_integrity_manifest=False,
            )
            check(
                "pre-manifest whitelist rejects an unexpected file before manifest construction",
                any("unexpected_prebuild.txt" in e for e in prebuild_errors), str(prebuild_errors),
            )
            strict_manifest = swc.build_diagnostic_integrity_manifest(
                diag_dir, schedule_fingerprint=bundle.fingerprint,
                environment_fingerprint=expected_env_fingerprint, clock=base.FakeClock(),
            )
            strict_paths = {entry["relative_path"] for entry in strict_manifest["files"]}
            check(
                "manifest builder never blesses an unexpected pre-existing file",
                "unexpected_prebuild.txt" not in strict_paths, str(sorted(strict_paths)),
            )
            swc.write_json_atomic(root_manifest_path, strict_manifest)
            strict_ok, strict_errors = swc.verify_diagnostic_integrity_manifest(
                diag_dir, strict_manifest, expected_schedule_fingerprint=bundle.fingerprint,
                expected_environment_fingerprint=expected_env_fingerprint,
                expected_episode_count=2, expected_stabilization_count=1,
                expected_block_summary_count=0, expected_server_log_count=1,
            )
            check(
                "verification detects the unlisted pre-existing file after strict manifest construction",
                not strict_ok and any("unexpected_prebuild.txt" in e for e in strict_errors), str(strict_errors),
            )
            prebuild_rogue.unlink()
            root_manifest_path.write_text(root_manifest_text, encoding="utf-8")

            # --- Blocker A: incomplete artifact counts fail closed. -------
            artifact_errors_ok = swc._validate_diagnostic_artifact_counts(diag_dir)
            check("_validate_diagnostic_artifact_counts reports zero errors for the complete, well-formed tree", artifact_errors_ok == [], str(artifact_errors_ok))

            extra_stabilization_file = diag_dir / "stabilization" / "extra_stabilization_result.json"
            extra_stabilization_file.write_text("{}", encoding="utf-8")
            artifact_errors_extra_stab = swc._validate_diagnostic_artifact_counts(diag_dir)
            check(
                "_validate_diagnostic_artifact_counts fails closed when an extra stabilization file is present",
                bool(artifact_errors_extra_stab), str(artifact_errors_extra_stab),
            )
            ok_extra_stab, errors_extra_stab = swc.verify_diagnostic_integrity_manifest(
                diag_dir, integrity_obj, expected_schedule_fingerprint=bundle.fingerprint,
                expected_environment_fingerprint=expected_env_fingerprint,
                expected_episode_count=2, expected_stabilization_count=1,
                expected_block_summary_count=0, expected_server_log_count=1,
            )
            check(
                "verify_diagnostic_integrity_manifest with expected counts also fails closed on an extra stabilization file",
                not ok_extra_stab, str(errors_extra_stab),
            )
            extra_stabilization_file.unlink()

            missing_episode_file = sorted((diag_dir / "episodes").glob("*.json"))[0]
            missing_episode_text = missing_episode_file.read_text(encoding="utf-8")
            missing_episode_file.unlink()
            artifact_errors_missing_ep = swc._validate_diagnostic_artifact_counts(diag_dir)
            check(
                "_validate_diagnostic_artifact_counts fails closed when an episode file is missing",
                bool(artifact_errors_missing_ep), str(artifact_errors_missing_ep),
            )
            missing_episode_file.write_text(missing_episode_text, encoding="utf-8")  # restore

            # Tamper with an episode file too (already covered previously,
            # re-confirmed here against the NEW finalization order).
            some_ep_file = sorted((diag_dir / "episodes").glob("*.json"))[0]
            original_ep_text = some_ep_file.read_text(encoding="utf-8")
            some_ep_file.write_text(original_ep_text + "\n// tampered\n", encoding="utf-8")
            ok_after_tamper, errors_after_tamper = swc.verify_diagnostic_integrity_manifest(
                diag_dir, integrity_obj, expected_schedule_fingerprint=bundle.fingerprint,
                expected_environment_fingerprint=expected_env_fingerprint,
            )
            check("diagnostic integrity manifest verification fails closed after tampering an episode file", not ok_after_tamper and bool(errors_after_tamper))
            some_ep_file.write_text(original_ep_text, encoding="utf-8")  # restore

        # A resumed diagnostic-pair-only run against the SAME (now
        # non-empty) directory must be refused (fresh-directory-only,
        # never resumable) -- there is no --resume path at all here.
        refused_second_run = False
        try:
            asyncio.run(swc.run_diagnostic_pair(
                bundle=bundle, output_dir=diag_dir, host="127.0.0.1", port=37998, api_key="fake-key",
                transport=_build_fake_transport(bundle, "qwen_off12_k8_rep01", k=8, clock=base.FakeClock()),
                metrics_transport=_build_fake_metrics_transport(k=8), tokenizer=base.FakeTokenizerAdapter(),
                server_adapter=base.FakeServerProcessAdapter(), sleeper=base.FakeSleeper(), clock=base.FakeClock(),
                run_server_path=server_path, env=fake_diag_env,
            ))
        except base.ServerLifecycleError:
            refused_second_run = True
        check("re-running run_diagnostic_pair against the same (now non-empty) directory is refused", refused_second_run)

    # ========================================================================
    # 2026-07-20 hardening pass: A/B/C/D classifier
    # ========================================================================
    # Timeline (perf_ns, arbitrary units): all 20 victims are dispatched
    # and their streams open together, early (100-319 range), well before
    # ANY first token (>=400) -- matching "no client admission semaphore,
    # all streams open before first token". The 8 active-cohort victims
    # (indices 0-7) get real tokens immediately and cross token-16
    # somewhere in 1000-1070; trigger_ns = 1070 (max of their crossings).
    # The 12 non-cohort victims (8-19) get zero tokens until AFTER the
    # trigger, then eventually complete (all victims always reach 64
    # output tokens): first_token ~2000+, token_16 ~2160+, end ~20000+.
    TRIGGER_NS = 1070

    # Blocker 2 (2026-07-20 fourth hardening pass): fixtures now carry
    # full identity, bound to the REAL qwen_off12_k8_rep01 Episode
    # objects from the actual validated bundle (not hand-built fakes),
    # so every identity-mutation adversarial test below is checked
    # against the exact same expected-episode-object contract
    # `classify_diagnostic_pair()` requires.
    diag_real_episodes = swc.find_block(bundle, "qwen_off12_k8_rep01")
    real_no_burst_ep = next(e for e in diag_real_episodes if e.condition == "no_burst")
    real_prefill_burst_ep = next(e for e in diag_real_episodes if e.condition == base.BURST_CONDITION)
    classifier_env_fp = swc.compute_environment_fingerprint(swc.FakeEnvironmentProbe().gather(schedule_dir))
    classifier_expected_server_command = swc.build_server_command(
        server_path, swc.DIAGNOSTIC_MODEL_KEY, swc.DIAGNOSTIC_OFFLOAD_GB,
        swc.DIAGNOSTIC_SERVER_MAX_NUM_SEQS, "127.0.0.1", 8000,
    )

    def _synthetic_victim(idx: int, *, in_cohort: bool, episode: "swc.Episode") -> dict:
        task_created_ns = 100 + idx
        request_dispatch_ns = 200 + idx
        stream_open_ns = 300 + idx
        if in_cohort:
            first_token_ns = 400 + idx
            token16_ns = 1000 + idx * 10
            end_ns = 5000 + idx * 10
            dcount = 16
            exposure_group = swc.SERVER_EXPOSURE_RUNNING_AT_TRIGGER
        else:
            first_token_ns = 2000 + idx * 10
            token16_ns = 2160 + idx * 10
            end_ns = 20000 + idx * 10
            dcount = 0
            exposure_group = swc.SERVER_EXPOSURE_DISPATCHED_NO_OUTPUT
        last_token_ns = end_ns
        tpot_ms = ((last_token_ns - first_token_ns) / 63) / 1e6
        prompt_ids = [1000 + idx] * episode.victim_input_len
        return {
            "request_id": f"{episode.episode_id}:victim:{idx}", "role": "victim",
            "prompt_seed": base.victim_prompt_seed(episode, idx),
            "generation_seed": base.victim_generation_seed(episode, idx),
            "prompt_token_ids_sent": prompt_ids,
            "prompt_token_ids_returned": list(prompt_ids),
            "prompt_sha256": base.prompt_sha256(prompt_ids),
            "request_index": idx, "status": base.REQUEST_STATUS_COMPLETE,
            "validation_errors": [], "http_status": 200,
            "timed_out": False, "cancelled": False, "done_received": True,
            "error_type": None, "error_message": None, "finish_reason": "length",
            "expected_prompt_tokens": 256, "expected_completion_tokens": 64,
            "output_token_ids": [1] * 64,
            "usage": {"prompt_tokens": 256, "completion_tokens": 64},
            "last_token_receive_ns": last_token_ns,
            "client_observed_tpot_ms": tpot_ms,
            "itl_available": True,
            "itl_ms": [tpot_ms] * 63,
            # Raw canonical fields (as execute_completion_request itself produces).
            "task_created_ns": task_created_ns, "request_dispatch_ns": request_dispatch_ns,
            "first_token_receive_ns": first_token_ns, "stream_end_ns": end_ns,
            # perf_ns aliases (must exactly equal the raw fields above).
            "task_created_perf_ns": task_created_ns, "http_dispatch_start_perf_ns": request_dispatch_ns,
            "stream_open_or_response_headers_perf_ns": stream_open_ns,
            "first_token_perf_ns": first_token_ns, "token_16_perf_ns": token16_ns, "stream_end_perf_ns": end_ns,
            # Exposure/boolean/derived fields, all independently reconstructible
            # from the above given TRIGGER_NS.
            "was_dispatched_at_trigger": request_dispatch_ns <= TRIGGER_NS,
            "had_first_token_at_trigger": first_token_ns <= TRIGGER_NS,
            "server_exposure_group": exposure_group,
            "decode_tokens_received_at_trigger": dcount,
            "dispatch_to_first_token_ms": (first_token_ns - request_dispatch_ns) / 1e6,
            "timestamp_monotonicity_errors": [],
            "server_max_num_seqs": swc.DIAGNOSTIC_SERVER_MAX_NUM_SEQS,
        }

    def _synthetic_burst(j: int, *, first_token_ns: int, episode: "swc.Episode") -> dict:
        dispatch_ns = first_token_ns - 2
        open_ns = first_token_ns - 1
        end_ns = first_token_ns + 8
        prompt_ids = [2000 + j] * episode.burst_input_len
        return {
            "request_id": f"{episode.episode_id}:burst:{j}", "role": "burst",
            "prompt_seed": base.burst_prompt_seed(episode, j),
            "generation_seed": base.burst_generation_seed(episode, j),
            "prompt_token_ids_sent": prompt_ids,
            "prompt_token_ids_returned": list(prompt_ids),
            "prompt_sha256": base.prompt_sha256(prompt_ids),
            "request_index": j, "status": base.REQUEST_STATUS_COMPLETE,
            "validation_errors": [], "http_status": 200,
            "timed_out": False, "cancelled": False, "done_received": True,
            "error_type": None, "error_message": None, "finish_reason": "length",
            "expected_prompt_tokens": 2048, "expected_completion_tokens": 16,
            "output_token_ids": [1] * 16,
            "usage": {"prompt_tokens": 2048, "completion_tokens": 16},
            "request_dispatch_ns": dispatch_ns, "first_token_receive_ns": first_token_ns, "stream_end_ns": end_ns,
            "burst_dispatch_start_perf_ns": dispatch_ns, "burst_stream_open_or_response_headers_perf_ns": open_ns,
            "burst_first_token_perf_ns": first_token_ns, "burst_end_perf_ns": end_ns,
            "timestamp_monotonicity_errors": [],
        }

    def _synthetic_episode(
        *, episode: "swc.Episode", condition: str, active_indices: set, burst_first_tokens=None, trigger_status="ok",
        active_cohort_size=8, active_cohort_indices_override=None, validation_errors=None,
        break_victim_status=False, break_burst_status=False, all_streams_open_override=None,
        no_semaphore_flag=True, corrupt_field: "tuple[int, str, object] | list[tuple[int, str, object]] | None" = None,
        corrupt_burst_field: tuple[int, str, object] | None = None,
        episode_id_override=None, block_id_override=None, schedule_row_override=None,
        schedule_row_field_override: tuple[str, object] | None = None, schedule_row_incomplete=False,
        run_mode_override=None, schedule_fingerprint_override=None, environment_fingerprint_override=None,
        result_schema_version_override=None, runner_version_override=None,
        record_type_override=None, remove_victim_field: tuple[int, str] | None = None,
        trigger_perf_ns_override=None,
    ) -> dict:
        victims = [_synthetic_victim(i, in_cohort=(i in active_indices), episode=episode) for i in range(20)]
        if break_victim_status:
            victims[0]["status"] = "failed"
        if corrupt_field is not None:
            corrupt_field_list = corrupt_field if isinstance(corrupt_field, list) else [corrupt_field]
            for idx, field, value in corrupt_field_list:
                victims[idx][field] = value
        if remove_victim_field is not None:
            idx, field = remove_victim_field
            del victims[idx][field]
        burst = []
        if condition == base.BURST_CONDITION:
            tokens = burst_first_tokens if burst_first_tokens is not None else [3002, 3003, 3004, 3005]
            burst = [_synthetic_burst(j, first_token_ns=t, episode=episode) for j, t in enumerate(tokens)]
            if break_burst_status:
                burst[0]["status"] = "failed"
            if corrupt_burst_field is not None:
                j, field, value = corrupt_burst_field
                burst[j][field] = value
        reported_active_indices = (
            active_cohort_indices_override if active_cohort_indices_override is not None else sorted(active_indices)
        )
        first_token_values = [v["first_token_perf_ns"] for v in victims if type(v.get("first_token_perf_ns")) is int]
        open_values = [v.get("stream_open_or_response_headers_perf_ns") for v in victims]
        if first_token_values and all(type(x) is int for x in open_values):
            earliest_first_token = min(first_token_values)
            tce_recomputed = all(x < earliest_first_token for x in open_values)
        else:
            tce_recomputed = False

        if schedule_row_override is not None:
            schedule_row = schedule_row_override
        else:
            schedule_row = swc.asdict(episode)
            if schedule_row_field_override is not None:
                field_name, value = schedule_row_field_override
                schedule_row = dict(schedule_row)
                schedule_row[field_name] = value
            if schedule_row_incomplete:
                schedule_row = dict(schedule_row)
                del schedule_row["offload_gb"]

        return {
            "status": base.REQUEST_STATUS_COMPLETE,
            "record_type": record_type_override if record_type_override is not None else swc.RECORD_TYPE_REGULAR_EPISODE,
            "validation_errors": validation_errors if validation_errors is not None else [],
            "no_client_admission_semaphore_used": no_semaphore_flag,
            "victim_requests": victims,
            "burst_requests": burst,
            "trigger": {
                "status": trigger_status, "active_cohort_size": active_cohort_size,
                "active_cohort_request_indices": reported_active_indices,
                "trigger_perf_ns": trigger_perf_ns_override if trigger_perf_ns_override is not None else TRIGGER_NS,
                "server_max_num_seqs": swc.DIAGNOSTIC_SERVER_MAX_NUM_SEQS,
                "trigger_after_decode_tokens": 16,
                "cohort_freeze_ns": max(
                    [v.get("first_token_perf_ns") for v in victims[:8] if type(v.get("first_token_perf_ns")) is int]
                    or [0]
                ),
                "metrics_quality": {"metrics_quality_status": "corroborated", "nearest_pre_trigger_sample": None},
            },
            "transport_concurrency_evidence": {
                "earliest_victim_first_token_ns": min(first_token_values),
                "victim_stream_open_count": 20,
                "all_20_streams_open_before_first_token": (
                    all_streams_open_override if all_streams_open_override is not None else tce_recomputed
                ),
                "peak_concurrent_open_completion_streams": 24 if condition == base.BURST_CONDITION else 20,
                "completion_pool_limits": {"max_connections": 32, "max_keepalive_connections": 32},
            },
            "episode_id": episode_id_override if episode_id_override is not None else episode.episode_id,
            "block_id": block_id_override if block_id_override is not None else episode.block_id,
            "schedule_row": schedule_row,
            "run_mode": run_mode_override if run_mode_override is not None else swc.RUN_MODE_DIAGNOSTIC_PAIR,
            "schedule_fingerprint": schedule_fingerprint_override if schedule_fingerprint_override is not None else bundle.fingerprint,
            "result_schema_version": result_schema_version_override if result_schema_version_override is not None else swc.RESULT_SCHEMA_VERSION,
            "runner_version": runner_version_override if runner_version_override is not None else swc.RUNNER_VERSION,
            "timing_instrumentation_name": swc.TIMING_INSTRUMENTATION_NAME,
            "timing_instrumentation_version": swc.TIMING_INSTRUMENTATION_VERSION,
            "stabilization_reference": {
                "block_id": episode.block_id,
                "path": f"/tmp/diagnostic/stabilization/{episode.block_id}.json",
                "functional_passed": True,
            },
            "server_metadata": {
                "environment_fingerprint": environment_fingerprint_override if environment_fingerprint_override is not None else classifier_env_fp,
                "model_key": episode.model_key,
                "model_full_id": episode.model_id,
                "offload_gb": episode.offload_gb,
                "server_max_num_seqs": episode.server_max_num_seqs,
                "host": "127.0.0.1",
                "port": 8000,
                "server_command": list(classifier_expected_server_command),
                "completion_pool_limits": {"max_connections": 32, "max_keepalive_connections": 32},
            },
        }

    def _classify(no_burst_result, prefill_burst_result, **overrides):
        kwargs = dict(
            no_burst_result=no_burst_result, prefill_burst_result=prefill_burst_result,
            expected_no_burst_episode=real_no_burst_ep, expected_prefill_burst_episode=real_prefill_burst_ep,
            expected_schedule_fingerprint=bundle.fingerprint, expected_environment_fingerprint=classifier_env_fp,
            expected_server_command=classifier_expected_server_command,
        )
        kwargs.update(overrides)
        return swc.classify_diagnostic_pair(**kwargs)

    active_set = set(range(8))
    good_no_burst = _synthetic_episode(episode=real_no_burst_ep, condition="no_burst", active_indices=active_set)

    # A: first burst output before the LAST active victim finishes (max active end = 5000+7*10=5070).
    a_pb = _synthetic_episode(episode=real_prefill_burst_ep, condition=base.BURST_CONDITION, active_indices=active_set, burst_first_tokens=[3000, 3001, 3002, 3003])
    a_result = _classify(good_no_burst, a_pb)
    check("classifier: A_OUTPUT_LEVEL_OVERLAP when burst output precedes the last active victim", a_result["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_A, str(a_result))

    # C: first burst output after the LAST of ALL victims finishes (max all end = 20000+19*10=20190).
    c_pb = _synthetic_episode(episode=real_prefill_burst_ep, condition=base.BURST_CONDITION, active_indices=active_set, burst_first_tokens=[25000, 25001, 25002, 25003])
    c_result = _classify(good_no_burst, c_pb)
    check("classifier: C_BURST_OUTPUT_AFTER_ALL_VICTIMS when burst output follows every victim", c_result["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_C, str(c_result))

    # B: between the last active victim's completion and the last overall victim's completion.
    b_pb = _synthetic_episode(episode=real_prefill_burst_ep, condition=base.BURST_CONDITION, active_indices=active_set, burst_first_tokens=[10000, 10001, 10002, 10003])
    b_result = _classify(good_no_burst, b_pb)
    check("classifier: B_NO_OUTPUT_LEVEL_OVERLAP_WITH_FIRST_COHORT otherwise", b_result["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_B, str(b_result))

    # D: each individual invalidity gate, one at a time.
    d_missing = _classify(None, b_pb)
    check("classifier: D when a result is missing", d_missing["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D and bool(d_missing["reasons"]))

    def _pb_with(**kwargs) -> dict:
        merged = dict(episode=real_prefill_burst_ep, condition=base.BURST_CONDITION, active_indices=active_set)
        merged.update(kwargs)
        return _synthetic_episode(**merged)

    d_trigger = _classify(good_no_burst, _pb_with(trigger_status="timeout"))
    check("classifier: D when trigger.status != 'ok'", d_trigger["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_cohort_size = _classify(good_no_burst, _pb_with(active_cohort_size=7))
    check("classifier: D when active_cohort_size != 8", d_cohort_size["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_victim_status = _classify(good_no_burst, _pb_with(break_victim_status=True))
    check("classifier: D when a victim status != 'complete'", d_victim_status["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_burst_status = _classify(good_no_burst, _pb_with(break_burst_status=True))
    check("classifier: D when a burst request status != 'complete'", d_burst_status["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_streams_open = _classify(good_no_burst, _pb_with(all_streams_open_override=False))
    check(
        "classifier: D when all_20_streams_open_before_first_token is not true",
        d_streams_open["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
    )

    d_validation_errors = _classify(good_no_burst, _pb_with(validation_errors=["synthetic"]))
    check("classifier: D when validation_errors is non-empty", d_validation_errors["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    # --- 2026-07-20 THIRD hardening pass: adversarial counterexamples from
    # both independent audits, targeting the newly independently-recomputed
    # gates specifically (never trusting a stored flag alone). -------------
    d_no_semaphore = _classify(good_no_burst, _pb_with(no_semaphore_flag=False))
    check("classifier: D when no_client_admission_semaphore_used is False", d_no_semaphore["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_no_semaphore))

    d_short_index_list = _classify(good_no_burst, _pb_with(active_cohort_indices_override=sorted(active_set)[:7]))
    check(
        "classifier: D when active_cohort_size==8 but only 7 active indices are actually listed",
        d_short_index_list["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_short_index_list),
    )

    d_dup_index = _classify(good_no_burst, _pb_with(active_cohort_indices_override=[0, 0, 1, 2, 3, 4, 5, 6]))
    check("classifier: D when the active index list contains a duplicate", d_dup_index["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_out_of_range_index = _classify(good_no_burst, _pb_with(active_cohort_indices_override=[0, 1, 2, 3, 4, 5, 6, 99]))
    check("classifier: D when the active index list contains an out-of-range value", d_out_of_range_index["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_nonzero_noncohort = _classify(good_no_burst, _pb_with(corrupt_field=(8, "decode_tokens_received_at_trigger", 1)))
    check(
        "classifier: D when a non-cohort victim has decode_tokens_received_at_trigger == 1 (nonzero)",
        d_nonzero_noncohort["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_nonzero_noncohort),
    )

    d_active_completes_early = _classify(good_no_burst, _pb_with(corrupt_field=(0, "stream_end_perf_ns", TRIGGER_NS - 10)))
    check(
        "classifier: D when an active-cohort victim completes at/before the trigger",
        d_active_completes_early["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_active_completes_early),
    )

    d_missing_token16 = _classify(good_no_burst, _pb_with(corrupt_field=(0, "token_16_perf_ns", None)))
    check(
        "classifier: D when an active-cohort victim's token_16_perf_ns is null",
        d_missing_token16["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_missing_token16),
    )

    d_null_stream_open_with_true_flag = _classify(
        good_no_burst,
        _pb_with(corrupt_field=(0, "stream_open_or_response_headers_perf_ns", None), all_streams_open_override=True),
    )
    check(
        "classifier: D when a victim stream-open timestamp is null while the aggregate flag is falsely kept true",
        d_null_stream_open_with_true_flag["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
        str(d_null_stream_open_with_true_flag),
    )

    d_missing_burst_timestamps = _classify(good_no_burst, _pb_with(corrupt_burst_field=(0, "burst_dispatch_start_perf_ns", None)))
    check(
        "classifier: D when a burst request's canonical dispatch timestamp is null",
        d_missing_burst_timestamps["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_missing_burst_timestamps),
    )

    d_burst_at_trigger = _classify(good_no_burst, _pb_with(corrupt_burst_field=(0, "burst_dispatch_start_perf_ns", TRIGGER_NS)))
    check(
        "classifier: D when a burst request's dispatch timestamp is AT (not strictly after) the trigger",
        d_burst_at_trigger["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_burst_at_trigger),
    )

    d_burst_alias_mismatch = _classify(good_no_burst, _pb_with(corrupt_burst_field=(0, "burst_dispatch_start_perf_ns", 999999)))
    check(
        "classifier: D when a burst alias field disagrees with its own canonical source field",
        d_burst_alias_mismatch["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_burst_alias_mismatch),
    )

    non_dict_victims_pb = _pb_with()
    non_dict_victims_pb["victim_requests"][5] = "not-a-dict"
    d_non_dict_victim = _classify(good_no_burst, non_dict_victims_pb)
    check("classifier: D when a victim_requests entry is not a dict", d_non_dict_victim["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    # ==========================================================================
    # 2026-07-20 fifth hardening pass: Blocker B -- the logical trigger is
    # fully reconstructed from raw fields (never trusted via trigger.status
    # or stored exposure labels alone).
    # ==========================================================================
    d_active_token16_after_trigger = _classify(good_no_burst, _pb_with(corrupt_field=(0, "token_16_perf_ns", TRIGGER_NS + 100)))
    check(
        "Blocker B: D when an active-cohort victim's token_16_perf_ns is a valid int but AFTER the trigger",
        d_active_token16_after_trigger["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_active_token16_after_trigger),
    )

    d_trigger_too_high = _classify(good_no_burst, _pb_with(trigger_perf_ns_override=TRIGGER_NS + 1))
    check(
        "Blocker B: D when trigger_perf_ns is greater than max(active token_16_perf_ns)",
        d_trigger_too_high["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_trigger_too_high),
    )

    d_trigger_too_low = _classify(good_no_burst, _pb_with(trigger_perf_ns_override=TRIGGER_NS - 1))
    check(
        "Blocker B: D when trigger_perf_ns is less than max(active token_16_perf_ns)",
        d_trigger_too_low["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_trigger_too_low),
    )

    d_noncohort_first_token_before = _classify(
        good_no_burst,
        _pb_with(corrupt_field=[(8, "first_token_perf_ns", TRIGGER_NS - 10), (8, "first_token_receive_ns", TRIGGER_NS - 10)]),
    )
    check(
        "Blocker B: D when a non-cohort victim's first_token_perf_ns is strictly BEFORE the trigger",
        d_noncohort_first_token_before["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_noncohort_first_token_before),
    )

    d_noncohort_first_token_at = _classify(
        good_no_burst,
        _pb_with(corrupt_field=[(8, "first_token_perf_ns", TRIGGER_NS), (8, "first_token_receive_ns", TRIGGER_NS)]),
    )
    check(
        "Blocker B: D when a non-cohort victim's first_token_perf_ns is exactly AT the trigger",
        d_noncohort_first_token_at["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_noncohort_first_token_at),
    )

    d_active_completion_at_trigger = _classify(good_no_burst, _pb_with(corrupt_field=(0, "stream_end_perf_ns", TRIGGER_NS)))
    check(
        "Blocker B: D when an active-cohort victim's completion is exactly AT the trigger",
        d_active_completion_at_trigger["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_active_completion_at_trigger),
    )

    # ==========================================================================
    # 2026-07-20 fifth hardening pass: Blocker C -- victim fields, aliases,
    # and exposure are fully reconstructed from raw timestamps.
    # ==========================================================================
    d_active_exposure_wrong = _classify(good_no_burst, _pb_with(corrupt_field=(0, "server_exposure_group", "dispatched_no_output_at_trigger")))
    check("Blocker C: D when an active victim's server_exposure_group is wrong", d_active_exposure_wrong["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_noncohort_exposure_wrong = _classify(good_no_burst, _pb_with(corrupt_field=(8, "server_exposure_group", "running_at_trigger_observed")))
    check("Blocker C: D when a non-cohort victim's server_exposure_group is wrong", d_noncohort_exposure_wrong["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_was_dispatched_false = _classify(good_no_burst, _pb_with(corrupt_field=(0, "was_dispatched_at_trigger", False)))
    check("Blocker C: D when was_dispatched_at_trigger is falsely set to False", d_was_dispatched_false["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_active_had_first_token_false = _classify(good_no_burst, _pb_with(corrupt_field=(0, "had_first_token_at_trigger", False)))
    check("Blocker C: D when an active victim's had_first_token_at_trigger is falsely False", d_active_had_first_token_false["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_noncohort_had_first_token_true = _classify(good_no_burst, _pb_with(corrupt_field=(8, "had_first_token_at_trigger", True)))
    check("Blocker C: D when a non-cohort victim's had_first_token_at_trigger is falsely True", d_noncohort_had_first_token_true["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_missing_request_dispatch_ns = _classify(good_no_burst, _pb_with(remove_victim_field=(0, "request_dispatch_ns")))
    check("Blocker C: D when request_dispatch_ns is missing", d_missing_request_dispatch_ns["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_missing_first_token_receive_ns = _classify(good_no_burst, _pb_with(remove_victim_field=(0, "first_token_receive_ns")))
    check("Blocker C: D when first_token_receive_ns is missing", d_missing_first_token_receive_ns["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_missing_stream_end_ns = _classify(good_no_burst, _pb_with(remove_victim_field=(0, "stream_end_ns")))
    check("Blocker C: D when stream_end_ns is missing", d_missing_stream_end_ns["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_missing_task_created_perf_ns = _classify(good_no_burst, _pb_with(remove_victim_field=(0, "task_created_perf_ns")))
    check("Blocker C: D when task_created_perf_ns is missing", d_missing_task_created_perf_ns["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    # Every OTHER required victim field, removed one at a time, must also
    # independently force D (per the requirement to test every field).
    for _field_to_remove in swc._REQUIRED_VICTIM_FIELDS:
        _d_missing_field = _classify(good_no_burst, _pb_with(remove_victim_field=(0, _field_to_remove)))
        check(
            f"Blocker C: D when victim field {_field_to_remove!r} is missing",
            _d_missing_field["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(_d_missing_field),
        )

    d_alias_task_created_contradicts = _classify(good_no_burst, _pb_with(corrupt_field=(0, "task_created_perf_ns", 999999)))
    check("Blocker C: D when the task_created alias contradicts its canonical source", d_alias_task_created_contradicts["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_alias_dispatch_contradicts = _classify(good_no_burst, _pb_with(corrupt_field=(0, "http_dispatch_start_perf_ns", 999999)))
    check("Blocker C: D when the dispatch alias contradicts its canonical source", d_alias_dispatch_contradicts["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_alias_first_token_contradicts = _classify(good_no_burst, _pb_with(corrupt_field=(0, "first_token_perf_ns", 999999)))
    check("Blocker C: D when the first-token alias contradicts its canonical source", d_alias_first_token_contradicts["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_alias_stream_end_contradicts = _classify(good_no_burst, _pb_with(corrupt_field=(0, "stream_end_perf_ns", 999999)))
    check("Blocker C: D when the stream-end alias contradicts its canonical source", d_alias_stream_end_contradicts["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_dispatch_to_first_token_contradicts = _classify(good_no_burst, _pb_with(corrupt_field=(0, "dispatch_to_first_token_ms", 999999.0)))
    check("Blocker C: D when dispatch_to_first_token_ms contradicts its reconstruction", d_dispatch_to_first_token_contradicts["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_stored_mono_errors_nonempty = _classify(good_no_burst, _pb_with(corrupt_field=(0, "timestamp_monotonicity_errors", ["synthetic mono error"])))
    check("Blocker C: D when the stored timestamp_monotonicity_errors is non-empty", d_stored_mono_errors_nonempty["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    # Independently reconstructed monotonicity violation: the RAW
    # timestamps themselves are genuinely out of order (dispatch AFTER
    # first-token), even though nothing else about the record complains.
    d_real_mono_violation = _classify(
        good_no_burst,
        _pb_with(corrupt_field=[
            (0, "request_dispatch_ns", 100000), (0, "http_dispatch_start_perf_ns", 100000),
        ]),
    )
    check(
        "Blocker C: D when the raw timestamps are genuinely out of order (independently reconstructed monotonicity)",
        d_real_mono_violation["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_real_mono_violation),
    )

    d_record_type_wrong = _classify(good_no_burst, _pb_with(record_type_override="something_else"))
    check("Blocker C: D when record_type is wrong", d_record_type_wrong["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    # Final coding pass: bind per-request and actual server metadata.
    d_victim_maxseq_wrong = _classify(
        good_no_burst, _pb_with(corrupt_field=(0, "server_max_num_seqs", 4)),
    )
    check(
        "final pass: D when a victim record carries server_max_num_seqs != 8",
        d_victim_maxseq_wrong["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
        str(d_victim_maxseq_wrong),
    )

    d_burst_stored_mono = _classify(
        good_no_burst,
        _pb_with(corrupt_burst_field=(0, "timestamp_monotonicity_errors", ["synthetic"])),
    )
    check(
        "final pass: D when a burst record has non-empty timestamp_monotonicity_errors",
        d_burst_stored_mono["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
        str(d_burst_stored_mono),
    )
    d_burst_missing_mono_obj = copy.deepcopy(a_pb)
    del d_burst_missing_mono_obj["burst_requests"][0]["timestamp_monotonicity_errors"]
    d_burst_missing_mono = _classify(good_no_burst, d_burst_missing_mono_obj)
    check(
        "final pass: D when a burst record is missing timestamp_monotonicity_errors",
        d_burst_missing_mono["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
        str(d_burst_missing_mono),
    )

    server_metadata_mutations = {
        "model_key": "llama",
        "model_full_id": "wrong/model",
        "offload_gb": 0,
        "server_max_num_seqs": 4,
        "host": "127.0.0.2",
        "port": 9999,
        "server_command": ["wrong"],
    }
    for metadata_field, bad_value in server_metadata_mutations.items():
        mutated_pb = copy.deepcopy(a_pb)
        mutated_pb["server_metadata"][metadata_field] = bad_value
        classified = _classify(good_no_burst, mutated_pb)
        check(
            f"final pass: D when server_metadata.{metadata_field} is inconsistent with the real diagnostic launch",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

    # Positive fixtures must STILL classify correctly after ALL Blocker
    # B/C reconstruction checks were added.
    check("classifier: positive A fixture still classifies correctly after Blocker B/C reconstruction", a_result["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_A)
    check("classifier: positive B fixture still classifies correctly after Blocker B/C reconstruction", b_result["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_B)
    check("classifier: positive C fixture still classifies correctly after Blocker B/C reconstruction", c_result["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_C)

    # ==========================================================================
    # Section 6 requirement: the classifier must collect ALL detected
    # reasons, not just the first one found, and must never raise an
    # exception on manipulated fields (always a controlled D).
    # ==========================================================================
    d_multi_mutation = _classify(
        good_no_burst,
        _pb_with(
            episode_id_override="wrong_episode_id",  # identity error
            corrupt_field=[  # victim/trigger errors, several independent ones at once
                (0, "server_exposure_group", "dispatched_no_output_at_trigger"),
                (8, "decode_tokens_received_at_trigger", 1),
                (1, "task_created_perf_ns", 999999),
            ],
            corrupt_burst_field=(0, "burst_dispatch_start_perf_ns", TRIGGER_NS),  # burst error
            validation_errors=["synthetic top-level error"],  # episode-level error
        ),
    )
    check(
        "Section 6: a result with 5 independent, simultaneous mutations still classifies as D (no exception)",
        d_multi_mutation["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_multi_mutation),
    )
    _multi_reasons_text = " | ".join(d_multi_mutation.get("reasons", []))
    check(
        "Section 6: the identity error (wrong episode_id) is present among the collected reasons",
        any("episode_id" in r for r in d_multi_mutation.get("reasons", [])), _multi_reasons_text,
    )
    check(
        "Section 6: the exposure-group error is present among the collected reasons",
        any("server_exposure_group" in r for r in d_multi_mutation.get("reasons", [])), _multi_reasons_text,
    )
    check(
        "Section 6: the non-cohort decode-token-count error is present among the collected reasons",
        any("decode_tokens_received_at_trigger" in r for r in d_multi_mutation.get("reasons", [])), _multi_reasons_text,
    )
    check(
        "Section 6: the burst-dispatch-at-trigger error is present among the collected reasons",
        any("burst" in r and "trigger" in r for r in d_multi_mutation.get("reasons", [])), _multi_reasons_text,
    )
    check(
        "Section 6: the top-level validation_errors error is present among the collected reasons",
        any("validation_errors" in r for r in d_multi_mutation.get("reasons", [])), _multi_reasons_text,
    )
    check(
        "Section 6: more than one reason was actually collected (not stopped at the first)",
        len(d_multi_mutation.get("reasons", [])) >= 4, str(len(d_multi_mutation.get("reasons", []))),
    )

    # ========================================================================
    # 2026-07-21 sixth hardening pass: strict damaged-JSON type handling.
    # Every malformed value below must produce a controlled D result --
    # never A/B/C and never an exception. These regressions cover the
    # independent KI-3 audit findings plus adjacent Python equality traps.
    # ========================================================================
    for bad_validation_errors in (None, {}, "", (), {"bad"}):
        malformed = copy.deepcopy(a_pb)
        malformed["validation_errors"] = bad_validation_errors
        classified = _classify(good_no_burst, malformed)
        check(
            f"strict JSON: D when validation_errors is {bad_validation_errors!r} instead of exactly []",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
        )
    missing_validation_errors = copy.deepcopy(a_pb)
    del missing_validation_errors["validation_errors"]
    classified_missing_validation = _classify(good_no_burst, missing_validation_errors)
    check(
        "strict JSON: D when validation_errors is missing",
        classified_missing_validation["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
        str(classified_missing_validation),
    )

    for bad_dispatch in (None, "bad"):
        malformed = copy.deepcopy(a_pb)
        malformed["victim_requests"][0]["request_dispatch_ns"] = bad_dispatch
        classified = _classify(good_no_burst, malformed)
        check(
            f"strict JSON: non-int request_dispatch_ns={bad_dispatch!r} yields D rather than an exception",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
        )

    malformed_victim_unhashable = copy.deepcopy(a_pb)
    malformed_victim_unhashable["victim_requests"][0]["request_index"] = []
    classified = _classify(good_no_burst, malformed_victim_unhashable)
    check(
        "strict JSON: unhashable victim request_index=[] yields D rather than an exception",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
    )

    malformed_burst_unhashable = copy.deepcopy(a_pb)
    malformed_burst_unhashable["burst_requests"][0]["request_index"] = {}
    classified = _classify(good_no_burst, malformed_burst_unhashable)
    check(
        "strict JSON: unhashable burst request_index={} yields D rather than an exception",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
    )

    for collection_name in ("victim_requests", "burst_requests"):
        malformed = copy.deepcopy(a_pb)
        malformed[collection_name][0]["request_index"] = 0.0
        classified = _classify(good_no_burst, malformed)
        check(
            f"strict JSON: {collection_name} request_index=0.0 is rejected despite 0.0 == 0",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
        )

    malformed_tce = copy.deepcopy(a_pb)
    malformed_tce["transport_concurrency_evidence"] = "bad"
    classified = _classify(good_no_burst, malformed_tce)
    check(
        "strict JSON: non-dict transport_concurrency_evidence yields D rather than AttributeError",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
    )

    malformed_k_float = copy.deepcopy(a_pb)
    malformed_k_float["victim_requests"][0]["server_max_num_seqs"] = 8.0
    classified = _classify(good_no_burst, malformed_k_float)
    check(
        "strict JSON: victim server_max_num_seqs=8.0 is rejected despite 8.0 == 8",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
    )

    malformed_active_size_float = copy.deepcopy(a_pb)
    malformed_active_size_float["trigger"]["active_cohort_size"] = 8.0
    classified = _classify(good_no_burst, malformed_active_size_float)
    check(
        "strict JSON: trigger.active_cohort_size=8.0 is rejected despite 8.0 == 8",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
    )

    malformed_noncohort_count_float = copy.deepcopy(a_pb)
    malformed_noncohort_count_float["victim_requests"][8]["decode_tokens_received_at_trigger"] = 0.0
    classified = _classify(good_no_burst, malformed_noncohort_count_float)
    check(
        "strict JSON: non-cohort decode_tokens_received_at_trigger=0.0 is rejected",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
    )

    malformed_tce_bool_int = copy.deepcopy(a_pb)
    malformed_tce_bool_int["transport_concurrency_evidence"]["all_20_streams_open_before_first_token"] = 1
    classified = _classify(good_no_burst, malformed_tce_bool_int)
    check(
        "strict JSON: stored all_20_streams_open_before_first_token=1 is rejected as non-bool",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
    )

    malformed_schedule_numeric_type = copy.deepcopy(a_pb)
    malformed_schedule_numeric_type["schedule_row"]["offload_gb"] = 12.0
    classified = _classify(good_no_burst, malformed_schedule_numeric_type)
    check(
        "strict JSON: schedule_row.offload_gb=12.0 is rejected despite numeric equality",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
    )

    malformed_schema_float = copy.deepcopy(a_pb)
    malformed_schema_float["result_schema_version"] = float(swc.RESULT_SCHEMA_VERSION)
    classified = _classify(good_no_burst, malformed_schema_float)
    check(
        "strict JSON: float result_schema_version is rejected despite numeric equality",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
    )

    for metadata_field, bad_value in (
        ("offload_gb", 12.0), ("server_max_num_seqs", 8.0), ("port", 8000.0),
    ):
        malformed = copy.deepcopy(a_pb)
        malformed["server_metadata"][metadata_field] = bad_value
        classified = _classify(good_no_burst, malformed)
        check(
            f"strict JSON: server_metadata.{metadata_field}={bad_value!r} is rejected as the wrong type",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(classified),
        )

    # --- Blocker 2 (2026-07-20 fourth hardening pass): identity-binding
    # adversarial tests -- a syntactically/timing-wise perfect result that
    # is nonetheless bound to the WRONG episode/run/fingerprint identity
    # must never be classified A, B, or C. -----------------------------------
    d_wrong_episode_id = _classify(good_no_burst, _pb_with(episode_id_override="wrong_episode_id"))
    check("identity: D when episode_id is wrong", d_wrong_episode_id["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D, str(d_wrong_episode_id))

    d_wrong_block_id = _classify(good_no_burst, _pb_with(block_id_override="qwen_off0_k4_rep02"))
    check("identity: D when block_id is wrong", d_wrong_block_id["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_wrong_offload = _classify(good_no_burst, _pb_with(schedule_row_field_override=("offload_gb", 0)))
    check("identity: D when schedule_row.offload_gb is changed", d_wrong_offload["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_wrong_server_max = _classify(good_no_burst, _pb_with(schedule_row_field_override=("server_max_num_seqs", 4)))
    check("identity: D when schedule_row.server_max_num_seqs is changed", d_wrong_server_max["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_wrong_repeat = _classify(good_no_burst, _pb_with(schedule_row_field_override=("repeat", 2)))
    check("identity: D when schedule_row.repeat is changed", d_wrong_repeat["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_swapped_condition = _classify(good_no_burst, _pb_with(schedule_row_field_override=("condition", "no_burst")))
    check("identity: D when condition is swapped (prefill_burst file claims no_burst)", d_swapped_condition["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_wrong_run_mode = _classify(good_no_burst, _pb_with(run_mode_override="smoke"))
    check("identity: D when run_mode is wrong", d_wrong_run_mode["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_wrong_schedule_fp = _classify(good_no_burst, _pb_with(schedule_fingerprint_override="sha256:" + "f" * 64))
    check("identity: D when schedule_fingerprint is wrong", d_wrong_schedule_fp["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_wrong_env_fp = _classify(good_no_burst, _pb_with(environment_fingerprint_override="sha256:" + "e" * 64))
    check("identity: D when environment_fingerprint is wrong", d_wrong_env_fp["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_incomplete_schedule_row = _classify(good_no_burst, _pb_with(schedule_row_incomplete=True))
    check("identity: D when schedule_row is incomplete (missing a field)", d_incomplete_schedule_row["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_wrong_schema_version = _classify(good_no_burst, _pb_with(result_schema_version_override=999))
    check("identity: D when result_schema_version is wrong", d_wrong_schema_version["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_wrong_runner_version = _classify(good_no_burst, _pb_with(runner_version_override="wrong-runner-v0"))
    check("identity: D when runner_version is wrong", d_wrong_runner_version["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    # Also verify the no_burst side is independently bound (not just prefill_burst).
    d_wrong_no_burst_episode_id = _classify(
        _synthetic_episode(episode=real_no_burst_ep, condition="no_burst", active_indices=active_set, episode_id_override="wrong"),
        a_pb,
    )
    check("identity: D when the no_burst file's episode_id is wrong", d_wrong_no_burst_episode_id["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    d_wrong_no_burst_schedule_fp = _classify(
        _synthetic_episode(episode=real_no_burst_ep, condition="no_burst", active_indices=active_set, schedule_fingerprint_override="sha256:" + "0" * 64),
        a_pb,
    )
    check("identity: D when the no_burst file's schedule_fingerprint is wrong", d_wrong_no_burst_schedule_fp["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    # Mismatched fingerprints BETWEEN the two files (both individually
    # "plausible" but disagreeing with each other) must also be D, since
    # both are checked against the SAME expected_schedule_fingerprint.
    d_mismatched_pair_fp = _classify(
        _synthetic_episode(episode=real_no_burst_ep, condition="no_burst", active_indices=active_set, schedule_fingerprint_override="sha256:" + "9" * 64),
        a_pb,
    )
    check("identity: D when the two files' schedule_fingerprints disagree with each other (and thus with the expected one)", d_mismatched_pair_fp["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D)

    # Positive fixtures must STILL classify correctly after all these
    # identity checks were added (never a false D on well-formed input).
    check("classifier: positive A fixture still classifies correctly after identity binding", a_result["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_A)
    check("classifier: positive B fixture still classifies correctly after identity binding", b_result["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_B)
    check("classifier: positive C fixture still classifies correctly after identity binding", c_result["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_C)

    # Paired effect summary sanity on the good A-classified pair.
    effect = swc.compute_paired_effect_summary(no_burst_result=good_no_burst, prefill_burst_result=a_pb)
    check("compute_paired_effect_summary marks itself available for a valid pair", effect.get("available") is True)
    check(
        "compute_paired_effect_summary never attaches an inferential CI field",
        "confidence_interval" not in effect and "ci" not in effect,
    )
    unavailable_effect = swc.compute_paired_effect_summary(no_burst_result=None, prefill_burst_result=a_pb)
    check("compute_paired_effect_summary reports unavailable when a result is missing", unavailable_effect.get("available") is False)

    # ====================================================================
    # Final functional-validity / exception-free D-path hardening
    # ====================================================================
    def _deepcopy(obj):
        return json.loads(json.dumps(obj))

    request_mutations = [
        ("validation_errors", ["bad"]),
        ("http_status", 500),
        ("timed_out", True),
        ("cancelled", True),
        ("done_received", False),
        ("error_type", "synthetic"),
        ("error_message", "synthetic"),
        ("finish_reason", "error"),
        ("output_token_ids", []),
        ("usage", {"prompt_tokens": 1, "completion_tokens": 1}),
    ]
    for field, value in request_mutations:
        mutated = _deepcopy(a_pb)
        mutated["victim_requests"][0][field] = value
        classified = _classify(good_no_burst, mutated)
        check(
            f"request validity: victim {field} mutation yields D",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

        mutated = _deepcopy(a_pb)
        mutated["burst_requests"][0][field] = value
        classified = _classify(good_no_burst, mutated)
        check(
            f"request validity: burst {field} mutation yields D",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

    workload_identity_mutations = [
        ("role", "burst"),
        ("request_id", "wrong"),
        ("prompt_seed", 999),
        ("generation_seed", 999),
        ("prompt_sha256", "0" * 64),
        ("prompt_token_ids_sent", []),
        ("prompt_token_ids_returned", []),
    ]
    for field, value in workload_identity_mutations:
        mutated = _deepcopy(a_pb)
        mutated["victim_requests"][0][field] = value
        classified = _classify(good_no_burst, mutated)
        check(
            f"workload identity: victim {field} mutation yields D",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

    for field, value in (("role", "victim"), ("request_id", "wrong")):
        mutated = _deepcopy(a_pb)
        mutated["burst_requests"][0][field] = value
        classified = _classify(good_no_burst, mutated)
        check(
            f"workload identity: burst {field} mutation yields D",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

    impossible_effect = _deepcopy(a_pb)
    for index in impossible_effect["trigger"]["active_cohort_request_indices"]:
        record = impossible_effect["victim_requests"][index]
        record["client_observed_tpot_ms"] = 999999.0
        record["itl_ms"] = [999999.0] * 63
    impossible_effect_classification = _classify(good_no_burst, impossible_effect)
    impossible_effect_direct = swc.compute_paired_effect_summary(
        no_burst_result=good_no_burst, prefill_burst_result=impossible_effect,
    )
    check(
        "effect consistency: internally impossible TPOT/ITL values force D in the main classifier",
        impossible_effect_classification["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
        str(impossible_effect_classification),
    )
    check(
        "effect consistency: direct effect summary reports unavailable for impossible TPOT/ITL values",
        impossible_effect_direct.get("available") is False and bool(impossible_effect_direct.get("reasons")),
        str(impossible_effect_direct),
    )

    for mutation_name, mutator in (
        ("itl_available false", lambda r: r.__setitem__("itl_available", False)),
        ("ITL length wrong", lambda r: r.__setitem__("itl_ms", r["itl_ms"][:-1])),
        ("last-token timestamp wrong", lambda r: r.__setitem__("last_token_receive_ns", r["last_token_receive_ns"] + 1_000_000)),
    ):
        mutated = _deepcopy(a_pb)
        mutator(mutated["victim_requests"][0])
        classified = _classify(good_no_burst, mutated)
        check(
            f"effect consistency: {mutation_name} yields D",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

    # Round-6 strict-JSON remainder: raw burst aliases must themselves be
    # strict ints; numerically equal floats are not acceptable JSON identity.
    for field in ("request_dispatch_ns", "first_token_receive_ns", "stream_end_ns"):
        mutated = _deepcopy(a_pb)
        mutated["burst_requests"][0][field] = float(mutated["burst_requests"][0][field])
        classified = _classify(good_no_burst, mutated)
        check(
            f"strict JSON: burst raw {field} as numerically equal float yields D",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

    trigger_identity_mutations = [
        ("server_max_num_seqs", 4),
        ("trigger_after_decode_tokens", 1),
        ("cohort_freeze_ns", -1),
    ]
    for field, value in trigger_identity_mutations:
        mutated = _deepcopy(a_pb)
        mutated["trigger"][field] = value
        classified = _classify(good_no_burst, mutated)
        check(
            f"trigger identity: {field} mutation yields D",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

    mutated = _deepcopy(a_pb)
    mutated["trigger"]["cohort_freeze_ns"] -= 1
    classified = _classify(good_no_burst, mutated)
    check(
        "trigger identity: positive but non-reconstructed cohort_freeze_ns yields D",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
        str(classified),
    )

    for metrics_value in (None, {}, {"metrics_quality_status": "wrong"}):
        mutated = _deepcopy(a_pb)
        mutated["trigger"]["metrics_quality"] = metrics_value
        classified = _classify(good_no_burst, mutated)
        check(
            f"metrics quality: invalid object {metrics_value!r} yields D",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

    mutated = _deepcopy(a_pb)
    mutated["victim_requests"][0]["decode_tokens_received_at_trigger"] = 999
    classified = _classify(good_no_burst, mutated)
    check(
        "trigger evidence: active decode_tokens_received_at_trigger above output length yields D",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
        str(classified),
    )

    mutated = _deepcopy(good_no_burst)
    mutated["transport_concurrency_evidence"]["peak_concurrent_open_completion_streams"] = 21
    classified = _classify(mutated, a_pb)
    check(
        "transport evidence: no_burst peak above exactly 20 yields D",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
        str(classified),
    )

    mutated = _deepcopy(a_pb)
    mutated["transport_concurrency_evidence"]["peak_concurrent_open_completion_streams"] = 1000
    classified = _classify(good_no_burst, mutated)
    check(
        "transport evidence: physically impossible prefill_burst peak yields D",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
        str(classified),
    )

    instrumentation_mutations = [
        ("timing_instrumentation_name", "wrong"),
        ("timing_instrumentation_version", 999),
    ]
    for field, value in instrumentation_mutations:
        mutated = _deepcopy(a_pb)
        mutated[field] = value
        classified = _classify(good_no_burst, mutated)
        check(
            f"instrumentation identity: {field} mutation yields D",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

    stabilization_mutations = [
        ("functional_passed", False),
        ("block_id", "wrong"),
        ("path", "/tmp/wrong.json"),
    ]
    for field, value in stabilization_mutations:
        mutated = _deepcopy(a_pb)
        mutated["stabilization_reference"][field] = value
        classified = _classify(good_no_burst, mutated)
        check(
            f"stabilization identity: {field} mutation yields D",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

    tce_mutations = [
        ("victim_stream_open_count", 1),
        ("earliest_victim_first_token_ns", -1),
        ("peak_concurrent_open_completion_streams", 1),
        ("completion_pool_limits", {"max_connections": 1, "max_keepalive_connections": 1}),
    ]
    for field, value in tce_mutations:
        mutated = _deepcopy(a_pb)
        mutated["transport_concurrency_evidence"][field] = value
        classified = _classify(good_no_burst, mutated)
        check(
            f"transport evidence: {field} mutation yields D",
            classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
            str(classified),
        )

    mutated = _deepcopy(a_pb)
    mutated["server_metadata"]["completion_pool_limits"] = {
        "max_connections": 1, "max_keepalive_connections": 1,
    }
    classified = _classify(good_no_burst, mutated)
    check(
        "server metadata: completion_pool_limits mutation yields D",
        classified["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D,
        str(classified),
    )

    malformed_effect = _deepcopy(a_pb)
    malformed_effect["victim_requests"][0]["itl_ms"] = [1.0, "bad"]
    effect_bad_itl = swc.compute_paired_effect_summary(
        no_burst_result=good_no_burst, prefill_burst_result=malformed_effect,
    )
    check(
        "effect summary: malformed itl_ms returns available=False without raising",
        effect_bad_itl.get("available") is False and bool(effect_bad_itl.get("reasons")),
        str(effect_bad_itl),
    )

    malformed_effect = _deepcopy(a_pb)
    malformed_effect["trigger"]["active_cohort_request_indices"] = [[], [], [], [], [], [], [], []]
    effect_bad_indices = swc.compute_paired_effect_summary(
        no_burst_result=good_no_burst, prefill_burst_result=malformed_effect,
    )
    check(
        "effect summary: unhashable active indices return available=False without raising",
        effect_bad_indices.get("available") is False and bool(effect_bad_indices.get("reasons")),
        str(effect_bad_indices),
    )

    malformed_effect = _deepcopy(a_pb)
    malformed_effect["victim_requests"][0]["client_observed_tpot_ms"] = "bad"
    effect_bad_tpot = swc.compute_paired_effect_summary(
        no_burst_result=good_no_burst, prefill_burst_result=malformed_effect,
    )
    check(
        "effect summary: malformed TPOT returns available=False without raising",
        effect_bad_tpot.get("available") is False and bool(effect_bad_tpot.get("reasons")),
        str(effect_bad_tpot),
    )

    # Full run-level D-path regression: once classification is D, the
    # effect-summary function must not be called at all.
    original_classifier = swc.classify_diagnostic_pair
    original_effect_summary = swc.compute_paired_effect_summary
    try:
        swc.classify_diagnostic_pair = lambda **_kwargs: {
            "classification": swc.DIAGNOSTIC_CLASSIFICATION_D,
            "reasons": ["synthetic D-path regression"],
        }

        def _must_not_run(**_kwargs):
            raise AssertionError("effect summary was called for a D-classified pair")

        swc.compute_paired_effect_summary = _must_not_run
        with tempfile.TemporaryDirectory() as td:
            d_path_clock = base.FakeClock()
            d_path_summary = asyncio.run(swc.run_diagnostic_pair(
                bundle=bundle, output_dir=Path(td) / "d_path", host="127.0.0.1", port=38001,
                api_key="fake-key",
                transport=_build_fake_transport(bundle, "qwen_off12_k8_rep01", k=8, clock=d_path_clock),
                metrics_transport=_build_fake_metrics_transport(k=8),
                tokenizer=base.FakeTokenizerAdapter(), server_adapter=base.FakeServerProcessAdapter(),
                sleeper=base.FakeSleeper(), clock=d_path_clock, run_server_path=server_path,
                env=fake_diag_env,
            ))
        check(
            "full D path: run_diagnostic_pair skips effect computation and returns unavailable summary",
            d_path_summary["classification"]["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D
            and d_path_summary["paired_effect_summary"].get("available") is False,
            str(d_path_summary.get("paired_effect_summary")),
        )
    finally:
        swc.classify_diagnostic_pair = original_classifier
        swc.compute_paired_effect_summary = original_effect_summary

    # Full-path semantic mutation probes: mutate the files after the block
    # protocol has written them but before classification/summary/final
    # manifest.  Each mutation must make the returned pair D and invalid.
    async def _run_fullpath_mutation(name: str, mutator, port: int):
        original_protocol = swc._run_server_waiting_block_protocol

        async def _wrapped_protocol(**kwargs):
            result = await original_protocol(**kwargs)
            mutator(kwargs["output_dir"])
            return result

        swc._run_server_waiting_block_protocol = _wrapped_protocol
        try:
            with tempfile.TemporaryDirectory() as td:
                clock = base.FakeClock()
                summary = await swc.run_diagnostic_pair(
                    bundle=bundle, output_dir=Path(td) / name, host="127.0.0.1", port=port,
                    api_key="fake-key",
                    transport=_build_fake_transport(bundle, "qwen_off12_k8_rep01", k=8, clock=clock),
                    metrics_transport=_build_fake_metrics_transport(k=8),
                    tokenizer=base.FakeTokenizerAdapter(), server_adapter=base.FakeServerProcessAdapter(),
                    sleeper=base.FakeSleeper(), clock=clock, run_server_path=server_path,
                    env=fake_diag_env,
                )
        finally:
            swc._run_server_waiting_block_protocol = original_protocol
        return summary

    def _mutate_prefill_episode(output_dir: Path, mutate_record) -> None:
        path = output_dir / "episodes" / "qwen_off12_k8_trigger16_prefill_burst_rep01.json"
        obj = json.loads(path.read_text(encoding="utf-8"))
        mutate_record(obj)
        swc.write_json_atomic(path, obj)

    identity_summary = asyncio.run(_run_fullpath_mutation(
        "identity_mutation",
        lambda output_dir: _mutate_prefill_episode(
            output_dir, lambda obj: obj["victim_requests"][0].__setitem__("role", "burst")
        ),
        38011,
    ))
    check(
        "full path: wrong stored workload identity forces D before final attestation",
        identity_summary["classification"]["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D
        and identity_summary.get("diagnostic_valid") is False,
        str(identity_summary.get("classification")),
    )

    def _corrupt_effect(obj):
        for index in obj["trigger"]["active_cohort_request_indices"]:
            obj["victim_requests"][index]["client_observed_tpot_ms"] = 999999.0
            obj["victim_requests"][index]["itl_ms"] = [999999.0] * 63

    effect_summary_fullpath = asyncio.run(_run_fullpath_mutation(
        "effect_mutation",
        lambda output_dir: _mutate_prefill_episode(output_dir, _corrupt_effect),
        38012,
    ))
    check(
        "full path: internally impossible effect values force D and unavailable effect summary",
        effect_summary_fullpath["classification"]["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D
        and effect_summary_fullpath["paired_effect_summary"].get("available") is False,
        str(effect_summary_fullpath),
    )

    def _mutate_stabilization(output_dir: Path) -> None:
        path = output_dir / "stabilization" / f"{swc.DIAGNOSTIC_BLOCK_ID}.json"
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj["functional_passed"] = False
        obj["stabilization_passed"] = False
        obj["status"] = "failed"
        swc.write_json_atomic(path, obj)

    stabilization_summary = asyncio.run(_run_fullpath_mutation(
        "stabilization_mutation", _mutate_stabilization, 38013,
    ))
    check(
        "full path: semantically failed stabilization artifact forces D",
        stabilization_summary["classification"]["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D
        and stabilization_summary.get("diagnostic_valid") is False,
        str(stabilization_summary.get("classification")),
    )

    def _mutate_run_manifest(output_dir: Path) -> None:
        path = output_dir / swc.DIAGNOSTIC_RUN_MANIFEST_FILENAME
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj["environment_fingerprint"] = "sha256:" + "0" * 64
        obj["host"] = "wrong"
        obj["port"] = 1
        swc.write_json_atomic(path, obj)

    manifest_summary = asyncio.run(_run_fullpath_mutation(
        "manifest_mutation", _mutate_run_manifest, 38014,
    ))
    check(
        "full path: semantically inconsistent diagnostic_run_manifest forces D",
        manifest_summary["classification"]["classification"] == swc.DIAGNOSTIC_CLASSIFICATION_D
        and manifest_summary.get("diagnostic_valid") is False,
        str(manifest_summary.get("classification")),
    )

    # ========================================================================
    # 2026-07-20 hardening pass: burst timing aliases + monotonicity
    # ========================================================================
    burst_record = {"request_index": 0, "request_dispatch_ns": 100, "first_token_receive_ns": 200, "stream_end_ns": 300}
    burst_list = [burst_record]
    mono_errors = swc._enrich_burst_transport_and_timing_fields(burst_list, transport=base.FakeTransport(), episode_id="unit_ep")
    check(
        "burst timing aliases exactly mirror their source timestamps",
        burst_record["burst_dispatch_start_perf_ns"] == 100
        and burst_record["burst_first_token_perf_ns"] == 200
        and burst_record["burst_end_perf_ns"] == 300,
        str(burst_record),
    )
    check("well-formed burst timestamps produce zero monotonicity errors", mono_errors == [])

    bad_burst_record = {"request_index": 0, "request_dispatch_ns": 500, "first_token_receive_ns": 200, "stream_end_ns": 300}
    bad_mono_errors = swc._enrich_burst_transport_and_timing_fields([bad_burst_record], transport=base.FakeTransport(), episode_id="unit_ep")
    check(
        "an out-of-order burst timestamp (dispatch after first-token) is caught as a monotonicity violation",
        bool(bad_mono_errors) and bool(bad_burst_record["timestamp_monotonicity_errors"]),
        str(bad_mono_errors),
    )

    # ========================================================================
    # 2026-07-20 third hardening pass: B3 -- true open-stream peak must stay
    # at zero while requests are blocked BEFORE response headers arrive.
    # ========================================================================
    async def _blocked_before_headers_probe():
        inner = base.FakeTransport()
        for i in range(3):
            inner.queue_script(
                f"blocked:{i}",
                base.FakeStreamScript(prompt_token_ids_echo=[1], token_events=[], hang=True),
            )
        ct = swc.FakeCompletionTransport(inner, clock=base.RealClock())

        async def _consume(req_id):
            async for _ in ct.stream_completion("http://x/v1/completions", {}, {"prompt": [1]}, 5.0, request_id=req_id):
                pass

        tasks = [asyncio.create_task(_consume(f"blocked:{i}")) for i in range(3)]
        await asyncio.sleep(0.05)
        peak_open = ct.peak_open_stream_count
        peak_inflight = ct.peak_inflight_completion_attempts
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        return peak_open, peak_inflight

    blocked_peak_open, blocked_peak_inflight = asyncio.run(_blocked_before_headers_probe())
    check(
        "B3: true open-stream peak stays 0 while 3 requests are blocked before response headers",
        blocked_peak_open == 0, str(blocked_peak_open),
    )
    check(
        "B3: peak_inflight_completion_attempts correctly reflects the 3 blocked dispatch attempts",
        blocked_peak_inflight == 3, str(blocked_peak_inflight),
    )
    check(
        "B3: get_diagnostics() exposes both the true open-stream peak and the in-flight-attempts peak under distinct, honestly-named keys",
        set(swc.FakeCompletionTransport().get_diagnostics().keys())
        >= {"peak_open_stream_count", "peak_inflight_completion_attempts"},
    )

    # ========================================================================
    # 2026-07-20 third hardening pass: N1 -- partial transport-startup
    # failure must close whichever transport(s) actually started.
    # ========================================================================
    class _FailingStartTransport:
        def __init__(self, fail: bool) -> None:
            self.fail = fail
            self.started = False
            self.closed = False

        async def start(self) -> None:
            if self.fail:
                raise RuntimeError("synthetic start() failure")
            self.started = True

        async def aclose(self) -> None:
            self.closed = True

    async def _n1_probe(completion_fails: bool, metrics_fails: bool):
        completion = _FailingStartTransport(fail=completion_fails)
        metrics = _FailingStartTransport(fail=metrics_fails)
        raised = None
        try:
            async def _factory():
                return "ok"
            await swc.run_with_transactional_transports(completion, metrics, _factory)
        except RuntimeError as exc:
            raised = exc
        return completion, metrics, raised

    completion_a, metrics_a, raised_a = asyncio.run(_n1_probe(completion_fails=False, metrics_fails=True))
    check(
        "N1: completion transport that already started IS closed even when metrics_transport.start() then fails",
        completion_a.started and completion_a.closed, str((completion_a.started, completion_a.closed)),
    )
    check("N1: metrics transport that never successfully started is not closed", not metrics_a.closed)
    check("N1: the original start() failure still propagates to the caller", raised_a is not None and "synthetic start() failure" in str(raised_a))

    completion_b, metrics_b, raised_b = asyncio.run(_n1_probe(completion_fails=True, metrics_fails=False))
    check("N1: if the completion transport itself fails to start, neither transport is closed (metrics never started)", not completion_b.closed and not metrics_b.closed)
    check("N1: metrics_transport.start() is never even attempted if completion transport failed first", not metrics_b.started)

    completion_c, metrics_c, raised_c = asyncio.run(_n1_probe(completion_fails=False, metrics_fails=False))
    check("N1: happy path -- both transports start and both close", completion_c.started and completion_c.closed and metrics_c.started and metrics_c.closed)
    check("N1: happy path raises nothing", raised_c is None)

    # ========================================================================
    # 2026-07-20 fourth hardening pass: Blocker 3 -- error-independent
    # transport cleanup. A failure in EITHER transport's aclose() must
    # never suppress the other transport's close attempt, and both
    # errors must be preserved/reported, never silently swallowed.
    # ========================================================================
    class _FailingCloseTransport:
        def __init__(self, fail_close: bool) -> None:
            self.fail_close = fail_close
            self.started = False
            self.close_attempted = False
            self.closed = False

        async def start(self) -> None:
            self.started = True

        async def aclose(self) -> None:
            self.close_attempted = True
            if self.fail_close:
                raise RuntimeError("synthetic close() failure")
            self.closed = True

    async def _close_failure_probe(completion_close_fails: bool, metrics_close_fails: bool):
        completion = _FailingCloseTransport(fail_close=completion_close_fails)
        metrics = _FailingCloseTransport(fail_close=metrics_close_fails)

        async def _factory():
            return {"overall_status": "block_complete"}

        result = await swc.run_with_transactional_transports(completion, metrics, _factory)
        return completion, metrics, result

    # 1. Metrics-close raises -> completion-close must still be attempted
    #    (and succeed), and the error must be visibly reported.
    completion_d, metrics_d, result_d = asyncio.run(_close_failure_probe(completion_close_fails=False, metrics_close_fails=True))
    check(
        "Blocker3: metrics_transport.aclose() raising does NOT suppress completion_transport.aclose()",
        completion_d.close_attempted and completion_d.closed, str((completion_d.close_attempted, completion_d.closed)),
    )
    check("Blocker3: metrics_transport.aclose() was attempted (and failed) as expected", metrics_d.close_attempted and not metrics_d.closed)
    check(
        "Blocker3: the metrics-close error is visibly reported on the result",
        any(e.get("transport") == "metrics_transport" for e in result_d.get("transport_close_errors", [])),
        str(result_d.get("transport_close_errors")),
    )

    # 2. Completion-close raises -> metrics-close must still be attempted
    #    (and succeed), and the error must be visibly reported.
    completion_e, metrics_e, result_e = asyncio.run(_close_failure_probe(completion_close_fails=True, metrics_close_fails=False))
    check(
        "Blocker3: completion_transport.aclose() raising does NOT suppress metrics_transport.aclose()",
        metrics_e.close_attempted and metrics_e.closed, str((metrics_e.close_attempted, metrics_e.closed)),
    )
    check("Blocker3: completion_transport.aclose() was attempted (and failed) as expected", completion_e.close_attempted and not completion_e.closed)
    check(
        "Blocker3: the completion-close error is visibly reported on the result",
        any(e.get("transport") == "completion_transport" for e in result_e.get("transport_close_errors", [])),
        str(result_e.get("transport_close_errors")),
    )

    # 3. BOTH close calls raise -> both are attempted, both errors appear.
    completion_f, metrics_f, result_f = asyncio.run(_close_failure_probe(completion_close_fails=True, metrics_close_fails=True))
    check("Blocker3: both close attempts happen even when both raise", completion_f.close_attempted and metrics_f.close_attempted)
    close_errors_f = result_f.get("transport_close_errors", [])
    check(
        "Blocker3: both errors are present in the report when both closes fail",
        {"completion_transport", "metrics_transport"} == {e.get("transport") for e in close_errors_f},
        str(close_errors_f),
    )

    # 4/5. Both succeed / never started -- already covered by the N1
    # happy-path and never-started scenarios above (same underlying
    # function); re-affirmed here for Blocker-3 traceability.
    check(
        "Blocker3: (re-affirmed) happy path -- both transports close successfully with zero reported errors",
        "transport_close_errors" not in (asyncio.run(_close_failure_probe(False, False))[2]),
    )

    # 6. SIGTERM-/abort-fake path: an already-triggered interrupt state
    # must still result in a clean, best-effort shutdown -- both
    # transports closed, and (since the abort fires before the server
    # would even start) no server process left dangling.
    async def _abort_fake_diagnostic_probe():
        pre_triggered = swc.InterruptState()
        pre_triggered.trigger("SIGTERM")
        completion = _FailingCloseTransport(fail_close=False)
        metrics = _FailingCloseTransport(fail_close=False)
        diag_transport = swc.FakeCompletionTransport(base.FakeTransport(), clock=base.FakeClock())
        diag_metrics_transport = base.FakeTransport()
        diag_server_adapter = base.FakeServerProcessAdapter()

        async def _factory():
            with tempfile.TemporaryDirectory() as td2:
                return await swc.run_diagnostic_pair(
                    bundle=bundle, output_dir=Path(td2) / "abort_diag", host="127.0.0.1", port=37999,
                    api_key="fake-key", transport=diag_transport, metrics_transport=diag_metrics_transport,
                    tokenizer=base.FakeTokenizerAdapter(), server_adapter=diag_server_adapter,
                    sleeper=base.FakeSleeper(), clock=base.FakeClock(), run_server_path=server_path,
                    env=fake_diag_env, interrupt_state=pre_triggered,
                )

        summary = await swc.run_with_transactional_transports(completion, metrics, _factory)
        return completion, metrics, diag_server_adapter, summary

    completion_g, metrics_g, diag_server_adapter_g, summary_g = asyncio.run(_abort_fake_diagnostic_probe())
    check("Blocker3/SIGTERM-fake: the diagnostic run reports interrupted overall_status", summary_g.get("overall_status") == "interrupted", str(summary_g))
    check("Blocker3/SIGTERM-fake: no server was ever started (interrupt fired before server start)", len(diag_server_adapter_g.started) == 0)
    check("Blocker3/SIGTERM-fake: the outer completion transport was still closed", completion_g.closed)
    check("Blocker3/SIGTERM-fake: the outer metrics transport was still closed", metrics_g.closed)

    # ========================================================================
    # 2026-07-20 hardening pass: original prefill_confirmation hashes
    # (already-audited source) remain byte-for-byte unchanged
    # ========================================================================
    _EXPECTED_ORIGINAL_HASHES = {
        "run_prefill_confirmation.py": "981aba99aff820ea8fea3bef6df0e1e8bfb127df059695d4736d7702ef300b75",
    }
    for name, expected_hash in _EXPECTED_ORIGINAL_HASHES.items():
        actual = base._sha256_file(BASE_DIR / name)
        check(f"original {name} hash unchanged from the audited baseline", actual == expected_hash, f"actual={actual}")

    print("=" * 78)
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


import unittest


class _MainSuiteTestCase(unittest.TestCase):
    """N1 fix (2026-07-20 third hardening pass): a `python3 -m unittest
    discover` invocation previously found zero tests in this project's
    self-contained-script-with-main() test files (matching the
    convention already established by the originally-audited
    test_run_prefill_confirmation.py/test_prefill_confirmation_timing.py).
    This thin TestCase wrapper makes the SAME exhaustive check suite
    discoverable by `unittest discover`, without changing how the file
    behaves when run directly as `python3 test_*.py`."""

    def test_all_checks_pass(self) -> None:
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    raise SystemExit(main())
