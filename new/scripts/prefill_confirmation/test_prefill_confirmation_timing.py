#!/usr/bin/env python3
"""
test_prefill_confirmation_timing.py

Standalone, offline (no GPU, no network, no real server) test file for
the task/semaphore/dispatch/wave/trigger-exposure timing instrumentation
added to run_prefill_confirmation.py. Imports the functions under test
directly from the production runner -- it does not reimplement any
timing, wave, or trigger-classification logic itself. Only test
fixtures, synthetic scenarios, and assertions live here.

Covers the 20 required scenarios (see project prompt "Schritt-2-Tests"):
  1. Active request with negligible semaphore wait.
  2. Later request with positive local_queue_wait_ms.
  3. Multiple waves at concurrency 4 (real async integration run).
  4. Different task_creation_offset_ms for active vs. later wave.
  5. Shared, identical victim_phase_start_ns across all victim requests.
  6. Monotonic timestamps (both a hand-built record and a real run).
  7. Deliberately non-monotonic timestamps -> visible validation error.
  8. A successful request.
  9. Failure before semaphore acquisition.
  10. Failure after semaphore acquisition but before dispatch.
  11. Failure after dispatch.
  12. Timeout/cancellation with request_terminal_ns still set.
  13. Request running at the trigger instant.
  14. Request queued at the trigger instant.
  15. Request admitted but not yet dispatched at the trigger instant.
  16. Request completed before the trigger.
  17. Request created only after the trigger.
  18. Correct wave_id/wave_position (unit + real-run cross-check).
  19. Missing/invalid request_index -> visible failure, never silent.
  20. Backward compatibility of pre-existing fields/validator behavior.

Usage:
    python3 test_prefill_confirmation_timing.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_prefill_confirmation as runner  # noqa: E402


# ============================================================================
# Test harness (plain assert-style checks; no third-party test framework)
# ============================================================================

_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _results.append((name, bool(condition), detail))
    marker = "OK" if condition else "FAIL"
    print(f"[{marker}] {name}" + (f" -- {detail}" if detail and not condition else ""))


# ============================================================================
# Fixture helpers (test data only -- no production logic reimplemented)
# ============================================================================

def make_episode(
    *,
    episode_id: str = "test_ep",
    concurrency: int = 4,
    victim_request_count: int = 4,
    condition: str = "no_burst",
    trigger_after_decode_tokens: int = 1,
) -> "runner.Episode":
    """Minimal, otherwise-arbitrary Episode fixture. Field values that
    don't matter for the timing instrumentation are filled with small,
    valid placeholders."""
    return runner.Episode(
        episode_id=episode_id,
        model_key="llama",
        model_id="test/model",
        offload_gb=0,
        state_label="low",
        concurrency=concurrency,
        trigger_after_decode_tokens=trigger_after_decode_tokens,
        condition=condition,
        repeat=1,
        random_seed=1,
        episode_seed=1,
        victim_workload_seed=1,
        burst_workload_seed=2,
        victim_request_count=victim_request_count,
        victim_input_len=8,
        victim_output_len=4,
        victim_temperature=0.0,
        burst_parallel_requests=2,
        burst_input_len=8,
        burst_output_len=2,
        burst_temperature=0.0,
        max_num_batched_tokens=2048,
        condition_first_in_block=condition,
        restart_server_before_block=1,
        block_id="test_block",
        order_in_block=1,
    )


def make_fake_transport_all_success() -> "runner.FakeTransport":
    transport = runner.FakeTransport()

    def factory(payload: dict) -> "runner.FakeStreamScript":
        return runner.FakeStreamScript(
            prompt_token_ids_echo=list(payload["prompt"]),
            token_events=[[1000 + i] for i in range(payload["max_tokens"])],
            usage={"prompt_tokens": len(payload["prompt"]), "completion_tokens": payload["max_tokens"]},
        )

    transport.default_script_factory = factory
    return transport


def make_run_context(transport: "runner.FakeTransport") -> "runner.RunContext":
    clock = runner.FakeClock(step_ns=1_000_000)  # 1ms per tick, deterministic
    tokenizer = runner.FakeTokenizerAdapter()
    valid_ids = runner.compute_valid_token_ids(tokenizer)
    return runner.RunContext(
        transport=transport,
        clock=clock,
        sleeper=runner.FakeSleeper(),
        base_url="http://fake",
        api_key="fake-key",
        model_full_id="test/model",
        valid_ids=valid_ids,
    )


def well_formed_timing_record(
    *,
    victim_phase_start_ns: int = 1_000,
    task_created_ns: int = 2_000,
    semaphore_acquired_ns: int = 3_000,
    request_dispatch_ns: int = 4_000,
    request_terminal_ns: int = 5_000,
) -> dict:
    """A minimal dict carrying only the 5 raw timestamp fields under
    test, in an otherwise well-formed (monotonic) arrangement. Derived
    ms fields are recomputed by the production helper itself (never by
    this test file) and merged in, so the record is self-consistent."""
    record: dict = {
        "victim_phase_start_ns": victim_phase_start_ns,
        "task_created_ns": task_created_ns,
        "semaphore_acquired_ns": semaphore_acquired_ns,
        "request_dispatch_ns": request_dispatch_ns,
        "request_terminal_ns": request_terminal_ns,
        "wave_id": 0,
        "wave_position": 0,
    }
    record.update(
        runner._compute_queue_timing_fields(
            task_created_ns=task_created_ns,
            victim_phase_start_ns=victim_phase_start_ns,
            semaphore_acquired_ns=semaphore_acquired_ns,
            request_dispatch_ns=request_dispatch_ns,
            request_terminal_ns=request_terminal_ns,
        )
    )
    return record


# ============================================================================
# 1/2: local_queue_wait_ms for an active (negligible wait) vs. later
#      (positive wait) request
# ============================================================================

def test_1_active_negligible_semaphore_wait() -> None:
    fields = runner._compute_queue_timing_fields(
        task_created_ns=1_000_000,
        victim_phase_start_ns=500_000,
        semaphore_acquired_ns=1_000_500,  # 0.5 microsecond after task creation
        request_dispatch_ns=1_001_000,
        request_terminal_ns=2_000_000,
    )
    check(
        "(1) active request has near-zero local_queue_wait_ms",
        fields["local_queue_wait_ms"] is not None and fields["local_queue_wait_ms"] < 0.01,
        str(fields["local_queue_wait_ms"]),
    )


def test_2_later_positive_semaphore_wait() -> None:
    fields = runner._compute_queue_timing_fields(
        task_created_ns=1_000_000,
        victim_phase_start_ns=500_000,
        semaphore_acquired_ns=6_000_000,  # 5ms of real queueing
        request_dispatch_ns=6_001_000,
        request_terminal_ns=8_000_000,
    )
    check(
        "(2) later-wave request shows a clearly positive local_queue_wait_ms",
        fields["local_queue_wait_ms"] is not None and fields["local_queue_wait_ms"] >= 4.9,
        str(fields["local_queue_wait_ms"]),
    )


# ============================================================================
# 3/4/5/8/18/20: a real async run through run_regular_episode, concurrency
#                4, 8 victims (2 waves) -- exercises the actual wiring,
#                not just the pure helpers.
# ============================================================================

def _run_two_wave_episode() -> dict:
    async def _go() -> dict:
        transport = make_fake_transport_all_success()
        ctx = make_run_context(transport)
        episode = make_episode(concurrency=4, victim_request_count=8, condition="no_burst")
        return await runner.run_regular_episode(
            ctx, episode, schedule_fingerprint="sha256:" + "0" * 64,
            server_metadata={}, stabilization_ref={}, run_mode="smoke",
        )

    return asyncio.run(_go())


def test_3_4_5_8_18_20_real_two_wave_run() -> None:
    result = _run_two_wave_episode()
    victims = {r["request_index"]: r for r in result["victim_requests"]}

    check(
        "(8) all 8 victim requests in the real run completed successfully",
        len(victims) == 8 and all(r["status"] == runner.REQUEST_STATUS_COMPLETE for r in victims.values()),
        str({i: r["status"] for i, r in victims.items()}),
    )

    check(
        "(3) two distinct waves are present at concurrency=4/n=8",
        {victims[i]["wave_id"] for i in range(8)} == {0, 1},
        str({i: victims[i]["wave_id"] for i in range(8)}),
    )

    check(
        "(18) wave_id/wave_position match request_index // 4 and % 4 for every request",
        all(
            victims[i]["wave_id"] == i // 4 and victims[i]["wave_position"] == i % 4
            for i in range(8)
        ),
        str({i: (victims[i]["wave_id"], victims[i]["wave_position"]) for i in range(8)}),
    )

    phase_starts = {victims[i]["victim_phase_start_ns"] for i in range(8)}
    check(
        "(5) every victim request shares the exact same victim_phase_start_ns",
        len(phase_starts) == 1,
        str(phase_starts),
    )

    active_offsets = [victims[i]["task_creation_offset_ms"] for i in range(4)]
    later_offsets = [victims[i]["task_creation_offset_ms"] for i in range(4, 8)]
    check(
        "(4) later-wave task_creation_offset_ms is clearly larger than active-wave's",
        all(v is not None for v in active_offsets + later_offsets)
        and min(later_offsets) > max(active_offsets),
        f"active={active_offsets}, later={later_offsets}",
    )

    check(
        "(6) real run produces monotonic task_created_ns <= semaphore_acquired_ns <= "
        "request_dispatch_ns <= request_terminal_ns for every request",
        all(
            victims[i]["victim_phase_start_ns"] <= victims[i]["task_created_ns"]
            <= victims[i]["semaphore_acquired_ns"] <= victims[i]["request_dispatch_ns"]
            <= victims[i]["request_terminal_ns"]
            for i in range(8)
        ),
    )

    legacy_fields = ("request_start_ns", "ttft_ms", "e2el_ms", "client_observed_tpot_ms",
                      "raw_sse_events", "itl_ms", "status", "finish_reason", "request_index")
    check(
        "(20) all pre-existing legacy fields are still present and populated",
        all(all(f in r for f in legacy_fields) for r in victims.values()),
    )


# ============================================================================
# 6/7: monotonic vs. deliberately non-monotonic timestamps
# ============================================================================

def test_6_monotonic_record_passes() -> None:
    episode = make_episode(concurrency=4)
    record = well_formed_timing_record()
    record.update(
        {
            "request_id": f"{episode.episode_id}:victim:0", "role": "victim", "request_index": 0,
            "prompt_seed": runner.victim_prompt_seed(episode, 0),
            "generation_seed": runner.victim_generation_seed(episode, 0),
            "status": runner.REQUEST_STATUS_COMPLETE, "timed_out": False, "cancelled": False,
            "done_received": True, "error_type": None, "error_message": None, "validation_errors": [],
            "finish_reason": "length",
            "prompt_token_ids_sent": [0] * episode.victim_input_len,
            "prompt_token_ids_returned": [0] * episode.victim_input_len,
            "prompt_sha256": runner.prompt_sha256([0] * episode.victim_input_len),
            "expected_prompt_tokens": episode.victim_input_len,
            "output_token_ids": [0] * episode.victim_output_len,
            "expected_completion_tokens": episode.victim_output_len,
            "usage": {"prompt_tokens": episode.victim_input_len, "completion_tokens": episode.victim_output_len},
            "was_created_at_trigger": None, "was_admitted_at_trigger": None,
            "was_dispatched_at_trigger": None, "was_running_at_trigger": None,
            "trigger_exposure_group": None,
        }
    )
    errors = runner.validate_victim_timing_instrumentation(
        record, episode=episode, request_index=0, trigger_perf_ns=None,
    )
    check("(6) a well-formed, monotonic record produces zero timing errors", errors == [], str(errors))


def test_7_non_monotonic_record_fails_loudly() -> None:
    episode = make_episode(concurrency=4)
    # Deliberately swapped: semaphore_acquired_ns AFTER request_dispatch_ns.
    record = well_formed_timing_record(
        task_created_ns=2_000, semaphore_acquired_ns=9_000, request_dispatch_ns=4_000, request_terminal_ns=10_000,
    )
    errors = runner.validate_victim_timing_instrumentation(
        record, episode=episode, request_index=0, trigger_perf_ns=None,
    )
    check(
        "(7) deliberately non-monotonic timestamps produce a visible validation error",
        any("monotonic" in e for e in errors),
        str(errors),
    )


# ============================================================================
# 9/10/11/12: error paths at different stages -- derived fields must be
# None (never a crash, never a fabricated negative duration), and
# request_terminal_ns must still be set.
# ============================================================================

def test_9_failure_before_semaphore_acquisition() -> None:
    fields = runner._compute_queue_timing_fields(
        task_created_ns=1_000, victim_phase_start_ns=500,
        semaphore_acquired_ns=None, request_dispatch_ns=None, request_terminal_ns=2_000,
    )
    check(
        "(9) failure before semaphore acquisition: queue-wait/dispatch fields are None, "
        "only phase-relative fields are computable",
        fields["local_queue_wait_ms"] is None
        and fields["admission_to_dispatch_ms"] is None
        and fields["task_creation_offset_ms"] is not None
        and fields["total_e2el_from_task_creation_ms"] is not None,
        str(fields),
    )


def test_10_failure_after_acquisition_before_dispatch() -> None:
    fields = runner._compute_queue_timing_fields(
        task_created_ns=1_000, victim_phase_start_ns=500,
        semaphore_acquired_ns=1_500, request_dispatch_ns=None, request_terminal_ns=2_000,
    )
    check(
        "(10) failure after acquisition but before dispatch: local_queue_wait_ms computable, "
        "admission_to_dispatch_ms is None",
        fields["local_queue_wait_ms"] is not None and fields["admission_to_dispatch_ms"] is None,
        str(fields),
    )


def test_11_failure_after_dispatch() -> None:
    fields = runner._compute_queue_timing_fields(
        task_created_ns=1_000, victim_phase_start_ns=500,
        semaphore_acquired_ns=1_500, request_dispatch_ns=1_800, request_terminal_ns=2_500,
    )
    check(
        "(11) failure after dispatch: every derived field is computable",
        all(v is not None for v in fields.values()),
        str(fields),
    )


def test_12_timeout_sets_request_terminal_ns() -> None:
    async def _go() -> dict:
        transport = runner.FakeTransport()
        transport.default_script_factory = lambda payload: runner.FakeStreamScript(
            prompt_token_ids_echo=list(payload["prompt"]), token_events=[], hang=True,
        )
        ctx = make_run_context(transport)
        ctx.http_timeout_s = 0.01
        return await runner._run_victim_request(
            ctx, make_episode(), 0,
            task_created_ns=1_000, victim_phase_start_ns=500, semaphore_acquired_ns=1_500,
            wave_id=0, wave_position=0,
        )

    record = asyncio.run(_go())
    check(
        "(12) a timed-out/hanging request still has request_terminal_ns set and status reflects the timeout",
        record.get("request_terminal_ns") is not None and record.get("timed_out") is True
        and record.get("status") == runner.REQUEST_STATUS_FAILED,
        str({"terminal": record.get("request_terminal_ns"), "timed_out": record.get("timed_out"), "status": record.get("status")}),
    )


# ============================================================================
# 13/14/15/16/17: trigger_exposure_group classification
# ============================================================================

def test_13_running_at_trigger() -> None:
    result = runner._compute_trigger_exposure(
        task_created_ns=100, semaphore_acquired_ns=200, request_dispatch_ns=300,
        request_terminal_ns=900, trigger_perf_ns=500,
    )
    check("(13) request dispatched before and still running at the trigger -> running_at_trigger",
          result["trigger_exposure_group"] == "running_at_trigger" and result["was_running_at_trigger"] is True,
          str(result))


def test_14_queued_at_trigger() -> None:
    result = runner._compute_trigger_exposure(
        task_created_ns=100, semaphore_acquired_ns=600, request_dispatch_ns=700,
        request_terminal_ns=900, trigger_perf_ns=500,
    )
    check("(14) task created before, semaphore not yet acquired at the trigger -> queued_at_trigger",
          result["trigger_exposure_group"] == "queued_at_trigger", str(result))


def test_15_admitted_not_dispatched_at_trigger() -> None:
    result = runner._compute_trigger_exposure(
        task_created_ns=100, semaphore_acquired_ns=200, request_dispatch_ns=700,
        request_terminal_ns=900, trigger_perf_ns=500,
    )
    check("(15) admitted before, dispatched after the trigger -> admitted_not_dispatched_at_trigger",
          result["trigger_exposure_group"] == "admitted_not_dispatched_at_trigger", str(result))


def test_16_completed_before_trigger() -> None:
    result = runner._compute_trigger_exposure(
        task_created_ns=100, semaphore_acquired_ns=150, request_dispatch_ns=200,
        request_terminal_ns=300, trigger_perf_ns=500,
    )
    check("(16) request fully finished before the trigger -> completed_before_trigger",
          result["trigger_exposure_group"] == "completed_before_trigger", str(result))


def test_17_created_after_trigger() -> None:
    result = runner._compute_trigger_exposure(
        task_created_ns=600, semaphore_acquired_ns=650, request_dispatch_ns=700,
        request_terminal_ns=900, trigger_perf_ns=500,
    )
    check("(17) task created only after the trigger -> created_after_trigger",
          result["trigger_exposure_group"] == "created_after_trigger", str(result))


# ============================================================================
# 18/19: wave computation, including a missing/invalid request_index
# ============================================================================

def test_18_wave_computation_unit() -> None:
    cases = {(0, 4): (0, 0), (3, 4): (0, 3), (4, 4): (1, 0), (19, 4): (4, 3)}
    all_ok = True
    detail = {}
    for (idx, conc), expected in cases.items():
        actual = runner._compute_wave(idx, conc)
        detail[(idx, conc)] = (actual, expected)
        if actual != expected:
            all_ok = False
    check("(18) wave_id/wave_position match request_index // concurrency and % concurrency", all_ok, str(detail))


def test_19_missing_request_index_fails_visibly() -> None:
    raised = False
    try:
        runner._compute_wave(None, 4)
    except ValueError:
        raised = True
    check("(19a) _compute_wave(None, concurrency) raises ValueError rather than defaulting", raised)

    episode = make_episode(concurrency=4)
    record = well_formed_timing_record()
    record["wave_id"] = None
    record["wave_position"] = None
    errors = runner.validate_victim_timing_instrumentation(
        record, episode=episode, request_index=None, trigger_perf_ns=None,  # type: ignore[arg-type]
    )
    check(
        "(19b) validate_victim_timing_instrumentation reports a concrete error for an "
        "invalid request_index instead of crashing or silently defaulting",
        any("cannot compute expected wave" in e for e in errors),
        str(errors),
    )


# ============================================================================
# 20: backward compatibility -- an old-style record (no new fields at all)
# must still validate exactly as before when trigger_perf_ns is not passed.
# ============================================================================

def test_20_backward_compatibility_old_style_record() -> None:
    episode = make_episode(concurrency=4)
    old_style_record = {
        "request_id": f"{episode.episode_id}:victim:0",
        "role": "victim",
        "request_index": 0,
        "prompt_seed": runner.victim_prompt_seed(episode, 0),
        "generation_seed": runner.victim_generation_seed(episode, 0),
        "status": runner.REQUEST_STATUS_COMPLETE,
        "timed_out": False,
        "cancelled": False,
        "done_received": True,
        "error_type": None,
        "error_message": None,
        "validation_errors": [],
        "finish_reason": "length",
        "prompt_token_ids_sent": [0] * episode.victim_input_len,
        "prompt_token_ids_returned": [0] * episode.victim_input_len,
        "prompt_sha256": runner.prompt_sha256([0] * episode.victim_input_len),
        "expected_prompt_tokens": episode.victim_input_len,
        "output_token_ids": [0] * episode.victim_output_len,
        "expected_completion_tokens": episode.victim_output_len,
        "usage": {"prompt_tokens": episode.victim_input_len, "completion_tokens": episode.victim_output_len},
        "request_start_ns": 1_000,
        "first_token_receive_ns": 1_500,
        "last_token_receive_ns": 2_000,
        "stream_end_ns": 2_100,
        # Deliberately NO task_created_ns / victim_phase_start_ns / etc.
    }
    errors_without_trigger_arg = runner.validate_complete_request_record(
        old_style_record, episode=episode, role="victim", request_index=0,
    )
    check(
        "(20) validate_complete_request_record on an old-style record (no new fields, "
        "no trigger_perf_ns argument) behaves exactly as before -- no new-instrumentation errors",
        not any(
            "timing" in e or "wave" in e or "trigger_exposure" in e or "monotonic" in e
            for e in errors_without_trigger_arg
        ),
        str(errors_without_trigger_arg),
    )


# ============================================================================
# main
# ============================================================================

# ============================================================================
# Mutation tests against the production classify_result_file() /
# run_regular_episode() contract -- built on a REAL episode result
# produced by an actual run_regular_episode() call (not a hand-built
# fixture), then selectively mutated. No production logic is
# reimplemented here; only the resulting JSON dict is mutated and fed
# back into the real classify_result_file().
# ============================================================================

def _run_real_regular_episode(*, victim_request_count: int = 8, concurrency: int = 4) -> tuple[dict, "runner.Episode"]:
    async def _go() -> dict:
        transport = make_fake_transport_all_success()
        ctx = make_run_context(transport)
        episode = make_episode(
            episode_id="mutation_ep", concurrency=concurrency,
            victim_request_count=victim_request_count, condition="no_burst",
        )
        result = await runner.run_regular_episode(
            ctx, episode, schedule_fingerprint="sha256:" + "0" * 64,
            server_metadata={}, stabilization_ref={}, run_mode="smoke",
        )
        return result

    result = asyncio.run(_go())
    episode = make_episode(episode_id="mutation_ep", concurrency=concurrency, victim_request_count=victim_request_count, condition="no_burst")
    return result, episode


def _classify_mutated(result: dict, episode: "runner.Episode", mutate) -> tuple[str, list[str]]:
    import copy
    import json as _json
    import tempfile as _tempfile

    mutated = copy.deepcopy(result)
    mutate(mutated)
    with _tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "mutation_ep.json"
        p.write_text(_json.dumps(mutated), encoding="utf-8")
        return runner.classify_result_file(p, episode, "sha256:" + "0" * 64, "smoke")


_BASELINE_RESULT, _BASELINE_EPISODE = None, None


def _baseline() -> tuple[dict, "runner.Episode"]:
    global _BASELINE_RESULT, _BASELINE_EPISODE
    if _BASELINE_RESULT is None:
        _BASELINE_RESULT, _BASELINE_EPISODE = _run_real_regular_episode()
    return _BASELINE_RESULT, _BASELINE_EPISODE


def test_mutation_0_baseline_is_valid_complete() -> None:
    result, episode = _baseline()
    cls, notes = _classify_mutated(result, episode, lambda m: None)
    check(
        "(mutation-baseline) an unmutated, genuinely correct episode result is accepted as valid_complete",
        cls == runner.CLASSIFICATION_VALID_COMPLETE, f"{cls}: {notes}",
    )


def test_mutation_1_non_monotonic_timestamps() -> None:
    result, episode = _baseline()

    def mutate(m: dict) -> None:
        r = m["victim_requests"][0]
        r["task_created_ns"] = r["request_terminal_ns"] + 1

    cls, notes = _classify_mutated(result, episode, mutate)
    check("(Mutation 1) non-monotonic task_created_ns > request_terminal_ns -> invalid",
          cls == runner.CLASSIFICATION_INVALID, f"{cls}: {notes}")


def test_mutation_2_manipulated_derived_field() -> None:
    result, episode = _baseline()

    def mutate(m: dict) -> None:
        m["victim_requests"][0]["local_queue_wait_ms"] = -123.0

    cls, notes = _classify_mutated(result, episode, mutate)
    check("(Mutation 2) manipulated local_queue_wait_ms=-123.0 -> invalid",
          cls == runner.CLASSIFICATION_INVALID, f"{cls}: {notes}")


def test_mutation_3_manipulated_exposure_group() -> None:
    result, episode = _baseline()

    def mutate(m: dict) -> None:
        m["victim_requests"][0]["trigger_exposure_group"] = "nonsense"

    cls, notes = _classify_mutated(result, episode, mutate)
    check("(Mutation 3) trigger_exposure_group='nonsense' -> invalid",
          cls == runner.CLASSIFICATION_INVALID, f"{cls}: {notes}")


def test_mutation_4_new_fields_removed() -> None:
    result, episode = _baseline()

    for field in ("timing_instrumentation_version", "timing_instrumentation_name",
                  "victim_phase_start_ns", "queue_timing_summary"):
        def mutate(m: dict, field=field) -> None:
            del m[field]

        cls, notes = _classify_mutated(result, episode, mutate)
        check(f"(Mutation 4) removing top-level {field!r} -> invalid",
              cls == runner.CLASSIFICATION_INVALID, f"{cls}: {notes}")


def test_mutation_5_manipulated_episode_summary() -> None:
    result, episode = _baseline()

    def mutate(m: dict) -> None:
        m["queue_timing_summary"]["request_count_by_wave"] = {"0": 999}

    cls, notes = _classify_mutated(result, episode, mutate)
    check("(Mutation 5) manipulated queue_timing_summary.request_count_by_wave -> invalid",
          cls == runner.CLASSIFICATION_INVALID, f"{cls}: {notes}")


def test_mutation_6_inconsistent_shared_phase_start() -> None:
    result, episode = _baseline()

    def mutate(m: dict) -> None:
        m["victim_requests"][0]["victim_phase_start_ns"] += 999

    cls, notes = _classify_mutated(result, episode, mutate)
    check("(Mutation 6) victim_phase_start_ns differs on exactly one victim request -> invalid",
          cls == runner.CLASSIFICATION_INVALID, f"{cls}: {notes}")


def test_production_path_status_gated_by_timing_validity() -> None:
    """Real run_regular_episode() run, with validate_victim_timing_instrumentation
    monkeypatched to report a fake error for one request -- the episode
    must NOT come back status=='complete'. Restores the original
    function immediately afterward regardless of outcome."""
    original = runner.validate_victim_timing_instrumentation

    def _always_fails(record, *, episode, request_index, trigger_perf_ns):
        if request_index == 0:
            return ["synthetic forced failure for test_production_path_status_gated_by_timing_validity"]
        return original(record, episode=episode, request_index=request_index, trigger_perf_ns=trigger_perf_ns)

    async def _go_ok() -> dict:
        transport = make_fake_transport_all_success()
        ctx = make_run_context(transport)
        episode = make_episode(episode_id="prod_path_ep", concurrency=4, victim_request_count=4, condition="no_burst")
        return await runner.run_regular_episode(
            ctx, episode, schedule_fingerprint="sha256:" + "0" * 64,
            server_metadata={}, stabilization_ref={}, run_mode="smoke",
        )

    ok_result = asyncio.run(_go_ok())
    check(
        "(production-path a) with real (unpatched) timing validation, a genuinely correct episode is status=='complete'",
        ok_result["status"] == runner.REQUEST_STATUS_COMPLETE, str(ok_result["status"]),
    )

    runner.validate_victim_timing_instrumentation = _always_fails
    try:
        async def _go_forced_fail() -> dict:
            transport = make_fake_transport_all_success()
            ctx = make_run_context(transport)
            episode = make_episode(episode_id="prod_path_ep2", concurrency=4, victim_request_count=4, condition="no_burst")
            return await runner.run_regular_episode(
                ctx, episode, schedule_fingerprint="sha256:" + "0" * 64,
                server_metadata={}, stabilization_ref={}, run_mode="smoke",
            )

        forced_fail_result = asyncio.run(_go_forced_fail())
    finally:
        runner.validate_victim_timing_instrumentation = original

    check(
        "(production-path b) with timing validation monkeypatched to fail for request 0, "
        "the episode is NOT status=='complete'",
        forced_fail_result["status"] != runner.REQUEST_STATUS_COMPLETE,
        str(forced_fail_result["status"]),
    )
    check(
        "(production-path c) the forced timing failure is recorded in validation_errors with its request index",
        any("victim[0] timing" in e for e in forced_fail_result["validation_errors"]),
        str(forced_fail_result["validation_errors"]),
    )


def test_mutation_7_summary_nan_rejected() -> None:
    result, episode = _baseline()

    def mutate(m: dict) -> None:
        m["queue_timing_summary"]["median_local_queue_wait_ms_by_wave"]["0"] = float("nan")

    cls, notes = _classify_mutated(result, episode, mutate)
    check("(Mutation 7) queue_timing_summary float = NaN -> invalid",
          cls == runner.CLASSIFICATION_INVALID, f"{cls}: {notes}")


def test_mutation_8_summary_count_as_float_rejected() -> None:
    result, episode = _baseline()

    def mutate(m: dict) -> None:
        m["queue_timing_summary"]["request_count_by_wave"]["0"] = 4.0

    cls, notes = _classify_mutated(result, episode, mutate)
    check("(Mutation 8) queue_timing_summary count = 4.0 (float, not int) -> invalid",
          cls == runner.CLASSIFICATION_INVALID, f"{cls}: {notes}")


def main() -> int:
    print("test_prefill_confirmation_timing.py -- offline timing/wave/trigger instrumentation tests")
    print("=" * 78)

    test_1_active_negligible_semaphore_wait()
    test_2_later_positive_semaphore_wait()
    test_3_4_5_8_18_20_real_two_wave_run()
    test_6_monotonic_record_passes()
    test_7_non_monotonic_record_fails_loudly()
    test_9_failure_before_semaphore_acquisition()
    test_10_failure_after_acquisition_before_dispatch()
    test_11_failure_after_dispatch()
    test_12_timeout_sets_request_terminal_ns()
    test_13_running_at_trigger()
    test_14_queued_at_trigger()
    test_15_admitted_not_dispatched_at_trigger()
    test_16_completed_before_trigger()
    test_17_created_after_trigger()
    test_18_wave_computation_unit()
    test_19_missing_request_index_fails_visibly()
    test_20_backward_compatibility_old_style_record()

    test_mutation_0_baseline_is_valid_complete()
    test_mutation_1_non_monotonic_timestamps()
    test_mutation_2_manipulated_derived_field()
    test_mutation_3_manipulated_exposure_group()
    test_mutation_4_new_fields_removed()
    test_mutation_5_manipulated_episode_summary()
    test_mutation_6_inconsistent_shared_phase_start()
    test_mutation_7_summary_nan_rejected()
    test_mutation_8_summary_count_as_float_rejected()
    test_production_path_status_gated_by_timing_validity()

    print("=" * 78)
    passed = sum(1 for _, ok, _ in _results if ok)
    print(f"{passed}/{len(_results)} checks passed")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
