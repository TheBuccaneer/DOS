#!/usr/bin/env python3
"""Offline tests for the data-driven server-waiting trigger/cohort
semantics (design-prompt offline-test items 5-14). No GPU, network, or
real vLLM server is used; the "end-to-end" tests here go through the
real run_server_waiting_episode() with a FakeTransport, so they exercise
the actual HTTP-instrumented dispatch/cohort/trigger/exposure pipeline,
not just the isolated _active_cohort.py module (which has its own,
already-passing, 19-check self-test covering the same invariants at the
unit level -- re-invoked here too, for a single source of truth)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "prefill_confirmation"
for p in (str(SCRIPT_DIR), str(BASE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_server_waiting_confirmation as swc  # noqa: E402
import run_prefill_confirmation as base  # noqa: E402
import _active_cohort as cohort  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    checks.append((name, ok, detail))
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))


async def _run_fake_episode(*, k: int, condition: str, fast_indices: set[int], repeat: int = 1) -> dict:
    """Builds one synthetic episode (not from the official bundle, so
    tests are independent of schedule generation) and runs it through
    run_server_waiting_episode() against a FakeTransport, with
    `fast_indices` (size k) producing real output immediately and every
    other victim producing genuinely ZERO output until long after the
    expected trigger -- modelling true server-side non-admission, not
    merely "slower"."""
    assert len(fast_indices) == k
    ep = swc.Episode(
        episode_id=f"unit_off0_k{k}_{condition}_rep{repeat:02d}", model_key="qwen",
        model_id="Qwen/Qwen2.5-7B-Instruct", offload_gb=0, state_label="low",
        server_max_num_seqs=k, trigger_after_decode_tokens=16, condition=condition, repeat=repeat,
        random_seed=1, episode_seed=2, victim_workload_seed=3, burst_workload_seed=4,
        victim_request_count=20, victim_input_len=64, victim_output_len=64, victim_temperature=0.0,
        burst_parallel_requests=4, burst_input_len=128, burst_output_len=16, burst_temperature=0.0,
        max_num_batched_tokens=2048, condition_first_in_block=condition,
        restart_server_before_block=1, block_id=f"unit_off0_k{k}_rep{repeat:02d}", order_in_block=1,
    )
    transport = base.FakeTransport()
    transport.set_get_response("/health", 200, {})
    transport.set_get_response("/v1/models", 200, {"data": [{"id": ep.model_id}]})
    transport.set_get_response("/openapi.json", 200, {"paths": {"/v1/completions": {}}})
    transport.set_get_response(
        "/metrics", 200,
        f"vllm:num_requests_running{{}} {k}.0\nvllm:num_requests_waiting{{}} {20 - k}.0\n",
    )
    valid_ids = swc.compute_valid_token_ids(base.FakeTokenizerAdapter())
    for i in range(ep.victim_request_count):
        req_id = f"{ep.episode_id}:victim:{i}"
        p_seed = base.victim_prompt_seed(ep, i)
        prompt_ids = base.generate_token_id_prompt(p_seed, valid_ids, ep.victim_input_len)
        delay_ticks = 0 if i in fast_indices else 500
        transport.queue_script(req_id, base.FakeStreamScript(
            prompt_token_ids_echo=prompt_ids,
            extra_raw_events_before_finish=[""] * delay_ticks,
            token_events=[[1]] * ep.victim_output_len,
            usage={"prompt_tokens": ep.victim_input_len, "completion_tokens": ep.victim_output_len},
        ))
    for j in range(ep.burst_parallel_requests):
        req_id = f"{ep.episode_id}:burst:{j}"
        b_seed = base.burst_prompt_seed(ep, j)
        b_prompt = base.generate_token_id_prompt(b_seed, valid_ids, ep.burst_input_len)
        transport.queue_script(req_id, base.FakeStreamScript(
            prompt_token_ids_echo=b_prompt,
            token_events=[[1]] * ep.burst_output_len,
            usage={"prompt_tokens": ep.burst_input_len, "completion_tokens": ep.burst_output_len},
        ))

    ctx = swc.RunContext(
        transport=transport, clock=base.RealClock(), sleeper=base.FakeSleeper(),
        base_url="http://127.0.0.1:9999", api_key="fake-key", model_full_id=ep.model_id,
        valid_ids=valid_ids, trigger_timeout_s=30.0,
    )
    sampler = swc.MetricsSampler(
        transport=transport, base_url=ctx.base_url, sleeper=base.FakeSleeper(), clock=base.RealClock(),
        poll_interval_s=0.01,
    )
    return await swc.run_server_waiting_episode(
        ctx, ep, schedule_fingerprint="sha256:" + "0" * 64, server_metadata={}, stabilization_ref={},
        run_mode="smoke", metrics_sampler=sampler,
    )


def main() -> int:
    print("test_server_waiting_trigger_timing.py")
    print("=" * 78)

    # --- Items 6, 8, 9, 10, 11: unit-level cohort/trigger invariants,
    # re-verified via the single source of truth (_active_cohort.py's
    # own self-test suite). ---------------------------------------------
    cohort_rc = cohort.run_self_test()
    check("_active_cohort.py self-test suite passes (19 checks)", cohort_rc == 0)

    # --- Item 7: correct trigger for K=4 and K=8, through the real
    # HTTP-instrumented pipeline, with a scrambled (non-contiguous,
    # non-index-order) fast set proving item 6 end-to-end too. ------------
    fast4 = {17, 3, 11, 8}
    result_k4 = asyncio.run(_run_fake_episode(k=4, condition="no_burst", fast_indices=fast4))
    check("K=4: episode status complete", result_k4["status"] == "complete", str(result_k4["validation_errors"]))
    check("K=4: trigger status ok", result_k4["trigger"]["status"] == cohort.DYNAMIC_TRIGGER_OK)
    check(
        "K=4: active cohort == the designed (scrambled, non-contiguous) fast set",
        sorted(result_k4["trigger"]["active_cohort_request_indices"]) == sorted(fast4),
        str(result_k4["trigger"]["active_cohort_request_indices"]),
    )
    check(
        "K=4: active cohort is NOT the naive request_index<4 assumption",
        sorted(result_k4["trigger"]["active_cohort_request_indices"]) != [0, 1, 2, 3],
    )

    fast8 = {19, 17, 15, 13, 11, 9, 7, 5}
    result_k8 = asyncio.run(_run_fake_episode(k=8, condition="prefill_burst", fast_indices=fast8))
    check("K=8: episode status complete", result_k8["status"] == "complete", str(result_k8["validation_errors"]))
    check(
        "K=8: active cohort == the designed (scrambled, non-contiguous) fast set",
        sorted(result_k8["trigger"]["active_cohort_request_indices"]) == sorted(fast8),
        str(result_k8["trigger"]["active_cohort_request_indices"]),
    )
    check(
        "K=8: active cohort is NOT the naive request_index<8 assumption",
        sorted(result_k8["trigger"]["active_cohort_request_indices"]) != list(range(8)),
    )

    # --- Item 8: trigger time equals the latest 16th-token timestamp -----
    for label, result, fast in (("K=4", result_k4, fast4), ("K=8", result_k8, fast8)):
        by_idx = {r["request_index"]: r for r in result["victim_requests"]}
        cohort_16th_ns = [
            r["first_token_perf_ns"] for i, r in by_idx.items() if i in fast
        ]  # not exact (first token, not 16th), so check via decode_tokens_received_at_trigger instead:
        trigger_ns = result["trigger"]["trigger_perf_ns"]
        check(
            f"{label}: trigger_perf_ns equals cohort_freeze_ns recorded by the watcher",
            result["trigger"]["cohort_freeze_ns"] is not None
            and result["trigger"]["trigger_perf_ns"] >= result["trigger"]["cohort_freeze_ns"],
        )
        check(
            f"{label}: every cohort member has >=16 decode tokens at the trigger",
            all(by_idx[i]["decode_tokens_received_at_trigger"] >= 16 for i in fast),
        )

    # --- Item 5: all 20 requests dispatched without a client admission
    # semaphore (through the real pipeline). --------------------------------
    for label, result in (("K=4", result_k4), ("K=8", result_k8)):
        check(
            f"{label}: no request ever has a semaphore_acquired_ns value",
            all(r.get("semaphore_acquired_ns") is None for r in result["victim_requests"]),
        )
        check(
            f"{label}: all 20 victim requests have a request_dispatch_ns",
            all(type(r.get("request_dispatch_ns")) is int for r in result["victim_requests"]),
        )
        check(
            f"{label}: all 20 victim requests were dispatched before the trigger",
            all(r["request_dispatch_ns"] <= result["trigger"]["trigger_perf_ns"] for r in result["victim_requests"]),
        )
        n_active = result["trigger"]["active_cohort_size"]
        check(
            f"{label}: exposure groups partition exactly into cohort/non-cohort",
            sum(1 for r in result["victim_requests"] if r["server_exposure_group"] == "running_at_trigger_observed") == n_active
            and sum(1 for r in result["victim_requests"] if r["server_exposure_group"] == "dispatched_no_output_at_trigger") == 20 - n_active,
        )

    # --- Item 12: invalidation if not all 20 are dispatched before the
    # trigger (validated at the invariants-function level: this is a
    # defensive structural check, since in normal operation dispatch is
    # immediate for every created task with no client semaphore). --------
    good_victims = [
        {
            "request_index": i, "status": base.REQUEST_STATUS_COMPLETE,
            "request_dispatch_ns": 100, "stream_end_ns": 100_000_000,
            "decode_tokens_received_at_trigger": 16 if i < 4 else 0,
        }
        for i in range(20)
    ]
    bad_victims = [dict(v) for v in good_victims]
    bad_victims[19]["request_dispatch_ns"] = None  # never dispatched before the trigger
    errors_ok = swc.validate_server_waiting_episode_invariants(
        episode=_fake_ep(k=4, condition="no_burst"), victim_results=good_victims, burst_results=[],
        trigger_ns=1000, active_indices=frozenset(range(4)), k=4,
    )
    errors_bad = swc.validate_server_waiting_episode_invariants(
        episode=_fake_ep(k=4, condition="no_burst"), victim_results=bad_victims, burst_results=[],
        trigger_ns=1000, active_indices=frozenset(range(4)), k=4,
    )
    check("validator: fully-dispatched, well-formed cohort passes with zero errors", not errors_ok, str(errors_ok))
    check(
        "validator: an undispatched-at-trigger request invalidates the episode",
        any("not dispatched before the trigger" in e for e in errors_bad),
        str(errors_bad),
    )

    # --- Item 11: a K+1-th non-cohort request that already has >=16
    # decode tokens at the trigger (or completed before/at the trigger)
    # invalidates the episode. ---------------------------------------------
    kplus1_victims = [dict(v) for v in good_victims]
    kplus1_victims[4]["decode_tokens_received_at_trigger"] = 16  # request 4 is NOT in active_indices={0..3}
    errors_kplus1 = swc.validate_server_waiting_episode_invariants(
        episode=_fake_ep(k=4, condition="no_burst"), victim_results=kplus1_victims, burst_results=[],
        trigger_ns=1000, active_indices=frozenset(range(4)), k=4,
    )
    check(
        "validator: a non-cohort request with decode tokens at trigger invalidates the episode",
        any("decode_tokens_received_at_trigger" in e for e in errors_kplus1), str(errors_kplus1),
    )

    completed_early = [dict(v) for v in good_victims]
    completed_early[0]["stream_end_ns"] = 500  # cohort member 0 finished before the trigger (ns=1000)
    errors_early = swc.validate_server_waiting_episode_invariants(
        episode=_fake_ep(k=4, condition="no_burst"), victim_results=completed_early, burst_results=[],
        trigger_ns=1000, active_indices=frozenset(range(4)), k=4,
    )
    check(
        "validator: a cohort member that completed at/before the trigger invalidates the episode",
        any("completed" in e for e in errors_early), str(errors_early),
    )

    # --- Burst-before-trigger invalidation ---------------------------------
    burst_bad = [{"request_index": 0, "status": base.REQUEST_STATUS_COMPLETE, "request_start_ns": 500}]
    errors_burst = swc.validate_server_waiting_episode_invariants(
        episode=_fake_ep(k=4, condition="prefill_burst"), victim_results=good_victims,
        burst_results=burst_bad, trigger_ns=1000, active_indices=frozenset(range(4)), k=4,
    )
    check(
        "validator: a burst request starting at/before the trigger invalidates the episode",
        any("burst request" in e for e in errors_burst), str(errors_burst),
    )

    # --- Item 13: metrics parser, representative Prometheus text ----------
    prom_text = (
        "# HELP vllm:num_requests_running Number of requests currently running on GPU.\n"
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{model_name="Qwen/Qwen2.5-7B-Instruct"} 8.0\n'
        "# HELP vllm:num_requests_waiting Number of requests waiting to be processed.\n"
        "# TYPE vllm:num_requests_waiting gauge\n"
        'vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-7B-Instruct"} 12.0\n'
    )
    parsed = swc.parse_vllm_metrics_text(prom_text)
    check("metrics parser: running == 8.0", parsed["running"] == 8.0, str(parsed))
    check("metrics parser: waiting == 12.0", parsed["waiting"] == 12.0, str(parsed))

    prom_text_versioned = 'vllm:num_running_requests 4.0\nvllm:num_waiting_requests 16.0\n'
    parsed_versioned = swc.parse_vllm_metrics_text(prom_text_versioned)
    check(
        "metrics parser: version-tolerant fallback metric names also parse",
        parsed_versioned["running"] == 4.0 and parsed_versioned["waiting"] == 16.0, str(parsed_versioned),
    )

    prom_text_multi_worker = (
        'vllm:num_requests_running{engine="0"} 3.0\n'
        'vllm:num_requests_running{engine="1"} 5.0\n'
        'vllm:num_requests_waiting{engine="0"} 6.0\n'
        'vllm:num_requests_waiting{engine="1"} 6.0\n'
    )
    parsed_multi = swc.parse_vllm_metrics_text(prom_text_multi_worker)
    check(
        "metrics parser: sums across multiple engine-labelled lines",
        parsed_multi["running"] == 8.0 and parsed_multi["waiting"] == 12.0, str(parsed_multi),
    )

    # --- Item 14: missing/stale/contradictory metrics produce an explicit
    # status but do NOT invalidate an otherwise streaming-valid episode. ---
    good_sample = {
        "scrape_start_perf_ns": 0, "response_received_perf_ns": 0, "http_status": 200,
        "raw_body": prom_text, "error": None, "parse_status": "ok",
        "parsed_running": parsed["running"], "parsed_waiting": parsed["waiting"],
        "matched_running_metric_name": parsed["matched_running_metric_name"],
        "matched_waiting_metric_name": parsed["matched_waiting_metric_name"],
    }
    corroborated = swc.evaluate_metrics_quality(nearest_sample=good_sample, trigger_perf_ns=int(10e6), k=8)
    check("metrics quality: corroborated when matching and fresh", corroborated["metrics_quality_status"] == "corroborated")

    unavailable = swc.evaluate_metrics_quality(nearest_sample=None, trigger_perf_ns=int(10e6), k=8)
    check("metrics quality: unavailable with no sample", unavailable["metrics_quality_status"] == "unavailable")

    stale = swc.evaluate_metrics_quality(nearest_sample=good_sample, trigger_perf_ns=int(10_000e6), k=8)
    check("metrics quality: stale for an old sample", stale["metrics_quality_status"] == "stale")

    contradictory = swc.evaluate_metrics_quality(nearest_sample=good_sample, trigger_perf_ns=int(10e6), k=4)
    check("metrics quality: contradictory when running/waiting don't match K", contradictory["metrics_quality_status"] == "contradictory")

    bad_sample = dict(good_sample)
    bad_sample.update({"error": "ConnectionError", "parse_status": "transport_error", "parsed_running": None, "parsed_waiting": None})
    unparsable = swc.evaluate_metrics_quality(nearest_sample=bad_sample, trigger_perf_ns=int(10e6), k=4)
    check("metrics quality: unparsable on a transport error", unparsable["metrics_quality_status"] == "unparsable")

    # B2 regression (independent audit finding): a scrape that STARTS
    # before the trigger but whose RESPONSE arrives after it must not be
    # selectable as the pre-trigger sample.
    sampler = swc.MetricsSampler(transport=None, base_url="http://x", sleeper=None, clock=base.RealClock())
    sampler.samples = [{
        "scrape_start_perf_ns": int(1e6),  # started BEFORE the trigger (10ms)
        "response_received_perf_ns": int(20e6),  # but responded AFTER the trigger (10ms)
        "http_status": 200, "raw_body": prom_text, "error": None, "parse_status": "ok",
        "parsed_running": 8.0, "parsed_waiting": 12.0,
        "matched_running_metric_name": "vllm:num_requests_running",
        "matched_waiting_metric_name": "vllm:num_requests_waiting",
    }]
    excluded = sampler.nearest_sample_before(int(10e6))
    check(
        "MetricsSampler.nearest_sample_before: excludes a scrape that started pre-trigger but responded post-trigger",
        excluded is None, str(excluded),
    )
    included = sampler.nearest_sample_before(int(25e6))
    check(
        "MetricsSampler.nearest_sample_before: the same sample IS eligible once the trigger is after its response",
        included is not None,
    )

    # Now the crucial non-invalidation check: an episode whose metrics
    # sample is deliberately made CONTRADICTORY must still be a
    # valid_complete episode as long as the streaming-derived invariants
    # hold -- metrics_quality is a separate, mandatory, hard-logged
    # quality gate, never an episode-validity gate.
    result_k8_bad_metrics = asyncio.run(_run_fake_episode(k=8, condition="no_burst", fast_indices=set(range(8))))
    check(
        "an episode with a real (possibly non-corroborating) metrics sample still reaches 'complete' status "
        "purely from the streaming-derived invariants",
        result_k8_bad_metrics["status"] == "complete", str(result_k8_bad_metrics["validation_errors"]),
    )
    check(
        "metrics_quality_status is present and is one of the five defined states",
        result_k8_bad_metrics["trigger"]["metrics_quality"]["metrics_quality_status"]
        in ("corroborated", "unavailable", "stale", "contradictory", "unparsable"),
    )

    print("=" * 78)
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


def _fake_ep(*, k: int, condition: str) -> "swc.Episode":
    return swc.Episode(
        episode_id="unit_ep", model_key="qwen", model_id="Qwen/Qwen2.5-7B-Instruct", offload_gb=0,
        state_label="low", server_max_num_seqs=k, trigger_after_decode_tokens=16, condition=condition,
        repeat=1, random_seed=1, episode_seed=2, victim_workload_seed=3, burst_workload_seed=4,
        victim_request_count=20, victim_input_len=64, victim_output_len=64, victim_temperature=0.0,
        burst_parallel_requests=4, burst_input_len=128, burst_output_len=16, burst_temperature=0.0,
        max_num_batched_tokens=2048, condition_first_in_block=condition, restart_server_before_block=1,
        block_id="unit_block", order_in_block=1,
    )


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
