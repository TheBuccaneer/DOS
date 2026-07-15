"""
Stage 2 (request/trigger/stabilization/smoke-block) self-test checks,
mechanically moved out of run_phase_a.py's former "Self-test" section,
verbatim -- no logic changes.
"""

from __future__ import annotations

from run_phase_a import *  # noqa: F401,F403
from run_phase_a import _watch_trigger  # noqa: F401

from phase_a_tests.fixtures import (
    _build_fixture_episodes,
    _make_fixture_block_bundle,
    _make_success_transport,
)


async def _stage2_async_checks() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))

    fake_clock = RealClock()

    # --- 1. Token-ID prompt has exactly 256 ids -----------------------------
    tok = FakeTokenizerAdapter(vocab_size=300, special_token_ids={0, 1, 2})
    valid_ids = compute_valid_token_ids(tok)
    p256 = generate_token_id_prompt(seed=1, valid_ids=valid_ids, length=256)
    check("(1) token-id prompt has exactly 256 ids", len(p256) == 256)

    # --- 2. Special ids are never chosen -------------------------------------
    tok2 = FakeTokenizerAdapter(vocab_size=50, special_token_ids={0, 1, 2, 3, 4})
    valid_ids2 = compute_valid_token_ids(tok2)
    p_many = generate_token_id_prompt(seed=7, valid_ids=valid_ids2, length=2000)
    check(
        "(2) special token ids are never chosen",
        all(x not in {0, 1, 2, 3, 4} for x in p_many),
    )

    # --- 3. Same seed -> identical prompt-id lists ---------------------------
    a1 = generate_token_id_prompt(seed=42, valid_ids=valid_ids, length=256)
    a2 = generate_token_id_prompt(seed=42, valid_ids=valid_ids, length=256)
    a3 = generate_token_id_prompt(seed=43, valid_ids=valid_ids, length=256)
    check("(3) identical seeds produce identical prompt-id lists", a1 == a2 and a1 != a3)

    # --- 4/5. Matched low/high episodes share victim/burst sequences --------
    fixture_seed = 555001
    fixture_eps = _build_fixture_episodes("matchtest", fixture_seed)
    by_key: dict[tuple[int, str, int], list[Episode]] = {}
    for ep in fixture_eps:
        by_key.setdefault((ep.concurrency, ep.condition, ep.repeat), []).append(ep)
    matched_pairs = [v for v in by_key.values() if len(v) == 2]
    check("(4/5 setup) fixture has matched low/high pairs", len(matched_pairs) > 0)
    victim_match_ok = True
    burst_match_ok = True
    for pair in matched_pairs:
        low_ep, high_ep = pair[0], pair[1]
        for i in range(3):
            if victim_prompt_seed(low_ep, i) != victim_prompt_seed(high_ep, i):
                victim_match_ok = False
            if victim_generation_seed(low_ep, i) != victim_generation_seed(high_ep, i):
                victim_match_ok = False
            if burst_prompt_seed(low_ep, i) != burst_prompt_seed(high_ep, i):
                burst_match_ok = False
            if burst_generation_seed(low_ep, i) != burst_generation_seed(high_ep, i):
                burst_match_ok = False
        low_victims = [
            generate_token_id_prompt(victim_prompt_seed(low_ep, i), valid_ids, 8) for i in range(3)
        ]
        high_victims = [
            generate_token_id_prompt(victim_prompt_seed(high_ep, i), valid_ids, 8) for i in range(3)
        ]
        if low_victims != high_victims:
            victim_match_ok = False
        low_bursts = [
            generate_token_id_prompt(burst_prompt_seed(low_ep, j), valid_ids, 8) for j in range(2)
        ]
        high_bursts = [
            generate_token_id_prompt(burst_prompt_seed(high_ep, j), valid_ids, 8) for j in range(2)
        ]
        if low_bursts != high_bursts:
            burst_match_ok = False
    check("(4) matched low/high episodes produce identical victim sequences", victim_match_ok)
    check("(5) matched low/high episodes produce identical burst sequences", burst_match_ok)

    # --- 6. Stabilization uses its own seed domain ---------------------------
    stab_p_seed = stabilization_prompt_seed(fixture_seed, "matchtest", "matchtest_block01_low", 0)
    victim_p_seed_same_i = victim_prompt_seed(fixture_eps[0], 0)
    expected_stab_seed = derive_seed(
        str(fixture_seed), "matchtest", "matchtest_block01_low", "stabilization-prompt", "0"
    )
    check(
        "(6) stabilization_prompt_seed matches its own documented derivation "
        "and differs from the victim-prompt seed domain",
        stab_p_seed == expected_stab_seed and stab_p_seed != victim_p_seed_same_i,
    )

    # --- 7-13. Server-side completeness validation ---------------------------
    async def _exec(script: FakeStreamScript, expected_prompt: int = 10, expected_completion: int = 4) -> dict:
        t = FakeTransport()
        t.queue_script("t713", script)
        return await execute_completion_request(
            transport=t, clock=fake_clock, url="http://x/v1/completions",
            api_key="k", model_full_id="m", prompt_token_ids=list(range(expected_prompt)),
            max_tokens=expected_completion, min_tokens=expected_completion, temperature=0.0,
            request_seed=1, request_id="t713", role="victim", request_index=0,
            prompt_seed=1, generation_seed=1, expected_prompt_tokens=expected_prompt,
            expected_completion_tokens=expected_completion, http_timeout_s=5.0,
        )

    r7 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4},
    ))
    check(
        "(7) matching server-side prompt ids -> complete",
        r7["status"] == REQUEST_STATUS_COMPLETE, str(r7["validation_errors"]),
    )

    r8 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=[99] * 10, token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4},
    ))
    check("(8) mismatched server-side prompt ids -> incomplete", r8["status"] == REQUEST_STATUS_INCOMPLETE)

    r9 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 999, "completion_tokens": 4},
    ))
    check("(9) wrong usage.prompt_tokens -> incomplete", r9["status"] == REQUEST_STATUS_INCOMPLETE)

    r10 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 999},
    ))
    check("(10) wrong usage.completion_tokens -> incomplete", r10["status"] == REQUEST_STATUS_INCOMPLETE)

    r11 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3]],
        usage={"prompt_tokens": 10, "completion_tokens": 3},
    ))
    check("(11) too-short output-id list -> incomplete", r11["status"] == REQUEST_STATUS_INCOMPLETE)

    r12 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4}, include_done=False,
    ))
    check("(12) missing [DONE] -> incomplete", r12["status"] == REQUEST_STATUS_INCOMPLETE)

    r13 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4}, finish_reason="stop",
    ))
    check("(13) wrong finish_reason -> incomplete", r13["status"] == REQUEST_STATUS_INCOMPLETE)

    # --- 14/15. ITL availability rule -----------------------------------------
    r14 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2], [3], [4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4},
    ))
    check(
        "(14) one token per event -> itl_available=true",
        r14["itl_available"] is True and r14["itl_ms"] is not None,
    )

    r15 = await _exec(FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1, 2], [3, 4]],
        usage={"prompt_tokens": 10, "completion_tokens": 4},
    ))
    check(
        "(15) multiple tokens in one event -> itl_available=false",
        r15["itl_available"] is False and r15["itl_ms"] is None and r15["token_batch_sizes"] == [2, 2],
    )

    # --- 16/17. TPOT ends at last token event (not [DONE]); E2EL ends at stream end
    step_clock = FakeClock(step_ns=1_000_000)  # 1ms advance per clock call
    t1617 = FakeTransport()
    t1617.queue_script("t1617", FakeStreamScript(
        prompt_token_ids_echo=list(range(10)), token_events=[[1], [2]],
        usage={"prompt_tokens": 10, "completion_tokens": 2}, extra_keepalives=3,
    ))
    r1617 = await execute_completion_request(
        transport=t1617, clock=step_clock, url="http://x/v1/completions",
        api_key="k", model_full_id="m", prompt_token_ids=list(range(10)),
        max_tokens=2, min_tokens=2, temperature=0.0, request_seed=1,
        request_id="t1617", role="victim", request_index=0, prompt_seed=1,
        generation_seed=1, expected_prompt_tokens=10, expected_completion_tokens=2,
        http_timeout_s=5.0,
    )
    manual_tpot_ms = (r1617["last_token_receive_ns"] - r1617["first_token_receive_ns"]) / 1e6
    check(
        "(16) client-observed TPOT is derived from first/last token "
        "timestamps, not from [DONE]",
        r1617["client_observed_tpot_ms"] == manual_tpot_ms
        and r1617["last_token_receive_ns"] < r1617["stream_end_ns"],
    )
    check(
        "(17) E2EL is derived from stream end, which is later than the last token event",
        r1617["e2el_ms"] == (r1617["stream_end_ns"] - r1617["request_start_ns"]) / 1e6
        and r1617["e2el_ms"] > manual_tpot_ms,
    )

    # --- 18. Trigger fires only after every first-wave request's first token
    ev0, ev1 = asyncio.Event(), asyncio.Event()
    order: list[str] = []

    async def _fast_set() -> str:
        ev0.set()
        order.append("fast")
        await asyncio.Event().wait()  # hang forever (real task keeps running)

    async def _slow_set() -> str:
        await asyncio.sleep(0.05)
        ev1.set()
        order.append("slow")
        await asyncio.Event().wait()

    t_fast = asyncio.create_task(_fast_set())
    t_slow = asyncio.create_task(_slow_set())
    events18 = {0: ev0, 1: ev1}
    tasks18 = {0: t_fast, 1: t_slow}
    start18 = time.monotonic()
    status18 = await _watch_trigger({0, 1}, events18, tasks18, timeout_s=2.0)
    elapsed18 = time.monotonic() - start18
    await cancel_all([t_fast, t_slow])
    check(
        "(18) trigger only fires once every first-wave request's first "
        "token has arrived (waits for the slow one)",
        status18 == "ok" and elapsed18 >= 0.04 and order == ["fast", "slow"],
    )

    # --- 19. Trigger timeout cancels all tasks --------------------------------
    async def _hangs_forever() -> None:
        await asyncio.Event().wait()

    ev_a, ev_b = asyncio.Event(), asyncio.Event()
    t_a = asyncio.create_task(_hangs_forever())
    t_b = asyncio.create_task(_hangs_forever())
    status19 = await _watch_trigger(
        {0, 1}, {0: ev_a, 1: ev_b}, {0: t_a, 1: t_b}, timeout_s=0.05
    )
    await cancel_all([t_a, t_b])
    check(
        "(19) trigger timeout is reported and both first-wave tasks can "
        "then be cancelled",
        status19 == "timeout" and t_a.done() and t_b.done(),
    )

    # --- 20. No request task keeps running after cancellation ---------------
    t20 = FakeTransport()
    t20.default_script_factory = lambda payload: FakeStreamScript(hang=True, prompt_token_ids_echo=None, token_events=[])

    async def _hanging_request(i: int) -> dict:
        return await execute_completion_request(
            transport=t20, clock=fake_clock, url="http://x", api_key="k", model_full_id="m",
            prompt_token_ids=[1, 2, 3], max_tokens=4, min_tokens=4, temperature=0.0,
            request_seed=1, request_id=f"hang{i}", role="victim", request_index=i,
            prompt_seed=1, generation_seed=1, expected_prompt_tokens=3,
            expected_completion_tokens=4, http_timeout_s=30.0,
        )

    hang_tasks = [asyncio.create_task(_hanging_request(i)) for i in range(4)]
    await asyncio.sleep(0.02)
    active_before = t20.active_stream_count
    await cancel_all(hang_tasks)
    check(
        "(20) after cancel_all(), no fake stream is still active and all "
        "tasks are done",
        active_before == 4 and t20.active_stream_count == 0 and all(t.done() for t in hang_tasks),
    )

    # --- 21/22. Burst starts only after trigger; no_burst issues no bursts ---
    tok21 = FakeTokenizerAdapter(vocab_size=2000, special_token_ids={0, 1, 2})
    valid21 = compute_valid_token_ids(tok21)
    transport21 = _make_success_transport()
    ctx21 = RunContext(
        transport=transport21, clock=RealClock(), sleeper=FakeSleeper(),
        base_url="http://127.0.0.1:1", api_key="k21", model_full_id="fake/model",
        valid_ids=valid21, trigger_timeout_s=5.0,
    )
    ep21 = Episode(
        episode_id="burst_after_trigger_ep", model="llama", offload_gb=0, state_label="low",
        concurrency=4, condition="fixed_burst", repeat=1, random_seed=1, episode_seed=1,
        victim_workload_seed=111, burst_workload_seed=222, victim_request_count=8,
        victim_input_len=16, victim_output_len=4, victim_temperature=0.0,
        burst_parallel_requests=2, burst_input_len=16, burst_output_len=4,
        burst_temperature=0.0, restart_server_before_block=1, block_id="fake_block21",
        order_in_block=1,
    )
    result21 = await run_regular_episode(
        ctx21, ep21, schedule_fingerprint="sha256:" + "0" * 64,
        server_metadata={}, stabilization_ref={},
    )
    burst_starts_after_trigger = (
        result21["burst_interval"] is not None
        and result21["burst_interval"]["start_ns"] >= result21["trigger"]["trigger_perf_ns"]
    )
    check(
        "(21) burst requests only start once the trigger has fired",
        result21["status"] == REQUEST_STATUS_COMPLETE and burst_starts_after_trigger,
        str(result21.get("validation_errors")),
    )

    ep22 = Episode(**{**vars(ep21), "condition": "no_burst", "episode_id": "no_burst_ep22"})
    result22 = await run_regular_episode(
        ctx21, ep22, schedule_fingerprint="sha256:" + "0" * 64,
        server_metadata={}, stabilization_ref={},
    )
    check(
        "(22) no_burst issues zero burst requests",
        result22["burst_requests"] == [] and result22["burst_interval"] is None,
    )

    # --- 23/24/25/29/30/31/32/33/34: full run_smoke_block scenarios ---------
    import tempfile as _tempfile

    tok_block = FakeTokenizerAdapter(vocab_size=2000, special_token_ids={0, 1, 2})
    secret_key = "self-test-secret-key-should-never-leak"
    run_server_path_fixture = Path("/nonexistent/run_server.sh")

    # (23) partial stabilization prevents every regular episode.
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle23, block_id23 = _make_fixture_block_bundle("llama", 700001)
        t23 = _make_success_transport()
        t23.queue_script(
            f"{block_id23}:stabilization:3",
            FakeStreamScript(
                prompt_token_ids_echo=list(range(256)), token_events=[[1]] * 64,
                finish_reason="stop", usage={"prompt_tokens": 256, "completion_tokens": 64},
            ),
        )
        server_adapter23 = FakeServerProcessAdapter()
        summary23 = await run_smoke_block(
            bundle=bundle23, block_id=block_id23, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18101, resume=False, api_key=secret_key,
            transport=t23, tokenizer=tok_block, server_adapter=server_adapter23,
            sleeper=FakeSleeper(), clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        episodes_written = list((tmp_path / "out" / "episodes").glob("*.json")) if (tmp_path / "out" / "episodes").exists() else []
        check(
            "(23) partial stabilization prevents every regular episode from running",
            summary23["overall_status"] == "stabilization_failed" and not episodes_written,
            str(summary23.get("overall_status")),
        )
        check("(31, block on stab-fail) episode dir stays empty under episodes/", not episodes_written)
        check(
            "(32) stabilization output is written under stabilization/",
            (tmp_path / "out" / "stabilization" / f"{block_id23}.json").exists(),
        )
        check(
            "(34a) API key never appears in the smoke summary",
            secret_key not in json.dumps(summary23),
        )

    # (24) drift alone does not block the episodes (abort_on_stability_drift=False).
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle24, block_id24 = _make_fixture_block_bundle("llama", 700002)
        t24 = _make_success_transport()
        valid_ids24 = compute_valid_token_ids(tok_block)
        # First half: 1 token/event (fast). Second half: dramatically more
        # elapsed wall time via extra keepalives -- still functionally
        # complete, but the two halves' timing looks very different.
        for i in range(10, 20):
            p_seed24 = stabilization_prompt_seed(bundle24.json_obj["seed"], "llama", block_id24, i)
            prompt_ids24 = generate_token_id_prompt(p_seed24, valid_ids24, STABILIZATION_INPUT_LEN)
            t24.queue_script(
                f"{block_id24}:stabilization:{i}",
                FakeStreamScript(
                    prompt_token_ids_echo=prompt_ids24, token_events=[[1]] * 64,
                    usage={"prompt_tokens": 256, "completion_tokens": 64},
                    extra_keepalives=25,
                ),
            )
        server_adapter24 = FakeServerProcessAdapter()
        summary24 = await run_smoke_block(
            bundle=bundle24, block_id=block_id24, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18102, resume=False, api_key=secret_key,
            transport=t24, tokenizer=tok_block, server_adapter=server_adapter24,
            sleeper=FakeSleeper(), clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        stab24 = json.loads((tmp_path / "out" / "stabilization" / f"{block_id24}.json").read_text())
        check(
            "(24) functional/stabilization pass even with large timing drift "
            "between halves (abort_on_stability_drift=False)",
            stab24["functional_passed"] is True and stab24["stabilization_passed"] is True,
            str([r["validation_errors"] for r in stab24["request_results"] if r["status"] != "complete"]),
        )
        check(
            "(24b) drift is documented, not gating: episodes still ran",
            summary24["overall_status"] == "block_complete",
        )


    # (25) a partial episode prevents the next episode of the block.
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle25, block_id25 = _make_fixture_block_bundle("llama", 700003)
        block_eps25 = find_block(bundle25, block_id25)
        second_ep25 = block_eps25[1]
        t25 = _make_success_transport()
        t25.queue_script(
            f"{second_ep25.episode_id}:victim:0",
            FakeStreamScript(
                prompt_token_ids_echo=list(range(second_ep25.victim_input_len)),
                token_events=[[1]] * (second_ep25.victim_output_len - 1),
                usage={
                    "prompt_tokens": second_ep25.victim_input_len,
                    "completion_tokens": second_ep25.victim_output_len - 1,
                },
            ),
        )
        server_adapter25 = FakeServerProcessAdapter()
        summary25 = await run_smoke_block(
            bundle=bundle25, block_id=block_id25, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18103, resume=False, api_key=secret_key,
            transport=t25, tokenizer=tok_block, server_adapter=server_adapter25,
            sleeper=FakeSleeper(), clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        st25 = summary25["episode_statuses"]
        check(
            "(25) a partial episode prevents the next episode(s) of the "
            "block from starting",
            st25[block_eps25[0].episode_id] == CLASSIFICATION_VALID_COMPLETE
            and st25[block_eps25[1].episode_id] == CLASSIFICATION_PARTIAL
            and st25[block_eps25[2].episode_id] == CLASSIFICATION_MISSING
            and st25[block_eps25[3].episode_id] == CLASSIFICATION_MISSING
            and not episode_result_path(tmp_path / "out", block_eps25[2].episode_id).exists(),
            str(st25),
        )
        # Per section 22, a 'partial' file is exactly as non-resumable as
        # 'invalid'/'corrupted' -- it must gate resume with a clear abort,
        # not be silently rerun.
        server_adapter25_resume = FakeServerProcessAdapter()
        raised25b = False
        try:
            await run_smoke_block(
                bundle=bundle25, block_id=block_id25, output_dir=tmp_path / "out",
                host="127.0.0.1", port=18104, resume=True, api_key=secret_key,
                transport=_make_success_transport(), tokenizer=tok_block,
                server_adapter=server_adapter25_resume, sleeper=FakeSleeper(), clock=RealClock(),
                run_server_path=run_server_path_fixture,
            )
        except ServerLifecycleError:
            raised25b = True
        check(
            "(25b) a leftover 'partial' episode file also gates --resume "
            "with a clear abort (never silently rerun)",
            raised25b and len(server_adapter25_resume.started) == 0,
        )

    # (29) --resume begins at the first genuinely missing episode, restarts
    # the server, and leaves already-valid_complete episodes untouched.
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle29, block_id29 = _make_fixture_block_bundle("llama", 700005)
        block_eps29 = find_block(bundle29, block_id29)
        server_adapter29a = FakeServerProcessAdapter()
        summary29a = await run_smoke_block(
            bundle=bundle29, block_id=block_id29, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18107, resume=False, api_key=secret_key,
            transport=_make_success_transport(), tokenizer=tok_block,
            server_adapter=server_adapter29a, sleeper=FakeSleeper(), clock=RealClock(),
            run_server_path=run_server_path_fixture,
        )
        check("(29 setup) initial full run completes", summary29a["overall_status"] == "block_complete")

        kept_path = episode_result_path(tmp_path / "out", block_eps29[0].episode_id)
        kept_text = kept_path.read_text()
        for ep in block_eps29[1:]:
            episode_result_path(tmp_path / "out", ep.episode_id).unlink()

        server_adapter29b = FakeServerProcessAdapter()
        summary29b = await run_smoke_block(
            bundle=bundle29, block_id=block_id29, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18108, resume=True, api_key=secret_key,
            transport=_make_success_transport(), tokenizer=tok_block,
            server_adapter=server_adapter29b, sleeper=FakeSleeper(), clock=RealClock(),
            run_server_path=run_server_path_fixture,
        )
        check(
            "(29) --resume begins at the first missing episode, restarts "
            "the server (stabilization is mandatory again), and leaves "
            "the already-valid_complete episode's file byte-for-byte untouched",
            summary29b["overall_status"] == "block_complete"
            and len(server_adapter29b.started) == 1
            and kept_path.read_text() == kept_text
            and all(
                v == CLASSIFICATION_VALID_COMPLETE for v in summary29b["episode_statuses"].values()
            ),
            str(summary29b["episode_statuses"]),
        )

    # (30) a foreign/invalid result file must never be silently overwritten,
    # and must gate resume before any server is started.
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle30, block_id30 = _make_fixture_block_bundle("llama", 700006)
        block_eps30 = find_block(bundle30, block_id30)
        bad_path = episode_result_path(tmp_path / "out", block_eps30[0].episode_id)
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_payload = json.dumps({"runner_version": "not-the-real-runner"})
        bad_path.write_text(bad_payload, encoding="utf-8")
        server_adapter30 = FakeServerProcessAdapter()
        raised30 = False
        try:
            await run_smoke_block(
                bundle=bundle30, block_id=block_id30, output_dir=tmp_path / "out",
                host="127.0.0.1", port=18109, resume=True, api_key=secret_key,
                transport=_make_success_transport(), tokenizer=tok_block,
                server_adapter=server_adapter30, sleeper=FakeSleeper(), clock=RealClock(),
                run_server_path=run_server_path_fixture,
            )
        except ServerLifecycleError:
            raised30 = True
        check(
            "(30) a foreign/invalid episode result file is rejected, not "
            "silently overwritten, and no server is started",
            raised30 and len(server_adapter30.started) == 0 and bad_path.read_text() == bad_payload,
        )

    # (31/32/33/34 full happy path) episodes/ + stabilization/ dirs, no temp
    # files after success, API key never leaks anywhere on disk.
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle31, block_id31 = _make_fixture_block_bundle("llama", 700004)
        block_eps31 = find_block(bundle31, block_id31)
        server_adapter31 = FakeServerProcessAdapter()
        summary31 = await run_smoke_block(
            bundle=bundle31, block_id=block_id31, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18106, resume=False, api_key=secret_key,
            transport=_make_success_transport(), tokenizer=tok_block,
            server_adapter=server_adapter31, sleeper=FakeSleeper(), clock=RealClock(),
            run_server_path=run_server_path_fixture,
        )
        episodes_ok = all(
            episode_result_path(tmp_path / "out", ep.episode_id).exists() for ep in block_eps31
        )
        check("(31) episode files live under episodes/", episodes_ok and summary31["overall_status"] == "block_complete")
        check(
            "(32, happy path) stabilization file lives under stabilization/",
            stabilization_result_path(tmp_path / "out", block_id31).exists(),
        )
        leftovers = list((tmp_path / "out").rglob("*.tmp.*"))
        check("(33) the atomic writer leaves no temp file behind after success", not leftovers, str(leftovers))
        all_text = "".join(p.read_text() for p in (tmp_path / "out").rglob("*.json"))
        check(
            "(34) the API key never appears in any serialized result on disk",
            secret_key not in all_text and secret_key not in json.dumps(summary31),
        )
        check(
            "(28) stop_server only signals the server's own process "
            "group (SIGTERM path), never a global kill",
            server_adapter31.started[0].terminated and not server_adapter31.started[0].killed,
        )

    # --- 26/27. Server command shape -----------------------------------------
    cmd = build_server_command(Path("/x/run_server.sh"), "llama", 12, "127.0.0.1", 8123)
    check(
        "(26) the server command is exactly `bash run_server.sh <model> "
        "<offload_gb> <host> <port>`",
        cmd == ["bash", "/x/run_server.sh", "llama", "12", "127.0.0.1", "8123"],
    )
    check(
        "(27) the server command never contains an API key",
        all("secret" not in part.lower() and "key" not in part.lower() for part in cmd),
    )

    # =========================================================================
    # Patch: Stage-2-Realpfad-Abschluss -- sections 1, 2, 3, 5, 6, 7
    # =========================================================================

    # --- Section 1: readiness polls through transient connection errors ----
    # (P1-1) two ConnectionRefusedError, then a successful health/models check.
    t_p1a = FakeTransport()
    t_p1a.queue_get_error(HEALTH_ENDPOINT, ConnectionRefusedError("refused"))
    t_p1a.queue_get_error(HEALTH_ENDPOINT, ConnectionRefusedError("refused"))
    t_p1a.set_get_response(HEALTH_ENDPOINT, 200, {})
    t_p1a.set_get_response(MODELS_ENDPOINT, 200, {"data": [{"id": "fake/model"}]})
    handle_p1a = FakeServerHandle(["bash", "x"])
    readiness_p1a = await wait_for_server_ready(
        t_p1a, handle_p1a, "http://x", "k", "fake/model", FakeSleeper(),
        timeout_s=5.0, poll_interval_s=0.001,
    )
    check(
        "(P1-1) readiness polls through two transient ConnectionRefusedError "
        "and then succeeds",
        readiness_p1a["detected_model"] == "fake/model" and readiness_p1a["poll_count"] >= 3,
        str(readiness_p1a),
    )

    # (P1-2) health 200 immediately; models 503 then 200.
    t_p1b = FakeTransport()
    t_p1b.set_get_response(HEALTH_ENDPOINT, 200, {})
    t_p1b.queue_get_status(MODELS_ENDPOINT, 503, {})
    t_p1b.set_get_response(MODELS_ENDPOINT, 200, {"data": [{"id": "fake/model"}]})
    handle_p1b = FakeServerHandle(["bash", "x"])
    readiness_p1b = await wait_for_server_ready(
        t_p1b, handle_p1b, "http://x", "k", "fake/model", FakeSleeper(),
        timeout_s=5.0, poll_interval_s=0.001,
    )
    check(
        "(P1-2) readiness polls through a transient /v1/models 503 and then succeeds",
        readiness_p1b["detected_model"] == "fake/model",
    )

    # (P1-3) server process dies mid-poll -> a clear error.
    class _DyingSleeper:
        def __init__(self, handle: FakeServerHandle) -> None:
            self.handle = handle

        async def sleep(self, seconds: float) -> None:
            self.handle.alive = False
            await asyncio.sleep(0)

    t_p1c = FakeTransport()
    t_p1c.queue_get_error(HEALTH_ENDPOINT, ConnectionRefusedError("refused"))
    handle_p1c = FakeServerHandle(["bash", "x"])
    raised_p1c = False
    try:
        await wait_for_server_ready(
            t_p1c, handle_p1c, "http://x", "k", "fake/model", _DyingSleeper(handle_p1c),
            timeout_s=5.0, poll_interval_s=0.001,
        )
    except ServerLifecycleError:
        raised_p1c = True
    check("(P1-3) the server process dying mid-poll raises a clear ServerLifecycleError", raised_p1c)

    # (P1-4) only transient errors until the deadline -> a clean timeout.
    class _AlwaysFailTransport:
        async def get_json(self, url: str, headers: dict, timeout_s: float):
            raise ConnectionRefusedError("refused")

    handle_p1d = FakeServerHandle(["bash", "x"])
    raised_p1d = False
    try:
        await wait_for_server_ready(
            _AlwaysFailTransport(), handle_p1d, "http://x", "k", "fake/model", FakeSleeper(),
            timeout_s=0.05, poll_interval_s=0.001,
        )
    except ServerLifecycleError:
        raised_p1d = True
    check("(P1-4) only-transient errors until the deadline -> a clean ServerLifecycleError timeout", raised_p1d)

    # (P1-5) cancellation is never swallowed.
    class _HangingTransport:
        async def get_json(self, url: str, headers: dict, timeout_s: float):
            await asyncio.Event().wait()

    handle_p1e = FakeServerHandle(["bash", "x"])
    readiness_task = asyncio.create_task(
        wait_for_server_ready(
            _HangingTransport(), handle_p1e, "http://x", "k", "fake/model", FakeSleeper(),
            timeout_s=30.0, poll_interval_s=1.0,
        )
    )
    await asyncio.sleep(0.02)
    readiness_task.cancel()
    cancelled_p1e = False
    try:
        await readiness_task
    except asyncio.CancelledError:
        cancelled_p1e = True
    check("(P1-5) readiness cancellation (asyncio.CancelledError) is never swallowed", cancelled_p1e)

    # --- Section 2: post-stabilization health gate --------------------------
    t_health_ok = FakeTransport()
    t_health_ok.set_get_response(HEALTH_ENDPOINT, 200, {})
    result_health_ok = await check_post_stabilization_health(t_health_ok, "http://x")
    check(
        "(P2-1) post-stabilization health check: HTTP 200 -> ok=True",
        result_health_ok["ok"] is True and result_health_ok["http_status"] == 200,
    )

    t_health_503 = FakeTransport()
    t_health_503.set_get_response(HEALTH_ENDPOINT, 503, {})
    result_health_503 = await check_post_stabilization_health(t_health_503, "http://x")
    check(
        "(P2-2) post-stabilization health check: HTTP 503 -> ok=False",
        result_health_503["ok"] is False and result_health_503["http_status"] == 503,
    )

    class _RaisingHealthTransport:
        async def get_json(self, url: str, headers: dict, timeout_s: float):
            raise ConnectionRefusedError("refused")

    result_health_exc = await check_post_stabilization_health(_RaisingHealthTransport(), "http://x")
    check(
        "(P2-3) post-stabilization health check: connection exception -> "
        "ok=False, no crash",
        result_health_exc["ok"] is False and result_health_exc["error_type"] == "ConnectionRefusedError",
    )

    class _CountingHealthTransport:
        """Wraps a base FakeTransport: /health returns 200 for the first
        `pass_count` calls (covering readiness's own polling), then
        `fail_status`/`fail_exc` for every call after that (covering the
        post-stabilization gate specifically)."""

        def __init__(self, base: "FakeTransport", pass_count: int, fail_status: int | None = None, fail_exc: BaseException | None = None) -> None:
            self.base = base
            self.pass_count = pass_count
            self.fail_status = fail_status
            self.fail_exc = fail_exc
            self.health_calls = 0

        async def get_json(self, url: str, headers: dict, timeout_s: float):
            if url.endswith(HEALTH_ENDPOINT):
                self.health_calls += 1
                if self.health_calls > self.pass_count:
                    if self.fail_exc is not None:
                        raise self.fail_exc
                    return self.fail_status, {}
                return 200, {}
            return await self.base.get_json(url, headers, timeout_s)

        async def stream_completion(self, *a, **kw):
            async for x in self.base.stream_completion(*a, **kw):
                yield x

    # (P2-4) health 200 after stabilization -> cooldown happens, episodes run.
    bundle_p2a, block_id_p2a = _make_fixture_block_bundle("llama", 800101)
    t_p2a = _CountingHealthTransport(_make_success_transport(), pass_count=1, fail_status=200)
    server_adapter_p2a = FakeServerProcessAdapter()
    sleeper_p2a = FakeSleeper()
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary_p2a = await run_smoke_block(
            bundle=bundle_p2a, block_id=block_id_p2a, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18301, resume=False, api_key="k",
            transport=t_p2a, tokenizer=tok_block, server_adapter=server_adapter_p2a,
            sleeper=sleeper_p2a, clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        check(
            "(P2-4) health 200 after stabilization -> cooldown happens and episodes run",
            summary_p2a["overall_status"] == "block_complete"
            and summary_p2a["readiness"]["post_stabilization_health"]["ok"] is True
            and COOLDOWN_S in sleeper_p2a.calls,
            str(summary_p2a.get("overall_status")),
        )

    # (P2-5) health 503 after stabilization -> no episode runs.
    bundle_p2b, block_id_p2b = _make_fixture_block_bundle("llama", 800102)
    t_p2b = _CountingHealthTransport(_make_success_transport(), pass_count=1, fail_status=503)
    server_adapter_p2b = FakeServerProcessAdapter()
    sleeper_p2b = FakeSleeper()
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary_p2b = await run_smoke_block(
            bundle=bundle_p2b, block_id=block_id_p2b, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18302, resume=False, api_key="k",
            transport=t_p2b, tokenizer=tok_block, server_adapter=server_adapter_p2b,
            sleeper=sleeper_p2b, clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        episodes_dir_p2b = tmp_path / "out" / "episodes"
        episodes_written_p2b = list(episodes_dir_p2b.glob("*.json")) if episodes_dir_p2b.exists() else []
        check(
            "(P2-5) health 503 after stabilization -> "
            "post_stabilization_health_failed, no episode runs, no cooldown",
            summary_p2b["overall_status"] == "post_stabilization_health_failed"
            and not episodes_written_p2b
            and COOLDOWN_S not in sleeper_p2b.calls,
        )

    # (P2-6) connection exception at the post-stabilization health check.
    bundle_p2c, block_id_p2c = _make_fixture_block_bundle("llama", 800103)
    t_p2c = _CountingHealthTransport(
        _make_success_transport(), pass_count=1, fail_exc=ConnectionRefusedError("refused")
    )
    server_adapter_p2c = FakeServerProcessAdapter()
    sleeper_p2c = FakeSleeper()
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary_p2c = await run_smoke_block(
            bundle=bundle_p2c, block_id=block_id_p2c, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18303, resume=False, api_key="k",
            transport=t_p2c, tokenizer=tok_block, server_adapter=server_adapter_p2c,
            sleeper=sleeper_p2c, clock=RealClock(), run_server_path=run_server_path_fixture,
        )
        episodes_dir_p2c = tmp_path / "out" / "episodes"
        episodes_written_p2c = list(episodes_dir_p2c.glob("*.json")) if episodes_dir_p2c.exists() else []
        check(
            "(P2-6) a connection exception at the post-stabilization health "
            "check does not crash and prevents every episode",
            summary_p2c["overall_status"] == "post_stabilization_health_failed" and not episodes_written_p2c,
        )

    # --- Section 3: robust, verified server stop ----------------------------
    def _always_free(host: str, port: int) -> bool:
        return True

    def _always_occupied(host: str, port: int) -> bool:
        return False

    handle_p3a = FakeServerHandle(["bash", "x"], already_dead=True)
    stop_p3a = await stop_server(handle_p3a, "127.0.0.1", 1, FakeSleeper(), port_free_check=_always_free)
    check(
        "(P3-1) an already-dead process gets no unnecessary signal, "
        "stop_success=True",
        stop_p3a["term_sent"] is False and not handle_p3a.terminated and stop_p3a["stop_success"] is True,
    )

    handle_p3b = FakeServerHandle(["bash", "x"], raise_on_terminate=ProcessLookupError())
    stop_p3b = await stop_server(handle_p3b, "127.0.0.1", 1, FakeSleeper(), port_free_check=_always_free)
    check(
        "(P3-2) ProcessLookupError on SIGTERM is handled cleanly (no crash, "
        "stop_success=True, no stop_error)",
        stop_p3b["stop_success"] is True and stop_p3b["stop_error"] is None,
    )

    handle_p3c = FakeServerHandle(["bash", "x"])
    stop_p3c = await stop_server(handle_p3c, "127.0.0.1", 1, FakeSleeper(), port_free_check=_always_free)
    check(
        "(P3-3) process dies + port confirmed free -> stop_success=True",
        stop_p3c["stop_success"] is True and stop_p3c["alive_after_stop"] is False,
    )

    handle_p3d = FakeServerHandle(["bash", "x"], dies_on_terminate=False, dies_on_kill=False)
    stop_p3d = await stop_server(
        handle_p3d, "127.0.0.1", 1, FakeSleeper(),
        timeout_s=0.01, kill_confirm_timeout_s=0.01, port_free_check=_always_free,
    )
    check(
        "(P3-4) a process that survives SIGKILL -> stop_success=False, "
        "forced_kill=True, alive_after_stop=True",
        stop_p3d["stop_success"] is False
        and stop_p3d["forced_kill"] is True
        and stop_p3d["alive_after_stop"] is True,
    )

    handle_p3e = FakeServerHandle(["bash", "x"])
    stop_p3e = await stop_server(
        handle_p3e, "127.0.0.1", 1, FakeSleeper(),
        port_free_check=_always_occupied, port_poll_timeout_s=0.01,
    )
    check(
        "(P3-5) process dies but the port stays occupied -> stop_success=False",
        stop_p3e["stop_success"] is False
        and stop_p3e["alive_after_stop"] is False
        and stop_p3e["port_free_after_stop"] is False,
    )

    class _StuckServerAdapter:
        def __init__(self) -> None:
            self.started: list[FakeServerHandle] = []

        def start(self, cmd: list[str], log_path: Path) -> FakeServerHandle:
            h = FakeServerHandle(cmd, dies_on_terminate=False, dies_on_kill=False)
            self.started.append(h)
            return h

    bundle_p3f, block_id_p3f = _make_fixture_block_bundle("llama", 800201)
    adapter_p3f = _StuckServerAdapter()
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary_p3f = await run_smoke_block(
            bundle=bundle_p3f, block_id=block_id_p3f, output_dir=tmp_path / "out",
            host="127.0.0.1", port=18310, resume=False, api_key="k",
            transport=_make_success_transport(), tokenizer=tok_block,
            server_adapter=adapter_p3f, sleeper=FakeSleeper(), clock=RealClock(),
            run_server_path=run_server_path_fixture,
            stop_timeout_s=0.01, stop_kill_confirm_timeout_s=0.01, stop_port_poll_timeout_s=0.01,
        )
        check(
            "(P3-6) a block that otherwise finished cleanly is downgraded to "
            "'server_stop_failed' when the server process never actually stops",
            summary_p3f["overall_status"] == "server_stop_failed"
            and summary_p3f["server_stop"]["stop_success"] is False,
            str(summary_p3f.get("overall_status")),
        )

    handle_p3g = FakeServerHandle(["bash", "x"])
    await stop_server(handle_p3g, "127.0.0.1", 1, FakeSleeper(), port_free_check=_always_free)
    check(
        "(P3-7) stop_server only ever signals the handle's own PGID "
        "(terminate_group()/kill_group() on that exact handle)",
        handle_p3g.terminated and not handle_p3g.killed,
    )

    # --- Section 5: SSE/JSON protocol errors devalue a request --------------
    async def _exec_raw(extra_raw_events: list[str], token_events=None) -> dict:
        expected_prompt, expected_completion = 10, 4
        t = FakeTransport()
        t.queue_script(
            "t_proto",
            FakeStreamScript(
                prompt_token_ids_echo=list(range(expected_prompt)),
                token_events=token_events if token_events is not None else [[1], [2], [3], [4]],
                usage={"prompt_tokens": expected_prompt, "completion_tokens": expected_completion},
                extra_raw_events_before_finish=extra_raw_events,
            ),
        )
        return await execute_completion_request(
            transport=t, clock=RealClock(), url="http://x/v1/completions",
            api_key="k", model_full_id="m", prompt_token_ids=list(range(expected_prompt)),
            max_tokens=expected_completion, min_tokens=expected_completion, temperature=0.0,
            request_seed=1, request_id="t_proto", role="victim", request_index=0,
            prompt_seed=1, generation_seed=1, expected_prompt_tokens=expected_prompt,
            expected_completion_tokens=expected_completion, http_timeout_s=5.0,
        )

    r_p5_1 = await _exec_raw(["{not valid json"])
    check(
        "(P5-1) an invalid-JSON SSE event, followed by an otherwise complete "
        "stream, is never 'complete'",
        r_p5_1["status"] != REQUEST_STATUS_COMPLETE
        and any("JSON parse error" in e for e in r_p5_1["validation_errors"]),
        str(r_p5_1["validation_errors"]),
    )

    r_p5_2 = await _exec_raw(
        [json.dumps({"choices": [{"index": 0, "token_ids": ["1"], "finish_reason": None}]})]
    )
    check(
        "(P5-2) token_ids containing non-int elements is a protocol error -> not 'complete'",
        r_p5_2["status"] != REQUEST_STATUS_COMPLETE
        and any("token_ids" in e for e in r_p5_2["validation_errors"]),
        str(r_p5_2["validation_errors"]),
    )

    r_p5_3 = await _exec_raw(
        [json.dumps({"choices": [{"index": 0, "token_ids": [], "finish_reason": None}], "usage": []})]
    )
    check(
        "(P5-3) usage as a list instead of a dict is a protocol error and "
        "never crashes -> not 'complete'",
        r_p5_3["status"] != REQUEST_STATUS_COMPLETE
        and any("usage" in e for e in r_p5_3["validation_errors"]),
        str(r_p5_3["validation_errors"]),
    )

    r_p5_4 = await _exec_raw([json.dumps({"prompt_token_ids": [999] * 10, "choices": []})])
    check(
        "(P5-4) contradicting prompt_token_ids across multiple events -> not 'complete'",
        r_p5_4["status"] != REQUEST_STATUS_COMPLETE
        and any("contradicting prompt_token_ids" in e for e in r_p5_4["validation_errors"]),
        str(r_p5_4["validation_errors"]),
    )

    r_p5_5 = await _exec_raw([])
    check(
        "(P5-5) a fully well-formed protocol stays 'complete'",
        r_p5_5["status"] == REQUEST_STATUS_COMPLETE, str(r_p5_5["validation_errors"]),
    )

    # --- Section 6: trigger failures preserve full raw abort data ----------
    tok_trig = FakeTokenizerAdapter(vocab_size=500, special_token_ids={0, 1, 2})
    valid_trig = compute_valid_token_ids(tok_trig)

    t_p6a = FakeTransport()
    t_p6a.default_script_factory = lambda payload: FakeStreamScript(
        hang=True, prompt_token_ids_echo=None, token_events=[]
    )
    ctx_p6a = RunContext(
        transport=t_p6a, clock=RealClock(), sleeper=FakeSleeper(), base_url="http://x",
        api_key="k", model_full_id="m", valid_ids=valid_trig, trigger_timeout_s=0.05,
    )
    ep_p6a = Episode(
        episode_id="trig_timeout_ep", model="llama", offload_gb=0, state_label="low",
        concurrency=3, condition="no_burst", repeat=1, random_seed=1, episode_seed=1,
        victim_workload_seed=111, burst_workload_seed=222, victim_request_count=6,
        victim_input_len=8, victim_output_len=4, victim_temperature=0.0,
        burst_parallel_requests=2, burst_input_len=8, burst_output_len=4,
        burst_temperature=0.0, restart_server_before_block=1, block_id="fake_block_p6a",
        order_in_block=1,
    )
    result_p6a = await run_regular_episode(
        ctx_p6a, ep_p6a, schedule_fingerprint="x", server_metadata={}, stabilization_ref={},
    )
    check(
        "(P6-1) trigger timeout: exactly N victim records are stored and "
        "the episode is marked failed",
        len(result_p6a["victim_requests"]) == ep_p6a.victim_request_count
        and result_p6a["trigger"]["status"] == "timeout"
        and result_p6a["status"] == "failed",
        str(len(result_p6a["victim_requests"])),
    )
    check("(P6-2) after a trigger timeout, no fake stream is still active", t_p6a.active_stream_count == 0)
    check(
        "(P6-2b) every stored victim record after a trigger timeout has a "
        "well-formed identity, even the ones that never started",
        all(
            r.get("request_id") == f"trig_timeout_ep:victim:{i}" and r.get("role") == "victim"
            for i, r in enumerate(result_p6a["victim_requests"])
        ),
    )

    t_p6b = FakeTransport()
    t_p6b.queue_script(
        "trig_partial_ep:victim:0",
        FakeStreamScript(
            prompt_token_ids_echo=list(range(8)), token_events=[[1]],
            usage={"prompt_tokens": 8, "completion_tokens": 1},
        ),
    )
    t_p6b.queue_script(
        "trig_partial_ep:victim:1", FakeStreamScript(hang=True, prompt_token_ids_echo=None, token_events=[])
    )
    t_p6b.queue_script(
        "trig_partial_ep:victim:2", FakeStreamScript(hang=True, prompt_token_ids_echo=None, token_events=[])
    )
    ctx_p6b = RunContext(
        transport=t_p6b, clock=RealClock(), sleeper=FakeSleeper(), base_url="http://x",
        api_key="k", model_full_id="m", valid_ids=valid_trig, trigger_timeout_s=2.0,
    )
    ep_p6b = Episode(
        episode_id="trig_partial_ep", model="llama", offload_gb=0, state_label="low",
        concurrency=3, condition="fixed_burst", repeat=1, random_seed=1, episode_seed=1,
        victim_workload_seed=111, burst_workload_seed=222, victim_request_count=3,
        victim_input_len=8, victim_output_len=4, victim_temperature=0.0,
        burst_parallel_requests=2, burst_input_len=8, burst_output_len=4,
        burst_temperature=0.0, restart_server_before_block=1, block_id="fake_block_p6b",
        order_in_block=1,
    )
    result_p6b = await run_regular_episode(
        ctx_p6b, ep_p6b, schedule_fingerprint="x", server_metadata={}, stabilization_ref={},
    )
    check(
        "(P6-3) a first-wave request that delivers one token and then ends "
        "incomplete triggers pretrigger_failure even while a sibling is "
        "still waiting",
        result_p6b["trigger"]["status"] == "pretrigger_failure",
        str(result_p6b["trigger"]),
    )
    check("(P6-4) zero burst requests are started after a pretrigger_failure", result_p6b["burst_requests"] == [])
    check(
        "(P6-5) raw SSE events from the request that did start are preserved",
        len(result_p6b["victim_requests"]) == 3
        and any(r.get("raw_sse_events") for r in result_p6b["victim_requests"]),
        str([len(r.get("raw_sse_events") or []) for r in result_p6b["victim_requests"]]),
    )

    # --- Section 7: exact first wave -----------------------------------------
    t_p7 = _make_success_transport()
    ctx_p7 = RunContext(
        transport=t_p7, clock=RealClock(), sleeper=FakeSleeper(), base_url="http://x",
        api_key="k", model_full_id=MODEL_FULL_ID["llama"], valid_ids=compute_valid_token_ids(tok_block),
        trigger_timeout_s=5.0,
    )
    ep_p7 = Episode(
        episode_id="firstwave_ep", model="llama", offload_gb=0, state_label="low",
        concurrency=4, condition="no_burst", repeat=1, random_seed=1, episode_seed=1,
        victim_workload_seed=333, burst_workload_seed=444, victim_request_count=10,
        victim_input_len=16, victim_output_len=4, victim_temperature=0.0,
        burst_parallel_requests=2, burst_input_len=16, burst_output_len=4,
        burst_temperature=0.0, restart_server_before_block=1, block_id="fake_block_p7",
        order_in_block=1,
    )
    result_p7 = await run_regular_episode(
        ctx_p7, ep_p7, schedule_fingerprint="x", server_metadata={}, stabilization_ref={},
    )
    first_started = t_p7.call_order[: ep_p7.concurrency]
    expected_first = [f"{ep_p7.episode_id}:victim:{i}" for i in range(ep_p7.concurrency)]
    check(
        "(P7-1) the first `concurrency` requests to actually start are "
        "exactly victim indices 0..concurrency-1",
        first_started == expected_first, str(t_p7.call_order),
    )
    check(
        "(P7-2) never more than `concurrency` victim streams are active at once",
        t_p7.max_active_stream_count <= ep_p7.concurrency, str(t_p7.max_active_stream_count),
    )
    check("(P7-setup) the episode itself still completes successfully", result_p7["status"] == REQUEST_STATUS_COMPLETE)

    # =========================================================================
    # Patch: real-vLLM prompt-token-id mapping (choices[0] vs top-level)
    # =========================================================================

    async def _exec_full_raw(
        raw_events: list[str], expected_prompt: int = 5, expected_completion: int = 3,
        sent_prompt: list[int] | None = None,
    ) -> dict:
        t = FakeTransport()
        t.queue_script(
            "t_map",
            FakeStreamScript(
                prompt_token_ids_echo=None, token_events=[], include_done=False,
                extra_raw_events_before_finish=raw_events,
            ),
        )
        return await execute_completion_request(
            transport=t, clock=RealClock(), url="http://x/v1/completions",
            api_key="k", model_full_id="m",
            prompt_token_ids=sent_prompt if sent_prompt is not None else list(range(expected_prompt)),
            max_tokens=expected_completion, min_tokens=expected_completion, temperature=0.0,
            request_seed=1, request_id="t_map", role="victim", request_index=0,
            prompt_seed=1, generation_seed=1, expected_prompt_tokens=expected_prompt,
            expected_completion_tokens=expected_completion, http_timeout_s=5.0,
        )

    prompt5 = [10, 11, 12, 13, 14]

    # --- extract_prompt_token_ids() unit checks (isolated, no I/O) ----------
    check(
        "(helper-a) extract_prompt_token_ids: top-level only -> recognized",
        extract_prompt_token_ids({"prompt_token_ids": prompt5}, {}, 0) == (prompt5, []),
    )
    check(
        "(helper-b) extract_prompt_token_ids: choices[0] only -> recognized",
        extract_prompt_token_ids({}, {"prompt_token_ids": prompt5}, 0) == (prompt5, []),
    )
    check(
        "(helper-c) extract_prompt_token_ids: both positions null -> (None, [])",
        extract_prompt_token_ids({"prompt_token_ids": None}, {"prompt_token_ids": None}, 0) == (None, []),
    )
    _bad_top, _bad_errs = extract_prompt_token_ids({"prompt_token_ids": "nope"}, {}, 0)
    check(
        "(helper-d) extract_prompt_token_ids never crashes on malformed input",
        _bad_top is None and len(_bad_errs) == 1,
    )

    # (1) prompt-token-ids reported only at the top level.
    events_1 = [
        json.dumps({"prompt_token_ids": prompt5, "choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r1 = await _exec_full_raw(events_1, sent_prompt=prompt5)
    check(
        "(1) prompt-token-ids reported only at the top level are recognized",
        r1["status"] == REQUEST_STATUS_COMPLETE and r1["prompt_token_ids_returned"] == prompt5,
        str(r1["validation_errors"]),
    )

    # (2) prompt-token-ids reported only inside choices[0] (real vLLM 0.17.1 shape).
    events_2 = [
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None, "prompt_token_ids": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length", "prompt_token_ids": None}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r2 = await _exec_full_raw(events_2, sent_prompt=prompt5)
    check(
        "(2) prompt-token-ids reported only inside choices[0] are recognized",
        r2["status"] == REQUEST_STATUS_COMPLETE and r2["prompt_token_ids_returned"] == prompt5,
        str(r2["validation_errors"]),
    )

    # (3) identical top-level and choices[0] values in the same event.
    events_3 = [
        json.dumps({"prompt_token_ids": prompt5, "choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r3 = await _exec_full_raw(events_3, sent_prompt=prompt5)
    check(
        "(3) identical top-level and choices[0] prompt_token_ids within the "
        "same event are both recognized, not treated as a conflict",
        r3["status"] == REQUEST_STATUS_COMPLETE and r3["prompt_token_ids_returned"] == prompt5,
        str(r3["validation_errors"]),
    )

    # (4) contradicting top-level vs choices[0] within the same event.
    events_4 = [
        json.dumps({"prompt_token_ids": prompt5, "choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": [999, 999, 999, 999, 999]}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r4 = await _exec_full_raw(events_4, sent_prompt=prompt5)
    check(
        "(4) contradicting top-level vs choices[0] prompt_token_ids within "
        "the same event -> not 'complete'",
        r4["status"] != REQUEST_STATUS_COMPLETE,
        str(r4["validation_errors"]),
    )

    # (5) choices[0].prompt_token_ids of the wrong type.
    events_5 = [
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": "not-a-list"}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r5 = await _exec_full_raw(events_5, sent_prompt=prompt5)
    check(
        "(5) choices[0].prompt_token_ids of the wrong type -> not 'complete'",
        r5["status"] != REQUEST_STATUS_COMPLETE,
        str(r5["validation_errors"]),
    )

    # (6) only the first event carries choices[0].prompt_token_ids, later events send null.
    events_6 = [
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None, "prompt_token_ids": None}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length", "prompt_token_ids": None}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r6 = await _exec_full_raw(events_6, sent_prompt=prompt5)
    check(
        "(6) only the first event carries choices[0].prompt_token_ids, "
        "later events send null -> 'complete'",
        r6["status"] == REQUEST_STATUS_COMPLETE,
        str(r6["validation_errors"]),
    )

    # (7) two events with an identical choices[0].prompt_token_ids list.
    events_7 = [
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r7 = await _exec_full_raw(events_7, sent_prompt=prompt5)
    check(
        "(7) two events with an identical choices[0].prompt_token_ids list -> 'complete'",
        r7["status"] == REQUEST_STATUS_COMPLETE,
        str(r7["validation_errors"]),
    )

    # (8) two events with different choices[0].prompt_token_ids lists.
    events_8 = [
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [100], "finish_reason": None, "prompt_token_ids": prompt5}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [101], "finish_reason": None, "prompt_token_ids": [1, 2, 3, 4, 5]}]}),
        json.dumps({"choices": [{"index": 0, "text": "", "token_ids": [102], "finish_reason": "length"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}),
        "[DONE]",
    ]
    r8 = await _exec_full_raw(events_8, sent_prompt=prompt5)
    check(
        "(8) two events with different choices[0].prompt_token_ids lists -> not 'complete'",
        r8["status"] != REQUEST_STATUS_COMPLETE,
        str(r8["validation_errors"]),
    )

    # (9) the real vLLM 0.17.1 event shape at realistic 256/64 dimensions:
    # choice-level prompt ids on event 0 only, one output token per event,
    # a trailing usage-only event with empty choices, then [DONE].
    prompt256 = list(range(2000, 2256))
    output64 = list(range(3000, 3064))
    events_9: list[str] = []
    for i, tok in enumerate(output64):
        choice = {
            "index": 0, "text": "", "token_ids": [tok],
            "finish_reason": "length" if i == len(output64) - 1 else None,
        }
        choice["prompt_token_ids"] = prompt256 if i == 0 else None
        events_9.append(json.dumps({"choices": [choice]}))
    events_9.append(json.dumps({"choices": [], "usage": {"prompt_tokens": 256, "completion_tokens": 64}}))
    events_9.append("[DONE]")
    r9 = await _exec_full_raw(events_9, expected_prompt=256, expected_completion=64, sent_prompt=prompt256)
    check(
        "(9) the real vLLM 0.17.1 event shape (choice-level prompt ids on "
        "event 0, null afterward, trailing usage-only event) -> 'complete'",
        r9["status"] == REQUEST_STATUS_COMPLETE,
        str(r9["validation_errors"]),
    )

    # (10) prompt_token_ids_returned holds exactly the 256 server-reported ids.
    check(
        "(10) prompt_token_ids_returned holds exactly the 256 server-reported ids",
        r9["prompt_token_ids_returned"] == prompt256,
    )

    # (11) resume-depth validation accepts the resulting real-shaped record.
    ep_map = _build_fixture_episodes("llama", 900001)[0]
    tok_map = FakeTokenizerAdapter(vocab_size=5000, special_token_ids={0, 1, 2})
    valid_map = compute_valid_token_ids(tok_map)
    p_seed_map = victim_prompt_seed(ep_map, 0)
    g_seed_map = victim_generation_seed(ep_map, 0)
    prompt_map = generate_token_id_prompt(p_seed_map, valid_map, ep_map.victim_input_len)
    output_map = list(range(4000, 4000 + ep_map.victim_output_len))
    events_11: list[str] = []
    for i, tok in enumerate(output_map):
        choice = {
            "index": 0, "text": "", "token_ids": [tok],
            "finish_reason": "length" if i == len(output_map) - 1 else None,
        }
        choice["prompt_token_ids"] = prompt_map if i == 0 else None
        events_11.append(json.dumps({"choices": [choice]}))
    events_11.append(
        json.dumps(
            {"choices": [], "usage": {"prompt_tokens": ep_map.victim_input_len, "completion_tokens": ep_map.victim_output_len}}
        )
    )
    events_11.append("[DONE]")

    t_map11 = FakeTransport()
    t_map11.queue_script(
        f"{ep_map.episode_id}:victim:0",
        FakeStreamScript(
            prompt_token_ids_echo=None, token_events=[], include_done=False,
            extra_raw_events_before_finish=events_11,
        ),
    )
    r11 = await execute_completion_request(
        transport=t_map11, clock=RealClock(), url="http://x/v1/completions",
        api_key="k", model_full_id="m", prompt_token_ids=prompt_map,
        max_tokens=ep_map.victim_output_len, min_tokens=ep_map.victim_output_len, temperature=0.0,
        request_seed=g_seed_map, request_id=f"{ep_map.episode_id}:victim:0", role="victim", request_index=0,
        prompt_seed=p_seed_map, generation_seed=g_seed_map,
        expected_prompt_tokens=ep_map.victim_input_len, expected_completion_tokens=ep_map.victim_output_len,
        http_timeout_s=5.0,
    )
    depth_errors_11 = validate_complete_request_record(r11, episode=ep_map, role="victim", request_index=0)
    check(
        "(11) resume-depth validation accepts the resulting real-shaped "
        "complete request record",
        r11["status"] == REQUEST_STATUS_COMPLETE and depth_errors_11 == [],
        str((r11.get("status"), depth_errors_11)),
    )

    return results


