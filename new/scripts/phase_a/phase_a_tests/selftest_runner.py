"""
run_self_test() -- the top-level --self-test entry point, mechanically
moved out of run_phase_a.py's former "Self-test" section, verbatim --
no logic changes. Still a single function containing the Stage 1
inline checks directly (exactly as before), then delegating to the
Stage 2 / Stage 3 async check suites and the integration tests below.
"""

from __future__ import annotations

from run_phase_a import *  # noqa: F401,F403
from run_phase_a import _check_model_structure, _run_official_cli  # noqa: F401

from phase_a_tests.fixtures import _build_fixture_episodes, _make_fixture_block_bundle
from phase_a_tests.stage2_checks import _stage2_async_checks
from phase_a_tests.stage3_checks import _stage3_async_checks
from phase_a_tests.integration_checks import (
    run_fake_block_integration_test,
    run_subprocess_signal_test,
)


def run_self_test() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))

    # --- derive_seed determinism ------------------------------------------
    a = derive_seed("20260711", "x")
    b = derive_seed("20260711", "x")
    c = derive_seed("20260711", "y")
    check("derive_seed is deterministic", a == b)
    check("derive_seed differs for different input", a != c)
    check(
        "derive_seed returns a non-negative int below 2**31-1",
        isinstance(a, int) and 0 <= a < 2**31 - 1,
    )

    # --- valid fixture is accepted ------------------------------------------
    fixture_seed = 12345
    good = _build_fixture_episodes("testmodel", fixture_seed)
    errors = _check_model_structure(
        "testmodel", good, fixture_seed,
        repeats=1, episodes_per_model=8, blocks_per_model=2,
        expected_state_sequence=["low", "high"],
    )
    check("valid fixture schedule is accepted", not errors, str(errors[:3]))

    # --- corrupted fixtures are rejected -------------------------------------
    def fixture_errors(mutate) -> list[str]:
        eps = _build_fixture_episodes("testmodel", fixture_seed)
        mutate(eps)
        return _check_model_structure(
            "testmodel", eps, fixture_seed,
            repeats=1, episodes_per_model=8, blocks_per_model=2,
            expected_state_sequence=["low", "high"],
        )

    check(
        "reordering two episodes within a block is rejected",
        bool(fixture_errors(lambda eps: eps.__setitem__(slice(0, 2), [eps[1], eps[0]]))),
    )
    check(
        "wrong repeat value is rejected",
        bool(fixture_errors(lambda eps: setattr(eps[0], "repeat", 2))),
    )
    check(
        "broken block contiguity (interleaving) is rejected",
        bool(fixture_errors(lambda eps: eps.__setitem__(1, eps.pop(5)))),
    )
    check(
        "wrong episode_seed is rejected",
        bool(fixture_errors(lambda eps: setattr(eps[0], "episode_seed", eps[0].episode_seed + 1))),
    )
    check(
        "wrong restart_server_before_block sequence is rejected",
        bool(fixture_errors(lambda eps: setattr(eps[0], "restart_server_before_block", 0))),
    )

    # --- fingerprint round-trip ----------------------------------------------
    payload = {"a": 1, "b": [1, 2, 3], "c": {"x": 1.5}}
    fp1 = recompute_fingerprint(payload)
    payload_with_fp = dict(payload)
    payload_with_fp["schedule_fingerprint"] = fp1
    fp2 = recompute_fingerprint(payload_with_fp)
    check("fingerprint recomputation ignores schedule_fingerprint key", fp1 == fp2)
    check("fingerprint has valid format", is_valid_fingerprint_format(fp1))
    tampered = dict(payload)
    tampered["a"] = 2
    fp3 = recompute_fingerprint(tampered)
    check("fingerprint changes when payload changes", fp1 != fp3)

    # --- csv/json consistency check -------------------------------------
    sample_json_row = {f: getattr(good[0], f) for f in EPISODE_FIELDS}
    sample_csv_row = {f: str(getattr(good[0], f)) for f in EPISODE_FIELDS}
    normalized, norm_errors = normalize_csv_row(sample_csv_row, 0)
    check("csv row normalizes back to matching types", not norm_errors)
    check(
        "normalized csv row equals json row",
        all(normalized[f] == sample_json_row[f] for f in EPISODE_FIELDS),
    )
    sample_csv_row_bad = dict(sample_csv_row)
    sample_csv_row_bad["victim_output_len"] = "999"
    normalized_bad, _ = normalize_csv_row(sample_csv_row_bad, 0)
    check(
        "csv/json mismatch is detectable after normalization",
        normalized_bad["victim_output_len"] != sample_json_row["victim_output_len"],
    )

    # --- strict type checking (bool must not pass as int) -------------------
    bad_bool_episode = dict(sample_json_row)
    bad_bool_episode["victim_request_count"] = True
    schema_errors = check_json_episode_schema(bad_bool_episode, 0)
    check(
        "bool is rejected where int is expected",
        any("victim_request_count" in e for e in schema_errors),
    )
    extra_field_episode = dict(sample_json_row)
    extra_field_episode["warmup_requests"] = 1
    schema_errors2 = check_json_episode_schema(extra_field_episode, 0)
    check(
        "unexpected extra field (e.g. warmup_requests) is rejected",
        any("warmup_requests" in e for e in schema_errors2),
    )
    missing_field_episode = dict(sample_json_row)
    del missing_field_episode["block_id"]
    schema_errors3 = check_json_episode_schema(missing_field_episode, 0)
    check(
        "missing field is rejected",
        any("block_id" in e for e in schema_errors3),
    )

    # --- result-file classification -----------------------------------------
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ep0 = good[0]
        ep1 = good[1]
        expected_fp = "sha256:" + "0" * 64

        _fixture_tok = FakeTokenizerAdapter(vocab_size=5000, special_token_ids={0, 1, 2})
        _fixture_valid_ids = compute_valid_token_ids(_fixture_tok)

        def make_valid_request_record(ep: Episode, role: str, index: int) -> dict:
            if role == "victim":
                p_seed = victim_prompt_seed(ep, index)
                g_seed = victim_generation_seed(ep, index)
                input_len, output_len = ep.victim_input_len, ep.victim_output_len
            else:
                p_seed = burst_prompt_seed(ep, index)
                g_seed = burst_generation_seed(ep, index)
                input_len, output_len = ep.burst_input_len, ep.burst_output_len
            prompt_ids = generate_token_id_prompt(p_seed, _fixture_valid_ids, input_len)
            output_ids = [7000 + k for k in range(output_len)]
            return {
                "request_id": f"{ep.episode_id}:{role}:{index}",
                "role": role,
                "request_index": index,
                "prompt_seed": p_seed,
                "generation_seed": g_seed,
                "prompt_token_ids_sent": prompt_ids,
                "prompt_token_ids_returned": list(prompt_ids),
                "prompt_sha256": prompt_sha256(prompt_ids),
                "expected_prompt_tokens": input_len,
                "expected_completion_tokens": output_len,
                "usage": {"prompt_tokens": input_len, "completion_tokens": output_len},
                "output_token_ids": output_ids,
                "output_text": "",
                "finish_reason": "length",
                "raw_sse_events": [],
                "done_received": True,
                "request_start_utc": "1970-01-01T00:00:00Z",
                "request_end_utc": "1970-01-01T00:00:01Z",
                "request_start_ns": 1000,
                "first_token_receive_ns": 1100,
                "last_token_receive_ns": 1200,
                "stream_end_ns": 1300,
                "ttft_ms": 0.1,
                "client_observed_tpot_ms": 0.01,
                "e2el_ms": 0.3,
                "itl_available": True,
                "itl_ms": [],
                "token_batch_sizes": None,
                "token_batch_interarrival_ms": None,
                "chunk_interarrival_ms": None,
                "http_status": 200,
                "timed_out": False,
                "cancelled": False,
                "error_type": None,
                "error_message": None,
                "validation_errors": [],
                "status": REQUEST_STATUS_COMPLETE,
            }

        def make_valid_result(ep: Episode) -> dict:
            burst_count = ep.burst_parallel_requests if ep.condition == "fixed_burst" else 0
            return {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "runner_version": RUNNER_VERSION,
                "run_mode": "smoke",
                "schedule_fingerprint": expected_fp,
                "episode_id": ep.episode_id,
                "schedule_row": asdict(ep),
                "record_type": RECORD_TYPE_REGULAR_EPISODE,
                "status": "complete",
                "trigger": {
                    "status": "ok", "trigger_utc": "1970-01-01T00:00:00Z",
                    "trigger_perf_ns": 1000, "waited_ms": 0.0,
                },
                "burst_interval": (
                    {"start_ns": 1100, "end_ns": 1300} if ep.condition == "fixed_burst" else None
                ),
                "victim_requests": [
                    make_valid_request_record(ep, "victim", i) for i in range(ep.victim_request_count)
                ],
                "burst_requests": [
                    make_valid_request_record(ep, "burst", j) for j in range(burst_count)
                ],
            }

        valid_result = make_valid_result(ep0)

        missing_path = tmp_path / "missing.json"
        cls, _ = classify_result_file(missing_path, ep0, expected_fp, "smoke")
        check("missing result file classified as 'missing'", cls == CLASSIFICATION_MISSING)

        valid_path = tmp_path / "valid.json"
        valid_path.write_text(json.dumps(valid_result), encoding="utf-8")
        cls, notes = classify_result_file(valid_path, ep0, expected_fp, "smoke")
        check(
            "well-formed complete result classified as 'valid_complete' (test 9)",
            cls == CLASSIFICATION_VALID_COMPLETE, str(notes),
        )

        corrupted_path = tmp_path / "corrupted.json"
        corrupted_path.write_text("{not valid json", encoding="utf-8")
        cls, _ = classify_result_file(corrupted_path, ep0, expected_fp, "smoke")
        check("malformed JSON classified as 'corrupted'", cls == CLASSIFICATION_CORRUPTED)

        partial_result = dict(valid_result)
        partial_result["status"] = "in_progress"
        partial_path = tmp_path / "partial.json"
        partial_path.write_text(json.dumps(partial_result), encoding="utf-8")
        cls, _ = classify_result_file(partial_path, ep0, expected_fp, "smoke")
        check("status != complete classified as 'partial'", cls == CLASSIFICATION_PARTIAL)

        bad_count_result = dict(valid_result)
        bad_count_result["victim_requests"] = [{}] * (ep0.victim_request_count - 1)
        bad_count_path = tmp_path / "badcount.json"
        bad_count_path.write_text(json.dumps(bad_count_result), encoding="utf-8")
        cls, _ = classify_result_file(bad_count_path, ep0, expected_fp, "smoke")
        check("wrong victim_requests count classified as 'invalid' (test 7)", cls == CLASSIFICATION_INVALID)

        missing_key_result = dict(valid_result)
        del missing_key_result["schedule_row"]
        mk_path = tmp_path / "missingkey.json"
        mk_path.write_text(json.dumps(missing_key_result), encoding="utf-8")
        cls, _ = classify_result_file(mk_path, ep0, expected_fp, "smoke")
        check("missing required result key classified as 'invalid'", cls == CLASSIFICATION_INVALID)

        # --- new resume-validation tests (this patch) -----------------------

        # Test 1: file at episode A's path contains a fully valid, complete
        # result -- but for episode B. Must be 'invalid' when validated
        # against the expected episode A, never silently accepted.
        wrong_episode_result = make_valid_result(ep1)
        wrong_episode_path = tmp_path / f"{ep0.episode_id}.json"
        wrong_episode_path.write_text(json.dumps(wrong_episode_result), encoding="utf-8")
        cls, notes = classify_result_file(wrong_episode_path, ep0, expected_fp, "smoke")
        check(
            "valid result for a different episode is rejected as 'invalid' "
            "when checked against the expected episode (test 1)",
            cls == CLASSIFICATION_INVALID, str(notes),
        )

        # Test 2: wrong runner_version.
        wrong_runner_result = dict(valid_result)
        wrong_runner_result["runner_version"] = "some-other-runner-9.9"
        wr_path = tmp_path / "wrongrunner.json"
        wr_path.write_text(json.dumps(wrong_runner_result), encoding="utf-8")
        cls, _ = classify_result_file(wr_path, ep0, expected_fp, "smoke")
        check("wrong runner_version classified as 'invalid' (test 2)", cls == CLASSIFICATION_INVALID)

        # Test 3: episode_id is a list -- must not crash, must be 'invalid'.
        list_episode_id_result = dict(valid_result)
        list_episode_id_result["episode_id"] = []
        lid_path = tmp_path / "listepid.json"
        lid_path.write_text(json.dumps(list_episode_id_result), encoding="utf-8")
        try:
            cls, _ = classify_result_file(lid_path, ep0, expected_fp, "smoke")
            crashed = False
        except Exception:
            crashed = True
            cls = None
        check(
            "episode_id as a list is classified as 'invalid' without crashing (test 3)",
            (not crashed) and cls == CLASSIFICATION_INVALID,
        )

        # Test 4: result_schema_version = true (bool) must NOT be accepted
        # as version 1, and must not crash.
        bool_version_result = dict(valid_result)
        bool_version_result["result_schema_version"] = True
        bv_path = tmp_path / "boolversion.json"
        bv_path.write_text(json.dumps(bool_version_result), encoding="utf-8")
        cls, _ = classify_result_file(bv_path, ep0, expected_fp, "smoke")
        check(
            "result_schema_version=true (bool) is rejected as 'invalid', "
            "not accepted as version 1 (test 4)",
            cls == CLASSIFICATION_INVALID,
        )

        # Test 5: result_schema_version = 1 as a real int is still accepted.
        real_int_version_result = dict(valid_result)
        real_int_version_result["result_schema_version"] = RESULT_SCHEMA_VERSION
        riv_path = tmp_path / "realintversion.json"
        riv_path.write_text(json.dumps(real_int_version_result), encoding="utf-8")
        cls, notes = classify_result_file(riv_path, ep0, expected_fp, "smoke")
        check(
            "result_schema_version as a real int stays accepted (test 5)",
            cls == CLASSIFICATION_VALID_COMPLETE, str(notes),
        )

        # Test 6: schedule_row is not a dict.
        list_row_result = dict(valid_result)
        list_row_result["schedule_row"] = ["not", "a", "dict"]
        lr_path = tmp_path / "listrow.json"
        lr_path.write_text(json.dumps(list_row_result), encoding="utf-8")
        cls, _ = classify_result_file(lr_path, ep0, expected_fp, "smoke")
        check("schedule_row as a non-dict is classified as 'invalid' (test 6)", cls == CLASSIFICATION_INVALID)

        # Test 7 (type-level, complementing the count-mismatch test above):
        # victim_requests is not an array at all.
        non_array_victim_result = dict(valid_result)
        non_array_victim_result["victim_requests"] = "not-a-list"
        nav_path = tmp_path / "nonarrayvictim.json"
        nav_path.write_text(json.dumps(non_array_victim_result), encoding="utf-8")
        cls, _ = classify_result_file(nav_path, ep0, expected_fp, "smoke")
        check("victim_requests as a non-array is classified as 'invalid' (test 7)", cls == CLASSIFICATION_INVALID)

        # Test 8: burst_requests is not an array at all.
        non_array_burst_result = dict(valid_result)
        non_array_burst_result["burst_requests"] = 42
        nab_path = tmp_path / "nonarrayburst.json"
        nab_path.write_text(json.dumps(non_array_burst_result), encoding="utf-8")
        cls, _ = classify_result_file(nab_path, ep0, expected_fp, "smoke")
        check("burst_requests as a non-array is classified as 'invalid' (test 8)", cls == CLASSIFICATION_INVALID)

        # Test 10: scan_existing_results() must not skip any schedule
        # episode just because a file with a foreign episode_id exists at
        # a different episode's expected path -- each expected filename is
        # validated strictly against its own specific episode.
        fake_bundle = LoadedBundle(
            schedule_dir=tmp_path,
            json_obj={},
            csv_fieldnames=[],
            csv_rows=[],
            audit_text="",
            episodes=[ep0, ep1],
            fingerprint=expected_fp,
        )
        scan_dir = tmp_path / "scan"
        scan_dir.mkdir()
        (scan_dir / EPISODES_SUBDIR).mkdir()
        # ep0's own expected file actually holds ep1's (valid) result --
        # must show up as 'invalid' for ep0, and ep1 itself must still be
        # correctly reported as 'missing' (not silently matched/skipped).
        (scan_dir / EPISODES_SUBDIR / f"{ep0.episode_id}.json").write_text(
            json.dumps(make_valid_result(ep1)), encoding="utf-8"
        )
        classifications = scan_existing_results(scan_dir, fake_bundle, "smoke")
        check(
            "scan_existing_results: episode with a foreign result file is "
            "'invalid', not silently matched (test 10)",
            classifications.get(ep0.episode_id) == CLASSIFICATION_INVALID,
            str(classifications),
        )
        check(
            "scan_existing_results: the actual owner episode is still "
            "reported 'missing', not skipped (test 10)",
            classifications.get(ep1.episode_id) == CLASSIFICATION_MISSING,
            str(classifications),
        )

        # --- Stage-2 patch, section 4: deep per-request resume validation ---
        ep_burst = next(e for e in good if e.condition == "fixed_burst")

        def write_and_classify(mutated: dict, ep: Episode = ep0) -> tuple[str, list[str]]:
            p = tmp_path / f"depth_{len(list(tmp_path.glob('depth_*.json')))}.json"
            p.write_text(json.dumps(mutated), encoding="utf-8")
            return classify_result_file(p, ep, expected_fp, "smoke")

        # (4-1) empty request dicts are no longer accepted.
        empty_dicts_result = dict(make_valid_result(ep0))
        empty_dicts_result["victim_requests"] = [{}] * ep0.victim_request_count
        cls, notes = write_and_classify(empty_dicts_result)
        check("(4-1) empty request dicts -> invalid", cls == CLASSIFICATION_INVALID, str(notes[:2]))

        # (4-2) a request with status='incomplete' is rejected even though
        # every list has the right length.
        incomplete_result = dict(make_valid_result(ep0))
        incomplete_result["victim_requests"] = list(incomplete_result["victim_requests"])
        incomplete_result["victim_requests"][5] = dict(incomplete_result["victim_requests"][5])
        incomplete_result["victim_requests"][5]["status"] = "incomplete"
        cls, _ = write_and_classify(incomplete_result)
        check("(4-2) a request with status='incomplete' -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-3) a tampered prompt_sha256.
        bad_hash_result = dict(make_valid_result(ep0))
        bad_hash_result["victim_requests"] = list(bad_hash_result["victim_requests"])
        bad_hash_result["victim_requests"][0] = dict(bad_hash_result["victim_requests"][0])
        bad_hash_result["victim_requests"][0]["prompt_sha256"] = "0" * 64
        cls, _ = write_and_classify(bad_hash_result)
        check("(4-3) wrong prompt_sha256 -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-4) prompt_token_ids_sent != prompt_token_ids_returned.
        mismatched_prompt_result = dict(make_valid_result(ep0))
        mismatched_prompt_result["victim_requests"] = list(mismatched_prompt_result["victim_requests"])
        mismatched_prompt_result["victim_requests"][0] = dict(mismatched_prompt_result["victim_requests"][0])
        mismatched_prompt_result["victim_requests"][0]["prompt_token_ids_returned"] = [999] * ep0.victim_input_len
        cls, _ = write_and_classify(mismatched_prompt_result)
        check("(4-4) prompt_token_ids_sent != prompt_token_ids_returned -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-5) wrong usage counters.
        bad_usage_result = dict(make_valid_result(ep0))
        bad_usage_result["victim_requests"] = list(bad_usage_result["victim_requests"])
        bad_usage_result["victim_requests"][0] = dict(bad_usage_result["victim_requests"][0])
        bad_usage_result["victim_requests"][0]["usage"] = {"prompt_tokens": 1, "completion_tokens": 1}
        cls, _ = write_and_classify(bad_usage_result)
        check("(4-5) wrong usage counters -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-6) wrong output length.
        bad_outlen_result = dict(make_valid_result(ep0))
        bad_outlen_result["victim_requests"] = list(bad_outlen_result["victim_requests"])
        bad_outlen_result["victim_requests"][0] = dict(bad_outlen_result["victim_requests"][0])
        bad_outlen_result["victim_requests"][0]["output_token_ids"] = [1, 2, 3]
        cls, _ = write_and_classify(bad_outlen_result)
        check("(4-6) wrong output_token_ids length -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-7) wrong role/index on a request record.
        bad_role_result = dict(make_valid_result(ep0))
        bad_role_result["victim_requests"] = list(bad_role_result["victim_requests"])
        bad_role_result["victim_requests"][0] = dict(bad_role_result["victim_requests"][0])
        bad_role_result["victim_requests"][0]["role"] = "burst"
        cls, _ = write_and_classify(bad_role_result)
        check("(4-7a) wrong role -> invalid", cls == CLASSIFICATION_INVALID)

        bad_index_result = dict(make_valid_result(ep0))
        bad_index_result["victim_requests"] = list(bad_index_result["victim_requests"])
        bad_index_result["victim_requests"][0] = dict(bad_index_result["victim_requests"][0])
        bad_index_result["victim_requests"][0]["request_index"] = 17
        cls, _ = write_and_classify(bad_index_result)
        check("(4-7b) wrong request_index -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-8) duplicate index.
        dup_index_result = dict(make_valid_result(ep0))
        dup_index_result["victim_requests"] = list(dup_index_result["victim_requests"])
        dup_index_result["victim_requests"][1] = dict(dup_index_result["victim_requests"][0])
        cls, _ = write_and_classify(dup_index_result)
        check("(4-8) duplicate request_index -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-9) wrong deterministic seed.
        bad_seed_result = dict(make_valid_result(ep0))
        bad_seed_result["victim_requests"] = list(bad_seed_result["victim_requests"])
        bad_seed_result["victim_requests"][0] = dict(bad_seed_result["victim_requests"][0])
        bad_seed_result["victim_requests"][0]["prompt_seed"] = 424242424
        cls, _ = write_and_classify(bad_seed_result)
        check("(4-9) wrong deterministic prompt_seed -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-10) timestamps out of order.
        bad_order_result = dict(make_valid_result(ep0))
        bad_order_result["victim_requests"] = list(bad_order_result["victim_requests"])
        bad_order_result["victim_requests"][0] = dict(bad_order_result["victim_requests"][0])
        bad_order_result["victim_requests"][0]["last_token_receive_ns"] = 1
        cls, _ = write_and_classify(bad_order_result)
        check("(4-10) out-of-order request timestamps -> invalid", cls == CLASSIFICATION_INVALID)

        # (4-11) a fully correct real Stage-2 schema instance is accepted,
        # including a fixed_burst episode with a real burst_interval.
        good_no_burst = write_and_classify(dict(make_valid_result(ep0)), ep0)
        check("(4-11a) fully correct no_burst episode -> valid_complete", good_no_burst[0] == CLASSIFICATION_VALID_COMPLETE, str(good_no_burst[1]))
        good_burst = write_and_classify(dict(make_valid_result(ep_burst)), ep_burst)
        check(
            "(4-11b) fully correct fixed_burst episode (4 burst requests + "
            "burst_interval) -> valid_complete",
            good_burst[0] == CLASSIFICATION_VALID_COMPLETE, str(good_burst[1]),
        )

        # trigger.status must be 'ok'; burst_interval must match condition.
        bad_trigger_result = dict(make_valid_result(ep0))
        bad_trigger_result["trigger"] = {"status": "timeout"}
        cls, _ = write_and_classify(bad_trigger_result)
        check("(4-extra) trigger.status != 'ok' -> invalid", cls == CLASSIFICATION_INVALID)

        no_burst_with_interval_result = dict(make_valid_result(ep0))
        no_burst_with_interval_result["burst_interval"] = {"start_ns": 1, "end_ns": 2}
        cls, _ = write_and_classify(no_burst_with_interval_result)
        check(
            "(4-extra) no_burst episode with a non-null burst_interval -> invalid",
            cls == CLASSIFICATION_INVALID,
        )

        fixed_burst_missing_interval_result = dict(make_valid_result(ep_burst))
        fixed_burst_missing_interval_result["burst_interval"] = None
        cls, _ = write_and_classify(fixed_burst_missing_interval_result, ep_burst)
        check(
            "(4-extra) fixed_burst episode with a null burst_interval -> invalid",
            cls == CLASSIFICATION_INVALID,
        )

        # --- output-dir marker conflict --------------------------------
        write_run_mode_marker(tmp_path, "official")
        conflict_raised = False
        try:
            check_output_dir_not_shared(tmp_path, "smoke")
        except OutputDirConflictError:
            conflict_raised = True
        check("output-dir marker conflict (official vs smoke) is rejected", conflict_raised)

        no_conflict_raised = False
        try:
            check_output_dir_not_shared(tmp_path, "official")
        except OutputDirConflictError:
            no_conflict_raised = True
        check("output-dir marker matching the same mode is accepted", not no_conflict_raised)

    # --- CLI mode mutual exclusivity / --resume guard -----------------------
    def parse_expect_systemexit(argv: list[str]) -> bool:
        try:
            parse_args(argv)
        except SystemExit:
            return True
        return False

    check("no mode flag is rejected", parse_expect_systemexit([]))
    check(
        "two mode flags together are rejected",
        parse_expect_systemexit(["--self-test", "--dry-run"]),
    )
    check(
        "--resume without --official-run/--smoke-test is rejected",
        parse_expect_systemexit(["--dry-run", "--resume"]),
    )
    check(
        "--resume with --official-run is accepted",
        not parse_expect_systemexit(["--official-run", "--resume"]),
    )

    # --- VLLM_API_KEY is read from the environment ONLY inside
    # read_api_key_from_env(), and that function is only ever invoked
    # from the real --smoke-test execution path -- never from
    # --self-test, --dry-run, or --official-run. (Stage 1 never read it
    # at all; Stage 2 legitimately needs it for the real smoke test, so
    # this check's scope narrows accordingly instead of disappearing.)
    own_source = SCRIPT_PATH.read_text(encoding="utf-8")
    try:
        read_api_key_source = inspect.getsource(read_api_key_from_env)
    except (OSError, TypeError):
        read_api_key_source = ""
    env_access_patterns = [
        r'os\.environ\.get\(\s*["\']VLLM_API_KEY',
        r'os\.environ\[\s*["\']VLLM_API_KEY',
        r'os\.getenv\(\s*["\']VLLM_API_KEY',
        r'\.get\(\s*["\']VLLM_API_KEY',
    ]
    source_without_that_function = own_source.replace(read_api_key_source, "", 1)
    found_env_access_elsewhere = any(
        re.search(p, source_without_that_function) for p in env_access_patterns
    )
    check(
        "VLLM_API_KEY is read from the environment only inside "
        "read_api_key_from_env(), nowhere else in this module",
        bool(read_api_key_source) and not found_env_access_elsewhere,
    )
    main_source = inspect.getsource(main)
    official_branch_source = main_source.split("if args.official_run:")[1].split(
        "assert args.smoke_test"
    )[0]
    check(
        "(35) --official-run never calls run_smoke_block() (smoke-only "
        "path) and never uses a global pkill/process-name search",
        "run_smoke_block(" not in official_branch_source
        and "pkill" not in official_branch_source
        and "killall" not in official_branch_source,
    )
    check(
        "(35b) --official-run reads the API key and runs the real "
        "official campaign machinery -- no longer artificially disabled",
        "read_api_key_from_env(" in official_branch_source
        and "_run_official_cli(" in official_branch_source,
    )

    # --- 36/37/38: CLI validation for the new Stage-2 flags -----------------
    check(
        "(36) '--smoke-test' without '--smoke-block' is rejected",
        parse_expect_systemexit(["--smoke-test"]),
    )
    check(
        "(36b) '--smoke-block' without '--smoke-test' is rejected",
        parse_expect_systemexit(["--dry-run", "--smoke-block", "x"]),
    )
    fixture_bundle_for_block_check, _bid = _make_fixture_block_bundle("llama", 42)
    invalid_block_raised = False
    try:
        find_and_validate_smoke_block(fixture_bundle_for_block_check, "does_not_exist")
    except ValueError:
        invalid_block_raised = True
    check("(37) an invalid/unknown --smoke-block is rejected", invalid_block_raised)
    check(
        "(38) --port outside 1-65535 is rejected",
        parse_expect_systemexit(["--smoke-test", "--smoke-block", "x", "--port", "0"])
        and parse_expect_systemexit(["--smoke-test", "--smoke-block", "x", "--port", "70000"]),
    )
    check(
        "(38b) --port within 1-65535 is accepted",
        not parse_expect_systemexit(["--smoke-test", "--smoke-block", "x", "--port", "8000"]),
    )

    # --- Stage 2: async request/trigger/stabilization/smoke-block checks ---
    results.extend(asyncio.run(_stage2_async_checks()))

    # --- Stage 3: official campaign checks -----------------------------
    results.extend(asyncio.run(_stage3_async_checks()))

    # --- Section 26: fake full-block integration test (no sleep, no GPU) ---
    fake_block_ok, fake_block_notes = run_fake_block_integration_test()
    check(
        "fake full-block integration test (simulated server, stabilization "
        "+ 4 episodes, all JSON outputs validated)",
        fake_block_ok, "; ".join(fake_block_notes),
    )

    # --- Section 10 (contract-blocker patch): real OS-signal subprocess tests ---
    sigint_ok, sigint_msg = run_subprocess_signal_test(signal.SIGINT, "SIGINT", 130)
    check(
        "(real-signal) a real os.kill(pid, SIGINT) to an isolated fake-"
        "campaign subprocess yields a persisted 'interrupted'/SIGINT "
        "summary and exit code 130",
        sigint_ok, sigint_msg,
    )
    sigterm_ok, sigterm_msg = run_subprocess_signal_test(signal.SIGTERM, "SIGTERM", 143)
    check(
        "(real-signal) a real os.kill(pid, SIGTERM) to an isolated fake-"
        "campaign subprocess yields a persisted 'interrupted'/SIGTERM "
        "summary and exit code 143",
        sigterm_ok, sigterm_msg,
    )

    # --- summary --------------------------------------------------------
    print("Self-test results")
    print("=" * 60)
    all_passed = True
    for name, passed, detail in results:
        status = "OK" if passed else "FAIL"
        if not passed:
            all_passed = False
        line = f"[{status}] {name}"
        if detail and not passed:
            line += f" -- {detail}"
        print(line)
    print("=" * 60)
    print(f"{sum(1 for _, p, _ in results if p)}/{len(results)} checks passed")
    print("SELF-TEST: PASS" if all_passed else "SELF-TEST: FAIL")
    return 0 if all_passed else 1


