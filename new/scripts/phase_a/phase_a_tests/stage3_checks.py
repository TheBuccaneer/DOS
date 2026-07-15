"""
Stage 3 (official-campaign) self-test checks, mechanically moved out of
run_phase_a.py's former "Self-test" section, verbatim -- no logic
changes.
"""

from __future__ import annotations

from run_phase_a import *  # noqa: F401,F403
from run_phase_a import _run_block_protocol, _run_official_cli  # noqa: F401

from phase_a_tests.fixtures import (
    _fixture_tokenizer_factory,
    _make_fixture_campaign_bundle,
    _make_success_transport,
    _success_script_factory,
)


async def _stage3_async_checks() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))

    import tempfile as _tempfile

    clock = RealClock()
    sleeper = FakeSleeper()
    secret_key = "stage3-secret-should-never-leak"

    # --- (1/2/3) fresh run: exact order, one start+stop per block, run_mode="official"
    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle1 = _make_fixture_campaign_bundle(900101)
        expected_order = all_block_ids_in_schedule_order(bundle1)
        server_adapter1 = FakeServerProcessAdapter()
        summary1 = await run_official_campaign(
            bundle=bundle1, output_dir=out, host="127.0.0.1", port=19101, resume=False,
            api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=server_adapter1, sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
        )
        check(
            "(1) a fresh official run executes every fixture block in exact "
            "schedule order and completes",
            summary1["overall_status"] == "complete"
            and list(summary1["block_statuses"].keys()) == expected_order,
            str(summary1.get("overall_status")),
        )
        check(
            "(2) exactly one server start and one verified server stop per block",
            len(server_adapter1.started) == len(expected_order)
            and all(h.terminated and not h.killed for h in server_adapter1.started),
        )
        all_json_text = "".join(p.read_text() for p in out.rglob("*.json"))
        episode_run_modes = [
            json.loads(p.read_text())["run_mode"] for p in (out / "episodes").glob("*.json")
        ]
        stab_run_modes = [
            json.loads(p.read_text())["run_mode"] for p in (out / "stabilization").glob("*.json")
        ]
        check(
            "(3) every episode and stabilization file records run_mode='official'",
            all(m == RUN_MODE_OFFICIAL for m in episode_run_modes)
            and all(m == RUN_MODE_OFFICIAL for m in stab_run_modes)
            and len(episode_run_modes) == 16,
        )
        check("(12a) the API key never appears anywhere on disk after a fresh run", secret_key not in all_json_text)
        check("(12b) the API key never appears in the returned summary", secret_key not in json.dumps(summary1))

        # --- (19) complete is impossible with < 100% valid episodes -------
        check(
            "(19) overall_status='complete' implies valid_complete_episodes "
            "== planned_episodes (never partial, e.g. 15/16)",
            summary1["overall_status"] != "complete"
            or summary1["valid_complete_episodes"] == summary1["planned_episodes"],
        )

        # --- (17) integrity manifest is complete and self-verifies --------
        integrity1 = json.loads((out / INTEGRITY_MANIFEST_FILENAME).read_text())
        verified1, verify_errors1 = verify_integrity_manifest(out, integrity1)
        check(
            "(17) a successful fake campaign produces a complete, "
            "self-verifying integrity manifest",
            verified1
            and integrity1["episode_file_count"] == 16
            and integrity1["stabilization_file_count"] == 4
            and integrity1["block_summary_count"] == 4,
            str(verify_errors1),
        )

        # --- (18) tampering with a captured file is detected --------------
        one_episode_path = next((out / "episodes").glob("*.json"))
        original_text = one_episode_path.read_text()
        one_episode_path.write_text(original_text[:-1] + ("0" if original_text[-1] != "0" else "1") + "}")
        verified_after_tamper, tamper_errors = verify_integrity_manifest(out, integrity1)
        one_episode_path.write_text(original_text)  # restore
        check(
            "(18) tampering with a captured file is detected by the "
            "integrity verification",
            not verified_after_tamper and len(tamper_errors) >= 1,
            str(tamper_errors)[:200],
        )

        # --- (3/4/6/11/23/27) idempotent resume: nothing to do -----------
        server_adapter_resume_full = FakeServerProcessAdapter()
        all_paths_before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
        summary_resume_full = await run_official_campaign(
            bundle=bundle1, output_dir=out, host="127.0.0.1", port=19102, resume=True,
            api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=server_adapter_resume_full, sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
        )
        all_paths_after = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
        check(
            "(4) a fully-complete campaign is skipped entirely on --resume: "
            "no server start at all",
            summary_resume_full["overall_status"] == "already_complete"
            and len(server_adapter_resume_full.started) == 0,
        )
        check(
            "(3/6/23) resuming an already-complete campaign is a genuine "
            "no-op: identical file set, every file (summary/integrity/"
            "manifest/marker/block_summaries/stabilization/episodes/"
            "server_logs) byte-for-byte untouched, no new/missing files",
            set(all_paths_before) == set(all_paths_after) and all_paths_before == all_paths_after,
            f"before={len(all_paths_before)} files, after={len(all_paths_after)} files, "
            f"diff_keys={sorted(str(k) for k in (set(all_paths_before) ^ set(all_paths_after)))[:5]}",
        )
        integrity_after_noop = json.loads((out / INTEGRITY_MANIFEST_FILENAME).read_text())
        verified_after_noop, verify_errors_after_noop = verify_integrity_manifest(out, integrity_after_noop)
        check(
            "(3b) the existing integrity manifest still verifies after the no-op resume",
            verified_after_noop, str(verify_errors_after_noop),
        )
        check(
            "(11) a matching official_run_manifest.json (same fake "
            "environment) allows resume to proceed",
            summary_resume_full["overall_status"] == "already_complete",
        )
        check(
            "(27) re-running the environment probe against an output "
            "directory now full of result files still yields the same "
            "environment_fingerprint (result files are never hash inputs)",
            summary_resume_full["environment_fingerprint"] == summary1["environment_fingerprint"],
        )

        # --- (24) idempotent resume: integrity manifest missing -----------
        (out / INTEGRITY_MANIFEST_FILENAME).unlink()
        server_adapter_rebuild = FakeServerProcessAdapter()
        summary_rebuild = await run_official_campaign(
            bundle=bundle1, output_dir=out, host="127.0.0.1", port=19103, resume=True,
            api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=server_adapter_rebuild, sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
        )
        check(
            "(24) a campaign complete except for a missing integrity "
            "manifest rebuilds and verifies it without starting a server",
            summary_rebuild["overall_status"] == "complete"
            and len(server_adapter_rebuild.started) == 0
            and (out / INTEGRITY_MANIFEST_FILENAME).exists(),
        )

    # --- (5/6) partial resume: some valid, some missing ----------------------
    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle2 = _make_fixture_campaign_bundle(900102)
        block_ids2 = all_block_ids_in_schedule_order(bundle2)
        await run_official_campaign(
            bundle=bundle2, output_dir=out, host="127.0.0.1", port=19110, resume=False,
            api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=FakeServerProcessAdapter(), sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
        )
        target_block = block_ids2[2]
        target_eps = find_block(bundle2, target_block)
        kept_path = episode_result_path(out, target_eps[0].episode_id)
        kept_text = kept_path.read_text()
        old_stab_text = stabilization_result_path(out, target_block).read_text()
        for ep in target_eps[1:]:
            episode_result_path(out, ep.episode_id).unlink()

        server_adapter_partial = FakeServerProcessAdapter()
        summary_partial = await run_official_campaign(
            bundle=bundle2, output_dir=out, host="127.0.0.1", port=19111, resume=True,
            api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=server_adapter_partial, sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
        )
        new_stab_text = stabilization_result_path(out, target_block).read_text()
        check(
            "(5) a partially-executed block restarts the server, "
            "stabilizes again, and runs only its missing episodes -- other "
            "blocks are skipped with no server start",
            summary_partial["overall_status"] == "complete"
            and len(server_adapter_partial.started) == 1
            and new_stab_text != old_stab_text,
            str(summary_partial["block_statuses"]),
        )
        check(
            "(6b) the already-valid episode in the partially-executed "
            "block stays byte-for-byte untouched",
            kept_path.read_text() == kept_text,
        )

    # --- (7/8/9) partial/invalid/corrupted anywhere blocks all resume --------
    async def _resume_should_be_rejected(mutate_fn) -> tuple[bool, int]:
        with _tempfile.TemporaryDirectory() as tmp2:
            out2 = Path(tmp2) / "out"
            b = _make_fixture_campaign_bundle(900200 + hash(mutate_fn.__name__) % 1000)
            await run_official_campaign(
                bundle=b, output_dir=out2, host="127.0.0.1", port=19120, resume=False,
                api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
                server_adapter=FakeServerProcessAdapter(), sleeper=sleeper, clock=clock,
                run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
            )
            mutate_fn(b, out2)
            adapter = FakeServerProcessAdapter()
            raised = False
            try:
                await run_official_campaign(
                    bundle=b, output_dir=out2, host="127.0.0.1", port=19121, resume=True,
                    api_key=secret_key, transport=_make_success_transport(),
                    tokenizer_factory=_fixture_tokenizer_factory, server_adapter=adapter,
                    sleeper=sleeper, clock=clock, run_server_path=Path("/x/run_server.sh"),
                    environment_probe=FakeEnvironmentProbe(),
                )
            except ServerLifecycleError:
                raised = True
            return raised, len(adapter.started)

    def _make_partial(bundle_, out_dir):
        eps = find_block(bundle_, all_block_ids_in_schedule_order(bundle_)[0])
        p = episode_result_path(out_dir, eps[0].episode_id)
        obj = json.loads(p.read_text())
        obj["status"] = "failed"
        p.write_text(json.dumps(obj))

    def _make_invalid(bundle_, out_dir):
        eps = find_block(bundle_, all_block_ids_in_schedule_order(bundle_)[0])
        p = episode_result_path(out_dir, eps[0].episode_id)
        obj = json.loads(p.read_text())
        obj["runner_version"] = "not-the-real-runner"
        p.write_text(json.dumps(obj))

    def _make_corrupted(bundle_, out_dir):
        eps = find_block(bundle_, all_block_ids_in_schedule_order(bundle_)[0])
        p = episode_result_path(out_dir, eps[0].episode_id)
        p.write_text("{not valid json,,,")

    raised_partial, started_partial = await _resume_should_be_rejected(_make_partial)
    check(
        "(7) a 'partial' episode file anywhere in the campaign rejects "
        "--resume before any server starts",
        raised_partial and started_partial == 0,
    )
    raised_invalid, started_invalid = await _resume_should_be_rejected(_make_invalid)
    check(
        "(8) an 'invalid' episode file anywhere in the campaign rejects "
        "--resume before any server starts",
        raised_invalid and started_invalid == 0,
    )
    raised_corrupted, started_corrupted = await _resume_should_be_rejected(_make_corrupted)
    check(
        "(9) a 'corrupted' episode file anywhere in the campaign rejects "
        "--resume before any server starts",
        raised_corrupted and started_corrupted == 0,
    )

    # --- (10/25/26) environment fingerprint mismatches reject resume ---------
    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle3 = _make_fixture_campaign_bundle(900300)
        await run_official_campaign(
            bundle=bundle3, output_dir=out, host="127.0.0.1", port=19130, resume=False,
            api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=FakeServerProcessAdapter(), sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
        )

        gpu_mismatch_probe = FakeEnvironmentProbe()
        gpu_mismatch_probe.env["resolved_gpu"] = dict(gpu_mismatch_probe.env["resolved_gpu"])
        gpu_mismatch_probe.env["resolved_gpu"]["uuid"] = "GPU-a-different-physical-card"
        adapter_gpu = FakeServerProcessAdapter()
        raised_gpu = False
        try:
            await run_official_campaign(
                bundle=bundle3, output_dir=out, host="127.0.0.1", port=19131, resume=True,
                api_key=secret_key, transport=_make_success_transport(),
                tokenizer_factory=_fixture_tokenizer_factory, server_adapter=adapter_gpu,
                sleeper=sleeper, clock=clock, run_server_path=Path("/x/run_server.sh"),
                environment_probe=gpu_mismatch_probe,
            )
        except ServerLifecycleError:
            raised_gpu = True
        check(
            "(10/25) resuming on a different physical GPU UUID (even a "
            "second card of the identical model) rejects resume before "
            "any server starts",
            raised_gpu and len(adapter_gpu.started) == 0,
        )

        file_mismatch_probe = FakeEnvironmentProbe()
        file_mismatch_probe.env["file_hashes"] = dict(file_mismatch_probe.env["file_hashes"])
        file_mismatch_probe.env["file_hashes"]["run_phase_a.py"] = "a-different-hash-someone-edited-the-runner"
        adapter_file = FakeServerProcessAdapter()
        raised_file = False
        try:
            await run_official_campaign(
                bundle=bundle3, output_dir=out, host="127.0.0.1", port=19132, resume=True,
                api_key=secret_key, transport=_make_success_transport(),
                tokenizer_factory=_fixture_tokenizer_factory, server_adapter=adapter_file,
                sleeper=sleeper, clock=clock, run_server_path=Path("/x/run_server.sh"),
                environment_probe=file_mismatch_probe,
            )
        except ServerLifecycleError:
            raised_file = True
        check(
            "(26) a changed hash for a tracked runner/shell file rejects "
            "resume before any server starts",
            raised_file and len(adapter_file.started) == 0,
        )

    # --- (13) a block failure prevents every later block ----------------------
    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle4 = _make_fixture_campaign_bundle(900400)
        block_ids4 = all_block_ids_in_schedule_order(bundle4)
        second_block_eps = find_block(bundle4, block_ids4[1])
        t4 = _make_success_transport()
        t4.queue_script(
            f"{second_block_eps[0].episode_id}:victim:0",
            FakeStreamScript(
                prompt_token_ids_echo=list(range(second_block_eps[0].victim_input_len)),
                token_events=[[1]] * (second_block_eps[0].victim_output_len - 1),
                usage={
                    "prompt_tokens": second_block_eps[0].victim_input_len,
                    "completion_tokens": second_block_eps[0].victim_output_len - 1,
                },
            ),
        )
        summary4 = await run_official_campaign(
            bundle=bundle4, output_dir=out, host="127.0.0.1", port=19140, resume=False,
            api_key=secret_key, transport=t4, tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=FakeServerProcessAdapter(), sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
        )
        check(
            "(13) a block failure halts the campaign and no later block "
            "is ever attempted",
            summary4["overall_status"] == "block_failed"
            and summary4["failed_block"] == block_ids4[1]
            and len(summary4["block_statuses"]) == 2
            and not (out / INTEGRITY_MANIFEST_FILENAME).exists(),
            str(summary4["block_statuses"]),
        )

    # --- (14) a server-stop failure prevents 'complete' -----------------------
    class _StuckServerAdapter:
        def __init__(self) -> None:
            self.started: list[FakeServerHandle] = []

        def start(self, cmd: list[str], log_path: Path) -> FakeServerHandle:
            h = FakeServerHandle(cmd, dies_on_terminate=False, dies_on_kill=False)
            self.started.append(h)
            return h

    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle5 = _make_fixture_campaign_bundle(900500)
        summary5 = await run_official_campaign(
            bundle=bundle5, output_dir=out, host="127.0.0.1", port=19150, resume=False,
            api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=_StuckServerAdapter(), sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
            stop_timeout_s=0.01, stop_kill_confirm_timeout_s=0.01, stop_port_poll_timeout_s=0.01,
        )
        check(
            "(14) a server that never verifiably stops prevents "
            "overall_status='complete', even if every episode succeeded",
            summary5["overall_status"] == "server_stop_failed"
            and not (out / INTEGRITY_MANIFEST_FILENAME).exists(),
            str(summary5.get("overall_status")),
        )

    # --- (15/16) SIGINT / SIGTERM -> interrupted, server stop, exit code ------
    async def _signal_scenario(signal_name: str, expected_exit_code: int) -> None:
        with _tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            bundle_sig = _make_fixture_campaign_bundle(900600 + (1 if signal_name == "SIGTERM" else 0))
            t_sig = FakeTransport()
            t_sig.default_script_factory = lambda payload: FakeStreamScript(
                hang=True, prompt_token_ids_echo=None, token_events=[]
            )
            t_sig.set_get_response(HEALTH_ENDPOINT, 200, {})
            t_sig.set_get_response(
                MODELS_ENDPOINT, 200, {"data": [{"id": MODEL_FULL_ID["llama"]}, {"id": MODEL_FULL_ID["qwen"]}]}
            )
            t_sig.set_get_response(OPENAPI_ENDPOINT, 200, {"paths": {}})
            interrupt_state = InterruptState()
            server_adapter_sig = FakeServerProcessAdapter()

            async def _trigger_soon() -> None:
                await asyncio.sleep(0.05)
                interrupt_state.trigger(signal_name)

            trigger_task = asyncio.create_task(_trigger_soon())
            summary_sig = await run_official_campaign(
                bundle=bundle_sig, output_dir=out, host="127.0.0.1", port=19160, resume=False,
                api_key=secret_key, transport=t_sig, tokenizer_factory=_fixture_tokenizer_factory,
                server_adapter=server_adapter_sig, sleeper=sleeper, clock=clock,
                run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
                interrupt_state=interrupt_state,
            )
            await trigger_task
            disk_summary = json.loads((out / OFFICIAL_RUN_SUMMARY_FILENAME).read_text())
            exit_code = official_run_exit_code(summary_sig)
            check(
                f"({'15' if signal_name == 'SIGINT' else '16'}a) {signal_name} "
                f"produces overall_status='interrupted' with the server stopped",
                summary_sig["overall_status"] == "interrupted"
                and summary_sig["interrupted_by"] == signal_name
                and len(server_adapter_sig.started) >= 1
                and server_adapter_sig.started[0].terminated,
                str(summary_sig.get("overall_status")),
            )
            check(
                f"({'15' if signal_name == 'SIGINT' else '16'}b) the persisted "
                f"official_run_summary.json also records 'interrupted'/{signal_name}",
                disk_summary["overall_status"] == "interrupted" and disk_summary["interrupted_by"] == signal_name,
            )
            check(
                f"({'15' if signal_name == 'SIGINT' else '16'}c) the exit-code "
                f"mapping for {signal_name} is {expected_exit_code}",
                exit_code == expected_exit_code,
                str(exit_code),
            )
            check(
                f"({'15' if signal_name == 'SIGINT' else '16'}d) no integrity "
                f"manifest is produced for an interrupted run",
                not (out / INTEGRITY_MANIFEST_FILENAME).exists(),
            )

    await _signal_scenario("SIGINT", 130)
    await _signal_scenario("SIGTERM", 143)

    # --- exit-code pure-function checks (no process/signal needed) -----------
    check("(15e) official_run_exit_code maps SIGINT -> 130", official_run_exit_code({"overall_status": "interrupted", "interrupted_by": "SIGINT"}) == 130)
    check("(16e) official_run_exit_code maps SIGTERM -> 143", official_run_exit_code({"overall_status": "interrupted", "interrupted_by": "SIGTERM"}) == 143)
    check("(exit-ok) official_run_exit_code maps complete -> 0", official_run_exit_code({"overall_status": "complete"}) == 0)
    check("(exit-ok2) official_run_exit_code maps already_complete -> 0", official_run_exit_code({"overall_status": "already_complete"}) == 0)
    check("(exit-fail) official_run_exit_code maps block_failed -> 1", official_run_exit_code({"overall_status": "block_failed"}) == 1)

    # --- (20) --smoke-test regression sentinel --------------------------------
    check(
        "(20) --smoke-test's own dedicated code path (run_smoke_block) is "
        "unchanged and still separate from the official campaign",
        inspect.getsource(run_smoke_block).count("_run_block_protocol(") == 1
        and "smoke_run_summary.json" in inspect.getsource(run_smoke_block),
    )

    # --- (21) --dry-run has no side effects (source-level sentinel) ----------
    main_source_for_dryrun = inspect.getsource(main)
    dry_run_branch = main_source_for_dryrun.split("if args.dry_run:")[1].split("if args.official_run:")[0]
    check(
        "(21) --dry-run's code path never starts a server, opens a "
        "transport, or writes a result file",
        "run_official_campaign(" not in dry_run_branch
        and "run_smoke_block(" not in dry_run_branch
        and "HttpxTransport(" not in dry_run_branch
        and "write_json_atomic(" not in dry_run_branch,
    )

    # --- (22) --official-run is no longer artificially disabled --------------
    official_branch_for_22 = main_source_for_dryrun.split("if args.official_run:")[1].split(
        "assert args.smoke_test"
    )[0]
    check(
        "(22) --official-run's code path actually runs the campaign and "
        "no longer just prints a 'disabled until Stage 3' message",
        "run_official_campaign(" in inspect.getsource(_run_official_cli)
        and "disabled until Stage 3" not in official_branch_for_22,
    )

    # =========================================================================
    # Patch: Stage 3 Contract Blockers Only
    # =========================================================================

    # --- Section 1: strict environment gate ----------------------------------
    def _good_env() -> dict:
        return FakeEnvironmentProbe().gather(Path("/x"))

    check(
        "(env-gate-ok) a well-formed fake environment passes the gate cleanly",
        validate_official_environment(_good_env()) == [],
    )
    env_dirty = _good_env()
    env_dirty["git_dirty"] = True
    check(
        "(11-1) git_dirty=True is a hard error, before any output change "
        "and before any server start",
        len(validate_official_environment(env_dirty)) >= 1
        and any("git_dirty" in e for e in validate_official_environment(env_dirty)),
    )
    env_no_gpu = _good_env()
    env_no_gpu["resolved_gpu"] = None
    check(
        "(11-2) resolved_gpu=None is a hard error",
        any("resolved_gpu" in e for e in validate_official_environment(env_no_gpu)),
    )
    env_missing_hash = _good_env()
    env_missing_hash["file_hashes"] = dict(env_missing_hash["file_hashes"])
    env_missing_hash["file_hashes"]["run_phase_a.py"] = None
    check(
        "(11-3) a required file hash being None is a hard error",
        len(validate_official_environment(env_missing_hash)) >= 1,
    )
    env_bad_commit = _good_env()
    env_bad_commit["git_commit"] = "not-40-hex-chars"
    check(
        "(env-gate) a git_commit that isn't a 40-char hex hash is a hard error",
        any("git_commit" in e for e in validate_official_environment(env_bad_commit)),
    )
    env_empty_version = _good_env()
    env_empty_version["vllm_version"] = ""
    check(
        "(env-gate) an empty version string is a hard error, not silently tolerated",
        any("vllm_version" in e for e in validate_official_environment(env_empty_version)),
    )

    # The gate must actually be enforced by the campaign, before any write.
    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle_gate = _make_fixture_campaign_bundle(901001)
        bad_probe = FakeEnvironmentProbe()
        bad_probe.env["git_dirty"] = True
        adapter_gate = FakeServerProcessAdapter()
        raised_gate = False
        try:
            await run_official_campaign(
                bundle=bundle_gate, output_dir=out, host="127.0.0.1", port=19170, resume=False,
                api_key=secret_key, transport=_make_success_transport(),
                tokenizer_factory=_fixture_tokenizer_factory, server_adapter=adapter_gate,
                sleeper=sleeper, clock=clock, run_server_path=Path("/x/run_server.sh"),
                environment_probe=bad_probe,
            )
        except ServerLifecycleError:
            raised_gate = True
        check(
            "(env-gate-enforced) a dirty tracked working tree rejects even "
            "a fresh run before any output-directory change or server start",
            raised_gate and len(adapter_gate.started) == 0 and not out.exists(),
        )

    # --- Section 2: real fresh-run emptiness contract ------------------------
    async def _fresh_run_should_be_rejected(seed_stray: int, populate) -> tuple[bool, int, bool]:
        with _tempfile.TemporaryDirectory() as tmp2:
            out2 = Path(tmp2) / "out"
            populate(out2)
            b = _make_fixture_campaign_bundle(seed_stray)
            adapter = FakeServerProcessAdapter()
            raised = False
            try:
                await run_official_campaign(
                    bundle=b, output_dir=out2, host="127.0.0.1", port=19180, resume=False,
                    api_key=secret_key, transport=_make_success_transport(),
                    tokenizer_factory=_fixture_tokenizer_factory, server_adapter=adapter,
                    sleeper=sleeper, clock=clock, run_server_path=Path("/x/run_server.sh"),
                    environment_probe=FakeEnvironmentProbe(),
                )
            except ServerLifecycleError:
                raised = True
            manifest_written = (out2 / OFFICIAL_RUN_MANIFEST_FILENAME).exists()
            return raised, len(adapter.started), manifest_written

    def _populate_stray_server_log(out_dir: Path) -> None:
        (out_dir / "server_logs").mkdir(parents=True)
        (out_dir / "server_logs" / "old.log").write_text("stale")

    def _populate_unknown_file(out_dir: Path) -> None:
        out_dir.mkdir(parents=True)
        (out_dir / "random_unknown_file.txt").write_text("???")

    raised_log, started_log, manifest_log = await _fresh_run_should_be_rejected(901010, _populate_stray_server_log)
    check(
        "(11-4) a fresh run with a stray server_logs/old.log aborts before "
        "any output change and before any server start",
        raised_log and started_log == 0 and not manifest_log,
    )
    raised_unk, started_unk, manifest_unk = await _fresh_run_should_be_rejected(901011, _populate_unknown_file)
    check(
        "(11-5) a fresh run with a completely unknown file aborts before "
        "any output change and before any server start",
        raised_unk and started_unk == 0 and not manifest_unk,
    )

    # --- Section 4: an existing but invalid integrity manifest is never resealed
    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle_reseal = _make_fixture_campaign_bundle(901020)
        await run_official_campaign(
            bundle=bundle_reseal, output_dir=out, host="127.0.0.1", port=19190, resume=False,
            api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=FakeServerProcessAdapter(), sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
        )
        integrity_path_reseal = out / INTEGRITY_MANIFEST_FILENAME
        original_integrity_bytes = integrity_path_reseal.read_bytes()
        # Tamper with a captured file's bytes without changing its parsed
        # JSON semantics (still classifies valid_complete) -- this must be
        # caught specifically by the integrity manifest's own hash check,
        # not by the episode-classification scan.
        stab_file = next((out / "stabilization").glob("*.json"))
        stab_file.write_text(stab_file.read_text() + " ")

        adapter_reseal = FakeServerProcessAdapter()
        raised_reseal = False
        reseal_error = ""
        try:
            await run_official_campaign(
                bundle=bundle_reseal, output_dir=out, host="127.0.0.1", port=19191, resume=True,
                api_key=secret_key, transport=_make_success_transport(),
                tokenizer_factory=_fixture_tokenizer_factory, server_adapter=adapter_reseal,
                sleeper=sleeper, clock=clock, run_server_path=Path("/x/run_server.sh"),
                environment_probe=FakeEnvironmentProbe(),
            )
        except ServerLifecycleError as exc:
            raised_reseal = True
            reseal_error = str(exc)
        check(
            "(7/8/11-7/11-8) an existing integrity manifest that fails deep "
            "verification (a captured file's hash no longer matches) is a "
            "hard resume abort, never automatically resealed, no server start",
            raised_reseal
            and len(adapter_reseal.started) == 0
            and integrity_path_reseal.read_bytes() == original_integrity_bytes
            and ("sha256" in reseal_error or "size mismatch" in reseal_error or "integrity" in reseal_error.lower()),
            reseal_error[:200],
        )

    # --- Section 6: stabilization deep validator rejects a bare status stub --
    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle_stub = _make_fixture_campaign_bundle(901030)
        await run_official_campaign(
            bundle=bundle_stub, output_dir=out, host="127.0.0.1", port=19200, resume=False,
            api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=FakeServerProcessAdapter(), sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
        )
        (out / INTEGRITY_MANIFEST_FILENAME).unlink()
        block_ids_stub = all_block_ids_in_schedule_order(bundle_stub)
        target_stub = block_ids_stub[0]
        stab_path_stub = stabilization_result_path(out, target_stub)
        stab_path_stub.write_text(json.dumps({"status": "complete"}))

        stub_errors = validate_complete_stabilization_file(
            {"status": "complete"}, bundle=bundle_stub, block_id=target_stub,
            model_key=find_block(bundle_stub, target_stub)[0].model, offload_gb=0,
            state_label="low", run_mode=RUN_MODE_OFFICIAL,
        )
        check(
            "(9) a fake stabilization file with only status='complete' is "
            "rejected by the deep validator (many missing fields)",
            len(stub_errors) >= 5,
        )

        adapter_stub = FakeServerProcessAdapter()
        summary_stub = await run_official_campaign(
            bundle=bundle_stub, output_dir=out, host="127.0.0.1", port=19201, resume=True,
            api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=adapter_stub, sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
        )
        new_stab_stub = json.loads(stab_path_stub.read_text())
        check(
            "(9b) resume never accepts the fake stub -- it re-executes "
            "exactly that block's stabilization for real",
            summary_stub["overall_status"] == "complete"
            and len(adapter_stub.started) == 1
            and len(new_stab_stub.get("request_results", [])) == STABILIZATION_REQUEST_COUNT,
        )

    # --- Section 7: block-summary deep validator rejects a bad server_stop ---
    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle_bs = _make_fixture_campaign_bundle(901040)
        await run_official_campaign(
            bundle=bundle_bs, output_dir=out, host="127.0.0.1", port=19210, resume=False,
            api_key=secret_key, transport=_make_success_transport(), tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=FakeServerProcessAdapter(), sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
        )
        block_ids_bs = all_block_ids_in_schedule_order(bundle_bs)
        target_bs = block_ids_bs[0]
        bs_path = out / "block_summaries" / f"{target_bs}.json"
        bs_obj = json.loads(bs_path.read_text())
        bs_obj["server_stop"]["stop_success"] = False
        bs_errors = validate_complete_block_summary(
            bs_obj, bundle=bundle_bs, block_id=target_bs,
            model_key=find_block(bundle_bs, target_bs)[0].model, offload_gb=0, state_label="low",
            repeat=find_block(bundle_bs, target_bs)[0].repeat, run_mode=RUN_MODE_OFFICIAL,
        )
        check(
            "(10) a block summary with server_stop.stop_success=false is "
            "rejected by the deep validator",
            any("stop_success" in e for e in bs_errors),
        )

    # --- Section 8: no server start after an interrupt (tokenizer-loading) ---
    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle_tok = _make_fixture_campaign_bundle(901050)
        interrupt_state_tok = InterruptState()

        def _tokenizer_factory_triggers_sigint(model_key: str) -> FakeTokenizerAdapter:
            interrupt_state_tok.trigger("SIGINT")
            return FakeTokenizerAdapter(vocab_size=2000, special_token_ids={0, 1, 2})

        adapter_tok = FakeServerProcessAdapter()
        summary_tok = await run_official_campaign(
            bundle=bundle_tok, output_dir=out, host="127.0.0.1", port=19220, resume=False,
            api_key=secret_key, transport=_make_success_transport(),
            tokenizer_factory=_tokenizer_factory_triggers_sigint, server_adapter=adapter_tok,
            sleeper=sleeper, clock=clock, run_server_path=Path("/x/run_server.sh"),
            environment_probe=FakeEnvironmentProbe(), interrupt_state=interrupt_state_tok,
        )
        check(
            "(11-11) a signal that fires during tokenizer loading is "
            "caught immediately afterward -- zero server starts",
            summary_tok["overall_status"] == "interrupted" and len(adapter_tok.started) == 0,
            str(summary_tok.get("overall_status")),
        )

    # --- Section 9: full cleanup when interrupted mid-episode -----------------
    with _tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        bundle_mid = _make_fixture_campaign_bundle(901060)
        t_mid = FakeTransport()
        t_mid.set_get_response(HEALTH_ENDPOINT, 200, {})
        t_mid.set_get_response(
            MODELS_ENDPOINT, 200, {"data": [{"id": MODEL_FULL_ID["llama"]}, {"id": MODEL_FULL_ID["qwen"]}]}
        )
        t_mid.set_get_response(OPENAPI_ENDPOINT, 200, {"paths": {}})

        def _mid_factory(payload: dict) -> FakeStreamScript:
            if payload["max_tokens"] == STABILIZATION_OUTPUT_LEN and len(payload["prompt"]) == STABILIZATION_INPUT_LEN:
                return _success_script_factory(payload)
            return FakeStreamScript(hang=True, prompt_token_ids_echo=None, token_events=[])

        t_mid.default_script_factory = _mid_factory
        interrupt_state_mid = InterruptState()

        async def _trigger_mid_episode() -> None:
            await asyncio.sleep(0.05)
            interrupt_state_mid.trigger("SIGTERM")

        trigger_task = asyncio.create_task(_trigger_mid_episode())
        adapter_mid = FakeServerProcessAdapter()
        summary_mid = await run_official_campaign(
            bundle=bundle_mid, output_dir=out, host="127.0.0.1", port=19230, resume=False,
            api_key=secret_key, transport=t_mid, tokenizer_factory=_fixture_tokenizer_factory,
            server_adapter=adapter_mid, sleeper=sleeper, clock=clock,
            run_server_path=Path("/x/run_server.sh"), environment_probe=FakeEnvironmentProbe(),
            interrupt_state=interrupt_state_mid,
        )
        await trigger_task
        check(
            "(11-12/9) a signal during a regular episode's active streams "
            "results in overall_status='interrupted' with zero active "
            "fake streams left running afterward",
            summary_mid["overall_status"] == "interrupted" and t_mid.active_stream_count == 0,
            str(summary_mid.get("overall_status")),
        )
        check(
            "(9b) the interrupted block's server was still verifiably "
            "stopped, and no integrity manifest was produced",
            len(adapter_mid.started) >= 1
            and adapter_mid.started[0].terminated
            and not (out / INTEGRITY_MANIFEST_FILENAME).exists(),
        )

    return results


