from __future__ import annotations

import asyncio
import inspect
import json
import socket
import tempfile
from pathlib import Path
from typing import Callable

import make_prefill_screen_schedule as generator
import run_prefill_screen as runner


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_frozen_bundle(directory: Path) -> str:
    episodes = generator.generate_schedule(
        generator.DEFAULT_MODELS[0],
        generator.DEFAULT_REPEATS,
        generator.DEFAULT_SEED,
    )
    errors = generator.validate_schedule(
        episodes,
        generator.DEFAULT_MODELS[0],
        generator.DEFAULT_REPEATS,
        generator.DEFAULT_SEED,
    )
    errors.extend(
        generator.validate_global(
            episodes,
            generator.DEFAULT_MODELS,
            generator.DEFAULT_REPEATS,
        )
    )
    if errors:
        raise AssertionError(f"generator fixture validation failed: {errors}")

    payload = generator.build_canonical_payload(
        episodes,
        generator.DEFAULT_MODELS,
        generator.DEFAULT_REPEATS,
        generator.DEFAULT_SEED,
    )
    fingerprint = generator.compute_fingerprint(payload)
    csv_text = generator.build_csv_text(episodes)
    json_text = generator.build_json_text(payload, fingerprint)
    audit_text = generator.build_audit_text(
        {generator.DEFAULT_MODELS[0]: episodes},
        generator.DEFAULT_REPEATS,
        generator.DEFAULT_SEED,
        episodes,
        fingerprint,
    )
    consistency_errors = generator.check_csv_json_consistency(csv_text, json_text)
    if consistency_errors:
        raise AssertionError(f"fixture CSV/JSON mismatch: {consistency_errors}")

    generator.write_and_replace_output_files(
        [
            (directory / "prefill_screen_schedule.csv", csv_text),
            (directory / "prefill_screen_schedule.json", json_text),
            (directory / "prefill_screen_schedule_audit.txt", audit_text),
        ]
    )
    return fingerprint


def _success_transport() -> runner.FakeTransport:
    transport = runner.FakeTransport()
    transport.set_get_response(runner.HEALTH_ENDPOINT, 200, {})
    transport.set_get_response(
        runner.MODELS_ENDPOINT,
        200,
        {"data": [{"id": runner.MODEL_FULL_ID["llama"]}]},
    )
    transport.set_get_response(
        runner.OPENAPI_ENDPOINT,
        200,
        {"paths": {runner.COMPLETIONS_ENDPOINT: {"post": {}}}},
    )
    return transport


def run_self_test() -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, bool(condition), detail))

    check(
        "derive_seed is deterministic and bounded",
        runner.derive_seed("a", "b") == runner.derive_seed("a", "b")
        and 0 <= runner.derive_seed("a", "b") < 2**31 - 1,
    )
    check("design version is frozen", runner.DESIGN_VERSION == "prefill-screen-v1")
    check("schedule seed is frozen", runner.OFFICIAL_SEED == 20260716)
    check("only llama is enabled", runner.OFFICIAL_MODELS == ["llama"])
    check("three repeats are frozen", runner.OFFICIAL_REPEATS == 3)
    check("24 episodes are frozen", runner.OFFICIAL_EPISODE_COUNT == 24)
    check("six blocks are frozen", runner.BLOCKS_PER_MODEL == 6)
    check(
        "conditions are no_burst and prefill_burst",
        runner.OFFICIAL_CONDITIONS == ["no_burst", "prefill_burst"],
    )
    check(
        "prefill burst shape is 4 x 2048/16",
        runner.OFFICIAL_BURST_CONFIGURATION
        == {
            "burst_parallel_requests": 4,
            "burst_input_len": 2048,
            "burst_output_len": 16,
            "burst_temperature": 0.0,
        },
    )
    check(
        "state sequence is low/high, high/low, low/high",
        runner.EXPECTED_STATE_SEQUENCE
        == ["low", "high", "high", "low", "low", "high"],
    )
    check(
        "environment hash inputs are exactly the six Prefill-Screen files",
        runner.EXPECTED_ENVIRONMENT_FILE_HASH_NAMES
        == frozenset(
            {
                "run_prefill_screen.py",
                "run_prefill_screen.sh",
                "run_server.sh",
                "prefill_screen_schedule.json",
                "prefill_screen_schedule.csv",
                "prefill_screen_schedule_audit.txt",
            }
        ),
    )
    check(
        "server port-release polling default remains 30 seconds",
        inspect.signature(runner.stop_server)
        .parameters["port_poll_timeout_s"]
        .default
        == 30.0,
    )
    check(
        "server command has the expected five arguments and no API key",
        runner.build_server_command(
            Path("/x/run_server.sh"), "llama", 12, "127.0.0.1", 8123
        )
        == ["bash", "/x/run_server.sh", "llama", "12", "127.0.0.1", "8123"],
    )
    check(
        "fake official environment passes the strict gate",
        runner.validate_official_environment(runner.FakeEnvironmentProbe().gather(Path(".")))
        == [],
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bundle_dir = root / "bundle"
        bundle_dir.mkdir()
        fingerprint = _write_frozen_bundle(bundle_dir)
        check(
            "generated fingerprint matches the runner's frozen fingerprint",
            fingerprint == runner.OFFICIAL_FINGERPRINT,
            fingerprint,
        )

        bundle, errors = runner.load_and_validate_bundle(bundle_dir)
        check("runner accepts the generated three-file bundle", not errors, str(errors))
        if bundle is not None:
            plan = runner.build_execution_plan(bundle)
            check("bundle contains exactly 24 episodes", len(bundle.episodes) == 24)
            check("execution plan contains exactly six blocks", len(plan["blocks"]) == 6)
            check("execution plan has six stabilization runs", plan["planned_stabilization_runs"] == 6)
            check("execution plan has 12 no_burst episodes", plan["no_burst_count"] == 12)
            check("execution plan has 12 prefill_burst episodes", plan["burst_condition_count"] == 12)
            check(
                "all schedule models are llama",
                {episode.model for episode in bundle.episodes} == {"llama"},
            )
            check(
                "all design cells have repeats 1,2,3",
                all(
                    {
                        ep.repeat
                        for ep in bundle.episodes
                        if ep.offload_gb == offload
                        and ep.concurrency == concurrency
                        and ep.condition == condition
                    }
                    == {1, 2, 3}
                    for offload in (0, 12)
                    for concurrency in (4, 8)
                    for condition in ("no_burst", "prefill_burst")
                ),
            )

            # Full 6-block / 24-episode campaign against fakes. This exercises
            # the actual official orchestrator, trigger/burst path, atomic
            # episode outputs, block summaries, shutdown verification, and
            # final integrity manifest without GPU, network, or sleeping.
            transport = _success_transport()
            server_adapter = runner.FakeServerProcessAdapter()
            fake_tokenizer = runner.FakeTokenizerAdapter(
                vocab_size=4096, special_token_ids={0, 1, 2}
            )
            api_key = "self-test-secret-that-must-not-leak"
            output_dir = root / "official_output"
            port = _free_tcp_port()

            summary = asyncio.run(
                runner.run_official_campaign(
                    bundle=bundle,
                    output_dir=output_dir,
                    host="127.0.0.1",
                    port=port,
                    resume=False,
                    api_key=api_key,
                    transport=transport,
                    tokenizer_factory=lambda _model: fake_tokenizer,
                    server_adapter=server_adapter,
                    sleeper=runner.FakeSleeper(),
                    clock=runner.RealClock(),
                    run_server_path=root / "run_server.sh",
                    environment_probe=runner.FakeEnvironmentProbe(),
                )
            )
            check(
                "fake official campaign completes all 24 episodes",
                summary.get("overall_status") == "complete"
                and summary.get("valid_complete_episodes") == 24
                and summary.get("missing_episodes") == 0,
                json.dumps(summary, sort_keys=True),
            )
            check(
                "fake official campaign starts and stops exactly six servers",
                len(server_adapter.started) == 6
                and all(handle.terminated and not handle.alive for handle in server_adapter.started),
            )
            check(
                "official output contains 24 episodes, 6 stabilization files, and 6 block summaries",
                len(list((output_dir / "episodes").glob("*.json"))) == 24
                and len(list((output_dir / "stabilization").glob("*.json"))) == 6
                and len(list((output_dir / "block_summaries").glob("*.json"))) == 6,
            )
            check(
                "final integrity manifest exists",
                (output_dir / runner.INTEGRITY_MANIFEST_FILENAME).is_file(),
            )

            episode_objects = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((output_dir / "episodes").glob("*.json"))
            ]
            classifications_ok = True
            for episode in bundle.episodes:
                classification, notes = runner.classify_result_file(
                    runner.episode_result_path(output_dir, episode.episode_id),
                    episode,
                    bundle.fingerprint,
                    runner.RUN_MODE_OFFICIAL,
                )
                if classification != runner.CLASSIFICATION_VALID_COMPLETE or notes:
                    classifications_ok = False
                    break
            check("every written episode passes deep resume validation", classifications_ok)

            burst_payloads = [
                payload
                for payload in transport.seen_payloads
                if len(payload.get("prompt", [])) == 2048
                and payload.get("max_tokens") == 16
            ]
            check(
                "the 12 prefill_burst episodes start exactly four burst requests each",
                len(burst_payloads) == 48,
                f"observed {len(burst_payloads)} burst-shaped payloads",
            )
            check(
                "stored aggregates use throughput, not the incorrect goodput name",
                all(
                    "victim_throughput_tokens_per_s" in obj.get("aggregate_metrics", {})
                    and "victim_goodput_tokens_per_s" not in obj.get("aggregate_metrics", {})
                    for obj in episode_objects
                ),
            )
            check(
                "the API key is absent from all serialized JSON outputs",
                api_key
                not in "".join(
                    path.read_text(encoding="utf-8")
                    for path in output_dir.rglob("*.json")
                ),
            )
            check(
                "no temporary atomic-write files remain",
                not list(output_dir.rglob("*.tmp.*")),
            )

            # A complete resume must be a server-free no-op.
            before = {
                path.relative_to(output_dir).as_posix(): path.read_bytes()
                for path in output_dir.rglob("*")
                if path.is_file()
            }
            resume_adapter = runner.FakeServerProcessAdapter()
            resumed = asyncio.run(
                runner.run_official_campaign(
                    bundle=bundle,
                    output_dir=output_dir,
                    host="127.0.0.1",
                    port=port,
                    resume=True,
                    api_key=api_key,
                    transport=_success_transport(),
                    tokenizer_factory=lambda _model: fake_tokenizer,
                    server_adapter=resume_adapter,
                    sleeper=runner.FakeSleeper(),
                    clock=runner.RealClock(),
                    run_server_path=root / "run_server.sh",
                    environment_probe=runner.FakeEnvironmentProbe(),
                )
            )
            after = {
                path.relative_to(output_dir).as_posix(): path.read_bytes()
                for path in output_dir.rglob("*")
                if path.is_file()
            }
            check(
                "complete resume is a byte-exact no-op with no server start",
                resumed.get("overall_status") == "already_complete"
                and not resume_adapter.started
                and before == after,
                json.dumps(resumed, sort_keys=True),
            )

    print("Prefill-Screen self-test results")
    print("=" * 72)
    all_passed = True
    for name, passed, detail in results:
        status = "OK" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {name}")
        if detail and not passed:
            print(f"       {detail}")
    print("=" * 72)
    print(f"{sum(1 for _, passed, _ in results if passed)}/{len(results)} checks passed")
    print("SELF-TEST: PASS" if all_passed else "SELF-TEST: FAIL")
    return 0 if all_passed else 1
