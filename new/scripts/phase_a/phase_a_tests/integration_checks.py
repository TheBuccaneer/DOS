"""
Fake full-block integration test and the real-subprocess SIGINT/SIGTERM
test, mechanically moved out of run_phase_a.py's former "Self-test"
section, verbatim -- no logic changes.
"""

from __future__ import annotations

from run_phase_a import *  # noqa: F401,F403

from phase_a_tests.fixtures import (
    _build_fixture_episodes,
    _make_fixture_block_bundle,
    _make_success_transport,
)


def run_fake_block_integration_test() -> tuple[bool, list[str]]:
    """Section 26: a full simulated block run -- server start/readiness
    simulated, 20 stabilization requests, simulated cooldown, four
    complete regular episodes, simulated server stop -- with every JSON
    output validated. No sleeping, no GPU, no real network/server."""
    notes: list[str] = []
    ok = True

    def note(msg: str) -> None:
        notes.append(msg)

    try:
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture_seed = 999001
            # "llama" is reused deliberately so MODEL_FULL_ID resolves; this
            # is still an isolated, self-built fixture bundle in a fresh
            # temp dir, not the real official schedule.
            bundle, block_id = _make_fixture_block_bundle("llama", fixture_seed)
            episodes = bundle.episodes[:BLOCK_SIZE]

            tok = FakeTokenizerAdapter(vocab_size=2000, special_token_ids={0, 1, 2})
            transport = _make_success_transport()
            server_adapter = FakeServerProcessAdapter()
            sleeper = FakeSleeper()
            clock = RealClock()
            fake_api_key = "fake-integration-secret-9f8e7d"

            summary = asyncio.run(
                run_smoke_block(
                    bundle=bundle, block_id=block_id, output_dir=tmp_path / "out",
                    host="127.0.0.1", port=18200, resume=False, api_key=fake_api_key,
                    transport=transport, tokenizer=tok, server_adapter=server_adapter,
                    sleeper=sleeper, clock=clock, run_server_path=tmp_path / "run_server.sh",
                )
            )

            if summary.get("overall_status") != "block_complete":
                ok = False
                note(f"expected overall_status='block_complete', got {summary.get('overall_status')!r}")

            if len(server_adapter.started) != 1:
                ok = False
                note(f"expected exactly one simulated server start, got {len(server_adapter.started)}")
            elif not server_adapter.started[0].terminated:
                ok = False
                note("simulated server was never terminated by stop_server()")

            stab_path = stabilization_result_path(tmp_path / "out", block_id)
            if not stab_path.exists():
                ok = False
                note("stabilization output file is missing")
            else:
                stab_obj = json.loads(stab_path.read_text(encoding="utf-8"))
                if stab_obj.get("status") != REQUEST_STATUS_COMPLETE:
                    ok = False
                    note("stabilization status != complete")
                if len(stab_obj.get("request_results", [])) != STABILIZATION_REQUEST_COUNT:
                    ok = False
                    note("stabilization did not run exactly 20 requests")
                if stab_obj.get("record_type") != RECORD_TYPE_STABILIZATION:
                    ok = False
                    note("stabilization record_type is wrong")

            for ep in episodes:
                p = episode_result_path(tmp_path / "out", ep.episode_id)
                if not p.exists():
                    ok = False
                    note(f"episode output file missing for {ep.episode_id}")
                    continue
                obj = json.loads(p.read_text(encoding="utf-8"))
                if obj.get("status") != REQUEST_STATUS_COMPLETE:
                    ok = False
                    note(f"episode {ep.episode_id} status != complete")
                if obj.get("result_schema_version") != RESULT_SCHEMA_VERSION:
                    ok = False
                    note(f"episode {ep.episode_id} has the wrong result_schema_version")
                if obj.get("runner_version") != RUNNER_VERSION:
                    ok = False
                    note(f"episode {ep.episode_id} has the wrong runner_version")
                if len(obj.get("victim_requests", [])) != ep.victim_request_count:
                    ok = False
                    note(f"episode {ep.episode_id} has the wrong victim_requests count")
                expected_burst = ep.burst_parallel_requests if ep.condition == "fixed_burst" else 0
                if len(obj.get("burst_requests", [])) != expected_burst:
                    ok = False
                    note(f"episode {ep.episode_id} has the wrong burst_requests count")
                if fake_api_key in json.dumps(obj):
                    ok = False
                    note(f"API key leaked into episode {ep.episode_id} result file")

            leftovers = list((tmp_path / "out").rglob("*.tmp.*"))
            if leftovers:
                ok = False
                note(f"atomic writer left temp file(s) behind: {leftovers}")

            if fake_api_key in json.dumps(summary):
                ok = False
                note("API key leaked into smoke_run_summary.json")

            if not (tmp_path / "out" / "smoke_run_summary.json").exists():
                ok = False
                note("smoke_run_summary.json was not written")

    except Exception as exc:  # noqa: BLE001 -- a failing integration test must report, not crash --self-test
        ok = False
        note(f"fake block integration test raised an unexpected exception: {exc!r}")

    return ok, notes


def run_subprocess_signal_test(
    signal_value: int, expected_signal_name: str, expected_exit_code: int,
) -> tuple[bool, str]:
    """Section 10: spawns a real, isolated child process running a fake
    official campaign (hanging fake transport, fake server adapter --
    no GPU, no network, no real vLLM server), sends it a genuine OS
    signal via os.kill(), and verifies the persisted 'interrupted'
    summary plus the child's exit code."""
    child_script = f'''\
import asyncio
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, {str(SCRIPT_PATH.parent)!r})
import run_phase_a as rp
from phase_a_tests.fixtures import _build_fixture_episodes


def _make_bundle(seed):
    episodes = _build_fixture_episodes("llama", seed) + _build_fixture_episodes("qwen", seed)
    return rp.LoadedBundle(
        schedule_dir=Path("/nonexistent-fixture-subprocess-only"),
        json_obj={{"seed": seed, "design_version": "fixture-subprocess-v1"}},
        csv_fieldnames=[], csv_rows=[], audit_text="", episodes=episodes,
        fingerprint="sha256:" + "d" * 64,
    )


async def _main_async(output_dir):
    bundle = _make_bundle(424242)
    t = rp.FakeTransport()
    t.default_script_factory = lambda payload: rp.FakeStreamScript(
        hang=True, prompt_token_ids_echo=None, token_events=[]
    )
    t.set_get_response(rp.HEALTH_ENDPOINT, 200, {{}})
    t.set_get_response(
        rp.MODELS_ENDPOINT, 200,
        {{"data": [{{"id": rp.MODEL_FULL_ID["llama"]}}, {{"id": rp.MODEL_FULL_ID["qwen"]}}]}},
    )
    t.set_get_response(rp.OPENAPI_ENDPOINT, 200, {{"paths": {{}}}})

    interrupt_state = rp.InterruptState()
    loop = asyncio.get_running_loop()
    for sig, name in ((signal.SIGINT, "SIGINT"), (signal.SIGTERM, "SIGTERM")):
        loop.add_signal_handler(sig, lambda name=name: interrupt_state.trigger(name))

    def tok_factory(model_key):
        return rp.FakeTokenizerAdapter(vocab_size=2000, special_token_ids={{0, 1, 2}})

    # Signal handlers are now installed -- tell the parent it is safe to
    # send a real OS signal at any point from here on.
    ready_marker = output_dir.parent / "child_ready.marker"
    ready_marker.parent.mkdir(parents=True, exist_ok=True)
    ready_marker.write_text("ready", encoding="utf-8")

    summary = await rp.run_official_campaign(
        bundle=bundle, output_dir=output_dir, host="127.0.0.1", port=19999, resume=False,
        api_key="subprocess-test-fake-key", transport=t, tokenizer_factory=tok_factory,
        server_adapter=rp.FakeServerProcessAdapter(), sleeper=rp.FakeSleeper(), clock=rp.RealClock(),
        run_server_path=Path("/nonexistent/run_server.sh"), environment_probe=rp.FakeEnvironmentProbe(),
        interrupt_state=interrupt_state,
    )
    return rp.official_run_exit_code(summary)


def main():
    output_dir = Path(sys.argv[1])
    exit_code = asyncio.run(_main_async(output_dir))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
'''
    import subprocess as _subprocess
    import tempfile as _tempfile_mod
    import time as _time_mod

    with _tempfile_mod.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        script_path = tmp_path / "signal_child.py"
        script_path.write_text(child_script, encoding="utf-8")
        out_dir = tmp_path / "out"
        ready_marker = tmp_path / "child_ready.marker"

        proc = _subprocess.Popen(
            [sys.executable, str(script_path), str(out_dir)],
            stdout=_subprocess.PIPE, stderr=_subprocess.PIPE,
        )
        try:
            # Poll for the child's readiness marker (written right after it
            # installs its signal handlers) instead of a fixed sleep --
            # avoids flakiness from import/startup time under load, and
            # guarantees the signal is never sent before a handler exists.
            deadline = _time_mod.monotonic() + 10.0
            while not ready_marker.exists():
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate()
                    return False, (
                        f"child process exited early (code={proc.returncode}) before "
                        f"becoming ready (stdout={stdout[-500:]!r} stderr={stderr[-500:]!r})"
                    )
                if _time_mod.monotonic() > deadline:
                    proc.kill()
                    proc.wait(timeout=5)
                    return False, "child process never became ready within 10s"
                _time_mod.sleep(0.02)

            os.kill(proc.pid, signal_value)
            try:
                returncode = proc.wait(timeout=15)
            except _subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
                return False, "child process did not exit within 15s after the signal"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

        if returncode != expected_exit_code:
            stdout, stderr = proc.communicate()
            return False, (
                f"exit code {returncode} != expected {expected_exit_code} "
                f"(stdout={stdout[-500:]!r} stderr={stderr[-500:]!r})"
            )

        summary_path = out_dir / OFFICIAL_RUN_SUMMARY_FILENAME
        if not summary_path.exists():
            return False, "official_run_summary.json was not written by the child process"
        try:
            disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"could not read/parse the child's summary file: {exc}"

        if disk_summary.get("overall_status") != "interrupted":
            return False, f"persisted overall_status {disk_summary.get('overall_status')!r} != 'interrupted'"
        if disk_summary.get("interrupted_by") != expected_signal_name:
            return False, (
                f"persisted interrupted_by {disk_summary.get('interrupted_by')!r} != "
                f"expected {expected_signal_name!r}"
            )

        return True, "ok"


