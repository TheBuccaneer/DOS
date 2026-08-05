#!/usr/bin/env python3
"""Offline regression suite for audit_server_waiting_diagnostic.py.

All mutations operate on a complete fake diagnostic directory containing both
20-victim episodes, four burst records, stabilization/provenance/summary files,
a server log, a run-mode marker, and a matching integrity manifest.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import audit_server_waiting_diagnostic as auditor


@dataclass
class Episode:
    episode_id: str
    model_key: str
    model_id: str
    offload_gb: int
    state_label: str
    server_max_num_seqs: int
    repeat: int
    condition: str
    order_in_block: int
    block_id: str
    victim_workload_seed: int
    burst_workload_seed: int
    victim_input_len: int = 4
    victim_output_len: int = 4
    burst_input_len: int = 5
    burst_output_len: int = 3


class FakeRunner:
    MODEL_KEY = "qwen"
    MANIFEST_SCHEMA_VERSION = 1
    RUNNER_VERSION = "run_server_waiting_confirmation-v4"
    RESULT_SCHEMA_VERSION = 6
    RUN_MODE_DIAGNOSTIC_PAIR = "diagnostic_pair"
    DIAGNOSTIC_BLOCK_ID = "qwen_off12_k8_rep01"
    DIAGNOSTIC_MODEL_KEY = "qwen"
    DIAGNOSTIC_OFFLOAD_GB = 12
    DIAGNOSTIC_SERVER_MAX_NUM_SEQS = 8
    DIAGNOSTIC_CLASSIFICATION_A = "A_OUTPUT_LEVEL_OVERLAP"
    DIAGNOSTIC_CLASSIFICATION_B = "B_NO_OUTPUT_LEVEL_OVERLAP_WITH_FIRST_COHORT"
    DIAGNOSTIC_CLASSIFICATION_C = "C_BURST_OUTPUT_AFTER_ALL_VICTIMS"
    DIAGNOSTIC_CLASSIFICATION_D = "D_AMBIGUOUS_OR_INVALID"
    RUN_MODE_MARKER_FILENAME = ".server_waiting_confirmation_run_mode"
    INTEGRITY_MANIFEST_FILENAME = "integrity_manifest.json"
    __file__ = str(Path(__file__).resolve().parent / "run_server_waiting_confirmation.py")

    def __init__(self, episodes: list[Episode], schedule_fp: str) -> None:
        self.episodes = episodes
        self.bundle = SimpleNamespace(
            episodes=episodes,
            fingerprint=schedule_fp,
            json_obj={"seed": 20260720, "design_version": "server-waiting-confirmation-v1"},
        )

    def load_and_validate_bundle(self, schedule_dir: Path, model_key: str):
        if not schedule_dir.is_dir():
            return None, ["schedule dir missing"]
        return self.bundle, []

    def validate_diagnostic_block_selection(self, bundle):
        return list(self.episodes), []

    @staticmethod
    def build_server_command(path: Path, model_key: str, offload: int, k: int, host: str, port: int):
        return ["bash", str(path), model_key, str(offload), str(k), host, str(port)]

    @staticmethod
    def compute_environment_fingerprint(env: dict) -> str:
        resolved = env.get("resolved_gpu") or {}
        payload = {
            "gpu_uuid": resolved.get("uuid"),
            "gpu_model": resolved.get("name"),
            "gpu_memory_total": resolved.get("memory_total"),
            "gpu_driver_version": resolved.get("driver_version"),
            "python_version": env.get("python_version"),
            "vllm_version": env.get("vllm_version"),
            "torch_version": env.get("torch_version"),
            "transformers_version": env.get("transformers_version"),
            "httpx_version": env.get("httpx_version"),
            "kernel": env.get("kernel"),
            "git_commit": env.get("git_commit"),
            "file_hashes": dict(sorted((env.get("file_hashes") or {}).items())),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"sha256:{digest}"

    @staticmethod
    def render_diagnostic_text_summary(summary: dict) -> str:
        return "fake complete diagnostic pair\n"

    @staticmethod
    def _validate_diagnostic_run_manifest_artifact(actual, expected):
        return [] if isinstance(actual, dict) else ["manifest invalid"]

    @staticmethod
    def _validate_diagnostic_stabilization_artifact(**kwargs):
        return [] if isinstance(kwargs.get("obj"), dict) else ["stabilization invalid"]

    @staticmethod
    def _validate_exact_stabilization_references(**kwargs):
        expected = str(kwargs["expected_path"].resolve())
        errors = []
        for label, result in kwargs["results"]:
            if not isinstance(result, dict) or result.get("stabilization_reference", {}).get("path") != expected:
                errors.append(f"{label}: stabilization reference mismatch")
        return errors

    @staticmethod
    def _validate_diagnostic_artifact_counts(output_dir: Path):
        expected = {
            ".server_waiting_confirmation_run_mode",
            "diagnostic_run_manifest.json",
            "diagnostic_pair_summary.json",
            "diagnostic_pair_summary.txt",
            "integrity_manifest.json",
            "episodes/qwen_off12_k8_no_burst_rep01.json",
            "episodes/qwen_off12_k8_prefill_burst_rep01.json",
            "stabilization/qwen_off12_k8_rep01.json",
            "server_logs/qwen_off12_k8_rep01.log",
        }
        actual = {p.relative_to(output_dir).as_posix() for p in output_dir.rglob("*") if p.is_file()}
        return [] if actual == expected else [f"artifact set mismatch: {sorted(actual ^ expected)}"]

    @staticmethod
    def verify_diagnostic_integrity_manifest(output_dir: Path, manifest: Any, **expected):
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
            return False, ["integrity malformed"]
        errors = []
        listed = {e.get("relative_path"): e for e in manifest["files"] if isinstance(e, dict)}
        current = {
            p.relative_to(output_dir).as_posix(): p
            for p in output_dir.rglob("*")
            if p.is_file() and p.name != "integrity_manifest.json"
        }
        if set(listed) != set(current):
            errors.append("integrity file set mismatch")
        for rel, p in current.items():
            entry = listed.get(rel, {})
            if entry.get("size_bytes") != p.stat().st_size or entry.get("sha256") != file_hash(p):
                errors.append(f"integrity mismatch: {rel}")
        if manifest.get("schedule_fingerprint") != expected.get("expected_schedule_fingerprint"):
            errors.append("schedule fingerprint mismatch")
        if manifest.get("environment_fingerprint") != expected.get("expected_environment_fingerprint"):
            errors.append("environment fingerprint mismatch")
        return not errors, errors

    def classify_diagnostic_pair(self, **kwargs):
        no = kwargs["no_burst_result"]
        pb = kwargs["prefill_burst_result"]
        if not isinstance(no, dict) or not isinstance(pb, dict):
            return {"classification": self.DIAGNOSTIC_CLASSIFICATION_D, "reasons": ["missing"]}
        try:
            active = set(pb["trigger"]["active_cohort_request_indices"])
            victims = {r["request_index"]: r for r in pb["victim_requests"]}
            bursts = pb["burst_requests"]
            first_burst = min(r["burst_first_token_perf_ns"] for r in bursts)
            last_active = max(victims[i]["stream_end_perf_ns"] for i in active)
            last_all = max(r["stream_end_perf_ns"] for r in victims.values())
        except Exception as exc:
            return {"classification": self.DIAGNOSTIC_CLASSIFICATION_D, "reasons": [str(exc)]}
        if first_burst < last_active:
            value = self.DIAGNOSTIC_CLASSIFICATION_A
        elif first_burst > last_all:
            value = self.DIAGNOSTIC_CLASSIFICATION_C
        else:
            value = self.DIAGNOSTIC_CLASSIFICATION_B
        return {"classification": value, "reasons": []}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prompt_for(ep: Episode, role: str, index: int) -> tuple[int, int, list[int]]:
    valid = auditor.compute_valid_token_ids(auditor.FakeTokenizerAdapter())
    if role == "victim":
        ps = auditor.victim_prompt_seed(ep, index)
        gs = auditor.victim_generation_seed(ep, index)
        length = ep.victim_input_len
    else:
        ps = auditor.burst_prompt_seed(ep, index)
        gs = auditor.burst_generation_seed(ep, index)
        length = ep.burst_input_len
    return ps, gs, auditor.generate_token_id_prompt(ps, valid, length)


def make_events(prompt: list[int], tokens: list[list[int]], start_ns: int, usage: dict) -> list[dict]:
    events = [{
        "event_index": 0,
        "receive_perf_counter_ns": start_ns - 10,
        "elapsed_since_request_start_ms": 0.01,
        "raw_data": json.dumps({"choices": [{"token_ids": [], "prompt_token_ids": prompt, "text": "", "finish_reason": None}]}),
        "parse_status": "ok",
        "token_ids": [],
        "prompt_token_ids": prompt,
        "text_delta": "",
        "finish_reason": None,
        "usage": None,
    }]
    t = start_ns
    for batch in tokens:
        events.append({
            "event_index": len(events),
            "receive_perf_counter_ns": t,
            "elapsed_since_request_start_ms": (t - (start_ns - 100)) / 1e6,
            "raw_data": json.dumps({"choices": [{"token_ids": list(batch), "text": "x", "finish_reason": None}]}),
            "parse_status": "ok",
            "token_ids": list(batch),
            "prompt_token_ids": None,
            "text_delta": "x",
            "finish_reason": None,
            "usage": None,
        })
        t += 1_000_000
    events.append({
        "event_index": len(events),
        "receive_perf_counter_ns": t,
        "elapsed_since_request_start_ms": 1.0,
        "raw_data": json.dumps({"choices": [{"token_ids": [], "text": "", "finish_reason": "length"}], "usage": usage}),
        "parse_status": "ok",
        "token_ids": [],
        "prompt_token_ids": None,
        "text_delta": "",
        "finish_reason": "length",
        "usage": usage,
    })
    events.append({
        "event_index": len(events),
        "receive_perf_counter_ns": t + 1,
        "elapsed_since_request_start_ms": 1.1,
        "raw_data": "[DONE]",
        "parse_status": "done",
        "token_ids": [],
        "prompt_token_ids": None,
        "text_delta": None,
        "finish_reason": None,
        "usage": None,
    })
    return events


def make_record(ep: Episode, role: str, index: int, first_ns: int, batches: list[list[int]] | None = None) -> dict:
    ps, gs, prompt = prompt_for(ep, role, index)
    completion = ep.victim_output_len if role == "victim" else ep.burst_output_len
    prompt_len = ep.victim_input_len if role == "victim" else ep.burst_input_len
    batches = batches or [[100 + j] for j in range(completion)]
    usage = {"prompt_tokens": prompt_len, "completion_tokens": completion}
    events = make_events(prompt, batches, first_ns, usage)
    raw = auditor.reconstruct_raw_sse_events(events)
    output = [x for batch in batches for x in batch]
    stream_end = raw.last_positive_token_receive_ns + 2_000_000
    rec = {
        "request_id": f"{ep.episode_id}:{role}:{index}",
        "role": role,
        "request_index": index,
        "prompt_seed": ps,
        "generation_seed": gs,
        "prompt_token_ids_sent": prompt,
        "prompt_token_ids_returned": prompt,
        "prompt_sha256": auditor.prompt_sha256(prompt),
        "expected_prompt_tokens": prompt_len,
        "expected_completion_tokens": completion,
        "usage": usage,
        "output_token_ids": output,
        "finish_reason": "length",
        "raw_sse_events": events,
        "done_received": True,
        "request_start_ns": first_ns - 100_000,
        "request_dispatch_ns": first_ns - 90_000,
        "first_token_receive_ns": raw.first_positive_token_receive_ns,
        "last_token_receive_ns": raw.last_positive_token_receive_ns,
        "first_token_perf_ns": raw.first_positive_token_receive_ns,
        "stream_end_ns": stream_end,
        "stream_end_perf_ns": stream_end,
        "client_observed_tpot_ms": (
            (raw.last_positive_token_receive_ns - raw.first_positive_token_receive_ns)
            / 1e6 / (completion - 1)
        ),
        "itl_available": raw.itl_available,
        "itl_ms": raw.itl_ms if raw.itl_available else None,
        "token_batch_sizes": None if raw.itl_available else raw.token_batch_sizes,
        "token_batch_interarrival_ms": None if raw.itl_available else raw.token_batch_interarrival_ms,
        "chunk_interarrival_ms": None if raw.itl_available else raw.token_batch_interarrival_ms,
        "status": "complete",
        "validation_errors": [],
    }
    if role == "burst":
        rec.update({
            "burst_first_token_perf_ns": raw.first_positive_token_receive_ns,
            "burst_dispatch_start_perf_ns": rec["request_dispatch_ns"],
            "burst_end_perf_ns": stream_end,
        })
    return rec


def rebuild_integrity(root: Path, schedule_fp: str, env_fp: str) -> None:
    path = root / "integrity_manifest.json"
    if path.exists():
        path.unlink()
    files = []
    for p in sorted((x for x in root.rglob("*") if x.is_file()), key=lambda x: x.relative_to(root).as_posix()):
        rel = p.relative_to(root).as_posix()
        files.append({"relative_path": rel, "size_bytes": p.stat().st_size, "sha256": file_hash(p)})
    write_json(path, {
        "file_count": len(files),
        "episode_file_count": 2,
        "stabilization_file_count": 1,
        "block_summary_count": 0,
        "schedule_fingerprint": schedule_fp,
        "environment_fingerprint": env_fp,
        "files": files,
    })


def build_complete_tree(root: Path) -> tuple[FakeRunner, dict[str, Path]]:
    schedule = root / "schedule"
    schedule.mkdir(parents=True)
    schedule_fp = "sha256:" + "7" * 64
    no_ep = Episode(
        "qwen_off12_k8_no_burst_rep01", "qwen", "Qwen/Qwen2.5-7B-Instruct", 12, "high", 8, 1,
        "no_burst", 1, "qwen_off12_k8_rep01", 111, 222,
    )
    pb_ep = copy.deepcopy(no_ep)
    pb_ep.episode_id = "qwen_off12_k8_prefill_burst_rep01"
    pb_ep.condition = "prefill_burst"
    pb_ep.order_in_block = 2
    runner = FakeRunner([no_ep, pb_ep], schedule_fp)

    diagnostic = root / "diagnostic"
    (diagnostic / "episodes").mkdir(parents=True)
    (diagnostic / "stabilization").mkdir()
    (diagnostic / "server_logs").mkdir()
    env = {
        "python_executable": "/fake/bin/python3",
        "python_version": "3.12.0",
        "platform": "FakeLinux-x86_64",
        "hostname": "fake-host",
        "kernel": "6.0.0-fake",
        "git_commit": "f" * 40,
        "git_dirty": False,
        "cuda_visible_devices": "0",
        "vllm_version": "0.17.1",
        "torch_version": "2.5.0",
        "transformers_version": "4.45.0",
        "httpx_version": "0.27.0",
        "gpu_list": [{
            "index": "0", "name": "Fake RTX 3090", "uuid": "GPU-fake",
            "memory_total": "24576 MiB", "driver_version": "550.00",
        }],
        "resolved_gpu": {
            "index": "0", "name": "Fake RTX 3090", "uuid": "GPU-fake",
            "memory_total": "24576 MiB", "driver_version": "550.00",
        },
        "file_hashes": {
            "run_server_waiting_confirmation.py": "a" * 64,
            "run_server_waiting_confirmation.sh": "b" * 64,
            "run_server_waiting_server.sh": "c" * 64,
            "server_waiting_schedule.json": "d" * 64,
            "server_waiting_schedule.csv": "e" * 64,
            "server_waiting_schedule_audit.txt": "0" * 64,
        },
    }
    runner.actual_file_hashes = dict(env["file_hashes"])
    env_fp = runner.compute_environment_fingerprint(env)
    stab_path = diagnostic / "stabilization" / "qwen_off12_k8_rep01.json"
    write_json(stab_path, {"status": "complete", "runner_version": runner.RUNNER_VERSION})
    stab_ref = {"path": str(stab_path.resolve())}

    def episode_obj(ep: Episode, burst: bool) -> dict:
        victims = []
        # active victims 0..7 end after burst's first output; non-active victims end later too.
        for i in range(20):
            first = 1_000_000_000 + i * 20_000_000
            victims.append(make_record(ep, "victim", i, first))
        bursts = []
        if burst:
            for j in range(4):
                bursts.append(make_record(ep, "burst", j, 1_050_000_000 + j * 2_000_000))
        return {
            "runner_version": runner.RUNNER_VERSION,
            "result_schema_version": 6,
            "run_mode": "diagnostic_pair",
            "schedule_fingerprint": schedule_fp,
            "episode_id": ep.episode_id,
            "schedule_row": ep.__dict__,
            "block_id": ep.block_id,
            "stabilization_reference": stab_ref,
            "trigger": {"active_cohort_request_indices": list(range(8)), "active_cohort_size": 8},
            "victim_requests": victims,
            "burst_requests": bursts,
            "status": "complete",
            "validation_errors": [],
        }

    no_obj = episode_obj(no_ep, False)
    pb_obj = episode_obj(pb_ep, True)
    no_path = diagnostic / "episodes" / f"{no_ep.episode_id}.json"
    pb_path = diagnostic / "episodes" / f"{pb_ep.episode_id}.json"
    write_json(no_path, no_obj)
    write_json(pb_path, pb_obj)

    manifest = {
        "manifest_schema_version": runner.MANIFEST_SCHEMA_VERSION,
        "runner_version": runner.RUNNER_VERSION,
        "result_schema_version": runner.RESULT_SCHEMA_VERSION,
        "schedule_fingerprint": schedule_fp,
        "design_version": runner.bundle.json_obj["design_version"],
        "schedule_seed": runner.bundle.json_obj["seed"],
        "run_mode": runner.RUN_MODE_DIAGNOSTIC_PAIR,
        "created_utc": "2026-07-21T12:00:00+00:00",
        "output_dir": str(diagnostic),
        "host": "127.0.0.1",
        "port": 8000,
        "python_executable": env["python_executable"],
        "python_version": env["python_version"],
        "platform": env["platform"],
        "hostname": env["hostname"],
        "kernel": env["kernel"],
        "git_commit": env["git_commit"],
        "git_dirty": env["git_dirty"],
        "CUDA_VISIBLE_DEVICES": env["cuda_visible_devices"],
        "vllm_version": env["vllm_version"],
        "torch_version": env["torch_version"],
        "transformers_version": env["transformers_version"],
        "httpx_version": env["httpx_version"],
        "gpu_list": env["gpu_list"],
        "resolved_gpu": env["resolved_gpu"],
        "file_hashes": env["file_hashes"],
        "environment_fingerprint": env_fp,
    }
    write_json(diagnostic / "diagnostic_run_manifest.json", manifest)
    stored = runner.classify_diagnostic_pair(no_burst_result=no_obj, prefill_burst_result=pb_obj)
    write_json(diagnostic / "diagnostic_pair_summary.json", {
        "runner_version": runner.RUNNER_VERSION,
        "result_schema_version": runner.RESULT_SCHEMA_VERSION,
        "run_mode": runner.RUN_MODE_DIAGNOSTIC_PAIR,
        "schedule_fingerprint": schedule_fp,
        "environment_fingerprint": env_fp,
        "diagnostic_block_id": runner.DIAGNOSTIC_BLOCK_ID,
        "classification": stored,
        "integrity_manifest_filename": runner.INTEGRITY_MANIFEST_FILENAME,
        "integrity_finalization_required": True,
    })
    (diagnostic / "diagnostic_pair_summary.txt").write_text("fake complete diagnostic pair\n", encoding="utf-8")
    (diagnostic / ".server_waiting_confirmation_run_mode").write_text("diagnostic_pair", encoding="utf-8")
    (diagnostic / "server_logs" / "qwen_off12_k8_rep01.log").write_text("fake server log\n", encoding="utf-8")
    rebuild_integrity(diagnostic, schedule_fp, env_fp)
    return runner, {
        "schedule": schedule,
        "diagnostic": diagnostic,
        "no": no_path,
        "pb": pb_path,
    }


def mutate_json(path: Path, fn: Callable[[dict], None]) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    fn(obj)
    write_json(path, obj)


def run_audit(root: Path, runner: FakeRunner, paths: dict[str, Path], name: str = "audit"):
    out = root / name
    return auditor.audit_diagnostic(
        paths["diagnostic"], paths["schedule"], out,
        runner_module=runner,
        tokenizer_factory=lambda _model: auditor.FakeTokenizerAdapter(),
        file_hash_provider=lambda _runner, _schedule: dict(runner.actual_file_hashes),
    )


class AuditorRegressionTests(unittest.TestCase):
    def with_tree(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        runner, paths = build_complete_tree(root)
        return td, root, runner, paths

    def test_01_complete_fake_pair_passes(self):
        td, root, runner, paths = self.with_tree()
        with td:
            code, result = run_audit(root, runner, paths)
            self.assertEqual(code, 0, result)
            self.assertEqual(result["overall_audit_status"], "PASS")
            self.assertEqual(len(result["per_request_status"]), 44)

    def test_02_original_tree_is_byte_identical(self):
        td, root, runner, paths = self.with_tree()
        with td:
            before = auditor.snapshot_tree(paths["diagnostic"])
            code, result = run_audit(root, runner, paths)
            after = auditor.snapshot_tree(paths["diagnostic"])
            self.assertEqual(code, 0, result)
            self.assertEqual(before, after)
            self.assertTrue(result["diagnostic_tree_read_only_verified"])

    def test_03_jointly_rehashed_wrong_victim_prompt_fails(self):
        td, root, runner, paths = self.with_tree()
        with td:
            def change(obj):
                r = obj["victim_requests"][0]
                bad = list(r["prompt_token_ids_sent"]); bad[0] += 10
                r["prompt_token_ids_sent"] = bad
                r["prompt_token_ids_returned"] = bad
                r["prompt_sha256"] = auditor.prompt_sha256(bad)
                r["raw_sse_events"][0]["prompt_token_ids"] = bad
            mutate_json(paths["no"], change); rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
            self.assertEqual(run_audit(root, runner, paths)[0], 2)

    def test_04_jointly_rehashed_wrong_burst_prompt_fails(self):
        td, root, runner, paths = self.with_tree()
        with td:
            def change(obj):
                r = obj["burst_requests"][0]
                bad = list(r["prompt_token_ids_sent"]); bad[-1] += 11
                r["prompt_token_ids_sent"] = bad
                r["prompt_token_ids_returned"] = bad
                r["prompt_sha256"] = auditor.prompt_sha256(bad)
                r["raw_sse_events"][0]["prompt_token_ids"] = bad
            mutate_json(paths["pb"], change); rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
            self.assertEqual(run_audit(root, runner, paths)[0], 2)

    def test_05_prompt_echo_mismatch_fails(self):
        td, root, runner, paths = self.with_tree()
        with td:
            mutate_json(paths["no"], lambda o: o["victim_requests"][1]["raw_sse_events"][0].update(prompt_token_ids=[999]*4))
            rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
            self.assertEqual(run_audit(root, runner, paths)[0], 2)

    def test_06_same_length_wrong_output_values_fail(self):
        td, root, runner, paths = self.with_tree()
        with td:
            mutate_json(paths["no"], lambda o: o["victim_requests"][2].update(output_token_ids=[9,9,9,9]))
            rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
            self.assertEqual(run_audit(root, runner, paths)[0], 2)

    def test_07_shifted_victim_first_last_aliases_fail(self):
        td, root, runner, paths = self.with_tree()
        with td:
            def change(o):
                r=o["victim_requests"][3]; r["first_token_receive_ns"]+=1; r["first_token_perf_ns"]+=1; r["last_token_receive_ns"]+=1
            mutate_json(paths["no"], change); rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
            self.assertEqual(run_audit(root, runner, paths)[0], 2)

    def test_08_shifted_burst_first_last_aliases_fail(self):
        td, root, runner, paths = self.with_tree()
        with td:
            def change(o):
                r=o["burst_requests"][1]; r["first_token_receive_ns"]+=5; r["first_token_perf_ns"]+=5; r["burst_first_token_perf_ns"]+=5; r["last_token_receive_ns"]+=5
            mutate_json(paths["pb"], change); rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
            self.assertEqual(run_audit(root, runner, paths)[0], 2)

    def test_09_itl_redistribution_same_sum_fails(self):
        td, root, runner, paths = self.with_tree()
        with td:
            def change(o):
                r=o["victim_requests"][4]; r["itl_ms"]=[0.5,1.5,1.0]
            mutate_json(paths["no"], change); rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
            self.assertEqual(run_audit(root, runner, paths)[0], 2)

    def test_10_wrong_tpot_fails(self):
        td, root, runner, paths = self.with_tree()
        with td:
            mutate_json(paths["no"], lambda o: o["victim_requests"][5].update(client_observed_tpot_ms=99.0))
            rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
            self.assertEqual(run_audit(root, runner, paths)[0], 2)

    def test_11_token_batch_fixture_reconstructs_and_passes(self):
        td, root, runner, paths = self.with_tree()
        with td:
            ep=runner.episodes[0]
            batch_record=make_record(ep,"victim",10,1_200_000_000,batches=[[100,101],[102],[103]])
            mutate_json(paths["no"], lambda o: o["victim_requests"].__setitem__(10,batch_record))
            rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
            code,result=run_audit(root,runner,paths)
            self.assertEqual(code,0,result)
            item=next(r for r in result["per_request_status"] if r["condition"]=="no_burst" and r["request_index"]==10)
            self.assertFalse(item["raw_reconstruction"]["itl_available"])
            self.assertEqual(item["raw_reconstruction"]["token_batch_sizes"],[2,1,1])

    def test_12_conflicting_batch_fields_fail(self):
        td, root, runner, paths = self.with_tree()
        with td:
            ep=runner.episodes[0]; rec=make_record(ep,"victim",10,1_200_000_000,batches=[[100,101],[102],[103]])
            rec["token_batch_sizes"]=[1,1,2]
            mutate_json(paths["no"], lambda o:o["victim_requests"].__setitem__(10,rec))
            rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
            self.assertEqual(run_audit(root,runner,paths)[0],2)

    def test_13_shifted_stored_burst_times_and_abc_contradict_raw(self):
        td, root, runner, paths = self.with_tree()
        with td:
            def change(o):
                for r in o["burst_requests"]:
                    r["burst_first_token_perf_ns"]=1_300_000_000
            mutate_json(paths["pb"],change)
            no=json.loads(paths["no"].read_text()); pb=json.loads(paths["pb"].read_text())
            stored=runner.classify_diagnostic_pair(no_burst_result=no,prefill_burst_result=pb)
            mutate_json(paths["diagnostic"] / "diagnostic_pair_summary.json",lambda o:o.update(classification=stored))
            rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
            code,result=run_audit(root,runner,paths)
            self.assertEqual(code,2)
            self.assertNotEqual(result["stored_semantic_classification"][0], result["raw_output_overlap_classification"][0])

    def test_14_malformed_events_are_controlled_scientific_failures(self):
        mutations = [
            lambda r: r.update(raw_sse_events={}),
            lambda r: r["raw_sse_events"][1].update(token_ids="bad"),
            lambda r: r["raw_sse_events"][1].update(receive_perf_counter_ns="bad"),
            lambda r: r["raw_sse_events"][1].update(event_index=99),
            lambda r: r["raw_sse_events"][1].update(parse_status="malformed_json"),
            lambda r: r["raw_sse_events"][1].update(raw_data="{broken"),
        ]
        for pos, mutation in enumerate(mutations):
            with self.subTest(pos=pos):
                td, root, runner, paths = self.with_tree()
                with td:
                    def change(o,m=mutation): m(o["victim_requests"][6])
                    mutate_json(paths["no"],change); rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())["environment_fingerprint"])
                    code,result=run_audit(root,runner,paths)
                    self.assertEqual(code,2,result)

    def test_15_missing_local_tokenizer_is_exit_1(self):
        td, root, runner, paths = self.with_tree()
        with td:
            def missing(_model): raise auditor.TechnicalAuditError("local tokenizer missing")
            code,result=auditor.audit_diagnostic(
                paths["diagnostic"], paths["schedule"], root/"audit",
                runner_module=runner, tokenizer_factory=missing,
                file_hash_provider=lambda _runner, _schedule: dict(runner.actual_file_hashes),
            )
            self.assertEqual(code,1)
            self.assertEqual(result["overall_audit_status"],"TECHNICAL_ERROR")

    def test_16_malformed_json_is_exit_1_not_exception(self):
        td, root, runner, paths = self.with_tree()
        with td:
            paths["no"].write_text("{broken",encoding="utf-8")
            code,result=run_audit(root,runner,paths)
            self.assertEqual(code,1)
            self.assertTrue(result["technical_errors"])

    def test_17_output_dir_inside_diagnostic_is_exit_1(self):
        td, root, runner, paths = self.with_tree()
        with td:
            code,result=auditor.audit_diagnostic(
                paths["diagnostic"], paths["schedule"], paths["diagnostic"]/"audit",
                runner_module=runner, tokenizer_factory=lambda _: auditor.FakeTokenizerAdapter(),
                file_hash_provider=lambda _runner, _schedule: dict(runner.actual_file_hashes),
            )
            self.assertEqual(code,1)

    def test_18_pure_reconstructor_rejects_decreasing_time_and_duplicate_done(self):
        ep=Episode("x","qwen","Qwen/Qwen2.5-7B-Instruct",12,"high",8,1,"no_burst",1,"b",1,2)
        rec=make_record(ep,"victim",0,1_000_000_000)
        events=copy.deepcopy(rec["raw_sse_events"])
        events[2]["receive_perf_counter_ns"]=events[1]["receive_perf_counter_ns"]-1
        events.append(copy.deepcopy(events[-1])); events[-1]["event_index"]=len(events)-1
        raw=auditor.reconstruct_raw_sse_events(events)
        self.assertTrue(any("decreases" in e for e in raw.errors))
        self.assertTrue(any("DONE events" in e for e in raw.errors))


    def test_19_manifest_hashes_must_match_actual_sources(self):
        td, root, runner, paths = self.with_tree()
        with td:
            manifest_path = paths["diagnostic"] / "diagnostic_run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["file_hashes"]["run_server_waiting_confirmation.py"] = "0" * 64
            env = {
                "python_executable": manifest["python_executable"],
                "python_version": manifest["python_version"],
                "platform": manifest["platform"],
                "hostname": manifest["hostname"],
                "kernel": manifest["kernel"],
                "git_commit": manifest["git_commit"],
                "git_dirty": manifest["git_dirty"],
                "cuda_visible_devices": manifest["CUDA_VISIBLE_DEVICES"],
                "vllm_version": manifest["vllm_version"],
                "torch_version": manifest["torch_version"],
                "transformers_version": manifest["transformers_version"],
                "httpx_version": manifest["httpx_version"],
                "gpu_list": manifest["gpu_list"],
                "resolved_gpu": manifest["resolved_gpu"],
                "file_hashes": manifest["file_hashes"],
            }
            wrong_fp = runner.compute_environment_fingerprint(env)
            manifest["environment_fingerprint"] = wrong_fp
            write_json(manifest_path, manifest)
            for ep_path in (paths["no"], paths["pb"]):
                mutate_json(ep_path, lambda obj, fp=wrong_fp: obj.get("server_metadata", {}).update(environment_fingerprint=fp))
            mutate_json(
                paths["diagnostic"] / "diagnostic_pair_summary.json",
                lambda obj, fp=wrong_fp: obj.update(environment_fingerprint=fp),
            )
            rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, wrong_fp)
            code, result = run_audit(root, runner, paths)
            self.assertEqual(code, 2, result)
            self.assertTrue(any("actual on-disk SHA-256" in e for e in result["errors"]), result)

    def test_20_missing_or_malformed_last_token_aliases_fail(self):
        cases = [
            ("no", "victim_requests", 0, None),
            ("no", "victim_requests", 10, None),
            ("pb", "burst_requests", 0, None),
            ("no", "victim_requests", 10, 1.0),
            ("pb", "burst_requests", 0, True),
        ]
        for condition, collection, index, replacement in cases:
            with self.subTest(condition=condition, collection=collection, index=index, replacement=replacement):
                td, root, runner, paths = self.with_tree()
                with td:
                    def change(obj, c=collection, i=index, value=replacement):
                        record = obj[c][i]
                        if value is None:
                            record.pop("last_token_receive_ns", None)
                        else:
                            record["last_token_receive_ns"] = value
                    mutate_json(paths[condition], change)
                    manifest = json.loads((paths["diagnostic"] / "diagnostic_run_manifest.json").read_text())
                    rebuild_integrity(paths["diagnostic"], runner.bundle.fingerprint, manifest["environment_fingerprint"])
                    code, result = run_audit(root, runner, paths)
                    self.assertEqual(code, 2, result)
                    self.assertTrue(any("last_token_receive_ns" in e for e in result["errors"]), result)

    def test_21_actual_runner_generated_tree_passes_and_detects_provenance_tamper(self):
        import generate_server_waiting_schedule as actual_gen
        import run_server_waiting_confirmation as actual_runner
        import test_run_server_waiting_confirmation as runner_tests

        base = actual_runner.base
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            schedule_dir = root / "schedule"
            self.assertEqual(actual_gen.main(["--output-dir", str(schedule_dir)]), 0)
            bundle, bundle_errors = actual_runner.load_and_validate_bundle(schedule_dir)
            self.assertIsNotNone(bundle, bundle_errors)
            diag_dir = root / "diagnostic"
            clock = base.FakeClock()
            transport = runner_tests._build_fake_transport(bundle, actual_runner.DIAGNOSTIC_BLOCK_ID, 8, clock)
            metrics = runner_tests._build_fake_metrics_transport(8)
            env = actual_runner.FakeEnvironmentProbe().gather(schedule_dir)
            actual_hashes = auditor._default_environment_file_hash_provider(actual_runner, schedule_dir)
            env["file_hashes"] = dict(actual_hashes)
            summary = asyncio.run(actual_runner.run_diagnostic_pair(
                bundle=bundle,
                output_dir=diag_dir,
                host="127.0.0.1",
                port=37998,
                api_key="fake-key",
                transport=transport,
                metrics_transport=metrics,
                tokenizer=base.FakeTokenizerAdapter(),
                server_adapter=base.FakeServerProcessAdapter(),
                sleeper=base.FakeSleeper(),
                clock=clock,
                run_server_path=Path(actual_runner.__file__).resolve().parent / "run_server_waiting_server.sh",
                env=env,
            ))
            self.assertTrue(summary.get("diagnostic_valid"), summary)
            before = auditor.snapshot_tree(diag_dir)
            code, result = auditor.audit_diagnostic(
                diag_dir,
                schedule_dir,
                root / "audit-pass",
                runner_module=actual_runner,
                tokenizer_factory=lambda _model: auditor.FakeTokenizerAdapter(),
            )
            self.assertEqual(code, 0, result)
            self.assertEqual(before, auditor.snapshot_tree(diag_dir))

            # Recreate a fresh tree, then jointly rewrite the stored provenance
            # chain.  Actual on-disk hashes remain unchanged and must expose it.
            tampered_dir = root / "diagnostic-tampered"
            clock2 = base.FakeClock()
            summary2 = asyncio.run(actual_runner.run_diagnostic_pair(
                bundle=bundle,
                output_dir=tampered_dir,
                host="127.0.0.1",
                port=37999,
                api_key="fake-key",
                transport=runner_tests._build_fake_transport(bundle, actual_runner.DIAGNOSTIC_BLOCK_ID, 8, clock2),
                metrics_transport=runner_tests._build_fake_metrics_transport(8),
                tokenizer=base.FakeTokenizerAdapter(),
                server_adapter=base.FakeServerProcessAdapter(),
                sleeper=base.FakeSleeper(),
                clock=clock2,
                run_server_path=Path(actual_runner.__file__).resolve().parent / "run_server_waiting_server.sh",
                env=env,
            ))
            self.assertTrue(summary2.get("diagnostic_valid"), summary2)
            manifest_path = tampered_dir / "diagnostic_run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["file_hashes"]["run_server_waiting_confirmation.py"] = "0" * 64
            fp_env = {
                "python_executable": manifest["python_executable"],
                "python_version": manifest["python_version"],
                "platform": manifest["platform"],
                "hostname": manifest["hostname"],
                "kernel": manifest["kernel"],
                "git_commit": manifest["git_commit"],
                "git_dirty": manifest["git_dirty"],
                "cuda_visible_devices": manifest["CUDA_VISIBLE_DEVICES"],
                "vllm_version": manifest["vllm_version"],
                "torch_version": manifest["torch_version"],
                "transformers_version": manifest["transformers_version"],
                "httpx_version": manifest["httpx_version"],
                "gpu_list": manifest["gpu_list"],
                "resolved_gpu": manifest["resolved_gpu"],
                "file_hashes": manifest["file_hashes"],
            }
            wrong_fp = actual_runner.compute_environment_fingerprint(fp_env)
            manifest["environment_fingerprint"] = wrong_fp
            write_json(manifest_path, manifest)
            for ep_path in sorted((tampered_dir / "episodes").glob("*.json")):
                mutate_json(ep_path, lambda obj, fp=wrong_fp: obj["server_metadata"].update(environment_fingerprint=fp))
            stab_path = next((tampered_dir / "stabilization").glob("*.json"))
            mutate_json(stab_path, lambda obj, fp=wrong_fp: obj["server_metadata"].update(environment_fingerprint=fp))
            summary_path = tampered_dir / "diagnostic_pair_summary.json"
            mutate_json(summary_path, lambda obj, fp=wrong_fp: obj.update(environment_fingerprint=fp))
            summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
            (tampered_dir / "diagnostic_pair_summary.txt").write_text(
                actual_runner.render_diagnostic_text_summary(summary_obj), encoding="utf-8"
            )
            integrity = actual_runner.build_diagnostic_integrity_manifest(
                tampered_dir,
                schedule_fingerprint=bundle.fingerprint,
                environment_fingerprint=wrong_fp,
                clock=base.FakeClock(),
            )
            write_json(tampered_dir / "integrity_manifest.json", integrity)
            code2, result2 = auditor.audit_diagnostic(
                tampered_dir,
                schedule_dir,
                root / "audit-fail",
                runner_module=actual_runner,
                tokenizer_factory=lambda _model: auditor.FakeTokenizerAdapter(),
            )
            self.assertEqual(code2, 2, result2)
            self.assertTrue(any("actual on-disk SHA-256" in e for e in result2["errors"]), result2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
