#!/usr/bin/env python3
"""
run_phase_a.py

Executes the frozen phase_a_schedule.csv plan (produced by
make_phase_a_schedule.py) exactly in file order against a real vLLM
OpenAI-compatible server, restarting the server at every block boundary,
running victim + burst streaming requests with a first-token trigger
synchronization, and writing one audited raw-metrics JSON per episode.

This script does NOT analyze the State x Burst effect. It only collects
auditable raw data and descriptive per-episode metrics.

--- Why this is Python, not a shell script ---
Unlike the profiling stage (one server started manually per config, then
a client benchmark run against it -- see run_server.sh /
run_client_profile_grid_v2.sh), Phase A must restart the server ~20
times *within a single run* (10 blocks x 2 models), with exact PID /
process-group tracking, per-block readiness timing, deterministic
prompt generation tied to workload seeds, first-token-triggered burst
synchronization, and crash-safe resumable output. That control-flow and
state belongs in Python. A thin shell wrapper (run_phase_a.sh) exists
only for environment setup (venv, GPU visibility, logging) and forwards
all arguments unchanged.

--- Frozen design constants ---
This script imports STATES / CONCURRENCIES / CONDITIONS / VICTIM_* /
BURST_* / WARMUP_REQUESTS_PER_BLOCK directly from make_phase_a_schedule.py
instead of duplicating them, to avoid the two scripts silently drifting
out of sync. run_phase_a.py must therefore live in the same directory as
make_phase_a_schedule.py (or be importable via PYTHONPATH).

--- Prompt generation ---
No existing reusable exact-length prompt function was found in the
project's current scripts: run_client_profile_grid_v2.sh delegates
random-length prompt generation to vLLM's own internal
`vllm bench serve --random-input-len/--random-output-len`, which is not
exposed as an importable Python function. generate_exact_length_prompt()
below is therefore a fresh implementation, using the standard technique
of sampling random vocabulary ids, decoding, then re-encoding and
topping up/truncating until the exact target token count is reached
(subword tokenizers can change token count on decode/re-encode).

--- Dependencies ---
Real runs need `transformers` (tokenizer) and `httpx` (async streaming
HTTP client). Both are imported lazily inside the functions that need
them, so --self-test and --dry-run run with only the Python standard
library.

Usage:
    # No GPU / vLLM / network required:
    python3 run_phase_a.py --self-test

    python3 run_phase_a.py --dry-run \
        --schedule /home/rock/projects/DOS/new/runs/phase_a/phase_a_schedule.csv

    # Real run:
    python3 run_phase_a.py \
        --schedule /home/rock/projects/DOS/new/runs/phase_a/phase_a_schedule.csv \
        --output-dir /home/rock/projects/DOS/new/runs/phase_a/results \
        --gpu-device 0 \
        --resume
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence


# ---------------------------------------------------------------------------
# Import frozen design constants from the sibling schedule generator
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from make_phase_a_schedule import (
        Episode as ScheduleRow,
        REQUIRED_FIELDS as SCHEDULE_REQUIRED_FIELDS,
        STATES,
        CONCURRENCIES,
        CONDITIONS,
        VICTIM_INPUT_LEN,
        VICTIM_OUTPUT_LEN,
        VICTIM_TEMPERATURE,
        BURST_PARALLEL_REQUESTS,
        BURST_INPUT_LEN,
        BURST_OUTPUT_LEN,
        BURST_TEMPERATURE,
        DEFAULT_SEED as SCHEDULE_DEFAULT_SEED,
        generate_schedule,
    )
except ImportError as exc:
    print(
        "FATAL: could not import make_phase_a_schedule.py. run_phase_a.py "
        "must live in the same directory as make_phase_a_schedule.py (or "
        "be importable via PYTHONPATH), since it reuses the frozen design "
        f"constants to avoid duplicating them out of sync. Import error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME_MAP: dict[str, str] = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}
EXPECTED_MODELS: tuple[str, ...] = tuple(MODEL_NAME_MAP.keys())
EXPECTED_TOTAL_EPISODES = 80  # frozen design: 2 models x 40 episodes

DEFAULT_SCHEDULE_PATH = Path(
    "/home/rock/projects/DOS/new/runs/phase_a/phase_a_schedule.csv"
)
DEFAULT_API_KEY = "pilotkey"
DEFAULT_GPU_MEM_UTIL = 0.90
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_VICTIM_REQUESTS = 20
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_SERVER_START_TIMEOUT_S = 300.0
DEFAULT_TRIGGER_TIMEOUT_S = 60.0

# vLLM server invocation, matching the flags used by the project's
# existing run_server.sh, but launched as `python -m vllm serve ...`
# rather than the bare `vllm` console script, so --python-executable can
# select a specific venv/interpreter. If your vLLM version does not
# support `python -m vllm serve ...` identically to the `vllm serve`
# console script, this is the one place to adjust.
VLLM_SERVE_MODULE_ARGS: tuple[str, ...] = ("-m", "vllm", "serve")

MAX_PROMPT_ADJUST_ITERS = 8


# ---------------------------------------------------------------------------
# Tokenizer abstraction
# ---------------------------------------------------------------------------

class SelfTestTokenizer:
    """
    Deterministic, dependency-free stand-in tokenizer used only by
    --self-test (and anything else that must run without GPU/vLLM/
    transformers). "Tokens" are opaque synthetic words; encode/decode
    round-trip exactly, so the exact-length prompt generator converges
    in a single iteration. NEVER used for real runs.
    """

    def __init__(self, vocab_size: int = 4000) -> None:
        self._vocab = [f"tok{i:05d}" for i in range(vocab_size)]

    def vocab_ids(self) -> list[str]:
        return list(self._vocab)

    def encode(self, text: str) -> list[str]:
        return text.split(" ") if text else []

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


class HFTokenizerWrapper:
    """
    Thin wrapper around a real HuggingFace tokenizer for actual runs.
    transformers is imported lazily so --self-test / --dry-run never
    require it to be installed.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for real (non-self-test) runs "
                "to build exact-length prompts. Install it with: "
                "pip install transformers"
            ) from exc

        self._tok = AutoTokenizer.from_pretrained(model_name)
        special_ids = set(self._tok.all_special_ids or [])
        vocab_size = self._tok.vocab_size
        self._vocab_ids = [i for i in range(vocab_size) if i not in special_ids]

    def vocab_ids(self) -> list[int]:
        return self._vocab_ids

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


def load_tokenizer(model_name: str, self_test: bool) -> Any:
    if self_test:
        return SelfTestTokenizer()
    return HFTokenizerWrapper(model_name)


# ---------------------------------------------------------------------------
# Deterministic exact-length prompt generation
# ---------------------------------------------------------------------------

@dataclass
class PromptResult:
    text: str
    token_count: int
    sha256_hex: str


def generate_exact_length_prompt(
    workload_seed: int,
    target_len: int,
    tokenizer: Any,
) -> PromptResult:
    """
    Deterministically derives prompt text from workload_seed that
    re-encodes to exactly target_len tokens under `tokenizer`.

    Standard technique: sample random vocabulary ids, decode to text,
    re-encode (decode/encode round-trips can change token count due to
    subword merges/normalization), top up or truncate, retry until the
    length matches exactly.
    """
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")

    rng = random.Random(workload_seed)
    vocab = tokenizer.vocab_ids()
    if not vocab:
        raise RuntimeError("tokenizer.vocab_ids() returned no usable ids")

    ids = [rng.choice(vocab) for _ in range(target_len)]

    for _ in range(MAX_PROMPT_ADJUST_ITERS):
        text = tokenizer.decode(ids)
        encoded = tokenizer.encode(text)

        if len(encoded) == target_len:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            return PromptResult(
                text=text, token_count=target_len, sha256_hex=digest
            )

        if len(encoded) < target_len:
            ids = encoded + [
                rng.choice(vocab) for _ in range(target_len - len(encoded))
            ]
        else:
            ids = encoded[:target_len]

    raise RuntimeError(
        f"Could not converge on exact prompt length {target_len} after "
        f"{MAX_PROMPT_ADJUST_ITERS} iterations (workload_seed={workload_seed})"
    )


def derive_warmup_seed(block_id: str) -> int:
    digest = hashlib.sha256(f"warmup:{block_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


# ---------------------------------------------------------------------------
# Schedule CSV parsing
# ---------------------------------------------------------------------------

_FIELD_CASTERS: dict[str, Any] = {
    "episode_id": str,
    "model": str,
    "offload_gb": int,
    "state_label": str,
    "concurrency": int,
    "condition": str,
    "repeat": int,
    "random_seed": int,
    "episode_seed": int,
    "victim_workload_seed": int,
    "burst_workload_seed": int,
    "victim_input_len": int,
    "victim_output_len": int,
    "victim_temperature": float,
    "burst_parallel_requests": int,
    "burst_input_len": int,
    "burst_output_len": int,
    "burst_temperature": float,
    "warmup_requests": int,
    "restart_server_before_block": int,
    "block_id": str,
    "order_in_block": int,
}


def parse_schedule_csv(path: Path) -> list[ScheduleRow]:
    if not path.exists():
        raise FileNotFoundError(f"schedule file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_cols = [
            f for f in SCHEDULE_REQUIRED_FIELDS if f not in fieldnames
        ]
        if missing_cols:
            raise ValueError(f"{path}: missing column(s): {missing_cols}")

        rows: list[ScheduleRow] = []
        for line_no, raw in enumerate(reader, start=2):
            try:
                kwargs = {
                    name: caster(raw[name])
                    for name, caster in _FIELD_CASTERS.items()
                }
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"{path}: line {line_no}: could not parse row: {exc}"
                ) from exc
            rows.append(ScheduleRow(**kwargs))

    return rows


# ---------------------------------------------------------------------------
# Schedule validation (independent re-check of the frozen plan)
# ---------------------------------------------------------------------------

def validate_schedule_rows(rows: list[ScheduleRow]) -> list[str]:
    errors: list[str] = []

    if len(rows) != EXPECTED_TOTAL_EPISODES:
        errors.append(
            f"expected {EXPECTED_TOTAL_EPISODES} episodes total, "
            f"found {len(rows)}"
        )

    ids = [r.episode_id for r in rows]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        errors.append(f"duplicate episode_id(s): {dupes}")

    models_found = sorted({r.model for r in rows})
    for m in models_found:
        if m not in EXPECTED_MODELS:
            errors.append(f"unexpected model in schedule: {m!r}")
    for m in EXPECTED_MODELS:
        if m not in models_found:
            errors.append(f"missing model in schedule: {m!r}")

    expected_offloads = {o for o, _ in STATES}
    expected_state_labels = {s for _, s in STATES}

    for r in rows:
        if r.offload_gb not in expected_offloads:
            errors.append(f"{r.episode_id}: unexpected offload_gb={r.offload_gb}")
        if r.state_label not in expected_state_labels:
            errors.append(f"{r.episode_id}: unexpected state_label={r.state_label!r}")
        if r.concurrency not in CONCURRENCIES:
            errors.append(f"{r.episode_id}: unexpected concurrency={r.concurrency}")
        if r.condition not in CONDITIONS:
            errors.append(f"{r.episode_id}: unexpected condition={r.condition!r}")
        if r.victim_input_len != VICTIM_INPUT_LEN or r.victim_output_len != VICTIM_OUTPUT_LEN:
            errors.append(f"{r.episode_id}: unexpected victim length config")
        if r.victim_temperature != VICTIM_TEMPERATURE:
            errors.append(f"{r.episode_id}: unexpected victim_temperature={r.victim_temperature}")
        if r.burst_parallel_requests != BURST_PARALLEL_REQUESTS:
            errors.append(f"{r.episode_id}: unexpected burst_parallel_requests={r.burst_parallel_requests}")
        if r.burst_input_len != BURST_INPUT_LEN or r.burst_output_len != BURST_OUTPUT_LEN:
            errors.append(f"{r.episode_id}: unexpected burst length config")
        if r.burst_temperature != BURST_TEMPERATURE:
            errors.append(f"{r.episode_id}: unexpected burst_temperature={r.burst_temperature}")

    for model in EXPECTED_MODELS:
        model_rows = [r for r in rows if r.model == model]
        errors.extend(_validate_model_blocks(model, model_rows))

    return errors


def _validate_model_blocks(model: str, rows: list[ScheduleRow]) -> list[str]:
    errors: list[str] = []

    blocks: dict[str, list[ScheduleRow]] = {}
    block_order: list[str] = []
    for r in rows:
        if r.block_id not in blocks:
            blocks[r.block_id] = []
            block_order.append(r.block_id)
        blocks[r.block_id].append(r)

    expected_block_size = len(CONCURRENCIES) * len(CONDITIONS)

    for block_id in block_order:
        block_rows = blocks[block_id]

        if len(block_rows) != expected_block_size:
            errors.append(
                f"model={model}: block {block_id} has {len(block_rows)} "
                f"rows, expected {expected_block_size}"
            )

        positions = sorted(r.order_in_block for r in block_rows)
        if positions != list(range(1, expected_block_size + 1)):
            errors.append(
                f"model={model}: block {block_id} order_in_block values "
                f"{positions} incomplete/invalid"
            )

        warmups = [r for r in block_rows if r.warmup_requests > 0]
        if len(warmups) != 1 or (warmups and warmups[0].order_in_block != 1):
            errors.append(f"model={model}: block {block_id} warmup flag incorrect")

        restarts = [r for r in block_rows if r.restart_server_before_block > 0]
        if len(restarts) != 1 or (restarts and restarts[0].order_in_block != 1):
            errors.append(f"model={model}: block {block_id} restart flag incorrect")

        states = {r.state_label for r in block_rows}
        offloads = {r.offload_gb for r in block_rows}
        repeats = {r.repeat for r in block_rows}
        if len(states) != 1 or len(offloads) != 1:
            errors.append(f"model={model}: block {block_id} mixes state/offload values")
        if len(repeats) != 1:
            errors.append(f"model={model}: block {block_id} mixes repeat values")

    max_repeat = max((r.repeat for r in rows), default=0)
    expected_sequence: list[str] = []
    for repeat in range(1, max_repeat + 1):
        order = STATES if repeat % 2 == 1 else tuple(reversed(STATES))
        expected_sequence.extend(label for _, label in order)

    actual_sequence = [blocks[b][0].state_label for b in block_order]
    if actual_sequence != expected_sequence:
        errors.append(
            f"model={model}: block state sequence {actual_sequence} "
            f"!= expected {expected_sequence}"
        )

    matched: dict[tuple[int, str, int], dict[str, ScheduleRow]] = {}
    for r in rows:
        key = (r.concurrency, r.condition, r.repeat)
        matched.setdefault(key, {})[r.state_label] = r

    for key, by_state in matched.items():
        if set(by_state.keys()) != {"low", "high"}:
            errors.append(
                f"model={model}: incomplete low/high pair for {key}: "
                f"{sorted(by_state)}"
            )
            continue
        low, high = by_state["low"], by_state["high"]
        if low.order_in_block != high.order_in_block:
            errors.append(
                f"model={model}: order_in_block mismatch for matched pair "
                f"{key}: {low.order_in_block} vs {high.order_in_block}"
            )
        if low.victim_workload_seed != high.victim_workload_seed:
            errors.append(f"model={model}: victim_workload_seed mismatch for matched pair {key}")
        if low.burst_workload_seed != high.burst_workload_seed:
            errors.append(f"model={model}: burst_workload_seed mismatch for matched pair {key}")

    return errors


def check_output_dir_compatibility(
    output_dir: Path,
    schedule_path: Path,
    resume: bool,
) -> tuple[list[str], str]:
    errors: list[str] = []
    schedule_bytes = schedule_path.read_bytes()
    current_fp = hashlib.sha256(schedule_bytes).hexdigest()

    fingerprint_path = output_dir / "schedule_fingerprint.txt"
    episodes_dir = output_dir / "episodes"
    has_existing_output = episodes_dir.exists() and any(episodes_dir.iterdir())

    if fingerprint_path.exists():
        existing_fp = fingerprint_path.read_text(encoding="utf-8").strip()
        if existing_fp != current_fp:
            errors.append(
                f"output-dir {output_dir} already contains results for a "
                f"different schedule (fingerprint mismatch: existing="
                f"{existing_fp[:12]}..., current={current_fp[:12]}...). "
                f"Use a different --output-dir or clean it manually."
            )
    elif has_existing_output and not resume:
        errors.append(
            f"output-dir {output_dir} already contains episode results "
            f"but no schedule_fingerprint.txt, and --resume was not given. "
            f"Refusing to overwrite silently."
        )

    return errors, current_fp


# ---------------------------------------------------------------------------
# Resume: episode file status
# ---------------------------------------------------------------------------

EpisodeFileStatus = Literal["missing", "complete", "partial", "corrupted"]


def check_episode_file_status(
    path: Path,
    expected_episode_id: str,
) -> EpisodeFileStatus:
    if not path.exists():
        return "missing"

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return "corrupted"

    if not isinstance(data, dict):
        return "corrupted"

    if data.get("episode_id") != expected_episode_id:
        return "corrupted"

    status = data.get("status")
    if status == "complete":
        return "complete"
    if status in ("failed", "partial"):
        return "partial"

    return "corrupted"


# ---------------------------------------------------------------------------
# Atomic output writers
# ---------------------------------------------------------------------------

def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp_path, path)


def append_manifest_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str))
        handle.write("\n")


# ---------------------------------------------------------------------------
# Execution plan (shared by --dry-run and --self-test; no I/O)
# ---------------------------------------------------------------------------

@dataclass
class PlanEntry:
    episode_id: str
    model: str
    block_id: str
    order_in_block: int
    offload_gb: int
    state_label: str
    concurrency: int
    condition: str
    repeat: int
    will_restart: bool
    will_warmup: bool
    planned_burst_requests: int
    server_command: list[str] | None


def build_server_command(
    python_executable: str,
    model_name: str,
    host: str,
    port: int,
    api_key: str,
    gpu_mem_util: float,
    tp_size: int,
    max_model_len: int,
    offload_gb: int,
) -> list[str]:
    return [
        python_executable,
        *VLLM_SERVE_MODULE_ARGS,
        model_name,
        "--host", host,
        "--port", str(port),
        "--api-key", api_key,
        "--dtype", "auto",
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", str(max_model_len),
        "--cpu-offload-gb", str(offload_gb),
    ]


def build_execution_plan(
    rows: list[ScheduleRow],
    python_executable: str,
    host: str,
    port: int,
    api_key: str,
    gpu_mem_util: float,
    tp_size: int,
    max_model_len: int,
) -> list[PlanEntry]:
    plan: list[PlanEntry] = []

    for r in rows:
        will_restart = r.restart_server_before_block > 0
        will_warmup = r.warmup_requests > 0
        planned_burst = r.burst_parallel_requests if r.condition == "fixed_burst" else 0

        server_command = None
        if will_restart:
            model_name = MODEL_NAME_MAP[r.model]
            server_command = build_server_command(
                python_executable, model_name, host, port, api_key,
                gpu_mem_util, tp_size, max_model_len, r.offload_gb,
            )

        plan.append(
            PlanEntry(
                episode_id=r.episode_id, model=r.model, block_id=r.block_id,
                order_in_block=r.order_in_block, offload_gb=r.offload_gb,
                state_label=r.state_label, concurrency=r.concurrency,
                condition=r.condition, repeat=r.repeat,
                will_restart=will_restart, will_warmup=will_warmup,
                planned_burst_requests=planned_burst,
                server_command=server_command,
            )
        )

    return plan


def print_dry_run(plan: list[PlanEntry]) -> None:
    print("=" * 100)
    print("DRY RUN: execution plan (no server started, no requests sent)")
    print("=" * 100)

    for i, e in enumerate(plan, start=1):
        marker = " [RESTART]" if e.will_restart else ""
        warm = " +warmup" if e.will_warmup else ""
        burst_info = (
            f"burst x{e.planned_burst_requests}"
            if e.planned_burst_requests else "no_burst"
        )
        print(
            f"{i:3d}. {e.episode_id:45s} block={e.block_id:20s} "
            f"pos={e.order_in_block} state={e.state_label:5s} "
            f"conc={e.concurrency} {e.condition:12s} {burst_info}{marker}{warm}"
        )
        if e.server_command:
            print(f"       server: {' '.join(e.server_command)}")

    n_restarts = sum(1 for e in plan if e.will_restart)
    n_warmups = sum(1 for e in plan if e.will_warmup)
    n_burst_episodes = sum(1 for e in plan if e.planned_burst_requests > 0)

    print()
    print(f"Total episodes: {len(plan)}")
    print(f"Planned server restarts: {n_restarts}")
    print(f"Planned warmups: {n_warmups}")
    print(f"Episodes with burst: {n_burst_episodes}")
    print(f"Episodes without burst: {len(plan) - n_burst_episodes}")


# ---------------------------------------------------------------------------
# Server process management
# ---------------------------------------------------------------------------

def start_server(
    command: list[str],
    stdout_log: Path,
    stderr_log: Path,
    gpu_device: str | None,
) -> subprocess.Popen:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if gpu_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_device

    stdout_f = stdout_log.open("ab")
    stderr_f = stderr_log.open("ab")

    process = subprocess.Popen(
        command,
        stdout=stdout_f,
        stderr=stderr_f,
        env=env,
        preexec_fn=os.setsid,  # own process group -> targeted stop, no pkill
    )
    return process


def stop_server(process: subprocess.Popen, timeout_s: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            pass


async def wait_for_server_ready(
    base_url: str,
    api_key: str,
    timeout_s: float,
    process: subprocess.Popen,
) -> bool:
    import httpx

    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False  # server process died during startup

            try:
                resp = await client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass

            try:
                resp = await client.get(
                    f"{base_url}/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass

            await asyncio.sleep(2.0)

    return False


# ---------------------------------------------------------------------------
# Streaming request execution
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    request_id: str
    role: str  # "victim" | "burst" | "warmup"
    index: int
    start_iso: str
    start_monotonic: float
    first_token_monotonic: float | None
    end_monotonic: float | None
    ttft_ms: float | None
    e2el_ms: float | None
    output_tokens: int
    tpot_ms: float | None
    itl_ms: list[float]
    error: str | None
    timed_out: bool
    prompt_sha256: str
    prompt_token_count: int
    overlaps_burst: bool = False
    overlap_seconds: float = 0.0


async def stream_chat_request(
    client: Any,
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: PromptResult,
    max_tokens: int,
    temperature: float,
    role: str,
    index: int,
    request_timeout_s: float,
    first_token_event: asyncio.Event | None = None,
) -> RequestMetrics:
    request_id = f"{role}-{index}-{int(time.time() * 1000)}"
    start_iso = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()

    first_token_time: float | None = None
    end_time: float | None = None
    token_timestamps: list[float] = []
    output_text_parts: list[str] = []
    error: str | None = None
    timed_out = False

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt.text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions",
            json=payload, headers=headers, timeout=request_timeout_s,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    now = time.monotonic()
                    if first_token_time is None:
                        first_token_time = now
                        if first_token_event is not None:
                            first_token_event.set()
                    token_timestamps.append(now)
                    output_text_parts.append(content)
        end_time = time.monotonic()
    except asyncio.TimeoutError:
        timed_out = True
        error = "timeout"
        end_time = time.monotonic()
    except Exception as exc:  # noqa: BLE001 - record any client/server error
        error = f"{type(exc).__name__}: {exc}"
        end_time = time.monotonic()

    output_tokens = len(token_timestamps)

    ttft_ms = (first_token_time - start) * 1000.0 if first_token_time is not None else None
    e2el_ms = (end_time - start) * 1000.0 if end_time is not None else None

    itl_ms: list[float] = [
        (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
        for i in range(1, len(token_timestamps))
    ]

    tpot_ms = None
    if output_tokens >= 2 and first_token_time is not None and end_time is not None:
        tpot_ms = ((end_time - first_token_time) * 1000.0) / (output_tokens - 1)

    return RequestMetrics(
        request_id=request_id, role=role, index=index,
        start_iso=start_iso, start_monotonic=start,
        first_token_monotonic=first_token_time, end_monotonic=end_time,
        ttft_ms=ttft_ms, e2el_ms=e2el_ms, output_tokens=output_tokens,
        tpot_ms=tpot_ms, itl_ms=itl_ms, error=error, timed_out=timed_out,
        prompt_sha256=prompt.sha256_hex, prompt_token_count=prompt.token_count,
    )


# ---------------------------------------------------------------------------
# Metric summarization
# ---------------------------------------------------------------------------

def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def summarize_metric(values: list[float]) -> dict[str, float | int | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "median": None, "p95": None, "p99": None}
    return {
        "n": len(clean),
        "median": statistics.median(clean),
        "p95": percentile(clean, 0.95),
        "p99": percentile(clean, 0.99),
    }


# ---------------------------------------------------------------------------
# Episode execution
# ---------------------------------------------------------------------------

async def run_episode(
    row: ScheduleRow,
    block_server_config: dict[str, Any],
    warmup_result: dict[str, Any] | None,
    base_url: str,
    api_key: str,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import httpx

    model_name = MODEL_NAME_MAP[row.model]

    victim_prompt = generate_exact_length_prompt(
        row.victim_workload_seed, row.victim_input_len, tokenizer
    )
    burst_prompt = None
    if row.condition == "fixed_burst":
        burst_prompt = generate_exact_length_prompt(
            row.burst_workload_seed, row.burst_input_len, tokenizer
        )

    episode_start_iso = datetime.now(timezone.utc).isoformat()

    first_wave_n = min(row.concurrency, args.victim_requests)
    first_wave_events = [asyncio.Event() for _ in range(first_wave_n)]
    sem = asyncio.Semaphore(row.concurrency)

    victim_metrics: list[RequestMetrics | None] = [None] * args.victim_requests

    burst_start_monotonic: float | None = None
    burst_end_monotonic: float | None = None
    burst_metrics: list[RequestMetrics] = []
    trigger_achieved = False
    trigger_time_monotonic: float | None = None

    async with httpx.AsyncClient(timeout=args.request_timeout_s) as client:

        async def run_victim(i: int) -> None:
            async with sem:
                event = first_wave_events[i] if i < first_wave_n else None
                metrics = await stream_chat_request(
                    client, base_url, api_key, model_name, victim_prompt,
                    row.victim_output_len, row.victim_temperature,
                    "victim", i, args.request_timeout_s, event,
                )
                victim_metrics[i] = metrics

        victim_tasks = [
            asyncio.create_task(run_victim(i))
            for i in range(args.victim_requests)
        ]

        try:
            await asyncio.wait_for(
                asyncio.gather(*[e.wait() for e in first_wave_events]),
                timeout=args.trigger_timeout_s,
            )
            trigger_achieved = True
            trigger_time_monotonic = time.monotonic()
        except asyncio.TimeoutError:
            trigger_achieved = False

        if trigger_achieved and row.condition == "fixed_burst":
            burst_start_monotonic = time.monotonic()
            burst_tasks = [
                asyncio.create_task(
                    stream_chat_request(
                        client, base_url, api_key, model_name, burst_prompt,
                        row.burst_output_len, row.burst_temperature,
                        "burst", i, args.request_timeout_s, None,
                    )
                )
                for i in range(row.burst_parallel_requests)
            ]
            burst_metrics = list(await asyncio.gather(*burst_tasks))
            burst_end_monotonic = time.monotonic()

        await asyncio.gather(*victim_tasks)

    episode_end_iso = datetime.now(timezone.utc).isoformat()

    for m in victim_metrics:
        if m is None or m.end_monotonic is None:
            continue
        if burst_start_monotonic is not None and burst_end_monotonic is not None:
            overlap_start = max(m.start_monotonic, burst_start_monotonic)
            overlap_end = min(m.end_monotonic, burst_end_monotonic)
            overlap = max(0.0, overlap_end - overlap_start)
            m.overlaps_burst = overlap > 0
            m.overlap_seconds = overlap

    victim_completed = [m for m in victim_metrics if m is not None and m.error is None]
    victim_failed = [m for m in victim_metrics if m is not None and m.error is not None]
    victim_timeouts = [m for m in victim_metrics if m is not None and m.timed_out]

    burst_completed = [m for m in burst_metrics if m.error is None]
    burst_failed = [m for m in burst_metrics if m.error is not None]

    victim_ttft = summarize_metric([m.ttft_ms for m in victim_completed if m.ttft_ms is not None])
    victim_tpot = summarize_metric([m.tpot_ms for m in victim_completed if m.tpot_ms is not None])
    victim_itl = summarize_metric([v for m in victim_completed for v in m.itl_ms])
    victim_e2el = summarize_metric([m.e2el_ms for m in victim_completed if m.e2el_ms is not None])

    total_victim_output_tokens = sum(m.output_tokens for m in victim_completed)
    if victim_completed:
        span_end = max(m.end_monotonic for m in victim_completed if m.end_monotonic is not None)
        span_start = min(m.start_monotonic for m in victim_completed)
        total_victim_duration_s = span_end - span_start
    else:
        total_victim_duration_s = 0.0
    victim_output_throughput = (
        total_victim_output_tokens / total_victim_duration_s
        if total_victim_duration_s > 0 else None
    )

    n_overlap = sum(1 for m in victim_metrics if m is not None and m.overlaps_burst)
    overlap_fraction = n_overlap / len(victim_metrics) if victim_metrics else 0.0

    burst_output_tokens = sum(m.output_tokens for m in burst_completed)
    burst_duration_s = (
        (burst_end_monotonic - burst_start_monotonic)
        if burst_start_monotonic is not None and burst_end_monotonic is not None
        else None
    )
    burst_output_throughput = (
        burst_output_tokens / burst_duration_s
        if burst_duration_s and burst_duration_s > 0 else None
    )

    if not trigger_achieved:
        status = "failed"
    elif victim_failed or (row.condition == "fixed_burst" and burst_failed):
        status = "partial"
    else:
        status = "complete"

    return {
        "episode_id": row.episode_id,
        "schedule_row": asdict(row),
        "server_config": block_server_config,
        "timestamps": {
            "episode_start": episode_start_iso,
            "episode_end": episode_end_iso,
        },
        "warmup_result": warmup_result,
        "trigger": {
            "concurrency_required": first_wave_n,
            "timeout_s": args.trigger_timeout_s,
            "achieved": trigger_achieved,
            "trigger_time_monotonic": trigger_time_monotonic,
        },
        "burst_interval": (
            {
                "start_monotonic": burst_start_monotonic,
                "end_monotonic": burst_end_monotonic,
            }
            if burst_start_monotonic is not None else None
        ),
        "victim_requests": [asdict(m) for m in victim_metrics if m is not None],
        "burst_requests": [asdict(m) for m in burst_metrics],
        "victim_summary_counts": {
            "completed": len(victim_completed),
            "failed": len(victim_failed),
            "timeouts": len(victim_timeouts),
            "total": len(victim_metrics),
        },
        "burst_summary_counts": {
            "completed": len(burst_completed),
            "failed": len(burst_failed),
            "total": len(burst_metrics),
        },
        "victim_aggregate_metrics": {
            "ttft_ms": victim_ttft,
            "tpot_ms": victim_tpot,
            "itl_ms": victim_itl,
            "e2el_ms": victim_e2el,
            "output_throughput_tokens_per_s": victim_output_throughput,
            "n_overlapping_burst": n_overlap,
            "overlap_fraction": overlap_fraction,
        },
        "burst_aggregate_metrics": {
            "output_tokens_total": burst_output_tokens,
            "output_throughput_tokens_per_s": burst_output_throughput,
        },
        "status": status,
    }


# ---------------------------------------------------------------------------
# Full schedule execution (real run)
# ---------------------------------------------------------------------------

async def execute_schedule(rows: list[ScheduleRow], args: argparse.Namespace) -> int:
    import httpx

    output_dir: Path = args.output_dir
    episodes_dir = output_dir / "episodes"
    server_logs_dir = output_dir / "server_logs"
    manifest_path = output_dir / "phase_a_run_manifest.jsonl"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    server_logs_dir.mkdir(parents=True, exist_ok=True)

    current_server: subprocess.Popen | None = None
    current_block_id: str | None = None
    current_server_config: dict[str, Any] = {}

    shutdown = asyncio.Event()

    def handle_signal(signame: str) -> None:
        print(f"\nReceived {signame}, shutting down gracefully...", file=sys.stderr)
        shutdown.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s.name))

    tokenizer_cache: dict[str, Any] = {}

    def get_tokenizer(model_key: str) -> Any:
        if model_key not in tokenizer_cache:
            tokenizer_cache[model_key] = load_tokenizer(
                MODEL_NAME_MAP[model_key], self_test=False
            )
        return tokenizer_cache[model_key]

    executed = 0
    exit_code = 0
    base_url = f"http://{args.host}:{args.port}"

    try:
        for row in rows:
            if shutdown.is_set():
                print("Shutdown requested; stopping before next episode.", file=sys.stderr)
                break

            if args.only_model and row.model != args.only_model:
                continue

            if args.max_episodes is not None and executed >= args.max_episodes:
                break

            episode_path = episodes_dir / f"{row.episode_id}.json"

            if args.resume:
                status = check_episode_file_status(episode_path, row.episode_id)
                if status == "complete":
                    print(f"SKIP (complete): {row.episode_id}")
                    continue
                if status in ("partial", "corrupted"):
                    print(
                        f"WARNING: {row.episode_id} found with "
                        f"status={status!r}; re-running (will overwrite)"
                    )

            needs_restart = (
                row.restart_server_before_block > 0
                or current_server is None
                or current_block_id != row.block_id
            )

            warmup_result: dict[str, Any] | None = None

            if needs_restart:
                if current_server is not None:
                    stop_server(current_server)
                    current_server = None

                model_name = MODEL_NAME_MAP[row.model]
                command = build_server_command(
                    args.python_executable, model_name, args.host, args.port,
                    args.api_key, args.gpu_memory_utilization,
                    args.tensor_parallel_size, args.max_model_len, row.offload_gb,
                )
                stdout_log = server_logs_dir / f"{row.block_id}.stdout.log"
                stderr_log = server_logs_dir / f"{row.block_id}.stderr.log"

                start_iso = datetime.now(timezone.utc).isoformat()
                current_server = start_server(command, stdout_log, stderr_log, args.gpu_device)

                ready = await wait_for_server_ready(
                    base_url, args.api_key, args.server_start_timeout_s, current_server,
                )
                ready_iso = datetime.now(timezone.utc).isoformat() if ready else None

                current_block_id = row.block_id
                current_server_config = {
                    "command": command, "pid": current_server.pid,
                    "start_time": start_iso, "ready_time": ready_iso,
                    "offload_gb": row.offload_gb, "model": row.model,
                    "host": args.host, "port": args.port, "block_id": row.block_id,
                }

                append_manifest_line(manifest_path, {
                    "event": "server_start", "block_id": row.block_id,
                    "model": row.model, "offload_gb": row.offload_gb,
                    "pid": current_server.pid, "command": command,
                    "start_time": start_iso, "ready_time": ready_iso, "ready": ready,
                })

                if not ready:
                    print(
                        f"ERROR: server did not become ready for block "
                        f"{row.block_id} within {args.server_start_timeout_s}s",
                        file=sys.stderr,
                    )
                    exit_code = 1
                    break

                tokenizer = get_tokenizer(row.model)
                warmup_prompt = generate_exact_length_prompt(
                    derive_warmup_seed(row.block_id), row.victim_input_len, tokenizer,
                )
                warmup_start = time.monotonic()
                async with httpx.AsyncClient(timeout=args.request_timeout_s) as client:
                    warmup_metric = await stream_chat_request(
                        client, base_url, args.api_key, model_name, warmup_prompt,
                        row.victim_output_len, row.victim_temperature,
                        "warmup", 0, args.request_timeout_s, None,
                    )
                warmup_result = asdict(warmup_metric)
                warmup_result["duration_s"] = time.monotonic() - warmup_start

            tokenizer = get_tokenizer(row.model)

            episode_result = await run_episode(
                row, current_server_config, warmup_result, base_url,
                args.api_key, tokenizer, args,
            )

            write_json_atomic(episode_path, episode_result)
            append_manifest_line(manifest_path, {
                "event": "episode_complete", "episode_id": row.episode_id,
                "status": episode_result["status"], "block_id": row.block_id,
            })
            print(f"DONE ({episode_result['status']}): {row.episode_id}")
            executed += 1

    finally:
        if current_server is not None:
            stop_server(current_server)

    return exit_code


# ---------------------------------------------------------------------------
# Self-test (no GPU / vLLM / network required)
# ---------------------------------------------------------------------------

def _self_test_generate_schedule_csv(tmp_dir: Path) -> Path:
    rows: list[ScheduleRow] = []
    for model in ("llama", "qwen"):
        rows.extend(generate_schedule(model, repeats=5, seed=SCHEDULE_DEFAULT_SEED))

    csv_path = tmp_dir / "self_test_schedule.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SCHEDULE_REQUIRED_FIELDS))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    return csv_path


def self_test() -> bool:
    import tempfile

    all_ok = True

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal all_ok
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_ok = False
        suffix = f" -- {detail}" if detail and not condition else ""
        print(f"[{status}] {name}{suffix}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. Schedule validation on a valid, freshly generated schedule
        csv_path = _self_test_generate_schedule_csv(tmp_path)
        rows = parse_schedule_csv(csv_path)
        errors = validate_schedule_rows(rows)
        check("schedule validation (valid schedule)", not errors, "; ".join(errors[:3]))
        check("schedule row count == 80", len(rows) == 80, f"got {len(rows)}")

        # 1b. Validation must catch a deliberately corrupted schedule
        corrupted_rows = list(rows)
        corrupted_rows[1] = corrupted_rows[0]  # inject a duplicate episode_id
        errors_corrupt = validate_schedule_rows(corrupted_rows)
        check(
            "schedule validation detects duplicate episode_id",
            any("duplicate" in e for e in errors_corrupt),
        )

        # 2. Prompt reproducibility
        tok = SelfTestTokenizer()
        p1 = generate_exact_length_prompt(12345, 64, tok)
        p2 = generate_exact_length_prompt(12345, 64, tok)
        check(
            "prompt reproducibility (same seed -> identical text/hash)",
            p1.text == p2.text and p1.sha256_hex == p2.sha256_hex,
        )
        check(
            "prompt has exact target token count",
            p1.token_count == 64 and len(tok.encode(p1.text)) == 64,
        )

        # 3. Identical victim/burst prompt hashes across state and condition
        llama_rows = [r for r in rows if r.model == "llama"]
        ref = llama_rows[0]
        matched_group = [
            r for r in llama_rows
            if r.concurrency == ref.concurrency and r.repeat == ref.repeat
        ]
        victim_hashes = {
            generate_exact_length_prompt(r.victim_workload_seed, r.victim_input_len, tok).sha256_hex
            for r in matched_group
        }
        burst_hashes = {
            generate_exact_length_prompt(r.burst_workload_seed, r.burst_input_len, tok).sha256_hex
            for r in matched_group
        }
        check(
            "identical victim prompt hash across state & condition",
            len(victim_hashes) == 1 and len(matched_group) == 4,
            f"{len(victim_hashes)} distinct hash(es) over {len(matched_group)} rows",
        )
        check("identical burst prompt hash across state & condition", len(burst_hashes) == 1)

        # 4. Victim and burst prompt streams must be independent
        check("victim and burst prompt streams differ", victim_hashes != burst_hashes)

        # 5. Dry-run plan preserves CSV row order
        plan = build_execution_plan(
            rows, sys.executable, "127.0.0.1", 8000, "pilotkey", 0.9, 1, 8192
        )
        check(
            "dry-run plan preserves CSV row order",
            [p.episode_id for p in plan] == [r.episode_id for r in rows],
        )

        # 6. Exactly 20 restarts / 20 warmups (10 blocks x 2 models)
        n_restarts = sum(1 for p in plan if p.will_restart)
        n_warmups = sum(1 for p in plan if p.will_warmup)
        check("exactly 20 planned server restarts", n_restarts == 20, f"got {n_restarts}")
        check("exactly 20 planned warmups", n_warmups == 20, f"got {n_warmups}")

        # 7. no_burst -> 0 planned burst requests, fixed_burst -> 4
        no_burst_ok = all(p.planned_burst_requests == 0 for p in plan if p.condition == "no_burst")
        fixed_burst_ok = all(p.planned_burst_requests == 4 for p in plan if p.condition == "fixed_burst")
        check("no_burst episodes plan 0 burst requests", no_burst_ok)
        check("fixed_burst episodes plan 4 burst requests", fixed_burst_ok)

        # 8. Resume: complete / partial / corrupted / missing episode files
        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()

        complete_ep = rows[0]
        complete_path = episodes_dir / f"{complete_ep.episode_id}.json"
        write_json_atomic(complete_path, {"episode_id": complete_ep.episode_id, "status": "complete"})
        check(
            "resume detects complete episode",
            check_episode_file_status(complete_path, complete_ep.episode_id) == "complete",
        )

        partial_ep = rows[1]
        partial_path = episodes_dir / f"{partial_ep.episode_id}.json"
        write_json_atomic(partial_path, {"episode_id": partial_ep.episode_id, "status": "partial"})
        check(
            "resume detects partial episode",
            check_episode_file_status(partial_path, partial_ep.episode_id) == "partial",
        )

        corrupted_ep = rows[2]
        corrupted_path = episodes_dir / f"{corrupted_ep.episode_id}.json"
        corrupted_path.write_text("{not valid json", encoding="utf-8")
        check(
            "resume detects corrupted episode",
            check_episode_file_status(corrupted_path, corrupted_ep.episode_id) == "corrupted",
        )

        missing_ep = rows[3]
        missing_path = episodes_dir / f"{missing_ep.episode_id}.json"
        check(
            "resume detects missing episode",
            check_episode_file_status(missing_path, missing_ep.episode_id) == "missing",
        )

        # 8b. Atomic write must not leave a .tmp artifact behind
        tmp_artifact = complete_path.with_suffix(complete_path.suffix + ".tmp")
        check("atomic write leaves no .tmp artifact", not tmp_artifact.exists())

        # 8c. episode_id mismatch inside a file must be treated as corrupted
        mismatch_path = episodes_dir / f"{rows[4].episode_id}.json"
        write_json_atomic(mismatch_path, {"episode_id": "wrong-id", "status": "complete"})
        check(
            "resume flags episode_id mismatch as corrupted",
            check_episode_file_status(mismatch_path, rows[4].episode_id) == "corrupted",
        )

    print()
    print("=" * 60)
    print("SELF-TEST OVERALL:", "PASS" if all_ok else "FAIL")
    print("=" * 60)
    return all_ok


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the frozen Phase A schedule against a real "
        "vLLM OpenAI-compatible server, in exact CSV file order."
    )
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--victim-requests", type=int, default=DEFAULT_VICTIM_REQUESTS)
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--server-start-timeout-s", type=float, default=DEFAULT_SERVER_START_TIMEOUT_S)
    parser.add_argument("--trigger-timeout-s", type=float, default=DEFAULT_TRIGGER_TIMEOUT_S)
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--gpu-device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-model", type=str, default=None, choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEM_UTIL)
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)

    args = parser.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = args.schedule.parent / "results"

    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        ok = self_test()
        return 0 if ok else 1

    if not args.schedule.exists():
        print(f"ERROR: schedule file not found: {args.schedule}", file=sys.stderr)
        return 1

    try:
        rows = parse_schedule_csv(args.schedule)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    errors = validate_schedule_rows(rows)
    if errors:
        print("SCHEDULE VALIDATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    if args.dry_run:
        plan = build_execution_plan(
            rows, args.python_executable, args.host, args.port, args.api_key,
            args.gpu_memory_utilization, args.tensor_parallel_size, args.max_model_len,
        )
        print_dry_run(plan)
        return 0

    output_errors, fingerprint = check_output_dir_compatibility(
        args.output_dir, args.schedule, args.resume
    )
    if output_errors:
        print("OUTPUT DIRECTORY VALIDATION FAILED:", file=sys.stderr)
        for e in output_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "schedule_fingerprint.txt").write_text(fingerprint, encoding="utf-8")

    return asyncio.run(execute_schedule(rows, args))


if __name__ == "__main__":
    sys.exit(main())
