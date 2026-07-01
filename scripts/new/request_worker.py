#!/usr/bin/env python3
"""
request_worker.py
=================
Robuster Request-Worker für die Burst-Kalibrierung.

Betriebsarten (Umgebungsvariable MODE):
  continuous_victim  – hält CONCURRENCY Requests konstant aktiv bis SIGTERM/SIGINT
  fixed_count_burst  – sendet exakt REQUEST_COUNT Requests nach gemeinsamer Freigabe

Python 3.11+
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import random
import signal
import string
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"[request_worker] Fehlende Umgebungsvariable: {name}")
    return value


def _require_int(name: str) -> int:
    try:
        return int(_require(name))
    except ValueError:
        sys.exit(f"[request_worker] {name} muss eine ganze Zahl sein.")


def _require_float(name: str) -> float:
    try:
        return float(_require(name))
    except ValueError:
        sys.exit(f"[request_worker] {name} muss eine Zahl sein.")


def _optional_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        sys.exit(f"[request_worker] {name} muss eine Zahl sein.")


def _check(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(f"[request_worker] {message}")


BASE_URL = _require("BASE_URL")
ENDPOINT = _require("ENDPOINT")
API_KEY = _require("API_KEY")
MODEL = _require("MODEL")
MODE = _require("MODE")
ROLE = _require("ROLE")
CONDITION = _require("CONDITION")
EPISODE_ID = _require("EPISODE_ID")
OUTFILE = _require("OUTFILE")
OFFLOAD_GB = _require_int("OFFLOAD_GB")
RUN_CONCURRENCY = _require_int("RUN_CONCURRENCY")
INPUT_LEN = _require_int("INPUT_LEN")
OUTPUT_LEN = _require_int("OUTPUT_LEN")
TEMPERATURE = _require_float("TEMPERATURE")
CONCURRENCY = _require_int("CONCURRENCY")
RANDOM_SEED = _require_int("RANDOM_SEED")
REQUEST_TIMEOUT_S = _optional_float("REQUEST_TIMEOUT_S", 600.0)
DRAIN_TIMEOUT_S = _optional_float("DRAIN_TIMEOUT_S", 600.0)
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID", "").strip()

_request_count_raw = os.environ.get("REQUEST_COUNT", "").strip()
REQUEST_COUNT: Optional[int] = None
if _request_count_raw and _request_count_raw != "0":
    try:
        REQUEST_COUNT = int(_request_count_raw)
    except ValueError:
        sys.exit("[request_worker] REQUEST_COUNT muss eine ganze Zahl sein.")

if MODE not in {"continuous_victim", "fixed_count_burst"}:
    sys.exit(
        f"[request_worker] Ungültiger MODE='{MODE}'. "
        "Erlaubt: continuous_victim | fixed_count_burst"
    )
if ROLE not in {"victim", "burst"}:
    sys.exit(f"[request_worker] Ungültiger ROLE='{ROLE}'. Erlaubt: victim | burst")
if MODE == "continuous_victim" and ROLE != "victim":
    sys.exit("[request_worker] MODE=continuous_victim erfordert ROLE=victim.")
if MODE == "fixed_count_burst" and ROLE != "burst":
    sys.exit("[request_worker] MODE=fixed_count_burst erfordert ROLE=burst.")

_check(CONCURRENCY >= 1, f"CONCURRENCY muss >= 1 sein (ist {CONCURRENCY}).")
_check(RUN_CONCURRENCY >= 1, f"RUN_CONCURRENCY muss >= 1 sein (ist {RUN_CONCURRENCY}).")
_check(INPUT_LEN >= 1, f"INPUT_LEN muss >= 1 sein (ist {INPUT_LEN}).")
_check(OUTPUT_LEN >= 1, f"OUTPUT_LEN muss >= 1 sein (ist {OUTPUT_LEN}).")
_check(REQUEST_TIMEOUT_S > 0, f"REQUEST_TIMEOUT_S muss > 0 sein (ist {REQUEST_TIMEOUT_S}).")
_check(DRAIN_TIMEOUT_S >= 0, f"DRAIN_TIMEOUT_S muss >= 0 sein (ist {DRAIN_TIMEOUT_S}).")
_check(OFFLOAD_GB >= 0, f"OFFLOAD_GB muss >= 0 sein (ist {OFFLOAD_GB}).")
_check(TEMPERATURE >= 0, f"TEMPERATURE muss >= 0 sein (ist {TEMPERATURE}).")

if MODE == "continuous_victim":
    if REQUEST_COUNT is not None:
        sys.exit(
            "[request_worker] continuous_victim erfordert REQUEST_COUNT leer oder 0 "
            f"(ist {REQUEST_COUNT})."
        )
else:
    if REQUEST_COUNT is None or REQUEST_COUNT < 1:
        sys.exit("[request_worker] fixed_count_burst erfordert REQUEST_COUNT >= 1.")
    if CONCURRENCY != REQUEST_COUNT:
        sys.exit(
            "[request_worker] fixed_count_burst erfordert CONCURRENCY == REQUEST_COUNT, "
            f"aber CONCURRENCY={CONCURRENCY} und REQUEST_COUNT={REQUEST_COUNT}."
        )


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _rand_prompt(target_tokens: int, rng: random.Random) -> str:
    """Legacy-kompatible Näherung: ungefähr vier Zeichen pro Ziel-Token."""
    target_chars = target_tokens * 4
    words: list[str] = []
    current_chars = 0
    while current_chars < target_chars:
        word = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 10)))
        words.append(word)
        current_chars += len(word) + 1
    return " ".join(words)


def _stats(data: list[float]) -> dict[str, Optional[float]]:
    if not data:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    array = np.asarray(data, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1) if len(array) > 1 else 0.0),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _error_text(exc: BaseException) -> str:
    message = str(exc).strip()
    return message if message else type(exc).__name__


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _new_request_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Ergebnisobjekt
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    request_id: str
    request_idx: int
    input_len: int
    target_output_len: int

    observed_output_events: int = 0
    actual_output_tokens: Optional[int] = None
    token_count_source: str = "unavailable"
    stream_done_observed: bool = False
    request_success: bool = False
    timed_out: bool = False
    cancelled: bool = False
    error_text: str = ""

    start_time_unix_ns: int = 0
    first_token_time_unix_ns: int = 0
    last_token_time_unix_ns: int = 0
    end_time_unix_ns: int = 0

    start_offset_ms: float = 0.0
    first_token_offset_ms: float = 0.0
    last_token_offset_ms: float = 0.0
    end_offset_ms: float = 0.0
    release_to_request_start_ms: Optional[float] = None

    ttft_ms: float = 0.0
    itl_sequence_ms: list[float] = field(default_factory=list)
    decode_time_ms: float = 0.0
    e2el_ms: float = 0.0

    _start_perf_ns: int = field(default=0, repr=False)
    _first_output_perf_ns: Optional[int] = field(default=None, repr=False)
    _last_output_perf_ns: Optional[int] = field(default=None, repr=False)

    @property
    def itl_mean_ms(self) -> Optional[float]:
        return float(np.mean(self.itl_sequence_ms)) if self.itl_sequence_ms else None

    @property
    def itl_median_ms(self) -> Optional[float]:
        return float(np.median(self.itl_sequence_ms)) if self.itl_sequence_ms else None

    @property
    def itl_p95_ms(self) -> Optional[float]:
        return (
            float(np.percentile(self.itl_sequence_ms, 95))
            if self.itl_sequence_ms
            else None
        )

    @property
    def tpot_ms(self) -> Optional[float]:
        if self.observed_output_events > 1:
            return self.decode_time_ms / (self.observed_output_events - 1)
        return None

    def finalize(self, worker_start_perf_ns: int) -> None:
        end_perf_ns = time.perf_counter_ns()
        self.end_time_unix_ns = time.time_ns()
        self.end_offset_ms = (end_perf_ns - worker_start_perf_ns) / 1_000_000

        if self._start_perf_ns:
            self.e2el_ms = (end_perf_ns - self._start_perf_ns) / 1_000_000
        if self._first_output_perf_ns is not None and self._last_output_perf_ns is not None:
            self.decode_time_ms = (
                self._last_output_perf_ns - self._first_output_perf_ns
            ) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request_idx": self.request_idx,
            "role": ROLE,
            "mode": MODE,
            "episode_id": EPISODE_ID,
            "input_len": self.input_len,
            "target_output_len": self.target_output_len,
            "observed_output_events": self.observed_output_events,
            "actual_output_tokens": self.actual_output_tokens,
            "token_count_source": self.token_count_source,
            "stream_done_observed": self.stream_done_observed,
            "request_success": self.request_success,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "error_text": self.error_text,
            "start_time_unix_ns": self.start_time_unix_ns,
            "first_token_time_unix_ns": self.first_token_time_unix_ns,
            "last_token_time_unix_ns": self.last_token_time_unix_ns,
            "end_time_unix_ns": self.end_time_unix_ns,
            "start_offset_ms": self.start_offset_ms,
            "first_token_offset_ms": self.first_token_offset_ms,
            "last_token_offset_ms": self.last_token_offset_ms,
            "end_offset_ms": self.end_offset_ms,
            "release_to_request_start_ms": self.release_to_request_start_ms,
            "ttft_ms": self.ttft_ms,
            "itl_sequence_ms": self.itl_sequence_ms,
            "itl_mean_ms": self.itl_mean_ms,
            "itl_median_ms": self.itl_median_ms,
            "itl_p95_ms": self.itl_p95_ms,
            "decode_time_ms": self.decode_time_ms,
            "tpot_ms": self.tpot_ms,
            "e2el_ms": self.e2el_ms,
        }


# ---------------------------------------------------------------------------
# HTTP-Ausführung
# ---------------------------------------------------------------------------


def _extract_completion_tokens(chunk: dict[str, Any]) -> Optional[int]:
    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return None
    value = usage.get("completion_tokens")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and int(value) >= 0:
        return int(value)
    return None


async def do_request(
    client: httpx.AsyncClient,
    request_idx: int,
    prompt: str,
    worker_start_perf_ns: int,
    burst_release_perf_ns: Optional[int] = None,
) -> RequestResult:
    result = RequestResult(
        request_id=_new_request_id(),
        request_idx=request_idx,
        input_len=INPUT_LEN,
        target_output_len=OUTPUT_LEN,
    )

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": OUTPUT_LEN,
        "temperature": TEMPERATURE,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    result._start_perf_ns = time.perf_counter_ns()
    result.start_time_unix_ns = time.time_ns()
    result.start_offset_ms = (
        result._start_perf_ns - worker_start_perf_ns
    ) / 1_000_000
    if burst_release_perf_ns is not None:
        result.release_to_request_start_ms = (
            result._start_perf_ns - burst_release_perf_ns
        ) / 1_000_000

    previous_output_perf_ns: Optional[int] = None

    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_S):
            async with client.stream(
                "POST",
                f"{BASE_URL}{ENDPOINT}",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(
                    connect=min(10.0, REQUEST_TIMEOUT_S),
                    read=REQUEST_TIMEOUT_S,
                    write=REQUEST_TIMEOUT_S,
                    pool=REQUEST_TIMEOUT_S,
                ),
            ) as response:
                response.raise_for_status()

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        result.stream_done_observed = True
                        break

                    try:
                        chunk = json.loads(data)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not isinstance(chunk, dict):
                        continue

                    completion_tokens = _extract_completion_tokens(chunk)
                    if completion_tokens is not None:
                        result.actual_output_tokens = completion_tokens
                        result.token_count_source = "stream_usage"

                    choices = chunk.get("choices") or []
                    choice0 = choices[0] if choices and isinstance(choices[0], dict) else {}
                    delta = choice0.get("delta") or {}
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if not content:
                        continue

                    now_perf_ns = time.perf_counter_ns()
                    now_unix_ns = time.time_ns()
                    result.observed_output_events += 1
                    result.last_token_time_unix_ns = now_unix_ns
                    result.last_token_offset_ms = (
                        now_perf_ns - worker_start_perf_ns
                    ) / 1_000_000
                    result._last_output_perf_ns = now_perf_ns

                    if result._first_output_perf_ns is None:
                        result._first_output_perf_ns = now_perf_ns
                        result.first_token_time_unix_ns = now_unix_ns
                        result.first_token_offset_ms = (
                            now_perf_ns - worker_start_perf_ns
                        ) / 1_000_000
                        result.ttft_ms = (
                            now_perf_ns - result._start_perf_ns
                        ) / 1_000_000
                    elif previous_output_perf_ns is not None:
                        result.itl_sequence_ms.append(
                            (now_perf_ns - previous_output_perf_ns) / 1_000_000
                        )

                    previous_output_perf_ns = now_perf_ns

        if result.observed_output_events == 0:
            result.error_text = (
                "StreamEmpty: HTTP-Antwort empfangen, aber keine Content-Events im Stream."
            )
        elif not result.stream_done_observed:
            result.error_text = (
                "StreamIncomplete: Content empfangen, aber [DONE] wurde nicht beobachtet."
            )
        else:
            result.request_success = True

    except httpx.TimeoutException as exc:
        result.timed_out = True
        result.error_text = f"httpx.TimeoutException: {_error_text(exc)}"
    except TimeoutError as exc:
        result.timed_out = True
        result.error_text = f"TimeoutError: {_error_text(exc)}"
    except asyncio.CancelledError:
        result.cancelled = True
        result.error_text = result.error_text or "CancelledError: task aborted during drain"
        # Absichtlich nicht erneut auslösen: partielle Messdaten bleiben im Resultat erhalten.
    except Exception as exc:  # noqa: BLE001 - Fehler wird strukturiert protokolliert
        result.error_text = f"{type(exc).__name__}: {_error_text(exc)}"
    finally:
        result.finalize(worker_start_perf_ns)

    return result


# ---------------------------------------------------------------------------
# Betriebsart: continuous_victim
# ---------------------------------------------------------------------------


async def run_continuous_victim(
    stop_event: asyncio.Event,
    worker_start_perf_ns: int,
) -> tuple[list[RequestResult], int, Optional[int], Optional[float]]:
    results: list[RequestResult] = []
    pending: set[asyncio.Task[RequestResult]] = set()
    submitted = 0

    limits = httpx.Limits(
        max_connections=max(CONCURRENCY + 4, CONCURRENCY),
        max_keepalive_connections=CONCURRENCY,
    )

    async with httpx.AsyncClient(limits=limits) as client:
        async def one(idx: int, prompt: str) -> RequestResult:
            try:
                return await do_request(
                    client=client,
                    request_idx=idx,
                    prompt=prompt,
                    worker_start_perf_ns=worker_start_perf_ns,
                )
            except asyncio.CancelledError:
                # Sicherheitsnetz für Cancel vor Eintritt in do_request().
                placeholder = RequestResult(
                    request_id=_new_request_id(),
                    request_idx=idx,
                    input_len=INPUT_LEN,
                    target_output_len=OUTPUT_LEN,
                    cancelled=True,
                    error_text="CancelledError: task cancelled before request start",
                )
                placeholder.finalize(worker_start_perf_ns)
                return placeholder

        while not stop_event.is_set():
            done = {task for task in pending if task.done()}
            for task in done:
                results.append(task.result())
            pending.difference_update(done)

            while len(pending) < CONCURRENCY and not stop_event.is_set():
                request_idx = submitted
                prompt = _rand_prompt(
                    INPUT_LEN, random.Random(RANDOM_SEED + request_idx)
                )
                pending.add(asyncio.create_task(one(request_idx, prompt)))
                submitted += 1

            await asyncio.sleep(0.005)

        if pending:
            if DRAIN_TIMEOUT_S > 0:
                done, not_done = await asyncio.wait(
                    pending,
                    timeout=DRAIN_TIMEOUT_S,
                    return_when=asyncio.ALL_COMPLETED,
                )
            else:
                done, not_done = set(), set(pending)

            for task in not_done:
                task.cancel()
            if not_done:
                await asyncio.gather(*not_done, return_exceptions=True)

            for task in done | not_done:
                try:
                    results.append(task.result())
                except asyncio.CancelledError:
                    # Sollte wegen one()-Sicherheitsnetz nicht eintreten.
                    placeholder = RequestResult(
                        request_id=_new_request_id(),
                        request_idx=-1,
                        input_len=INPUT_LEN,
                        target_output_len=OUTPUT_LEN,
                        cancelled=True,
                        error_text="CancelledError: task produced no result",
                    )
                    placeholder.finalize(worker_start_perf_ns)
                    results.append(placeholder)
                except Exception as exc:  # noqa: BLE001
                    placeholder = RequestResult(
                        request_id=_new_request_id(),
                        request_idx=-1,
                        input_len=INPUT_LEN,
                        target_output_len=OUTPUT_LEN,
                        error_text=f"WorkerTaskError: {type(exc).__name__}: {_error_text(exc)}",
                    )
                    placeholder.finalize(worker_start_perf_ns)
                    results.append(placeholder)

    return results, submitted, None, None


# ---------------------------------------------------------------------------
# Betriebsart: fixed_count_burst
# ---------------------------------------------------------------------------


async def run_fixed_count_burst(
    worker_start_perf_ns: int,
) -> tuple[list[RequestResult], int, int, float]:
    assert REQUEST_COUNT is not None

    prompts = [
        _rand_prompt(INPUT_LEN, random.Random(RANDOM_SEED + idx))
        for idx in range(REQUEST_COUNT)
    ]
    all_ready_event = asyncio.Event()
    start_release_event = asyncio.Event()
    ready_lock = asyncio.Lock()
    ready_count = 0
    release_perf_ns: Optional[int] = None

    limits = httpx.Limits(
        max_connections=REQUEST_COUNT,
        max_keepalive_connections=REQUEST_COUNT,
    )

    async with httpx.AsyncClient(limits=limits) as client:
        async def one(idx: int) -> RequestResult:
            nonlocal ready_count
            async with ready_lock:
                ready_count += 1
                if ready_count == REQUEST_COUNT:
                    all_ready_event.set()

            # Auch der letzte Task wartet zwingend auf die separate Freigabe.
            await start_release_event.wait()
            assert release_perf_ns is not None
            return await do_request(
                client=client,
                request_idx=idx,
                prompt=prompts[idx],
                worker_start_perf_ns=worker_start_perf_ns,
                burst_release_perf_ns=release_perf_ns,
            )

        tasks = [asyncio.create_task(one(idx)) for idx in range(REQUEST_COUNT)]
        await all_ready_event.wait()

        release_perf_ns = time.perf_counter_ns()
        release_unix_ns = time.time_ns()
        release_offset_ms = (
            release_perf_ns - worker_start_perf_ns
        ) / 1_000_000
        start_release_event.set()

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[RequestResult] = []
    for idx, item in enumerate(raw_results):
        if isinstance(item, RequestResult):
            results.append(item)
        else:
            placeholder = RequestResult(
                request_id=_new_request_id(),
                request_idx=idx,
                input_len=INPUT_LEN,
                target_output_len=OUTPUT_LEN,
                cancelled=isinstance(item, asyncio.CancelledError),
                error_text=(
                    "CancelledError: burst task produced no result"
                    if isinstance(item, asyncio.CancelledError)
                    else f"WorkerTaskError: {type(item).__name__}: {_error_text(item)}"
                ),
            )
            placeholder.finalize(worker_start_perf_ns)
            results.append(placeholder)

    return results, REQUEST_COUNT, release_unix_ns, release_offset_ms


# ---------------------------------------------------------------------------
# JSON-Ausgabe
# ---------------------------------------------------------------------------


def write_json(
    results: list[RequestResult],
    submitted: int,
    worker_start_utc: str,
    worker_end_utc: str,
    duration_s: float,
    burst_release_time_unix_ns: Optional[int],
    burst_release_offset_ms: Optional[float],
) -> None:
    completed = [result for result in results if result.request_success]
    timed_out = [result for result in results if result.timed_out]
    cancelled = [result for result in results if result.cancelled]
    failed = [
        result
        for result in results
        if not result.request_success and not result.timed_out and not result.cancelled
    ]

    started = len(results)
    terminal_total = len(completed) + len(failed) + len(timed_out) + len(cancelled)
    if started != submitted:
        raise RuntimeError(
            "Invariant verletzt: len(results) != submitted "
            f"({started} != {submitted})."
        )
    if terminal_total != started:
        raise RuntimeError(
            "Invariant verletzt: completed + failed + timed_out + cancelled != started "
            f"({terminal_total} != {started})."
        )

    duration_safe = max(duration_s, 1e-9)
    actual_token_values = [
        result.actual_output_tokens
        for result in results
        if result.actual_output_tokens is not None
    ]

    document = {
        "schema_version": "1.1",
        "experiment_id": EXPERIMENT_ID,
        "episode_id": EPISODE_ID,
        "mode": MODE,
        "role": ROLE,
        "condition": CONDITION,
        "model_name": MODEL,
        "offload_gb": OFFLOAD_GB,
        "run_concurrency": RUN_CONCURRENCY,
        "worker_concurrency": CONCURRENCY,
        "request_count_target": REQUEST_COUNT,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "temperature": TEMPERATURE,
        "random_seed": RANDOM_SEED,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "drain_timeout_s": DRAIN_TIMEOUT_S,
        "worker_start_time_utc": worker_start_utc,
        "worker_end_time_utc": worker_end_utc,
        "duration_s": round(duration_s, 6),
        "burst_release_time_unix_ns": burst_release_time_unix_ns,
        "burst_release_offset_ms": burst_release_offset_ms,
        "submitted": submitted,
        "started": started,
        "completed": len(completed),
        "failed": len(failed),
        "timed_out": len(timed_out),
        "cancelled": len(cancelled),
        "total_input_tokens_requested": sum(
            result.input_len for result in results
        ),
        "total_output_events_observed": sum(
            result.observed_output_events for result in results
        ),
        "total_actual_output_tokens": sum(actual_token_values),
        "actual_output_tokens_available_requests": len(actual_token_values),
        "actual_output_tokens_unavailable_requests": started - len(actual_token_values),
        "request_throughput": round(len(completed) / duration_safe, 6),
        "ttft_ms": _stats([result.ttft_ms for result in completed]),
        "tpot_ms": _stats(
            [
                value
                for result in completed
                if (value := result.tpot_ms) is not None
            ]
        ),
        "itl_ms": _stats(
            [value for result in completed for value in result.itl_sequence_ms]
        ),
        "e2el_ms": _stats([result.e2el_ms for result in completed]),
        "individual_request_results": [result.to_dict() for result in results],
    }

    output_path = Path(OUTFILE)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)

    print(
        f"[{ROLE}/{MODE}] JSON → {output_path} "
        f"(submitted={submitted} started={started} OK={len(completed)} "
        f"FAIL={len(failed)} TIMEOUT={len(timed_out)} CANCELLED={len(cancelled)})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    loop = asyncio.get_running_loop()
    worker_start_perf_ns = time.perf_counter_ns()
    worker_start_utc = _utc_now()
    stop_event = asyncio.Event()

    def on_stop() -> None:
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGTERM, on_stop)
        loop.add_signal_handler(signal.SIGINT, on_stop)
    except NotImplementedError:
        # Fallback für Plattformen ohne loop.add_signal_handler().
        signal.signal(signal.SIGTERM, lambda *_: loop.call_soon_threadsafe(on_stop))
        signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(on_stop))

    if MODE == "continuous_victim":
        results, submitted, release_unix_ns, release_offset_ms = (
            await run_continuous_victim(stop_event, worker_start_perf_ns)
        )
    else:
        results, submitted, release_unix_ns, release_offset_ms = (
            await run_fixed_count_burst(worker_start_perf_ns)
        )

    worker_end_perf_ns = time.perf_counter_ns()
    worker_end_utc = _utc_now()
    duration_s = (worker_end_perf_ns - worker_start_perf_ns) / 1_000_000_000

    write_json(
        results=results,
        submitted=submitted,
        worker_start_utc=worker_start_utc,
        worker_end_utc=worker_end_utc,
        duration_s=duration_s,
        burst_release_time_unix_ns=release_unix_ns,
        burst_release_offset_ms=release_offset_ms,
    )


if __name__ == "__main__":
    asyncio.run(main())
