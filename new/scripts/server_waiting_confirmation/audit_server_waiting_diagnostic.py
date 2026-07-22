#!/usr/bin/env python3
"""Independent read-only raw-trace auditor for the frozen server-WAITING pair.

The auditor never writes below --diagnostic-dir.  It reconstructs prompt
identity, output token sequences, positive-token receive times, ITL or batch
semantics, TPOT, and output-level overlap directly from raw_sse_events.
Existing server-WAITING validators are invoked read-only when the audited
runner and its protected base dependency are available.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

AUDITOR_VERSION = "audit_server_waiting_diagnostic-v1"
EXIT_PASS = 0
EXIT_TECHNICAL = 1
EXIT_SCIENTIFIC = 2
FLOAT_ABS_TOL = 1e-9

RAW_CLASS_A = "A_RAW_OUTPUT_OVERLAP"
RAW_CLASS_B = "B_RAW_OUTPUT_BETWEEN_ACTIVE_AND_ALL"
RAW_CLASS_C = "C_RAW_OUTPUT_AFTER_ALL_VICTIMS"


class TechnicalAuditError(RuntimeError):
    """A technical condition prevents a meaningful scientific audit."""


class TokenizerAdapter(Protocol):
    def vocab_size(self) -> int: ...
    def special_token_ids(self) -> set[int]: ...


class HFTokenizerAdapter:
    """Local-only Hugging Face tokenizer adapter; network fallback is forbidden."""

    def __init__(self, model_full_id: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TechnicalAuditError(
                "the 'transformers' package is required for the local tokenizer"
            ) from exc
        try:
            self._tok = AutoTokenizer.from_pretrained(
                model_full_id, local_files_only=True
            )
        except Exception as exc:  # transformers uses several exception classes
            raise TechnicalAuditError(
                f"local tokenizer for {model_full_id!r} is unavailable: {type(exc).__name__}: {exc}"
            ) from exc
        self._vocab_size = len(self._tok)
        specials = set(self._tok.all_special_ids or [])
        added = getattr(self._tok, "added_tokens_encoder", {}) or {}
        specials.update(added.values())
        self._special_ids = specials

    def vocab_size(self) -> int:
        return self._vocab_size

    def special_token_ids(self) -> set[int]:
        return set(self._special_ids)


class FakeTokenizerAdapter:
    """Injectable deterministic tokenizer used only by offline tests."""

    def __init__(self, vocab_size: int = 1000, special_token_ids: Iterable[int] = (0, 1, 2)) -> None:
        self._vocab_size = vocab_size
        self._special_ids = set(special_token_ids)

    def vocab_size(self) -> int:
        return self._vocab_size

    def special_token_ids(self) -> set[int]:
        return set(self._special_ids)


def derive_seed(*parts: str) -> int:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def compute_valid_token_ids(tokenizer: TokenizerAdapter) -> list[int]:
    size = tokenizer.vocab_size()
    specials = tokenizer.special_token_ids()
    if type(size) is not int or size <= 0:
        raise TechnicalAuditError("tokenizer vocab size is not a positive strict integer")
    if not isinstance(specials, set) or any(type(x) is not int for x in specials):
        raise TechnicalAuditError("tokenizer special token IDs are not set[int]")
    valid = [i for i in range(size) if i not in specials]
    if not valid:
        raise TechnicalAuditError("tokenizer has no non-special token IDs")
    return valid


def generate_token_id_prompt(seed: int, valid_ids: Sequence[int], length: int) -> list[int]:
    if type(seed) is not int or type(length) is not int or length < 0:
        raise TechnicalAuditError("invalid strict seed or prompt length")
    if not valid_ids or any(type(x) is not int for x in valid_ids):
        raise TechnicalAuditError("valid token-ID set is empty or malformed")
    rng = random.Random(seed)
    n = len(valid_ids)
    return [valid_ids[rng.randrange(n)] for _ in range(length)]


def prompt_sha256(prompt_token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(prompt_token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def victim_prompt_seed(episode: Any, index: int) -> int:
    return derive_seed(str(_ep_get(episode, "victim_workload_seed")), "victim-prompt", str(index))


def victim_generation_seed(episode: Any, index: int) -> int:
    return derive_seed(str(_ep_get(episode, "victim_workload_seed")), "victim-generation", str(index))


def burst_prompt_seed(episode: Any, index: int) -> int:
    return derive_seed(str(_ep_get(episode, "burst_workload_seed")), "burst-prompt", str(index))


def burst_generation_seed(episode: Any, index: int) -> int:
    return derive_seed(str(_ep_get(episode, "burst_workload_seed")), "burst-generation", str(index))


def _ep_get(episode: Any, name: str) -> Any:
    if isinstance(episode, dict):
        return episode.get(name)
    return getattr(episode, name, None)


def _episode_as_dict(episode: Any) -> dict[str, Any]:
    if isinstance(episode, dict):
        return dict(episode)
    if is_dataclass(episode):
        return asdict(episode)
    return dict(vars(episode))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()




EnvironmentFileHashProvider = Callable[[Any, Path], dict[str, str]]


def _default_environment_file_hash_provider(runner: Any, schedule_dir: Path) -> dict[str, str]:
    """Hash the exact source and schedule files executed by the audited run.

    The persisted manifest is not trusted as the source of these digests.  A
    missing/unreadable source is a technical audit failure; a digest mismatch
    is a scientific/provenance failure reported by the caller.
    """
    try:
        script_dir = Path(runner.__file__).resolve().parent
    except Exception as exc:
        raise TechnicalAuditError(f"cannot resolve audited runner path: {exc}") from exc
    base_dir_value = getattr(runner, "_BASE_DIR", script_dir.parent / "prefill_confirmation")
    base_dir = Path(base_dir_value).resolve()
    targets = {
        "run_server_waiting_confirmation.py": script_dir / "run_server_waiting_confirmation.py",
        "run_server_waiting_confirmation.sh": script_dir / "run_server_waiting_confirmation.sh",
        "run_server_waiting_server.sh": script_dir / "run_server_waiting_server.sh",
        "server_waiting_confirmation_schedule.json": schedule_dir / "server_waiting_confirmation_schedule.json",
        "server_waiting_confirmation_schedule.csv": schedule_dir / "server_waiting_confirmation_schedule.csv",
        "server_waiting_confirmation_schedule_audit.txt": schedule_dir / "server_waiting_confirmation_schedule_audit.txt",
        "_active_cohort.py": script_dir / "_active_cohort.py",
        "run_prefill_confirmation.py": base_dir / "run_prefill_confirmation.py",
    }
    expected_names = getattr(runner, "EXPECTED_ENVIRONMENT_FILE_HASH_NAMES", frozenset(targets))
    if set(targets) != set(expected_names):
        raise TechnicalAuditError(
            "auditor source-hash target set does not match runner expectation: "
            f"targets={sorted(targets)}, expected={sorted(expected_names)}"
        )
    hashes: dict[str, str] = {}
    for name, path in targets.items():
        if not path.is_file():
            raise TechnicalAuditError(f"required provenance source is missing: {name} -> {path}")
        try:
            hashes[name] = _sha256_file(path)
        except OSError as exc:
            raise TechnicalAuditError(f"cannot hash required provenance source {name}: {exc}") from exc
    return dict(sorted(hashes.items()))

def snapshot_tree(root: Path) -> dict[str, dict[str, Any]]:
    """Return exact regular-file identity for a read-only before/after proof."""
    if not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
        raise TechnicalAuditError(f"diagnostic directory is not readable: {root}")
    snapshot: dict[str, dict[str, Any]] = {}
    for p in sorted(root.rglob("*"), key=lambda x: x.relative_to(root).as_posix()):
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            raise TechnicalAuditError(f"diagnostic tree contains a symlink: {rel}")
        if p.is_file():
            try:
                snapshot[rel] = {"size_bytes": p.stat().st_size, "sha256": _sha256_file(p)}
            except OSError as exc:
                raise TechnicalAuditError(f"cannot read diagnostic file {rel}: {exc}") from exc
    return snapshot


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TechnicalAuditError(f"cannot read JSON file {path}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise TechnicalAuditError(f"malformed JSON in {path}: {exc}") from exc


def _strict_int(value: object) -> bool:
    return type(value) is int


def _strict_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _float_equal(a: object, b: object) -> bool:
    return _strict_number(a) and _strict_number(b) and math.isclose(
        float(a), float(b), rel_tol=0.0, abs_tol=FLOAT_ABS_TOL
    )


def _list_float_equal(actual: object, expected: Sequence[float]) -> bool:
    return (
        isinstance(actual, list)
        and len(actual) == len(expected)
        and all(_float_equal(a, b) for a, b in zip(actual, expected))
    )


@dataclass
class RawSSEReconstruction:
    prompt_echo: list[int] | None
    output_token_ids: list[int]
    positive_event_times_ns: list[int]
    token_batch_sizes: list[int]
    token_batch_interarrival_ms: list[float]
    itl_available: bool
    itl_ms: list[float]
    first_positive_token_receive_ns: int | None
    last_positive_token_receive_ns: int | None
    finish_reason: str | None
    usage: dict | None
    done_received: bool
    errors: list[str]


def _strict_json_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(a, e) for a, e in zip(actual, expected)
        )
    return actual == expected


def reconstruct_raw_sse_events(raw_sse_events: object) -> RawSSEReconstruction:
    """Pure fail-closed reconstruction using the base executor's event semantics.

    For every ``parse_status='ok'`` event, ``raw_data`` is parsed again and
    bound strictly to the normalized event fields.  Thus neither raw JSON nor
    its normalized token/prompt/timing aliases can be jointly rewritten
    without detection.
    """
    errors: list[str] = []
    output: list[int] = []
    positive_times: list[int] = []
    batch_sizes: list[int] = []
    prompt_echo: list[int] | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    done_count = 0
    previous_receive_ns: int | None = None

    if not isinstance(raw_sse_events, list):
        return RawSSEReconstruction(
            None, [], [], [], [], False, [], None, None, None, None, False,
            ["raw_sse_events is not a list"],
        )

    allowed_statuses = {"ok", "keepalive", "done"}
    for pos, event in enumerate(raw_sse_events):
        prefix = f"raw_sse_events[{pos}]"
        if not isinstance(event, dict):
            errors.append(f"{prefix} is not a dict")
            continue
        if event.get("event_index") != pos or type(event.get("event_index")) is not int:
            errors.append(f"{prefix}.event_index is not the strict sequential index {pos}")
        status = event.get("parse_status")
        if type(status) is not str or status not in allowed_statuses:
            errors.append(f"{prefix}.parse_status is invalid: {status!r}")

        raw_data = event.get("raw_data")
        if type(raw_data) is not str:
            errors.append(f"{prefix}.raw_data is not a strict string")
        elapsed_ms = event.get("elapsed_since_request_start_ms")
        if not _strict_number(elapsed_ms) or float(elapsed_ms) < 0.0:
            errors.append(f"{prefix}.elapsed_since_request_start_ms is not a finite non-negative number")

        receive_ns = event.get("receive_perf_counter_ns")
        if type(receive_ns) is not int:
            errors.append(f"{prefix}.receive_perf_counter_ns is not a strict int")
            valid_receive = False
        else:
            valid_receive = True
            if previous_receive_ns is not None and receive_ns < previous_receive_ns:
                errors.append(f"{prefix}.receive_perf_counter_ns decreases")
            previous_receive_ns = receive_ns

        normalized_token_ids = event.get("token_ids")
        if not isinstance(normalized_token_ids, list) or any(type(x) is not int for x in normalized_token_ids):
            errors.append(f"{prefix}.token_ids is not list[int]")
            normalized_token_ids = []
        normalized_prompt_ids = event.get("prompt_token_ids")
        if normalized_prompt_ids is not None and (
            not isinstance(normalized_prompt_ids, list)
            or any(type(x) is not int for x in normalized_prompt_ids)
        ):
            errors.append(f"{prefix}.prompt_token_ids is not null or list[int]")
            normalized_prompt_ids = None
        normalized_finish = event.get("finish_reason")
        if normalized_finish is not None and type(normalized_finish) is not str:
            errors.append(f"{prefix}.finish_reason is not null or str")
            normalized_finish = None
        normalized_usage = event.get("usage")
        if normalized_usage is not None:
            if not isinstance(normalized_usage, dict):
                errors.append(f"{prefix}.usage is not null or dict")
                normalized_usage = None
            else:
                for counter_key in ("prompt_tokens", "completion_tokens"):
                    if counter_key in normalized_usage and type(normalized_usage[counter_key]) is not int:
                        errors.append(f"{prefix}.usage.{counter_key} is not a strict int")

        if status == "keepalive":
            if raw_data != "":
                errors.append(f"{prefix} keepalive raw_data is not the empty string")
            if (normalized_token_ids or normalized_prompt_ids is not None
                    or normalized_finish is not None or normalized_usage is not None):
                errors.append(f"{prefix} keepalive carries semantic payload")
            continue

        if status == "done":
            done_count += 1
            if raw_data != "[DONE]":
                errors.append(f"{prefix} DONE event raw_data is not '[DONE]'")
            if pos != len(raw_sse_events) - 1:
                errors.append(f"{prefix} DONE event is not the final event")
            if (normalized_token_ids or normalized_prompt_ids is not None
                    or normalized_finish is not None or normalized_usage is not None):
                errors.append(f"{prefix} DONE event carries semantic payload")
            continue

        if status != "ok":
            continue

        parsed_token_ids: list[int] = []
        parsed_prompt_ids: list[int] | None = None
        parsed_text: object = None
        parsed_finish: str | None = None
        parsed_usage: dict | None = None
        parsed_valid = True
        if type(raw_data) is str:
            try:
                obj = json.loads(raw_data)
            except json.JSONDecodeError as exc:
                errors.append(f"{prefix}.raw_data JSON parse error: {exc}")
                parsed_valid = False
                obj = {}
            if not isinstance(obj, dict):
                errors.append(f"{prefix}.raw_data JSON payload is not an object")
                parsed_valid = False
                obj = {}

            choices_raw = obj.get("choices")
            if choices_raw is not None and not isinstance(choices_raw, list):
                errors.append(f"{prefix}.raw_data choices is not a list")
                parsed_valid = False
                choices_raw = []
            choices = choices_raw or []
            choice0: dict = {}
            if choices:
                if not isinstance(choices[0], dict):
                    errors.append(f"{prefix}.raw_data choices[0] is not an object")
                    parsed_valid = False
                else:
                    choice0 = choices[0]

            token_raw = choice0.get("token_ids")
            if token_raw is None:
                parsed_token_ids = []
            elif isinstance(token_raw, list) and all(type(x) is int for x in token_raw):
                parsed_token_ids = list(token_raw)
            else:
                errors.append(f"{prefix}.raw_data token_ids is not list[int]")
                parsed_valid = False

            def prompt_value(value: object, where: str) -> list[int] | None:
                nonlocal parsed_valid
                if value is None:
                    return None
                if isinstance(value, list) and all(type(x) is int for x in value):
                    return list(value)
                errors.append(f"{prefix}.raw_data {where}.prompt_token_ids is not list[int]")
                parsed_valid = False
                return None

            top_prompt = prompt_value(obj.get("prompt_token_ids"), "top-level")
            choice_prompt = prompt_value(choice0.get("prompt_token_ids"), "choices[0]")
            if top_prompt is not None and choice_prompt is not None and top_prompt != choice_prompt:
                errors.append(f"{prefix}.raw_data top-level and choices[0] prompt echoes contradict")
                parsed_valid = False
            parsed_prompt_ids = top_prompt if top_prompt is not None else choice_prompt
            parsed_text = choice0.get("text")
            finish_raw = choice0.get("finish_reason")
            if finish_raw is None or type(finish_raw) is str:
                parsed_finish = finish_raw
            else:
                errors.append(f"{prefix}.raw_data finish_reason is not null or str")
                parsed_valid = False

            usage_raw = obj.get("usage")
            if usage_raw is not None:
                if not isinstance(usage_raw, dict):
                    errors.append(f"{prefix}.raw_data usage is not a dict")
                    parsed_valid = False
                else:
                    for counter_key in ("prompt_tokens", "completion_tokens"):
                        if counter_key in usage_raw and type(usage_raw[counter_key]) is not int:
                            errors.append(f"{prefix}.raw_data usage.{counter_key} is not a strict int")
                            parsed_valid = False
                    parsed_usage = dict(usage_raw)

        if parsed_valid:
            if not _strict_json_equal(event.get("token_ids"), parsed_token_ids):
                errors.append(f"{prefix}.token_ids does not match independently parsed raw_data")
            if not _strict_json_equal(event.get("prompt_token_ids"), parsed_prompt_ids):
                errors.append(f"{prefix}.prompt_token_ids does not match independently parsed raw_data")
            if not _strict_json_equal(event.get("text_delta"), parsed_text):
                errors.append(f"{prefix}.text_delta does not match independently parsed raw_data")
            if not _strict_json_equal(event.get("finish_reason"), parsed_finish):
                errors.append(f"{prefix}.finish_reason does not match independently parsed raw_data")
            if not _strict_json_equal(event.get("usage"), parsed_usage):
                errors.append(f"{prefix}.usage does not match independently parsed raw_data")

        token_ids = parsed_token_ids if parsed_valid else normalized_token_ids
        prompt_ids = parsed_prompt_ids if parsed_valid else normalized_prompt_ids
        event_finish = parsed_finish if parsed_valid else normalized_finish
        event_usage = parsed_usage if parsed_valid else normalized_usage

        if prompt_ids is not None:
            if prompt_echo is None:
                prompt_echo = list(prompt_ids)
            elif prompt_echo != prompt_ids:
                errors.append(f"{prefix}.prompt_token_ids contradicts the earlier prompt echo")
        if event_finish:
            if finish_reason is None:
                finish_reason = event_finish
            elif finish_reason != event_finish:
                errors.append(f"{prefix}.finish_reason contradicts the earlier finish reason")
        if event_usage is not None:
            if usage is None:
                usage = dict(event_usage)
            elif usage != event_usage:
                errors.append(f"{prefix}.usage contradicts the earlier usage object")
        output.extend(token_ids)
        if token_ids and valid_receive:
            positive_times.append(receive_ns)
            batch_sizes.append(len(token_ids))

    if done_count != 1:
        errors.append(f"raw trace contains {done_count} DONE events; expected exactly one")

    batch_interarrivals = [
        (b - a) / 1e6 for a, b in zip(positive_times, positive_times[1:])
    ]
    itl_available = bool(positive_times) and all(size == 1 for size in batch_sizes)
    itl_ms = list(batch_interarrivals) if itl_available else []
    return RawSSEReconstruction(
        prompt_echo=prompt_echo,
        output_token_ids=output,
        positive_event_times_ns=positive_times,
        token_batch_sizes=batch_sizes,
        token_batch_interarrival_ms=batch_interarrivals,
        itl_available=itl_available,
        itl_ms=itl_ms,
        first_positive_token_receive_ns=positive_times[0] if positive_times else None,
        last_positive_token_receive_ns=positive_times[-1] if positive_times else None,
        finish_reason=finish_reason,
        usage=usage,
        done_received=done_count == 1,
        errors=errors,
    )


def _expected_request(episode: Any, role: str, index: int, valid_ids: Sequence[int]) -> dict[str, Any]:
    if role == "victim":
        prompt_seed = victim_prompt_seed(episode, index)
        generation_seed = victim_generation_seed(episode, index)
        prompt_len = _ep_get(episode, "victim_input_len")
        completion_len = _ep_get(episode, "victim_output_len")
    elif role == "burst":
        prompt_seed = burst_prompt_seed(episode, index)
        generation_seed = burst_generation_seed(episode, index)
        prompt_len = _ep_get(episode, "burst_input_len")
        completion_len = _ep_get(episode, "burst_output_len")
    else:
        raise TechnicalAuditError(f"unsupported role {role!r}")
    if type(prompt_len) is not int or type(completion_len) is not int:
        raise TechnicalAuditError(f"episode has malformed {role} token lengths")
    prompt = generate_token_id_prompt(prompt_seed, valid_ids, prompt_len)
    episode_id = _ep_get(episode, "episode_id")
    return {
        "request_id": f"{episode_id}:{role}:{index}",
        "prompt_seed": prompt_seed,
        "generation_seed": generation_seed,
        "prompt_token_ids": prompt,
        "prompt_sha256": prompt_sha256(prompt),
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_len,
    }


def audit_request_record(
    record: object, *, episode: Any, role: str, index: int, valid_ids: Sequence[int]
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return {"role": role, "request_index": index, "status": "FAIL", "errors": ["record is not a dict"]}

    expected = _expected_request(episode, role, index, valid_ids)
    for key, value in (
        ("request_id", expected["request_id"]),
        ("role", role),
        ("request_index", index),
        ("prompt_seed", expected["prompt_seed"]),
        ("generation_seed", expected["generation_seed"]),
        ("expected_prompt_tokens", expected["prompt_tokens"]),
        ("expected_completion_tokens", expected["completion_tokens"]),
    ):
        actual = record.get(key)
        if type(actual) is not type(value) or actual != value:
            errors.append(f"{key}={actual!r} != strict expected {value!r}")

    expected_prompt = expected["prompt_token_ids"]
    if record.get("prompt_token_ids_sent") != expected_prompt:
        errors.append("prompt_token_ids_sent does not match deterministic reconstruction")
    if record.get("prompt_token_ids_returned") != expected_prompt:
        errors.append("prompt_token_ids_returned does not match deterministic reconstruction")
    if record.get("prompt_sha256") != expected["prompt_sha256"]:
        errors.append("prompt_sha256 does not match deterministic reconstruction")

    raw = reconstruct_raw_sse_events(record.get("raw_sse_events"))
    errors.extend(raw.errors)
    if raw.prompt_echo != expected_prompt:
        errors.append("raw SSE prompt echo does not match deterministic reconstruction")
    if record.get("prompt_token_ids_returned") != raw.prompt_echo:
        errors.append("prompt_token_ids_returned does not match raw SSE prompt echo")

    output_ids = record.get("output_token_ids")
    if not isinstance(output_ids, list) or any(type(x) is not int for x in output_ids):
        errors.append("output_token_ids is not list[int]")
    elif output_ids != raw.output_token_ids:
        errors.append("output_token_ids does not equal concatenated positive SSE token IDs")
    if len(raw.output_token_ids) != expected["completion_tokens"]:
        errors.append(
            f"raw output token count {len(raw.output_token_ids)} != expected {expected['completion_tokens']}"
        )

    first = raw.first_positive_token_receive_ns
    last = raw.last_positive_token_receive_ns
    if first is None or last is None:
        errors.append("raw trace has no positive output-token event")
    required_first_aliases = ["first_token_receive_ns"]
    if role == "victim":
        required_first_aliases.append("first_token_perf_ns")
    else:
        required_first_aliases.append("burst_first_token_perf_ns")
    for key in required_first_aliases:
        value = record.get(key)
        if type(value) is not int:
            errors.append(f"{key} is missing or not a strict integer")
        elif value != first:
            errors.append(f"{key} does not match raw first positive token time")

    # Additional aliases are validated whenever the runner records them, but
    # only the frozen role-specific fields above are mandatory.
    optional_first_aliases = ["first_token_perf_ns"] if role == "burst" else []
    for key in optional_first_aliases:
        if key in record:
            value = record.get(key)
            if type(value) is not int or value != first:
                errors.append(f"{key} does not strictly match raw first positive token time")

    value = record.get("last_token_receive_ns")
    if type(value) is not int:
        errors.append("last_token_receive_ns is missing or not a strict integer")
    elif value != last:
        errors.append("last_token_receive_ns does not match raw last positive token time")
    for key in ("last_token_perf_ns", "burst_last_token_perf_ns"):
        if key in record:
            value = record.get(key)
            if type(value) is not int or value != last:
                errors.append(f"{key} does not strictly match raw last positive token time")
    if "stream_end_ns" in record and "stream_end_perf_ns" in record             and record.get("stream_end_ns") != record.get("stream_end_perf_ns"):
        errors.append("stream_end_ns and stream_end_perf_ns aliases disagree")
    if role == "burst" and "burst_end_perf_ns" in record and "stream_end_perf_ns" in record             and record.get("burst_end_perf_ns") != record.get("stream_end_perf_ns"):
        errors.append("burst_end_perf_ns does not match stream_end_perf_ns")

    if record.get("finish_reason") != raw.finish_reason:
        errors.append("stored finish_reason does not match raw SSE")
    if raw.finish_reason != "length":
        errors.append(f"raw finish_reason {raw.finish_reason!r} != 'length'")
    if record.get("usage") != raw.usage:
        errors.append("stored usage does not match raw SSE")
    usage = raw.usage
    if not isinstance(usage, dict):
        errors.append("raw usage is missing or not a dict")
    else:
        if type(usage.get("prompt_tokens")) is not int or usage.get("prompt_tokens") != expected["prompt_tokens"]:
            errors.append("usage.prompt_tokens does not strictly match expected prompt length")
        if type(usage.get("completion_tokens")) is not int or usage.get("completion_tokens") != expected["completion_tokens"]:
            errors.append("usage.completion_tokens does not strictly match expected completion length")
    if record.get("done_received") is not True or not raw.done_received:
        errors.append("DONE binding failed")

    expected_batch_sizes = raw.token_batch_sizes
    expected_batch_interarrival = raw.token_batch_interarrival_ms
    if record.get("itl_available") is not raw.itl_available:
        errors.append("itl_available does not match raw token-batch semantics")
    if raw.itl_available:
        if not _list_float_equal(record.get("itl_ms"), raw.itl_ms):
            errors.append("itl_ms does not match elementwise raw single-token interarrivals")
        if record.get("token_batch_sizes") is not None:
            errors.append("token_batch_sizes must be null for singleton-token events")
        if record.get("token_batch_interarrival_ms") is not None:
            errors.append("token_batch_interarrival_ms must be null for singleton-token events")
        if record.get("chunk_interarrival_ms") is not None:
            errors.append("chunk_interarrival_ms must be null for singleton-token events")
    else:
        if record.get("itl_ms") is not None:
            errors.append("itl_ms must be null when a positive SSE event contains a token batch")
        if record.get("token_batch_sizes") != expected_batch_sizes:
            errors.append("token_batch_sizes does not match raw batch reconstruction")
        if not _list_float_equal(record.get("token_batch_interarrival_ms"), expected_batch_interarrival):
            errors.append("token_batch_interarrival_ms does not match raw batch reconstruction")
        if not _list_float_equal(record.get("chunk_interarrival_ms"), expected_batch_interarrival):
            errors.append("chunk_interarrival_ms does not match raw batch reconstruction")

    reconstructed_tpot: float | None = None
    if first is not None and last is not None and expected["completion_tokens"] > 1:
        reconstructed_tpot = (last - first) / 1e6 / (expected["completion_tokens"] - 1)
        if not _float_equal(record.get("client_observed_tpot_ms"), reconstructed_tpot):
            errors.append("client_observed_tpot_ms does not match raw positive-token timestamps")

    raw_trace_valid = (
        not raw.errors
        and raw.prompt_echo == expected_prompt
        and isinstance(output_ids, list)
        and output_ids == raw.output_token_ids
        and len(raw.output_token_ids) == expected["completion_tokens"]
        and raw.finish_reason == "length"
        and isinstance(raw.usage, dict)
        and raw.done_received
    )

    return {
        "request_id": expected["request_id"],
        "role": role,
        "request_index": index,
        "status": "PASS" if not errors else "FAIL",
        "raw_trace_valid": raw_trace_valid,
        "errors": errors,
        "prompt_reconstruction": {
            "prompt_seed": expected["prompt_seed"],
            "generation_seed": expected["generation_seed"],
            "prompt_sha256": expected["prompt_sha256"],
            "prompt_token_count": len(expected_prompt),
        },
        "raw_reconstruction": {
            "output_token_count": len(raw.output_token_ids),
            "positive_event_count": len(raw.positive_event_times_ns),
            "token_batch_sizes": raw.token_batch_sizes,
            "first_positive_token_receive_ns": first,
            "last_positive_token_receive_ns": last,
            "itl_available": raw.itl_available,
            "itl_ms": raw.itl_ms,
            "token_batch_interarrival_ms": raw.token_batch_interarrival_ms,
            "client_observed_tpot_ms_reconstructed": reconstructed_tpot,
            "finish_reason": raw.finish_reason,
            "usage": raw.usage,
            "done_received": raw.done_received,
        },
    }


def _classification_bucket(value: object) -> str | None:
    if type(value) is not str:
        return None
    if value.startswith("A_"):
        return "A"
    if value.startswith("B_"):
        return "B"
    if value.startswith("C_"):
        return "C"
    return None


def raw_overlap_classification(prefill_burst: dict, request_results: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    trigger = prefill_burst.get("trigger")
    if not isinstance(trigger, dict):
        return {"classification": None, "errors": ["prefill_burst.trigger is missing or not a dict"]}
    active = trigger.get("active_cohort_request_indices")
    if (
        not isinstance(active, list)
        or len(active) != 8
        or any(type(x) is not int for x in active)
        or len(set(active)) != 8
    ):
        return {"classification": None, "errors": ["active cohort is not eight unique strict integers"]}

    raw_valid = [r for r in request_results if r.get("raw_trace_valid") is True]
    by_key = {(r.get("role"), r.get("request_index")): r for r in raw_valid}
    victim_last: dict[int, int] = {}
    burst_first: list[int] = []
    for index in range(20):
        item = by_key.get(("victim", index))
        ns = ((item or {}).get("raw_reconstruction") or {}).get("last_positive_token_receive_ns")
        if type(ns) is not int:
            errors.append(f"victim {index} lacks a validated raw last-token time")
        else:
            victim_last[index] = ns
    for index in range(4):
        item = by_key.get(("burst", index))
        ns = ((item or {}).get("raw_reconstruction") or {}).get("first_positive_token_receive_ns")
        if type(ns) is not int:
            errors.append(f"burst {index} lacks a validated raw first-token time")
        else:
            burst_first.append(ns)
    missing_active = sorted(set(active) - set(victim_last))
    if missing_active:
        errors.append(f"validated raw victim times missing for active indices {missing_active}")
    if errors:
        return {"classification": None, "errors": errors}

    first_burst = min(burst_first)
    last_active = max(victim_last[i] for i in active)
    last_all = max(victim_last.values())
    if first_burst < last_active:
        classification = RAW_CLASS_A
    elif first_burst > last_all:
        classification = RAW_CLASS_C
    else:
        classification = RAW_CLASS_B
    return {
        "classification": classification,
        "errors": [],
        "first_burst_output_ns": first_burst,
        "last_active_victim_output_ns": last_active,
        "last_all_victim_output_ns": last_all,
    }


def _load_runner_module() -> Any:
    try:
        return importlib.import_module("run_server_waiting_confirmation")
    except Exception as exc:
        raise TechnicalAuditError(
            "cannot import run_server_waiting_confirmation and its protected base dependency: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _prepare_output_dir(output_dir: Path, diagnostic_dir: Path) -> None:
    diag_resolved = diagnostic_dir.resolve()
    out_resolved = output_dir.resolve()
    if out_resolved == diag_resolved or diag_resolved in out_resolved.parents:
        raise TechnicalAuditError("--audit-output-dir must be outside the diagnostic tree")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise TechnicalAuditError("--audit-output-dir exists and is not a directory")
        if any(output_dir.iterdir()):
            raise TechnicalAuditError("--audit-output-dir must be fresh and empty")
    else:
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise TechnicalAuditError(f"cannot create --audit-output-dir: {exc}") from exc


def _find_episode_file(diagnostic_dir: Path, episode_id: str) -> Path:
    return diagnostic_dir / "episodes" / f"{episode_id}.json"



def _validate_manifest_independently(
    manifest: object, *, runner: Any, bundle: Any, diagnostic_dir: Path,
    actual_file_hashes: dict[str, str],
) -> list[str]:
    """Independently bind the persisted pre-start provenance manifest."""
    if not isinstance(manifest, dict):
        return ["diagnostic_run_manifest.json is not a JSON object"]
    errors: list[str] = []

    expected_values = {
        "manifest_schema_version": getattr(runner, "MANIFEST_SCHEMA_VERSION", 1),
        "runner_version": getattr(runner, "RUNNER_VERSION", None),
        "result_schema_version": getattr(runner, "RESULT_SCHEMA_VERSION", None),
        "schedule_fingerprint": getattr(bundle, "fingerprint", None),
        "design_version": getattr(bundle, "json_obj", {}).get("design_version"),
        "schedule_seed": getattr(bundle, "json_obj", {}).get("seed"),
        "run_mode": getattr(runner, "RUN_MODE_DIAGNOSTIC_PAIR", "diagnostic_pair"),
    }
    for key, expected in expected_values.items():
        actual = manifest.get(key)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"diagnostic_run_manifest.{key}={actual!r} != strict expected {expected!r}")

    for key in ("created_utc", "output_dir", "host", "python_executable",
                "python_version", "platform", "hostname", "kernel",
                "git_commit", "vllm_version", "torch_version",
                "transformers_version", "httpx_version"):
        value = manifest.get(key)
        if type(value) is not str or not value.strip():
            errors.append(f"diagnostic_run_manifest.{key} is missing or not a non-empty strict string")
    if type(manifest.get("port")) is not int or not (1 <= manifest.get("port", 0) <= 65535):
        errors.append("diagnostic_run_manifest.port is not a strict TCP port integer")
    if manifest.get("git_dirty") is not False:
        errors.append("diagnostic_run_manifest.git_dirty must be exactly false")
    cuda_visible = manifest.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and type(cuda_visible) is not str:
        errors.append("diagnostic_run_manifest.CUDA_VISIBLE_DEVICES is not null or a strict string")
    if not isinstance(manifest.get("gpu_list"), list):
        errors.append("diagnostic_run_manifest.gpu_list is not a list")
    if not isinstance(manifest.get("resolved_gpu"), dict):
        errors.append("diagnostic_run_manifest.resolved_gpu is not a dict")
    file_hashes = manifest.get("file_hashes")
    if not isinstance(file_hashes, dict) or not file_hashes:
        errors.append("diagnostic_run_manifest.file_hashes is missing or empty")
    else:
        for name, digest in file_hashes.items():
            if type(name) is not str or type(digest) is not str or len(digest) != 64 \
                    or any(c not in "0123456789abcdef" for c in digest):
                errors.append(f"diagnostic_run_manifest.file_hashes[{name!r}] is not lowercase SHA-256")
        if set(file_hashes) != set(actual_file_hashes):
            errors.append(
                "diagnostic_run_manifest.file_hashes names do not match the actual audited source/schedule set: "
                f"stored={sorted(file_hashes)}, actual={sorted(actual_file_hashes)}"
            )
        for name in sorted(set(file_hashes) & set(actual_file_hashes)):
            if file_hashes.get(name) != actual_file_hashes.get(name):
                errors.append(
                    f"diagnostic_run_manifest.file_hashes[{name!r}]={file_hashes.get(name)!r} "
                    f"!= actual on-disk SHA-256 {actual_file_hashes.get(name)!r}"
                )

    output_dir = manifest.get("output_dir")
    if type(output_dir) is str:
        try:
            if Path(output_dir).resolve() != diagnostic_dir.resolve():
                errors.append("diagnostic_run_manifest.output_dir does not resolve to --diagnostic-dir")
        except OSError as exc:
            errors.append(f"diagnostic_run_manifest.output_dir cannot be resolved: {exc}")

    env = {
        "python_executable": manifest.get("python_executable"),
        "python_version": manifest.get("python_version"),
        "platform": manifest.get("platform"),
        "hostname": manifest.get("hostname"),
        "kernel": manifest.get("kernel"),
        "git_commit": manifest.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
        "cuda_visible_devices": manifest.get("CUDA_VISIBLE_DEVICES"),
        "vllm_version": manifest.get("vllm_version"),
        "torch_version": manifest.get("torch_version"),
        "transformers_version": manifest.get("transformers_version"),
        "httpx_version": manifest.get("httpx_version"),
        "gpu_list": manifest.get("gpu_list"),
        "resolved_gpu": manifest.get("resolved_gpu"),
        "file_hashes": manifest.get("file_hashes"),
    }
    stored_fp = manifest.get("environment_fingerprint")
    if type(stored_fp) is not str or not stored_fp.startswith("sha256:") or len(stored_fp) != 71:
        errors.append("diagnostic_run_manifest.environment_fingerprint has invalid format")
    compute_fp = getattr(runner, "compute_environment_fingerprint", None)
    if callable(compute_fp):
        try:
            recomputed_fp = compute_fp(env)
        except Exception as exc:
            errors.append(f"environment fingerprint reconstruction failed: {type(exc).__name__}: {exc}")
        else:
            if stored_fp != recomputed_fp:
                errors.append(
                    f"diagnostic_run_manifest.environment_fingerprint={stored_fp!r} "
                    f"!= independently recomputed {recomputed_fp!r}"
                )
    return errors

def _technical_result(message: str) -> dict[str, Any]:
    return {
        "auditor_version": AUDITOR_VERSION,
        "overall_audit_status": "TECHNICAL_ERROR",
        "scientifically_evaluable": False,
        "technical_errors": [message],
        "errors": [],
    }


def _render_text(result: dict[str, Any]) -> str:
    lines = [
        "Server-WAITING Raw-Trace Auditor",
        "=" * 60,
        f"auditor_version: {result.get('auditor_version')}",
        f"runner_version: {result.get('runner_version')}",
        f"overall_audit_status: {result.get('overall_audit_status')}",
        f"scientifically_evaluable: {result.get('scientifically_evaluable')}",
        f"stored_semantic_classification: {result.get('stored_semantic_classification')}",
        f"raw_output_overlap_classification: {result.get('raw_output_overlap_classification')}",
        f"diagnostic_tree_read_only_verified: {result.get('diagnostic_tree_read_only_verified')}",
        "",
    ]
    for section in ("technical_errors", "errors"):
        values = result.get(section) or []
        lines.append(f"{section} ({len(values)}):")
        for value in values:
            lines.append(f"  - {value}")
        lines.append("")
    per_requests = result.get("per_request_status") or []
    passed = sum(1 for r in per_requests if r.get("status") == "PASS")
    lines.append(f"per_request_status: {passed}/{len(per_requests)} PASS")
    return "\n".join(lines) + "\n"


def write_auditor_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    (output_dir / "auditor_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "auditor_result.txt").write_text(_render_text(result), encoding="utf-8")


def audit_diagnostic(
    diagnostic_dir: Path,
    schedule_dir: Path,
    audit_output_dir: Path,
    *,
    runner_module: Any | None = None,
    tokenizer_factory: Callable[[str], TokenizerAdapter] = HFTokenizerAdapter,
    file_hash_provider: EnvironmentFileHashProvider = _default_environment_file_hash_provider,
) -> tuple[int, dict[str, Any]]:
    """Perform the complete audit and return (exit_code, result)."""
    try:
        _prepare_output_dir(audit_output_dir, diagnostic_dir)
        before = snapshot_tree(diagnostic_dir)
        runner = runner_module if runner_module is not None else _load_runner_module()

        bundle, bundle_errors = runner.load_and_validate_bundle(schedule_dir, runner.MODEL_KEY)
        if bundle is None:
            raise TechnicalAuditError(f"validated schedule bundle could not be loaded: {bundle_errors}")
        selected, selection_errors = runner.validate_diagnostic_block_selection(bundle)
        if selected is None:
            raise TechnicalAuditError(f"diagnostic block selection failed: {selection_errors}")
        expected_no = next(ep for ep in selected if _ep_get(ep, "condition") == "no_burst")
        expected_pb = next(ep for ep in selected if _ep_get(ep, "condition") == "prefill_burst")

        actual_file_hashes = file_hash_provider(runner, schedule_dir)
        if not isinstance(actual_file_hashes, dict) or not actual_file_hashes:
            raise TechnicalAuditError("file_hash_provider returned no source/schedule hashes")
        if any(type(k) is not str or type(v) is not str for k, v in actual_file_hashes.items()):
            raise TechnicalAuditError("file_hash_provider returned malformed hash entries")

        model_full_id = _ep_get(expected_pb, "model_id")
        tokenizer = tokenizer_factory(model_full_id)
        valid_ids = compute_valid_token_ids(tokenizer)

        required = [
            diagnostic_dir / "diagnostic_run_manifest.json",
            diagnostic_dir / "diagnostic_pair_summary.json",
            diagnostic_dir / "diagnostic_pair_summary.txt",
            diagnostic_dir / "integrity_manifest.json",
            diagnostic_dir / "stabilization" / f"{runner.DIAGNOSTIC_BLOCK_ID}.json",
            _find_episode_file(diagnostic_dir, _ep_get(expected_no, "episode_id")),
            _find_episode_file(diagnostic_dir, _ep_get(expected_pb, "episode_id")),
        ]
        missing = [p.relative_to(diagnostic_dir).as_posix() for p in required if not p.is_file()]
        if missing:
            result = {
                "auditor_version": AUDITOR_VERSION,
                "runner_version": getattr(runner, "RUNNER_VERSION", None),
                "overall_audit_status": "FAIL",
                "scientifically_evaluable": False,
                "errors": [f"required diagnostic artifact is missing: {x}" for x in missing],
                "technical_errors": [],
            }
            after = snapshot_tree(diagnostic_dir)
            result["diagnostic_tree_read_only_verified"] = before == after
            write_auditor_outputs(audit_output_dir, result)
            return EXIT_SCIENTIFIC, result

        manifest = _load_json(diagnostic_dir / "diagnostic_run_manifest.json")
        summary = _load_json(diagnostic_dir / "diagnostic_pair_summary.json")
        integrity = _load_json(diagnostic_dir / "integrity_manifest.json")
        stabilization = _load_json(diagnostic_dir / "stabilization" / f"{runner.DIAGNOSTIC_BLOCK_ID}.json")
        no_result = _load_json(_find_episode_file(diagnostic_dir, _ep_get(expected_no, "episode_id")))
        pb_result = _load_json(_find_episode_file(diagnostic_dir, _ep_get(expected_pb, "episode_id")))

        errors: list[str] = []
        provenance_errors: list[str] = []
        schedule_fp = getattr(bundle, "fingerprint")
        environment_fp = manifest.get("environment_fingerprint") if isinstance(manifest, dict) else None
        if type(environment_fp) is not str or not environment_fp:
            errors.append("diagnostic_run_manifest.environment_fingerprint is missing")
        host = manifest.get("host") if isinstance(manifest, dict) else None
        port = manifest.get("port") if isinstance(manifest, dict) else None
        if type(host) is not str or type(port) is not int:
            errors.append("diagnostic_run_manifest host/port are malformed")
        expected_server_command = runner.build_server_command(
            Path(runner.__file__).resolve().parent / "run_server_waiting_server.sh",
            runner.DIAGNOSTIC_MODEL_KEY,
            runner.DIAGNOSTIC_OFFLOAD_GB,
            runner.DIAGNOSTIC_SERVER_MAX_NUM_SEQS,
            host if type(host) is str else "127.0.0.1",
            port if type(port) is int else 8000,
        )

        # Existing read-only scientific gates.
        classification = runner.classify_diagnostic_pair(
            no_burst_result=no_result,
            prefill_burst_result=pb_result,
            expected_no_burst_episode=expected_no,
            expected_prefill_burst_episode=expected_pb,
            expected_schedule_fingerprint=schedule_fp,
            expected_environment_fingerprint=environment_fp,
            expected_server_command=expected_server_command,
        )
        summary_errors: list[str] = []
        if not isinstance(summary, dict):
            summary_errors.append("diagnostic_pair_summary.json is not a JSON object")
        else:
            summary_expected = {
                "runner_version": getattr(runner, "RUNNER_VERSION", None),
                "result_schema_version": getattr(runner, "RESULT_SCHEMA_VERSION", None),
                "run_mode": getattr(runner, "RUN_MODE_DIAGNOSTIC_PAIR", "diagnostic_pair"),
                "schedule_fingerprint": schedule_fp,
                "environment_fingerprint": environment_fp,
                "diagnostic_block_id": getattr(runner, "DIAGNOSTIC_BLOCK_ID", None),
                "integrity_manifest_filename": getattr(runner, "INTEGRITY_MANIFEST_FILENAME", "integrity_manifest.json"),
                "integrity_finalization_required": True,
            }
            for key, expected_value in summary_expected.items():
                actual_value = summary.get(key)
                if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                    summary_errors.append(
                        f"diagnostic_pair_summary.{key}={actual_value!r} != strict expected {expected_value!r}"
                    )
            if not _strict_json_equal(summary.get("classification"), classification):
                summary_errors.append(
                    "diagnostic_pair_summary.classification does not exactly match read-only runner reconstruction"
                )
            render_summary = getattr(runner, "render_diagnostic_text_summary", None)
            if callable(render_summary):
                try:
                    expected_text = render_summary(summary)
                    actual_text = (diagnostic_dir / "diagnostic_pair_summary.txt").read_text(encoding="utf-8")
                except Exception as exc:
                    summary_errors.append(f"diagnostic_pair_summary.txt validation failed: {type(exc).__name__}: {exc}")
                else:
                    if actual_text != expected_text:
                        summary_errors.append(
                            "diagnostic_pair_summary.txt does not exactly match rendering of diagnostic_pair_summary.json"
                        )
        marker_name = getattr(runner, "RUN_MODE_MARKER_FILENAME", ".server_waiting_confirmation_run_mode")
        try:
            marker_value = (diagnostic_dir / marker_name).read_text(encoding="utf-8")
        except OSError as exc:
            summary_errors.append(f"run-mode marker is unreadable: {exc}")
        else:
            expected_mode = getattr(runner, "RUN_MODE_DIAGNOSTIC_PAIR", "diagnostic_pair")
            if marker_value != expected_mode:
                summary_errors.append(f"run-mode marker {marker_value!r} != exact expected {expected_mode!r}")
        errors.extend(summary_errors)
        provenance_errors.extend(summary_errors)

        stored_classification = ((summary or {}).get("classification") or {}).get("classification")
        recomputed_stored = classification.get("classification") if isinstance(classification, dict) else None
        if stored_classification != recomputed_stored:
            errors.append(
                f"stored summary classification {stored_classification!r} != read-only runner reconstruction {recomputed_stored!r}"
            )
        if recomputed_stored == getattr(runner, "DIAGNOSTIC_CLASSIFICATION_D", "D_AMBIGUOUS_OR_INVALID"):
            errors.extend([f"runner semantic gate: {x}" for x in classification.get("reasons", [])])

        # Independently reconstruct the pre-start manifest, then reuse the
        # runner's pure stabilization/reference validators read-only.
        provenance_errors.extend(_validate_manifest_independently(
            manifest, runner=runner, bundle=bundle, diagnostic_dir=diagnostic_dir,
            actual_file_hashes=actual_file_hashes,
        ))
        provenance_errors.extend(runner._validate_diagnostic_stabilization_artifact(
            obj=stabilization,
            bundle=bundle,
            expected_episode=expected_no,
            expected_environment_fingerprint=environment_fp,
            expected_server_command=expected_server_command,
        ))
        stabilization_path = diagnostic_dir / "stabilization" / f"{runner.DIAGNOSTIC_BLOCK_ID}.json"
        provenance_errors.extend(runner._validate_exact_stabilization_references(
            results=(("no_burst episode", no_result), ("prefill_burst episode", pb_result)),
            expected_path=stabilization_path,
        ))
        errors.extend(provenance_errors)

        integrity_ok, integrity_errors = runner.verify_diagnostic_integrity_manifest(
            diagnostic_dir,
            integrity,
            expected_schedule_fingerprint=schedule_fp,
            expected_environment_fingerprint=environment_fp,
            expected_episode_count=2,
            expected_stabilization_count=1,
            expected_block_summary_count=0,
            expected_server_log_count=1,
        )
        errors.extend([f"integrity: {x}" for x in integrity_errors])
        whitelist_errors = runner._validate_diagnostic_artifact_counts(diagnostic_dir)
        errors.extend([f"artifact whitelist: {x}" for x in whitelist_errors])

        per_request: list[dict[str, Any]] = []
        for episode, result in ((expected_no, no_result), (expected_pb, pb_result)):
            condition = _ep_get(episode, "condition")
            victims = result.get("victim_requests") if isinstance(result, dict) else None
            bursts = result.get("burst_requests") if isinstance(result, dict) else None
            if not isinstance(victims, list) or len(victims) != 20:
                errors.append(f"{condition}: victim_requests is not exactly 20 records")
                victims = victims if isinstance(victims, list) else []
            victim_by_index = {
                r.get("request_index"): r for r in victims
                if isinstance(r, dict) and type(r.get("request_index")) is int
            }
            for index in range(20):
                item = audit_request_record(
                    victim_by_index.get(index), episode=episode, role="victim", index=index, valid_ids=valid_ids
                )
                item["condition"] = condition
                per_request.append(item)
                errors.extend([f"{condition} victim[{index}]: {e}" for e in item["errors"]])
            expected_bursts = 4 if condition == "prefill_burst" else 0
            if not isinstance(bursts, list) or len(bursts) != expected_bursts:
                errors.append(f"{condition}: burst_requests count is not exactly {expected_bursts}")
                bursts = bursts if isinstance(bursts, list) else []
            burst_by_index = {
                r.get("request_index"): r for r in bursts
                if isinstance(r, dict) and type(r.get("request_index")) is int
            }
            for index in range(expected_bursts):
                item = audit_request_record(
                    burst_by_index.get(index), episode=episode, role="burst", index=index, valid_ids=valid_ids
                )
                item["condition"] = condition
                per_request.append(item)
                errors.extend([f"{condition} burst[{index}]: {e}" for e in item["errors"]])

        pb_requests = [r for r in per_request if r.get("condition") == "prefill_burst"]
        raw_class = raw_overlap_classification(pb_result, pb_requests)
        errors.extend([f"raw overlap: {x}" for x in raw_class.get("errors", [])])
        raw_classification = raw_class.get("classification")
        if _classification_bucket(stored_classification) != _classification_bucket(raw_classification):
            errors.append(
                f"stored semantic classification {stored_classification!r} contradicts raw output overlap {raw_classification!r}"
            )

        active_itl_details: dict[str, Any] = {}
        all_active_itl = True
        for condition, result in (("no_burst", no_result), ("prefill_burst", pb_result)):
            active = ((result.get("trigger") or {}).get("active_cohort_request_indices")
                      if isinstance(result, dict) else None)
            if not isinstance(active, list):
                active = []
            statuses = []
            for index in active:
                item = next((r for r in per_request if r.get("condition") == condition
                             and r.get("role") == "victim" and r.get("request_index") == index), None)
                value = bool(item and item.get("status") == "PASS"
                             and (item.get("raw_reconstruction") or {}).get("itl_available") is True)
                statuses.append({"request_index": index, "itl_available": value})
                all_active_itl = all_active_itl and value
            if len(statuses) != 8:
                all_active_itl = False
            active_itl_details[condition] = statuses
        if not all_active_itl:
            errors.append("not all eight active victims in both conditions have raw-reconstructed itl_available=True")

        after = snapshot_tree(diagnostic_dir)
        readonly_ok = before == after
        if not readonly_ok:
            errors.append("diagnostic tree file list or SHA-256 changed during audit")

        scientific_ok = not errors and all_active_itl and integrity_ok
        result = {
            "auditor_version": AUDITOR_VERSION,
            "runner_version": getattr(runner, "RUNNER_VERSION", None),
            "diagnostic_tree_identity": {
                "diagnostic_dir": str(diagnostic_dir.resolve()),
                "file_count": len(before),
                "tree_sha256": hashlib.sha256(
                    json.dumps(before, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
            "schedule_fingerprint": schedule_fp,
            "environment_fingerprint": environment_fp,
            "actual_source_and_schedule_hashes": actual_file_hashes,
            "integrity_result": {"passed": integrity_ok, "errors": integrity_errors},
            "provenance_result": {"passed": not provenance_errors, "errors": provenance_errors},
            "stabilization_result": {"passed": not any("stabilization" in e for e in provenance_errors)},
            "per_request_status": per_request,
            "prompt_reconstruction": {"passed": not any("prompt" in e.lower() for e in errors)},
            "output_sse_binding": {"passed": not any("output" in e.lower() or "sse" in e.lower() for e in errors)},
            "itl_batch_reconstruction": {"passed": not any("itl" in e.lower() or "batch" in e.lower() for e in errors)},
            "tpot_reconstruction": {"passed": not any("tpot" in e.lower() for e in errors)},
            "stored_semantic_classification": stored_classification,
            "raw_output_overlap_classification": raw_classification,
            "raw_overlap_details": raw_class,
            "active_victim_itl_evaluability": {
                "all_eight_per_condition": all_active_itl,
                "conditions": active_itl_details,
            },
            "scientifically_evaluable": scientific_ok,
            "diagnostic_tree_read_only_verified": readonly_ok,
            "overall_audit_status": "PASS" if scientific_ok else "FAIL",
            "technical_errors": [],
            "errors": errors,
        }
        write_auditor_outputs(audit_output_dir, result)
        return (EXIT_PASS if scientific_ok else EXIT_SCIENTIFIC), result
    except TechnicalAuditError as exc:
        result = _technical_result(str(exc))
        try:
            if audit_output_dir.exists() and audit_output_dir.is_dir():
                write_auditor_outputs(audit_output_dir, result)
        except Exception:
            pass
        return EXIT_TECHNICAL, result
    except Exception as exc:  # fail closed, never uncontrolled from CLI
        result = _technical_result(f"internal auditor error: {type(exc).__name__}: {exc}")
        try:
            if audit_output_dir.exists() and audit_output_dir.is_dir():
                write_auditor_outputs(audit_output_dir, result)
        except Exception:
            pass
        return EXIT_TECHNICAL, result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--schedule-dir", type=Path, required=True)
    parser.add_argument("--audit-output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    code, result = audit_diagnostic(
        args.diagnostic_dir, args.schedule_dir, args.audit_output_dir
    )
    stream = sys.stdout if code == EXIT_PASS else sys.stderr
    stream.write(json.dumps({
        "auditor_version": AUDITOR_VERSION,
        "overall_audit_status": result.get("overall_audit_status"),
        "scientifically_evaluable": result.get("scientifically_evaluable"),
        "exit_code": code,
        "errors": result.get("errors", []),
        "technical_errors": result.get("technical_errors", []),
    }, indent=2) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
