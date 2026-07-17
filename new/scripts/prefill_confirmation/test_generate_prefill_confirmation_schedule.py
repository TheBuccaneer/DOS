#!/usr/bin/env python3
"""Offline contract and fault-injection tests for the confirmation scheduler."""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import py_compile
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_prefill_confirmation_schedule as scheduler  # noqa: E402

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    _RESULTS.append((name, ok, detail))
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))


def artifacts(model_key: str) -> dict:
    episodes = scheduler.build_episodes(scheduler.OFFICIAL_SEED, model_key)
    errors = scheduler.validate_schedule(episodes, scheduler.OFFICIAL_SEED, model_key)
    payload = scheduler.build_canonical_payload(episodes, scheduler.OFFICIAL_SEED, model_key)
    fingerprint = scheduler.compute_schedule_fingerprint(payload)
    final_payload = dict(payload)
    final_payload["schedule_fingerprint"] = fingerprint
    return {
        "episodes": episodes,
        "errors": errors,
        "payload": payload,
        "fingerprint": fingerprint,
        "json": json.dumps(final_payload, indent=2, ensure_ascii=False),
        "csv": scheduler.render_csv(episodes),
        "audit": scheduler.render_audit(
            episodes,
            scheduler.OFFICIAL_SEED,
            model_key,
            fingerprint,
            errors,
        ),
    }


def bundle_files(prefix: str = "NEW") -> list[tuple[str, str]]:
    return [
        (scheduler.BUNDLE_FILENAMES[0], f"{prefix}-JSON"),
        (scheduler.BUNDLE_FILENAMES[1], f"{prefix}-CSV"),
        (scheduler.BUNDLE_FILENAMES[2], f"{prefix}-AUDIT"),
    ]


def residue(directory: Path) -> list[str]:
    return sorted(
        path.name
        for path in directory.iterdir()
        if ".tmp." in path.name or ".bak." in path.name
    )


def target_state(directory: Path) -> dict[str, str | None]:
    return {
        name: ((directory / name).read_text(encoding="utf-8") if (directory / name).exists() else None)
        for name in scheduler.BUNDLE_FILENAMES
    }


def test_compile() -> None:
    here = Path(__file__).resolve().parent
    paths = [here / "generate_prefill_confirmation_schedule.py", Path(__file__).resolve()]
    ok = True
    detail = []
    for path in paths:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            ok = False
            detail.append(str(exc))
    check("py_compile scheduler and test file", ok, "; ".join(detail))


def test_model_contract(model_key: str) -> dict:
    data = artifacts(model_key)
    episodes = data["episodes"]
    blocks: dict[str, list] = {}
    for episode in episodes:
        blocks.setdefault(episode.block_id, []).append(episode)

    check(f"{model_key}: validation passes", data["errors"] == [], str(data["errors"][:5]))
    check(f"{model_key}: exactly 96 episodes", len(episodes) == 96, str(len(episodes)))
    check(f"{model_key}: exactly 48 blocks", len(blocks) == 48, str(len(blocks)))
    check(
        f"{model_key}: bundle contains only selected model",
        all(
            episode.model_key == model_key
            and episode.model_id == scheduler.MODEL_REGISTRY[model_key]["model_id"]
            for episode in episodes
        ),
    )
    check(
        f"{model_key}: episode and block IDs use model prefix",
        all(
            episode.episode_id.startswith(model_key + "_")
            and episode.block_id.startswith(model_key + "_")
            for episode in episodes
        ),
    )
    check(
        f"{model_key}: frozen grid only",
        all(
            episode.offload_gb in (0, 8, 12)
            and episode.concurrency in (4, 8)
            and episode.trigger_after_decode_tokens == 16
            and 1 <= episode.repeat <= 8
            for episode in episodes
        ),
    )
    check(
        f"{model_key}: exact victim configuration",
        all(
            episode.victim_request_count == 20
            and episode.victim_input_len == 256
            and episode.victim_output_len == 64
            and episode.victim_temperature == 0.0
            for episode in episodes
        ),
    )
    check(
        f"{model_key}: exact burst configuration",
        all(
            episode.burst_parallel_requests == 4
            and episode.burst_input_len == 2048
            and episode.burst_output_len == 16
            and episode.burst_temperature == 0.0
            and episode.max_num_batched_tokens == 2048
            for episode in episodes
        ),
    )

    block_ok = all(
        len(group) == 2
        and [episode.order_in_block for episode in group] == [1, 2]
        and [episode.restart_server_before_block for episode in group] == [1, 0]
        and sorted(episode.condition for episode in group) == ["no_burst", "prefill_burst"]
        for group in blocks.values()
    )
    check(f"{model_key}: paired contiguous block contract", block_ok)

    first_by_cell: dict[tuple[int, int], list[str]] = {}
    cells_by_repeat: dict[int, set[tuple[int, int]]] = {repeat: set() for repeat in range(1, 9)}
    for group in blocks.values():
        first = group[0]
        first_by_cell.setdefault((first.offload_gb, first.concurrency), []).append(first.condition)
        cells_by_repeat[first.repeat].add((first.offload_gb, first.concurrency))
    check(
        f"{model_key}: exact 4/4 condition-first balance per cell",
        len(first_by_cell) == 6
        and all(
            conditions.count("no_burst") == 4
            and conditions.count("prefill_burst") == 4
            for conditions in first_by_cell.values()
        ),
        str(first_by_cell),
    )
    check(
        f"{model_key}: every repeat contains all six cells",
        all(cells == set(scheduler.all_cells()) for cells in cells_by_repeat.values()),
        str(cells_by_repeat),
    )

    victim_by_repeat: dict[int, set[int]] = {}
    burst_by_repeat: dict[int, set[int]] = {}
    for episode in episodes:
        victim_by_repeat.setdefault(episode.repeat, set()).add(episode.victim_workload_seed)
        burst_by_repeat.setdefault(episode.repeat, set()).add(episode.burst_workload_seed)
    check(
        f"{model_key}: victim seed constant per repeat and distinct across repeats",
        all(len(values) == 1 for values in victim_by_repeat.values())
        and len({next(iter(values)) for values in victim_by_repeat.values()}) == 8,
    )
    check(
        f"{model_key}: burst seed constant per repeat and distinct across repeats",
        all(len(values) == 1 for values in burst_by_repeat.values())
        and len({next(iter(values)) for values in burst_by_repeat.values()}) == 8,
    )
    check(
        f"{model_key}: victim and burst seeds never coincide",
        all(episode.victim_workload_seed != episode.burst_workload_seed for episode in episodes),
    )
    check(
        f"{model_key}: episode IDs and seeds unique",
        len({episode.episode_id for episode in episodes}) == 96
        and len({episode.episode_seed for episode in episodes}) == 96,
    )

    check(
        f"{model_key}: fingerprint reproducible",
        data["fingerprint"] == scheduler.compute_schedule_fingerprint(data["payload"]),
    )
    check(
        f"{model_key}: CSV/JSON consistency",
        scheduler.check_csv_json_consistency(data["csv"], episodes) == [],
    )
    parsed_csv = list(csv.DictReader(io.StringIO(data["csv"])))
    check(
        f"{model_key}: exact CSV header and row count",
        list(parsed_csv[0].keys()) == list(scheduler.EPISODE_FIELDS) and len(parsed_csv) == 96,
    )
    check(
        f"{model_key}: audit has one fingerprint and final PASS",
        data["audit"].count("schedule_fingerprint:") == 1
        and data["audit"].rstrip().endswith("OVERALL: PASS"),
    )
    check(
        f"{model_key}: planned checkpoints and initial repeats frozen",
        data["payload"]["planned_repeat_checkpoints"] == [8, 12, 16]
        and data["payload"]["included_repeats"] == list(range(1, 9)),
    )

    again = artifacts(model_key)
    check(
        f"{model_key}: repeated generation byte-identical",
        data["json"] == again["json"]
        and data["csv"] == again["csv"]
        and data["audit"] == again["audit"],
    )

    mutated = copy.deepcopy(episodes)
    mutated[0].model_key = "qwen" if model_key == "llama" else "llama"
    check(
        f"{model_key}: cross-model contamination rejected",
        bool(scheduler.validate_schedule(mutated, scheduler.OFFICIAL_SEED, model_key)),
    )
    return data


def test_cross_model(llama: dict, qwen: dict) -> None:
    check("Llama and Qwen fingerprints differ", llama["fingerprint"] != qwen["fingerprint"])
    llama_victim = {episode.victim_workload_seed for episode in llama["episodes"]}
    qwen_victim = {episode.victim_workload_seed for episode in qwen["episodes"]}
    llama_burst = {episode.burst_workload_seed for episode in llama["episodes"]}
    qwen_burst = {episode.burst_workload_seed for episode in qwen["episodes"]}
    check("Llama and Qwen victim seed streams are disjoint", llama_victim.isdisjoint(qwen_victim))
    check("Llama and Qwen burst seed streams are disjoint", llama_burst.isdisjoint(qwen_burst))
    raised = False
    try:
        scheduler.build_episodes(scheduler.OFFICIAL_SEED, "unknown")
    except ValueError:
        raised = True
    check("unknown model_key rejected", raised)


def test_noforce_existing_aborts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        first = directory / scheduler.BUNDLE_FILENAMES[0]
        first.write_text("OLD", encoding="utf-8")
        raised = False
        try:
            scheduler.write_bundle_atomic(directory, bundle_files(), force=False)
        except FileExistsError:
            raised = True
        check("writer: without force existing target aborts", raised)
        check("writer: no-force preserves old state", first.read_text(encoding="utf-8") == "OLD")
        check("writer: no-force leaves no residue", residue(directory) == [], str(residue(directory)))


def _install_failure_test(*, force: bool, initial: dict[str, str], exc_factory) -> tuple[object, dict, list[str]]:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name, content in initial.items():
            (directory / name).write_text(content, encoding="utf-8")
        before = target_state(directory)
        original_replace = scheduler.os.replace
        install_count = 0

        def injected_replace(src, dst):
            nonlocal install_count
            src_path = Path(src)
            if ".tmp." in src_path.name:
                install_count += 1
                if install_count == 2:
                    raise exc_factory()
            return original_replace(src, dst)

        scheduler.os.replace = injected_replace
        caught = None
        try:
            scheduler.write_bundle_atomic(directory, bundle_files(), force=force)
        except BaseException as exc:  # deliberate fault-injection capture
            caught = exc
        finally:
            scheduler.os.replace = original_replace
        return caught, {"before": before, "after": target_state(directory)}, residue(directory)


def test_install_rollbacks() -> None:
    caught, states, leftovers = _install_failure_test(force=False, initial={}, exc_factory=lambda: OSError("install-2"))
    check("writer: second install failure propagates", isinstance(caught, OSError))
    check("writer: empty pre-state restored after install failure", states["after"] == states["before"], str(states))
    check("writer: install failure leaves no residue", leftovers == [], str(leftovers))

    full_old = {name: f"OLD-{index}" for index, name in enumerate(scheduler.BUNDLE_FILENAMES)}
    caught, states, leftovers = _install_failure_test(force=True, initial=full_old, exc_factory=lambda: OSError("install-2"))
    check("writer: force install failure propagates", isinstance(caught, OSError))
    check("writer: complete old bundle restored", states["after"] == states["before"], str(states))
    check("writer: force rollback leaves no residue", leftovers == [], str(leftovers))

    partial_old = {
        scheduler.BUNDLE_FILENAMES[0]: "OLD-JSON",
        scheduler.BUNDLE_FILENAMES[2]: "OLD-AUDIT",
    }
    caught, states, leftovers = _install_failure_test(force=True, initial=partial_old, exc_factory=lambda: OSError("install-2"))
    check("writer: partial old-state failure propagates", isinstance(caught, OSError))
    check("writer: exact partial old state restored", states["after"] == states["before"], str(states))
    check("writer: partial rollback leaves no residue", leftovers == [], str(leftovers))

    caught, states, leftovers = _install_failure_test(force=True, initial=full_old, exc_factory=KeyboardInterrupt)
    check("writer: KeyboardInterrupt re-raised", isinstance(caught, KeyboardInterrupt))
    check("writer: KeyboardInterrupt restores complete old state", states["after"] == states["before"], str(states))
    check("writer: KeyboardInterrupt leaves no residue", leftovers == [], str(leftovers))


def test_successful_force() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for index, name in enumerate(scheduler.BUNDLE_FILENAMES):
            (directory / name).write_text(f"OLD-{index}", encoding="utf-8")
        scheduler.write_bundle_atomic(directory, bundle_files(), force=True)
        expected = {name: content for name, content in bundle_files()}
        check("writer: successful force installs all new files", target_state(directory) == expected)
        check("writer: successful force leaves no residue", residue(directory) == [], str(residue(directory)))


def test_fsync_failure_cleanup() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        original_fsync = scheduler.os.fsync
        calls = 0

        def injected_fsync(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("fsync-2")
            return original_fsync(fd)

        scheduler.os.fsync = injected_fsync
        caught = None
        try:
            scheduler.write_bundle_atomic(directory, bundle_files(), force=False)
        except BaseException as exc:
            caught = exc
        finally:
            scheduler.os.fsync = original_fsync
        check("writer: second fsync failure propagates", isinstance(caught, OSError))
        check("writer: fsync failure leaves no targets", target_state(directory) == {name: None for name in scheduler.BUNDLE_FILENAMES}, str(target_state(directory)))
        check("writer: fsync failure removes every temp and backup", residue(directory) == [], str(residue(directory)))


def test_backup_cleanup_failure_after_commit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for index, name in enumerate(scheduler.BUNDLE_FILENAMES):
            (directory / name).write_text(f"OLD-{index}", encoding="utf-8")

        original_unlink = Path.unlink
        backup_unlinks = 0

        def injected_unlink(self: Path, *args, **kwargs):
            nonlocal backup_unlinks
            if ".bak." in self.name:
                backup_unlinks += 1
                if backup_unlinks == 2:
                    raise OSError("backup-cleanup-2")
            return original_unlink(self, *args, **kwargs)

        Path.unlink = injected_unlink
        caught = None
        try:
            scheduler.write_bundle_atomic(directory, bundle_files(), force=True)
        except BaseException as exc:
            caught = exc
        finally:
            Path.unlink = original_unlink

        expected = {name: content for name, content in bundle_files()}
        leftovers = residue(directory)
        check("writer: post-commit backup cleanup error is visible", isinstance(caught, OSError))
        check("writer: complete new bundle retained after cleanup error", target_state(directory) == expected, str(target_state(directory)))
        check("writer: no temp remains after committed install", not any(".tmp." in name for name in leftovers), str(leftovers))
        check("writer: undeleted backups remain visible", any(".bak." in name for name in leftovers), str(leftovers))


def main() -> int:
    print("test_generate_prefill_confirmation_schedule.py")
    print("=" * 78)
    test_compile()
    llama = test_model_contract("llama")
    qwen = test_model_contract("qwen")
    test_cross_model(llama, qwen)
    test_noforce_existing_aborts()
    test_install_rollbacks()
    test_successful_force()
    test_fsync_failure_cleanup()
    test_backup_cleanup_failure_after_commit()
    print("=" * 78)
    passed = sum(1 for _name, ok, _detail in _RESULTS if ok)
    print(f"{passed}/{len(_RESULTS)} checks passed")
    return 0 if passed == len(_RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
