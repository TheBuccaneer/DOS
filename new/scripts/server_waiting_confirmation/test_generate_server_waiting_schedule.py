#!/usr/bin/env python3
"""Offline contract tests for generate_server_waiting_schedule.py.

No GPU, network, tokenizer, or vLLM server is used.
"""
from __future__ import annotations

import csv
import io
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import generate_server_waiting_schedule as gen  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    ok = bool(condition)
    checks.append((name, ok, detail))
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))


def main() -> int:
    print("test_generate_server_waiting_schedule.py")
    print("=" * 78)

    episodes = gen.build_episodes(gen.OFFICIAL_SEED)
    validation_errors = gen.validate_schedule(episodes, gen.OFFICIAL_SEED)
    check("official schedule validates with zero errors", not validation_errors, str(validation_errors))

    # --- Item 1: exact schedule dimensions and cell coverage ---------------
    check("exactly 32 episodes", len(episodes) == 32)
    blocks = {}
    for ep in episodes:
        blocks.setdefault(ep.block_id, []).append(ep)
    check("exactly 16 blocks", len(blocks) == 16)
    cells_by_repeat: dict[int, set[tuple[int, int]]] = {}
    for ep in episodes:
        cells_by_repeat.setdefault(ep.repeat, set()).add((ep.offload_gb, ep.server_max_num_seqs))
    check(
        "4 repeats, each with all 4 (offload, server_max_num_seqs) cells exactly once",
        set(cells_by_repeat) == {1, 2, 3, 4} and all(len(c) == 4 for c in cells_by_repeat.values()),
    )
    check(
        "cell coverage is exactly offload{0,12} x server_max_num_seqs{4,8}",
        set(gen.all_cells()) == {(0, 4), (0, 8), (12, 4), (12, 8)},
    )

    # --- Item 2: exact paired condition order and 2/2 balance --------------
    for ep_list in blocks.values():
        check(
            f"block {ep_list[0].block_id}: exactly 2 episodes, both conditions, order [1,2]",
            len(ep_list) == 2 and sorted(e.condition for e in ep_list) == ["no_burst", "prefill_burst"]
            and sorted(e.order_in_block for e in ep_list) == [1, 2],
        )
    first_condition_by_cell: dict[tuple[int, int], list[str]] = {}
    for ep_list in blocks.values():
        first_ep = next(e for e in ep_list if e.order_in_block == 1)
        key = (first_ep.offload_gb, first_ep.server_max_num_seqs)
        first_condition_by_cell.setdefault(key, []).append(first_ep.condition)
    check(
        "every cell has exactly 2/2 no_burst-first / prefill_burst-first balance across its 4 repeats",
        all(
            v.count("no_burst") == 2 and v.count("prefill_burst") == 2
            for v in first_condition_by_cell.values()
        ),
        str(first_condition_by_cell),
    )

    # --- Item 3: deterministic schedule fingerprint -------------------------
    payload_a = gen.build_canonical_payload(episodes, gen.OFFICIAL_SEED)
    fp_a = gen.compute_schedule_fingerprint(payload_a)
    episodes_b = gen.build_episodes(gen.OFFICIAL_SEED)
    payload_b = gen.build_canonical_payload(episodes_b, gen.OFFICIAL_SEED)
    fp_b = gen.compute_schedule_fingerprint(payload_b)
    check("fingerprint is deterministic across independent re-generation", fp_a == fp_b)
    check("fingerprint has the expected sha256: format", gen.is_valid_fingerprint_format(fp_a)) \
        if hasattr(gen, "is_valid_fingerprint_format") else check(
            "fingerprint has the expected sha256: format",
            fp_a.startswith("sha256:") and len(fp_a) == 71,
        )
    check(
        "fingerprint matches the value recorded in run_server_waiting_confirmation.py",
        fp_a == "sha256:7c5a6e411cc35f6c2d12c7d768434d67cbb58862c6435cd87d5db95672089557",
        fp_a,
    )

    different_seed_episodes = gen.build_episodes(gen.OFFICIAL_SEED + 1)
    different_seed_payload = gen.build_canonical_payload(different_seed_episodes, gen.OFFICIAL_SEED + 1)
    fp_different = gen.compute_schedule_fingerprint(different_seed_payload)
    check("a different seed produces a different fingerprint", fp_different != fp_a)

    # --- Seed-namespace independence from the original study ----------------
    check(
        "seed differs from the original prefill-confirmation study's seed (20260718)",
        gen.OFFICIAL_SEED != 20260718,
    )
    check(
        "design_version differs from the original study's",
        gen.DESIGN_VERSION != "prefill-confirmation-v1",
    )
    check(
        "SEED_NAMESPACE_TAG is mixed into episode/victim/burst seed derivation",
        gen._ns_seed(gen.OFFICIAL_SEED, "victim", "1")
        != gen.derive_seed(str(gen.OFFICIAL_SEED), "victim", "1"),
    )

    # --- Victim/burst seed constancy-per-repeat + distinctness -------------
    victim_seeds = {ep.repeat: ep.victim_workload_seed for ep in episodes}
    check("victim_workload_seed constant within each repeat", all(
        len({e.victim_workload_seed for e in episodes if e.repeat == r}) == 1 for r in range(1, 5)
    ))
    check("victim_workload_seed distinct across all 4 repeats", len(set(victim_seeds.values())) == 4)
    check("victim_workload_seed always != burst_workload_seed", all(
        e.victim_workload_seed != e.burst_workload_seed for e in episodes
    ))

    # --- CSV / JSON / audit consistency -------------------------------------
    csv_text = gen.render_csv(episodes)
    csv_consistency_errors = gen.check_csv_json_consistency(csv_text, episodes)
    check("rendered CSV is consistent with the episode objects", not csv_consistency_errors, str(csv_consistency_errors))
    reader = csv.DictReader(io.StringIO(csv_text))
    check("CSV header matches EPISODE_FIELDS exactly", list(reader.fieldnames or []) == list(gen.EPISODE_FIELDS))

    audit_text = gen.render_audit(episodes, gen.OFFICIAL_SEED, fp_a, validation_errors)
    check("audit report declares OVERALL: PASS", "OVERALL: PASS" in audit_text)
    check("audit report contains the fingerprint line", f"schedule_fingerprint: {fp_a}" in audit_text)

    # --- Structural mutation rejection (mirrors the runner's independent
    # check_structural_schedule / validate_schedule contract) ---------------
    mutated = list(episodes)
    mutated[0] = replace(mutated[0], server_max_num_seqs=16)
    mutation_errors = gen.validate_schedule(mutated, gen.OFFICIAL_SEED)
    check("forbidden server_max_num_seqs mutation is rejected by validate_schedule", bool(mutation_errors))

    mutated2 = list(episodes)
    mutated2[0] = replace(mutated2[0], trigger_after_decode_tokens=1)
    mutation_errors2 = gen.validate_schedule(mutated2, gen.OFFICIAL_SEED)
    check("forbidden trigger_after_decode_tokens mutation is rejected by validate_schedule", bool(mutation_errors2))

    mutated3 = list(episodes)
    mutated3[0] = replace(mutated3[0], offload_gb=8)
    mutation_errors3 = gen.validate_schedule(mutated3, gen.OFFICIAL_SEED)
    check("forbidden offload_gb=8 mutation is rejected (only 0/12 are allowed)", bool(mutation_errors3))

    # --- write_bundle_atomic: round trip through a real temp directory ------
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "bundle"
        files = [
            (gen.BUNDLE_FILENAMES[0], __import__("json").dumps({**payload_a, "schedule_fingerprint": fp_a}, indent=2, sort_keys=True)),
            (gen.BUNDLE_FILENAMES[1], csv_text),
            (gen.BUNDLE_FILENAMES[2], audit_text),
        ]
        gen.write_bundle_atomic(out_dir, files, force=False)
        check("write_bundle_atomic wrote all 3 files", all((out_dir / n).is_file() for n in gen.BUNDLE_FILENAMES))
        raised = False
        try:
            gen.write_bundle_atomic(out_dir, files, force=False)
        except FileExistsError:
            raised = True
        check("write_bundle_atomic refuses to overwrite without --force", raised)
        gen.write_bundle_atomic(out_dir, files, force=True)
        check("write_bundle_atomic --force overwrites cleanly", all((out_dir / n).is_file() for n in gen.BUNDLE_FILENAMES))

    # --- CLI smoke: generator main() actually runs end-to-end ---------------
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "cli_bundle"
        rc = gen.main(["--output-dir", str(out_dir)])
        check("generator CLI main() exits 0", rc == 0)
        check("generator CLI wrote all 3 bundle files", all((out_dir / n).is_file() for n in gen.BUNDLE_FILENAMES))

    print("=" * 78)
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


import unittest


class _MainSuiteTestCase(unittest.TestCase):
    """N1 fix (2026-07-20 third hardening pass): a `python3 -m unittest
    discover` invocation previously found zero tests in this project's
    self-contained-script-with-main() test files (matching the
    convention already established by the originally-audited
    test_run_prefill_confirmation.py/test_prefill_confirmation_timing.py).
    This thin TestCase wrapper makes the SAME exhaustive check suite
    discoverable by `unittest discover`, without changing how the file
    behaves when run directly as `python3 test_*.py`."""

    def test_all_checks_pass(self) -> None:
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    raise SystemExit(main())
