"""Read-only post-hoc checks for one eviction-control JSONL (no network, no writes).

Usage: python3 -I -B posthoc_check.py <cache_isolation_gate dir> <run.jsonl> <expected_A3_hits>
Checks: (1) counter continuity between the settled AFTER poll of request i and the BEFORE poll
of request i+1 (no foreign traffic in the gaps); (2) A3 equals the pre-registered value.
Exit 0 only if both hold.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from cache_eviction_control import ORDER, snapshot  # noqa: E402

rows = [json.loads(x) for x in Path(sys.argv[2]).read_text().splitlines()]
expected_a3 = int(sys.argv[3])
before = {r['request']: r['sample'] for r in rows if r['record_type'] == 'metrics_before'}
after = {}
for r in rows:
    if r['record_type'] == 'metrics_after_poll':
        after[r['request']] = r['sample']            # last poll wins = settled sample
problems = []
for prev, nxt in zip(ORDER, ORDER[1:]):
    if prev in after and nxt in before:
        a, b = snapshot(after[prev])['counters'], snapshot(before[nxt])['counters']
        if a != b:
            problems.append(f'gap {prev}->{nxt}: counters changed {a} -> {b}')
end = [r for r in rows if r['record_type'] == 'run_end']
a3 = next((r['cache_hit_delta'] for r in rows if r['record_type'] == 'request_assessment' and r['request'] == 'A3'), None)
if not end or end[-1]['status'] != 'valid_control_run':
    problems.append('run_end missing or not valid_control_run')
if a3 != expected_a3:
    problems.append(f'A3 hits {a3} != pre-registered {expected_a3}')
print(json.dumps({'continuity_and_prediction_ok': not problems, 'A3_hits': a3, 'problems': problems}, indent=2))
sys.exit(0 if not problems else 1)
