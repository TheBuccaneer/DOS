#!/usr/bin/env python3
"""Extract exact vLLM 0.17.1 iteration details; no invented timestamps."""
import argparse
import hashlib
import json
from pathlib import Path
import re

PATTERN = re.compile(r'Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, (\d+) generation requests, (\d+) generation tokens, iteration elapsed time: ([0-9.]+) ms')
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def parse_text(text):
    records, errors = [], []
    previous = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = ANSI.sub('', raw)
        if 'Iteration(' not in line:
            continue
        m = PATTERN.search(line)
        if not m:
            errors.append({'line': number, 'error': 'unrecognized_iteration_line'}); continue
        index = int(m[1])
        if previous is not None and index != previous+1:
            errors.append({'line': number, 'error': 'iteration_gap_restart_or_multiple_engines'})
        previous = index
        records.append({'line': number, 'iteration': index, 'context_requests': int(m[2]),
                        'context_tokens': int(m[3]), 'generation_requests': int(m[4]),
                        'generation_tokens': int(m[5]), 'elapsed_ms': float(m[6]),
                        'raw_line': raw})
    if not records:
        errors.append({'error': 'no_iteration_details'})
    trajectory = [[r['context_tokens'], r['generation_tokens']] for r in records]
    return {'records': records, 'errors': errors, 'iteration_count': len(records),
            'trajectory_sha256': hashlib.sha256(json.dumps(trajectory).encode()).hexdigest() if records else None,
            'server_gpu_kernel_time_measured': False, 'budget_factor_validated': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    a = p.parse_args()
    try:
        blob = a.log.read_bytes()
        report = parse_text(blob.decode('utf-8'))
        report.update(source_sha256=hashlib.sha256(blob).hexdigest(), source_file=str(a.log))
    except (OSError, ValueError) as exc:
        print(json.dumps({'errors': [str(exc)]})); return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(bool(report['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
