#!/usr/bin/env python3
"""Keep raw metric series identities; missing or reset counters yield no delta."""
import argparse
import json
import math
from pathlib import Path
import re

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z_0-9:]*)(\{.*\})?\s+([^\s]+)(?:\s+\d+)?\s*$')


def parse_metrics(body):
    if not isinstance(body, str):
        raise ValueError('metric body missing')
    result = {}
    for line in body.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        m = LINE.fullmatch(line)
        if not m:
            raise ValueError('unsupported or malformed Prometheus line')
        key = m[1] + (m[2] or '')
        value = float(m[3])
        if not math.isfinite(value) or key in result:
            raise ValueError('nonfinite or duplicate metric series')
        result[key] = value
    return result


def summarize(rows):
    series, failures, samples = {}, [], []
    prior = None
    starts = [r for r in rows if isinstance(r, dict) and r.get('record_type') == 'capture_start']
    ends = [r for r in rows if isinstance(r, dict) and r.get('record_type') == 'capture_end']
    for row in rows:
        if not isinstance(row, dict):
            failures.append('invalid_record'); continue
        if row.get('record_type') in ('capture_start', 'capture_end'):
            continue
        ns = row.get('response_received_perf_ns')
        if type(ns) is not int or ns < 0 or (prior is not None and ns < prior):
            failures.append('invalid_or_nonmonotonic_timestamp'); continue
        prior = ns
        if row.get('http_status') != 200 or row.get('error'):
            failures.append({'ns': ns, 'error': row.get('error') or 'http_error'}); continue
        try:
            values = parse_metrics(row.get('raw_body'))
        except ValueError as exc:
            failures.append({'ns': ns, 'error': str(exc)}); continue
        samples.append(ns)
        for key, value in values.items():
            if key.startswith('vllm:'):
                series.setdefault(key, []).append({'ns': ns, 'value': value})
    result = {}
    for key, points in sorted(series.items()):
        name = key.split('{', 1)[0]
        values = [p['value'] for p in points]
        complete = len(points) == len(samples) and not failures
        reset = any(b < a for a, b in zip(values, values[1:])) if name.endswith('_total') else None
        delta = values[-1]-values[0] if name.endswith('_total') and complete and len(values) >= 2 and not reset else None
        result[key] = {'points': points, 'min': min(values), 'max': max(values),
                       'counter_delta': delta, 'counter_reset_detected': reset, 'complete_series': complete}
    return {'mode': 'offline_metrics', 'clock_domain': starts[0].get('clock_domain') if len(starts) == 1 else None,
            'capture_closed': len(starts) == 1 and len(ends) == 1,
            'valid_sample_count': len(samples), 'failures': failures, 'series': result,
            'absent_metric_is_zero': False, 'causal_mechanism_validated': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    a = p.parse_args()
    try:
        rows = [json.loads(line) for line in a.input.read_text().splitlines() if line.strip()]
        report = summarize(rows)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({'status': 'invalid_input', 'detail': str(exc)})); return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(bool(report['failures']) or report['valid_sample_count'] < 2 or not report['capture_closed'])


if __name__ == '__main__':
    raise SystemExit(main())
