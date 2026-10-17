#!/usr/bin/env python3
"""Offline trajectory comparison, insufficient alone for factor acceptance."""
import argparse
from collections import Counter
import json
from pathlib import Path
from parse_iteration_logs import parse_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--small-log', type=Path, required=True)
    p.add_argument('--large-log', type=Path, required=True)
    a = p.parse_args()
    try:
        small, large = [parse_text(path.read_text()) for path in (a.small_log, a.large_log)]
        if small['errors'] or large['errors']:
            raise ValueError('missing, incomplete or unsupported iteration details')
        def summary(report):
            tokens = [r['context_tokens'] for r in report['records'] if r['context_tokens'] > 0]
            return {'iteration_count': report['iteration_count'], 'prefill_iteration_count': len(tokens),
                    'context_token_histogram': dict(sorted(Counter(tokens).items())),
                    'total_context_tokens': sum(tokens), 'max_context_tokens': max(tokens) if tokens else None,
                    'trajectory_sha256': report['trajectory_sha256']}
        sa, la = summary(small), summary(large)
        report = {'small': sa, 'large': la, 'trajectory_different': sa['trajectory_sha256'] != la['trajectory_sha256'],
                  'prefill_histogram_different': sa['context_token_histogram'] != la['context_token_histogram'],
                  'budget_factor_validated': False,
                  'note': 'Require comparable offered work, cache state, effective capacity and repeated observations; differences can be incidental.'}
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'invalid_input', 'detail': str(exc)})); return 1
    print(json.dumps(report, indent=2)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
