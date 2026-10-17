#!/usr/bin/env python3
"""Read-only verification of the historical components reused by this package."""
import argparse
import hashlib
import json
from pathlib import Path
from cache_smoke import BASE_SHA256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    reference = json.loads((Path(__file__).parent/'reference/SOURCE.json').read_text())
    expected = {reference['path']: reference['sha256'],
                'new/scripts/prefill_confirmation/run_prefill_confirmation.py': BASE_SHA256}
    rows = []
    for name, digest in expected.items():
        try:
            actual = hashlib.sha256((a.root/name).read_bytes()).hexdigest()
        except OSError:
            actual = None
        rows.append({'path': name, 'expected_sha256': digest, 'actual_sha256': actual, 'matches': actual == digest})
    print(json.dumps({'components': rows, 'runtime_gate_passed': False}, indent=2))
    return int(any(not r['matches'] for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
