#!/usr/bin/env python3
"""Map existing episode files to the paired-analysis table without dropping failures."""
import argparse
import hashlib
import json
from pathlib import Path
from extract_primary_endpoint import extract_episode
from validate_contract import validate


def build(contract_path, map_path):
    blob = contract_path.read_bytes()
    contract = json.loads(blob)
    validation = validate(contract)
    if validation['errors']:
        raise ValueError('contract incomplete: ' + ', '.join(validation['errors']))
    entries = json.loads(map_path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError('episode map must be a nonempty list')
    output = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('episode_file'), str):
            raise ValueError('missing_episode_file')
        path = (map_path.parent / entry['episode_file']).resolve()
        row = {k: entry.get(k) for k in ('replicate_id', 'capacity_label', 'cache_policy', 'budget_label',
                                         'condition', 'nuisance_config_sha256', 'effective_kv_capacity_tokens',
                                         'effective_max_num_batched_tokens')}
        row.update(cohort_contract_sha256=hashlib.sha256(blob).hexdigest(),
                   workload_sha256=contract['workload_sha256'], model_revision=contract['model_revision'],
                   episode_file=str(path), status='invalid_input', endpoint_ms=None)
        try:
            e = json.loads(path.read_text())
            condition = e.get('schedule_row', {}).get('condition')
            expected = {'reference': 'no_burst', 'additional': 'prefill_burst'}.get(row['condition'])
            if expected is None or condition != expected:
                raise ValueError('episode_condition_mismatch')
            result = extract_episode(e, contract['cohort_indices'], contract['token_index_1based'], contract['origin_field'])
            row.update(status=result['status'], endpoint_ms=result['cohort_elapsed_ms'], endpoint_detail=result)
        except (ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
            row['detail'] = str(exc)
        output.append(row)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--episode-map', type=Path, required=True)
    a = p.parse_args()
    try:
        rows = build(a.contract, a.episode_map)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({'status': 'invalid_input', 'detail': str(exc)})); return 1
    print(json.dumps(rows, indent=2, allow_nan=False))
    return int(any(r['status'] != 'observed' for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
