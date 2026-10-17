#!/usr/bin/env python3
"""Validate preregistered contract completeness, never claim a runtime gate pass."""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import re


def validate(obj):
    errors = []
    if not isinstance(obj, dict):
        return {'errors': ['contract_not_object'], 'runtime_gate_passed': False}
    for name in ('model_revision', 'origin_field', 'warmup_and_reset_protocol', 'independent_replicate_definition', 'precision_target'):
        if not isinstance(obj.get(name), str) or not obj[name].strip():
            errors.append('missing_' + name)
    try:
        dt = datetime.fromisoformat(obj.get('predeclared_utc', '').replace('Z', '+00:00'))
        if dt.tzinfo is None:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        errors.append('missing_or_invalid_predeclared_utc')
    cohort = obj.get('cohort_indices')
    if not isinstance(cohort, list) or not cohort or any(type(i) is not int or i < 0 for i in cohort) or len(set(cohort)) != len(cohort):
        errors.append('invalid_cohort_indices')
    k = obj.get('token_index_1based')
    if type(k) is not int or k < 2:
        errors.append('later_token_index_required')
    if obj.get('cpu_offload_gb') != 0 or type(obj.get('cpu_offload_gb')) is not int:
        errors.append('cpu_weight_offload_must_be_zero')
    if obj.get('failure_policy') != 'retain_invalid_no_numeric':
        errors.append('failure_policy_required')
    work = obj.get('workload_sha256')
    if not isinstance(work, str) or not re.fullmatch('[0-9a-f]{64}', work):
        errors.append('missing_workload_sha256')
    for field in ('budget_tokens', 'kv_capacity_tokens'):
        levels = obj.get(field)
        keys = ('small', 'large') if field == 'budget_tokens' else ('near', 'sufficient')
        if not isinstance(levels, dict) or any(type(levels.get(key)) is not int or levels[key] < 1 for key in keys):
            errors.append('unresolved_' + field)
        elif levels[keys[0]] >= levels[keys[1]]:
            errors.append('invalid_level_order_' + field)
    return {'errors': errors, 'contract_complete': not errors, 'runtime_gate_passed': False,
            'cells': [{'cache_policy': p, 'budget_label': b, 'capacity_label': c}
                      for p, b, c in itertools.product(('shared', 'isolated'), ('small', 'large'), ('sufficient', 'near'))],
            'note': 'Completeness does not prove preregistration timing, cache reuse, scheduler effects or effective capacity.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    a = p.parse_args()
    try:
        report = validate(json.loads(a.contract.read_text()))
    except (OSError, ValueError) as exc:
        report = {'errors': [str(exc)], 'runtime_gate_passed': False}
    print(json.dumps(report, indent=2))
    return int(bool(report['errors']))


if __name__ == '__main__':
    raise SystemExit(main())
