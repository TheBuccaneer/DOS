#!/usr/bin/env python3
"""Offline H/E/Gamma; bootstrap complete independent replicate blocks."""
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics

CELLS = list(itertools.product(('shared', 'isolated'), ('small', 'large')))
HASH_KEYS = ('cohort_contract_sha256', 'workload_sha256', 'nuisance_config_sha256')


def quantify(rows, *, bootstrap=5000, seed=20261010):
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty endpoint rows required')
    if type(bootstrap) is not int or not 100 <= bootstrap <= 100000:
        raise ValueError('bootstrap must be 100..100000')
    groups = defaultdict(dict)
    common = None
    capacities, budgets = {}, {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('row_not_object')
        for key in HASH_KEYS:
            if not isinstance(row.get(key), str) or not re.fullmatch('[0-9a-f]{64}', row[key]):
                raise ValueError('missing_or_invalid_' + key)
        if not isinstance(row.get('model_revision'), str) or not row['model_revision']:
            raise ValueError('missing_model_revision')
        identity = tuple(row[k] for k in HASH_KEYS) + (row['model_revision'],)
        if common is None:
            common = identity
        if identity != common:
            raise ValueError('cohort_workload_model_or_nuisance_mismatch')
        replicate = row.get('replicate_id')
        capacity, policy, budget, condition = (row.get(k) for k in ('capacity_label', 'cache_policy', 'budget_label', 'condition'))
        if not isinstance(replicate, str) or not replicate or capacity not in ('sufficient', 'near') or (policy, budget) not in CELLS or condition not in ('reference', 'additional'):
            raise ValueError('invalid_factor_or_replicate')
        value = row.get('endpoint_ms')
        if row.get('status') != 'observed' or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid_endpoint_retained_no_complete_case_filter')
        tokens, batched = row.get('effective_kv_capacity_tokens'), row.get('effective_max_num_batched_tokens')
        if type(tokens) is not int or tokens < 1 or type(batched) is not int or batched < 1:
            raise ValueError('missing_effective_capacity_or_budget')
        if capacity in capacities and capacities[capacity] != tokens:
            raise ValueError('capacity_not_constant_within_level')
        if budget in budgets and budgets[budget] != batched:
            raise ValueError('budget_value_inconsistent')
        capacities[capacity], budgets[budget] = tokens, batched
        key = (policy, budget, condition)
        block = groups[(capacity, replicate)]
        if key in block:
            raise ValueError('duplicate_cell_condition')
        block[key] = value
    if budgets.get('small', math.inf) >= budgets.get('large', -math.inf):
        raise ValueError('budget_order_or_missing_level')
    if set(capacities) != {'sufficient', 'near'} or capacities['near'] >= capacities['sufficient']:
        raise ValueError('capacity_order_or_missing_level')
    by_capacity = defaultdict(list)
    replicate_sets = defaultdict(set)
    expected = set(itertools.product(('shared', 'isolated'), ('small', 'large'), ('reference', 'additional')))
    for (capacity, replicate), block in sorted(groups.items()):
        if set(block) != expected:
            raise ValueError('incomplete_replicate_block:' + capacity + ':' + replicate)
        h = {(p, b): block[p, b, 'additional']-block[p, b, 'reference'] for p, b in CELLS}
        es = h['shared', 'small']-h['shared', 'large']
        ei = h['isolated', 'small']-h['isolated', 'large']
        by_capacity[capacity].append({'replicate_id': replicate,
                                    **{'H_' + p + '_' + b: h[p, b] for p, b in CELLS},
                                    'E_shared': es, 'E_isolated': ei, 'Gamma': ei-es})
        replicate_sets[capacity].add(replicate)
    if replicate_sets['near'] != replicate_sets['sufficient']:
        raise ValueError('incomplete_eight_cell_replicate_ids')
    rng = random.Random(seed)
    output = {}
    for capacity, blocks in sorted(by_capacity.items()):
        if len(blocks) < 2:
            raise ValueError('at_least_two_independent_replicates_required')
        keys = [k for k in blocks[0] if k != 'replicate_id']
        draws = {k: [] for k in keys}
        for _ in range(bootstrap):
            sample = [rng.randrange(len(blocks)) for _ in blocks]
            for key in keys:
                draws[key].append(statistics.mean(blocks[i][key] for i in sample))
        effects = {}
        for key in keys:
            values = sorted(draws[key])
            effects[key] = {'mean_ms': statistics.mean(b[key] for b in blocks),
                            'percentile_95_ci_ms': [values[int(.025*(bootstrap-1))], values[int(.975*(bootstrap-1))]]}
        output[capacity] = {'independent_replicates': len(blocks), 'blocks': blocks, 'effects': effects,
                            'point_estimate_reversal': effects['E_shared']['mean_ms'] < 0 < effects['E_isolated']['mean_ms'],
                            'marginal_ci_sign_pattern': effects['E_shared']['percentile_95_ci_ms'][1] < 0 < effects['E_isolated']['percentile_95_ci_ms'][0],
                            'protection_reversal_established': False,
                            'precision_target_achieved': None}
    return {'status': 'computed', 'unit': 'ms', 'bootstrap_unit': 'complete_replicate_block',
            'bootstrap_draws': bootstrap, 'bootstrap_seed': seed, 'capacity_results': output,
            'causal_design_validated': False,
            'inference_note': 'Exploratory marginal percentile intervals; no simultaneous coverage or causal/reversal claim.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20261010)
    a = p.parse_args()
    try:
        report = quantify(json.loads(a.input.read_text()), bootstrap=a.bootstrap, seed=a.seed)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({'status': 'invalid_input', 'detail': str(exc), 'causal_design_validated': False})); return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
