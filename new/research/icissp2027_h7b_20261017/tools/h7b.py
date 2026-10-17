"""Offline H7b feasibility analysis. No server, CUDA, or network imports.

Input is the documented H7b capture contract, not legacy overlap episodes.
Synthetic records require explicit opt-in and cannot become serving evidence.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference'))
from reanalyze import stream

SCHEMA = 'dos-h7b-trial-v1'
WINDOWS = (16, 32)
THETAS = (.5, .7, .9, .95, .99)
FEATURES = {
    'ttft_learned': ('ttft_ms',),
    'temporal_learned': ('ttft_ms', 'past_ttft_median_ms'),
    'joint_learned': ('ttft_ms', 'past_ttft_median_ms', 'reference_pre_ms'),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def checked_stream(events):
    require(bool(events), 'empty SSE stream')
    stamps = [x['receive_perf_counter_ns'] for x in events]
    require(all(isinstance(t, int) and t >= 0 for t in stamps), 'invalid timestamp')
    require(all(a <= b for a, b in zip(stamps, stamps[1:])), 'nonmonotone SSE')
    parsed = stream(events)
    require(not parsed['errors'], 'SSE: ' + ','.join(parsed['errors']))
    require(bool(parsed['times']), 'no output tokens')
    return parsed


def public_timing(reference_events, query_events, query_dispatch_ns,
                  trigger_receipt_ns):
    """Return only attacker-observable timings; usage/labels never enter features."""
    d = checked_stream(reference_events)
    q = checked_stream(query_events)
    require(len(d['times']) >= 28, 'reference does not reach token 28')
    require(len(d['times']) >= 8 and all(n == 1 for n in d['batch_sizes'][:8]),
            'reference tokens 1..8 not individually timed')
    t = d['times']
    require(trigger_receipt_ns == t[7], 'trigger is not reference token 8')
    require(trigger_receipt_ns <= query_dispatch_ns < t[8], 'late/early D8 dispatch')
    require(query_dispatch_ns <= q['times'][0], 'query output precedes dispatch')
    require(query_dispatch_ns < t[-1], 'requests do not overlap')
    intervals = [(t[i] - t[i - 1]) / 1e6 for i in range(4, 8)]
    require(all(v > 0 for v in intervals), 'nonpositive pre-dispatch interval')
    ttft = (q['times'][0] - query_dispatch_ns) / 1e6
    require(ttft > 0, 'nonpositive TTFT')
    return {'ttft_ms': ttft, 'reference_pre_ms': statistics.median(intervals)}


def prepare(records, *, allow_synthetic=False):
    require(bool(records), 'no trials')
    ordered = sorted(records, key=lambda x: x['query_dispatch_ns'])
    ids, block_splits, observed = set(), {}, []
    for r in ordered:
        require(r.get('schema') == SCHEMA, 'not an H7b trial; legacy data cannot be relabeled')
        require(r['trial_id'] not in ids, 'duplicate trial ID')
        ids.add(r['trial_id'])
        require(r['source_kind'] in ('synthetic', 'serving'), 'unknown provenance')
        require(allow_synthetic or r['source_kind'] != 'synthetic', 'synthetic opt-in required')
        require(r['split'] in ('calibration', 'test'), 'invalid split')
        require(r['namespace'] in ('shared', 'isolated'), 'invalid namespace')
        b = r['time_block']
        require(block_splits.get(b, r['split']) == r['split'], 'block crosses train/test')
        block_splits[b] = r['split']
        require(r['split'] != 'calibration' or r['namespace'] == 'shared',
                'calibration must use the declared shared setting')
        public = public_timing(r['reference_events'], r['query_events'],
                               r['query_dispatch_ns'], r['trigger_receipt_ns'])
        q = checked_stream(r['query_events'])
        d = checked_stream(r['reference_events'])
        require(len(q['ids']) == 1 and len(d['ids']) == 32, 'wrong output lengths')
        require(q['usage']['prompt_tokens'] == 512 and d['usage']['prompt_tokens'] == 192,
                'wrong prompt lengths')
        bit = r['evaluator']['secret_bit']
        require(type(bit) is int and bit in (0, 1), 'invalid secret bit')
        details = q['usage'].get('prompt_tokens_details', {})
        require('cached_tokens' in details, 'missing evaluator cache ground truth')
        cached = details['cached_tokens']
        require(type(cached) is int and cached in (0, 496), 'unexpected cached-token count')
        expected = 496 if bit and r['namespace'] == 'shared' else 0
        # Retain mismatched manipulations as outcomes; do not turn them into clean negatives.
        complete_ns = r['query_events'][-1]['receive_perf_counter_ns']
        dispatch_ns = r['query_dispatch_ns']
        prior = sorted((v for v in observed if v['complete_ns'] < dispatch_ns),
                       key=lambda v: v['complete_ns'])
        public['past_ttft_median_ms'] = (
            statistics.median(v['public']['ttft_ms'] for v in prior[-4:]) if prior else None)
        observed.append(dict(trial_id=r['trial_id'], block=b, split=r['split'],
                             namespace=r['namespace'], source_kind=r['source_kind'],
                             dispatch_ns=dispatch_ns, complete_ns=complete_ns,
                             public=public, label=bit, reuse_hit=int(cached > 0),
                             manipulation_ok=cached == expected))
    calibration = [v for v in observed if v['split'] == 'calibration']
    test = [v for v in observed if v['split'] == 'test']
    require(len(calibration) >= 16 and bool(test), 'insufficient calibration or absent test')
    require(len({v['block'] for v in calibration}) >= 2 and
            len({v['block'] for v in test}) >= 2, 'need separate time blocks')
    require(max(v['complete_ns'] for v in calibration) < min(v['dispatch_ns'] for v in test),
            'calibration not complete before test')
    require({v['label'] for v in calibration} == {0, 1}, 'single-class calibration')
    # A missing first history value uses a calibration-only constant, never test data.
    fallback = statistics.median(v['public']['ttft_ms'] for v in calibration)
    for v in observed:
        if v['public']['past_ttft_median_ms'] is None:
            v['public']['past_ttft_median_ms'] = fallback
    return observed


def kde_overlap(misses, hits, bandwidth):
    """Same frozen bandwidth, Gaussian KDE, fixed numeric integration rule."""
    require(bandwidth > 0 and math.isfinite(bandwidth), 'invalid KDE bandwidth')
    if min(len(misses), len(hits)) < 4:
        return None
    values = np.asarray(misses + hits, dtype=float)
    require(bool(np.isfinite(values).all()), 'invalid KDE sample')
    # A single grid across the full range undersamples narrow modes when one
    # request has a remote tail latency. Resolve every kernel locally instead.
    centers = np.unique(values)
    grid = np.unique((centers[:, None] + bandwidth *
                      np.linspace(-8, 8, 129)[None, :]).ravel())
    def density(values):
        z = (grid[:, None] - np.asarray(values)[None, :]) / bandwidth
        return np.exp(-.5 * z * z).mean(axis=1) / (bandwidth * math.sqrt(2 * math.pi))
    value = np.trapezoid(np.minimum(density(misses), density(hits)), grid)
    return float(np.clip(value, 0, 1))


def causal_gates(rows, bandwidth, windows=WINDOWS):
    result = {}
    for row in rows:
        # Strictly earlier completed measurements; no current/future query in its gate.
        prior = sorted((p for p in rows if p['complete_ns'] < row['dispatch_ns']),
                       key=lambda p: p['complete_ns'])
        result[row['trial_id']] = {}
        for w in windows:
            pool = prior[-w:]
            overlap = None
            if len(pool) == w:
                overlap = kde_overlap([p['public']['ttft_ms'] for p in pool if not p['reuse_hit']],
                                      [p['public']['ttft_ms'] for p in pool if p['reuse_hit']], bandwidth)
            result[row['trial_id']][str(w)] = overlap
    return result


def matrix(rows, fields):
    return np.asarray([[math.log1p(r['public'][f]) for f in fields] for r in rows])


def auc(rows, scores):
    if len(rows) < 4 or len({r['label'] for r in rows}) < 2:
        return None
    return float(roc_auc_score([r['label'] for r in rows], scores))


def summary(rows, scores):
    return {'n': len(rows), 'n_positive': sum(r['label'] for r in rows),
            'n_blocks': len({r['block'] for r in rows}), 'auc': auc(rows, scores)}


def analyze(records, *, allow_synthetic=False):
    rows = prepare(records, allow_synthetic=allow_synthetic)
    cal = [r for r in rows if r['split'] == 'calibration']
    test = [r for r in rows if r['split'] == 'test']
    times = [r['public']['ttft_ms'] for r in cal]
    bandwidth = max(1e-6, 1.06 * statistics.stdev(times) * len(times) ** -.2)
    cutoff = float(np.quantile([r['public']['reference_pre_ms'] for r in cal], .25))
    gates = causal_gates(rows, bandwidth)
    predictions = {'ttft_rank': [-r['public']['ttft_ms'] for r in test]}
    models = {}
    for name, fields in FEATURES.items():
        model = GradientBoostingClassifier(n_estimators=50, max_depth=2,
                                           min_samples_leaf=3, learning_rate=.05,
                                           random_state=20261017)
        model.fit(matrix(cal, fields), [r['label'] for r in cal])
        predictions[name] = model.predict_proba(matrix(test, fields))[:, 1].tolist()
        models[name] = model
    # Deterministic, label-blind permutation within test block AND namespace.
    shuffled = [dict(r, public=dict(r['public'])) for r in test]
    groups = defaultdict(list)
    for i, r in enumerate(test):
        groups[(r['block'], r['namespace'])].append(i)
    rng = np.random.default_rng(20261017)
    for indices in groups.values():
        values = [test[i]['public']['reference_pre_ms'] for i in indices]
        for i, value in zip(indices, rng.permutation(values)):
            shuffled[i]['public']['reference_pre_ms'] = float(value)
    predictions['joint_reference_permuted'] = models['joint_learned'].predict_proba(
        matrix(shuffled, FEATURES['joint_learned']))[:, 1].tolist()
    views = {}
    for namespace in ('shared', 'isolated'):
        indices = [i for i, r in enumerate(test) if r['namespace'] == namespace]
        selected = [i for i in indices if test[i]['public']['reference_pre_ms'] <= cutoff]
        permuted_selected = [i for i in indices if
                             shuffled[i]['public']['reference_pre_ms'] <= cutoff]
        views[namespace] = {}
        for tag, ix in [('all', indices), ('preselected', selected)]:
            view = {name: summary([test[i] for i in ix], [scores[i] for i in ix])
                    for name, scores in predictions.items()}
            view['selection_fraction'] = len(ix) / len(indices) if indices else None
            views[namespace][tag] = view
        # This negative control permutes the *whole phase-selection rule*, not
        # just the joint model's reference feature on the original subset.
        views[namespace]['preselected_reference_permuted'] = {
            'joint_reference_permuted': summary(
                [test[i] for i in permuted_selected],
                [predictions['joint_reference_permuted'][i] for i in permuted_selected]),
            'selection_fraction': (len(permuted_selected) / len(indices)
                                   if indices else None),
            'selection_changed_n': len(set(selected) ^ set(permuted_selected)),
        }
        for w in WINDOWS:
            for theta in THETAS:
                ix = [i for i in selected if gates[test[i]['trial_id']][str(w)] is not None
                      and gates[test[i]['trial_id']][str(w)] > theta]
                views[namespace][f'preselected_off_w{w}_theta{theta}'] = {
                    name: summary([test[i] for i in ix], [scores[i] for i in ix])
                    for name, scores in predictions.items()}
    has_synthetic = any(r['source_kind'] == 'synthetic' for r in rows)
    return dict(schema='dos-h7b-analysis-v1',
                evidence_kind='synthetic_plumbing_only' if has_synthetic else 'unreplicated_serving_analysis',
                hypothesis_demonstrated=False, submission_ready=False,
                n_calibration=len(cal), n_test=len(test), feature_fields=FEATURES,
                bandwidth_ms=bandwidth, preselection_cutoff_ms=cutoff,
                manipulation_failures=[r['trial_id'] for r in rows if not r['manipulation_ok']],
                views=views,
                per_trial=[dict(trial_id=r['trial_id'], namespace=r['namespace'],
                                manipulation_ok=r['manipulation_ok'],
                                preselected=r['public']['reference_pre_ms'] <= cutoff,
                                preselected_reference_permuted=(
                                    shuffled[i]['public']['reference_pre_ms'] <= cutoff),
                                public=r['public'], gate_overlaps=gates[r['trial_id']],
                                predictions={name: values[i] for name, values in predictions.items()})
                           for i, r in enumerate(test)],
                limitations=['Feasibility only; no InputSnatch replication or tuned baseline.',
                             'No causal or leakage confirmation from AUC point estimates.',
                             'Small selected groups may be uninformative; no uncertainty study yet.',
                             'Shadow KDE rule is not an implementation of PrefixWall flags.',
                             'Capture contract consistency does not authenticate server/seed provenance.',
                             'All completed outcomes retained; manipulation failures block a clean claim.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-synthetic', action='store_true')
    args = p.parse_args()
    data = args.input.read_bytes()
    result = analyze([json.loads(line) for line in data.decode().splitlines() if line.strip()],
                     allow_synthetic=args.allow_synthetic)
    result['input_sha256'] = hashlib.sha256(data).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('evidence_kind', 'n_calibration', 'n_test',
                                           'hypothesis_demonstrated', 'submission_ready')}))


if __name__ == '__main__':
    main()
