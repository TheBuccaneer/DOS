#!/usr/bin/env python3
"""Read-only historical integrity audit; invalid files stay in the report."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from trace_contract import check_stream, request_identities

CAMPAIGNS = (
    'chunk_budget_screen/results/official_20260716T075455Z',
    'server_waiting_confirmation/llama/official_campaign_20260804T141510Z',
    'server_waiting_confirmation/qwen/official_campaign_fixed_20260803T183032Z',
)
METRIC = re.compile(r'^(vllm:[a-zA-Z_:0-9]+)', re.M)


def audit_campaign(root, relative):
    base = root / 'new/runs' / relative
    files = sorted((base / 'episodes').glob('*.json'))
    errors, groups, metrics = [], defaultdict(list), set()
    statuses, quality, orders = Counter(), Counter(), Counter()
    requests = tokens = samples = after = matched = mismatch = differing = 0
    seen_ids = set()
    if not files:
        errors.append({'error': 'no_episodes', 'path': relative})
    for file in files:
        try:
            e = json.loads(file.read_text())
            if not isinstance(e, dict):
                raise ValueError('episode_not_object')
            schedule = e.get('schedule_row')
            if not isinstance(schedule, dict) or schedule.get('condition') not in ('no_burst', 'prefill_burst'):
                raise ValueError('invalid_condition_or_schedule')
            for key in ('episode_id', 'block_id'):
                if not isinstance(e.get(key), str) or not e[key]:
                    raise ValueError('missing_' + key)
            if e['episode_id'] in seen_ids:
                raise ValueError('duplicate_episode_id')
            seen_ids.add(e['episode_id'])
            order = schedule.get('within_block_order', schedule.get('order_in_block'))
            if type(order) is not int or order not in (1, 2):
                raise ValueError('invalid_pair_order')
            request_identities(e)
            statuses[str(e.get('status'))] += 1
            if e.get('status') != 'complete':
                errors.append({'file': file.name, 'error': 'incomplete_episode'})
            groups[e['block_id']].append((e, order))
            for role in ('victim_requests', 'burst_requests'):
                rows = e.get(role)
                if not isinstance(rows, list):
                    raise ValueError('missing_' + role)
                indices = set()
                for record in rows:
                    if not isinstance(record, dict):
                        raise ValueError('invalid_request_record')
                    index = record.get('request_index')
                    if type(index) is not int or index < 0 or index in indices:
                        raise ValueError('invalid_or_duplicate_' + role + '_index')
                    indices.add(index)
                    requests += 1
                    output = record.get('output_token_ids')
                    tokens += len(output) if isinstance(output, list) else 0
                    issues = check_stream(record)
                    if issues:
                        errors.append({'file': file.name, 'role': role, 'index': index, 'errors': issues})
            trigger = e.get('trigger', {})
            if not isinstance(trigger, dict):
                raise ValueError('invalid_trigger')
            q = trigger.get('metrics_quality', {})
            quality[str(q.get('metrics_quality_status', 'absent')) if isinstance(q, dict) else 'invalid'] += 1
            trace = e.get('metrics_full_samples', trigger.get('metrics_raw_samples', []))
            if not isinstance(trace, list):
                raise ValueError('invalid_metrics_samples')
            for sample in trace:
                if not isinstance(sample, dict):
                    raise ValueError('invalid_metrics_sample')
                samples += 1
                received, origin = sample.get('response_received_perf_ns'), trigger.get('trigger_perf_ns')
                if type(received) is int and type(origin) is int and received > origin:
                    after += 1
                body = sample.get('raw_body')
                if isinstance(body, str):
                    metrics.update(METRIC.findall(body))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({'file': file.name, 'error': 'invalid_input', 'detail': str(exc)})
    for block, pair in groups.items():
        if len(pair) != 2 or {e['schedule_row']['condition'] for e, _ in pair} != {'no_burst', 'prefill_burst'} or {order for _, order in pair} != {1, 2}:
            errors.append({'block': block, 'error': 'invalid_pair'})
            continue
        orders[min(pair, key=lambda item: item[1])[0]['schedule_row']['condition']] += 1
        if request_identities(pair[0][0]) == request_identities(pair[1][0]):
            matched += 1
        else:
            mismatch += 1
            errors.append({'block': block, 'error': 'pair_request_identity_mismatch'})
        cohorts = [e.get('trigger', {}).get('active_cohort_request_indices') for e, _ in pair]
        if all(isinstance(c, list) and all(type(i) is int for i in c) for c in cohorts):
            differing += set(cohorts[0]) != set(cohorts[1])
    manifest = base / 'official_run_manifest.json'
    obj = {}
    if manifest.exists():
        try:
            obj = json.loads(manifest.read_text())
            if not isinstance(obj, dict):
                raise ValueError('manifest_not_object')
        except (OSError, ValueError) as exc:
            errors.append({'file': manifest.name, 'error': 'invalid_manifest', 'detail': str(exc)})
            obj = {}
    return {'path': relative, 'episode_count': len(files), 'request_count': requests,
            'output_token_count': tokens, 'episode_statuses': dict(statuses),
            'matched_request_identity_pairs': matched, 'mismatched_request_identity_pairs': mismatch,
            'pair_first_conditions': dict(orders), 'pairs_with_different_observed_cohorts': differing,
            'stored_metrics_samples': samples, 'stored_samples_after_logical_trigger': after,
            'metrics_quality': dict(quality), 'metric_names': sorted(metrics),
            'historical_environment': {k: obj.get(k) for k in ('vllm_version', 'torch_version', 'python_version', 'git_commit')},
            'integrity_errors': errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--campaign', action='append', help='Relative below new/runs; repeatable')
    a = p.parse_args()
    root = a.root.resolve()
    if not root.is_dir():
        p.error('root must be a directory')
    campaigns = a.campaign or CAMPAIGNS
    if any(Path(c).is_absolute() or '..' in Path(c).parts for c in campaigns):
        p.error('campaign must be relative below new/runs')
    proc = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
    reports = [audit_campaign(root, c) for c in campaigns]
    output = {'created_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
              'mode': 'offline_read_only', 'new_experiment_gate': 'NOT_CONFIRMED',
              'git_head': proc.stdout.strip() if proc.returncode == 0 else None, 'campaigns': reports}
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return int(any(r['integrity_errors'] for r in reports))


if __name__ == '__main__':
    raise SystemExit(main())
