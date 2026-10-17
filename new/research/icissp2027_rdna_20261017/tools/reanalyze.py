"""Offline, independent reconstruction. Never imports a runner or contacts a server.

Run from this directory's parent: python3 tools/reanalyze.py --repo /path/to/DOS
Raw source files are read only. Output contains pointers and hashes, not copied prompts.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def stream(events):
    ids, times, sizes = [], [], []
    usage, done, errors = None, False, []
    for event in events:
        raw = event['raw_data']
        if raw == '[DONE]':
            done = True
            continue
        if done:
            errors.append('data_after_done')
        try:
            data = json.loads(raw)
        except ValueError:
            errors.append('malformed_json')
            continue
        if data.get('usage'):
            usage = data['usage']
        for c in data.get('choices', []):
            tokens = c.get('token_ids') or []
            ids.extend(tokens)
            times.extend([event['receive_perf_counter_ns']] * len(tokens))
            if tokens:
                sizes.append(len(tokens))
    if not done:
        errors.append('missing_done')
    if usage is None or usage.get('completion_tokens') != len(ids):
        errors.append('usage_token_mismatch')
    return dict(ids=ids, times=times, batch_sizes=sizes, usage=usage,
                done=done, errors=errors)


def metrics(text, name):
    values = [float(m.group(1)) for m in re.finditer(
        r'^' + re.escape(name) + r'(?:\{[^\n]*\})?\s+([\d.eE+\-]+)(?:\s|$)',
        text, re.M)]
    if not values:
        raise ValueError('missing metric: ' + name)
    return sum(values)


def analyze_episode(rows):
    final = rows[-1]
    e = final['episode']
    plan = e['plan']
    errors = []
    requests = {}
    all_events = 0
    for role, request in e['requests'].items():
        wire = [r for r in rows if r['record_type'] == 'sse_wire' and r['role'] == role]
        all_events += len(wire)
        s = stream(wire)
        errors.extend(role + ':' + x for x in s['errors'])
        if s['ids'] != request['output_token_ids']:
            errors.append(role + ':token_copy_mismatch')
        if len(s['ids']) != plan['outputs'][role]:
            errors.append(role + ':unexpected_output_length')
        if [(x['raw_data'], x['receive_perf_counter_ns']) for x in wire] != [
                (x['raw_data'], x['receive_perf_counter_ns'])
                for x in request['raw_sse_events']]:
            errors.append(role + ':event_copy_mismatch')
        if request['prompt_token_ids_sent'] != plan['prompts'][role]:
            errors.append(role + ':prompt_mismatch')
        usage = s['usage'] or {}
        cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        requests[role] = dict(tokens=len(s['ids']), events=len(wire),
                              cached_tokens=cached, prompt_tokens=usage.get('prompt_tokens'),
                              batch_sizes=s['batch_sizes'], times=s['times'])
    d = requests['D']
    exact = len(d['times']) >= 28 and all(x == 1 for x in d['batch_sizes'])
    endpoint = (d['times'][27] - d['times'][7]) / 1e6 if exact else None
    triggers = e['triggers']
    if not exact or len(triggers) != 1 or triggers[0]['count'] != 8 or (
            triggers[0]['trigger_perf_ns'] != d['times'][7]):
        errors.append('trigger_or_endpoint_not_exact')
    expected_hits = 112 if plan['condition'] == 'shared' else 0
    if d['cached_tokens'] != 0:
        errors.append('D_not_cold')
    if 'B' in requests and requests['B']['cached_tokens'] != expected_hits:
        errors.append('unexpected_B_cache')
    if endpoint is not None and abs(endpoint - final['assessment']['raw_endpoint_ms']) > 1e-8:
        errors.append('endpoint_copy_mismatch')
    samples = e['metrics_full_samples']
    preemptions = (metrics(samples[-1]['raw_body'], 'vllm:num_preemptions_total') -
                   metrics(samples[0]['raw_body'], 'vllm:num_preemptions_total'))
    trajectory = []
    for m in re.finditer(r'Iteration\((\d+)\): (\d+) context requests, (\d+) context tokens, '
                         r'(\d+) generation requests, (\d+) generation tokens', e['iteration_log']):
        trajectory.append(tuple(map(int, m.groups())))
    concurrent = sum(1 for _, cr, ct, gr, gt in trajectory if gr == 2 or (cr == 1 and gr == 1))
    context = sum(x[2] for x in trajectory)
    generation = sum(x[4] for x in trajectory)
    output_after_first = sum(q['tokens'] - 1 for q in requests.values())
    nominal = sum(q['prompt_tokens'] - q['cached_tokens'] for q in requests.values())
    # This is an accounting discrepancy, not proof of which request was recomputed.
    excess_context = context - nominal
    for q in requests.values():
        del q['times']
    return dict(condition=plan['condition'], **plan['config'],
                requests=requests, wire_events=all_events, endpoint_ms=endpoint,
                independent_integrity_errors=errors, preemptions=preemptions,
                iteration_context_tokens=context, initial_uncached_tokens=nominal,
                excess_context_tokens=excess_context, concurrent_iterations=concurrent,
                iteration_generation_tokens=generation,
                new_output_tokens_after_first=output_after_first,
                generation_accounting_residual=generation - output_after_first,
                nonunit_generation_iterations=[dict(iteration=x[0], requests=x[3], tokens=x[4])
                                               for x in trajectory if x[3] and x[3] != x[4]],
                runner_valid=final['assessment']['valid_episode'],
                runner_issues=final['assessment']['issues'],
                runner_mechanism_status=final['assessment']['mechanism_status'],
                measurement_complete=not errors,
                mechanism_attribution='unresolved' if preemptions else 'no_preemption_observed',
                causal_effect_estimated=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results, evictions = [], []
    for p in sorted((args.repo / 'new/runs/cache_isolation_gate').glob('overlap*.jsonl')):
        rows = [json.loads(s) for s in p.open() if s.strip()]
        a = analyze_episode(rows)
        a.update(file=str(p.relative_to(args.repo)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        results.append(a)
    for p in sorted((args.repo / 'new/runs/cache_isolation_gate').glob('eviction*.jsonl')):
        rows = [json.loads(s) for s in p.open() if s.strip()]
        last = rows[-1]
        evictions.append(dict(file=str(p.relative_to(args.repo)),
                              sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                              assessments=last['assessments'], status=last['status'],
                              verification='stored assessments only; not independent request reconstruction'))
    out = dict(episodes=results, evictions=evictions,
               totals=dict(episodes=len(results), requests=sum(len(x['requests']) for x in results),
                           tokens=sum(q['tokens'] for x in results for q in x['requests'].values()),
                           wire_events=sum(x['wire_events'] for x in results),
                           integrity_errors=sum(len(x['independent_integrity_errors']) for x in results)))
    (root / 'evidence/reanalysis.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out['totals']))
    for a in results:
        print(a['blocks'], a['budget'], a['condition'], round(a['endpoint_ms'], 3),
              a['preemptions'], a['excess_context_tokens'], a['independent_integrity_errors'])


if __name__ == '__main__':
    main()
