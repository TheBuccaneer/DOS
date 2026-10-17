"""Offline gates for one fixed D/B overlap episode; no network or H1 claims."""
import ast
import hashlib
import json
import re

from cache_eviction_control import (_one_series, prompt_hash, snapshot,
                                    verify_config, verify_server_identity)
from extract_primary_endpoint import receipt_at_token
from parse_iteration_logs import ANSI, parse_text
from summarize_metrics import parse_metrics
from trace_contract import check_stream, request_identities
from cache_smoke import BASE_SHA256

CONDITIONS = ('victim-only', 'shared', 'isolated', 'no-prefix-shared', 'no-prefix-isolated')
SEED = 20261017


def _config_fields(text, *, dictionary=False):
    """Split repr fields without evaluating vLLM objects or enum reprs."""
    parts, start, stack, quote, escaped = [], 0, [], None, False
    pairs = {')': '(', ']': '[', '}': '{'}
    for i, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            quote = char
        elif char in '([{':
            stack.append(char)
        elif char in ')]}':
            _require(bool(stack) and stack.pop() == pairs[char], 'unbalanced engine config')
        elif char == ',' and not stack:
            parts.append(text[start:i].strip())
            start = i + 1
    _require(not quote and not stack, 'incomplete engine config')
    parts.append(text[start:].strip())
    fields = {}
    for part in filter(None, parts):
        key, separator, value = part.partition(':' if dictionary else '=')
        _require(bool(separator), 'unsupported engine config field')
        key = ast.literal_eval(key.strip()) if dictionary else key.strip()
        _require(isinstance(key, str) and re.fullmatch(r'[A-Za-z_]\w*', key)
                 and key not in fields, 'invalid/duplicate engine config field')
        fields[key] = value.strip()
    return fields


def normalize_engine_config(text, config):
    """Mask only checked factor values and the v0.17.1 compile-range endpoint.

    Preserve unknown fields verbatim. Never replace a number globally: it may
    also be a seed, capture size or unrelated model setting.
    """
    def normalize(fields, section):
        result = dict(fields)
        factors = {}
        if section in ('engine', 'scheduler_config'):
            factors['max_num_batched_tokens'] = ('budget', config['budget'])
        if section in ('engine', 'cache_config'):
            factors.update(num_gpu_blocks_override=('blocks', config['blocks']),
                           num_gpu_blocks=('blocks', config['blocks']))
        for key, (factor, expected) in factors.items():
            if key in fields:
                actual = ast.literal_eval(fields[key])
                # EngineCore logs before KV initialization (v0.17.1 core.py:101,
                # 120-124). Only this resolved field may still be unset here;
                # the override and later startup/live block checks stay strict.
                uninitialized = key == 'num_gpu_blocks' and actual is None
                _require(uninitialized or (type(actual) is int and actual == expected),
                         'engine factor differs from declared ' + factor)
                result[key] = {'factor': factor}
        if section == 'compilation_config' and 'compile_ranges_split_points' in fields:
            points = ast.literal_eval(fields['compile_ranges_split_points'])
            _require(isinstance(points, list) and points and all(type(x) is int for x in points)
                     and points[-1] == config['budget'], 'compile-range endpoint differs from budget')
            # vllm/config/vllm.py:1492 sets the last boundary to the budget.
            # Other explicit/intermediate split points must still match.
            result['compile_ranges_split_points'] = [*points[:-1], {'factor': 'budget'}]
        if section == 'engine':
            for name, constructor in (('scheduler_config', 'SchedulerConfig'),
                                      ('cache_config', 'CacheConfig'),
                                      ('compilation_config', 'CompilationConfig')):
                if name not in fields:
                    continue
                value = fields[name]
                if value.startswith('{') and value.endswith('}'):
                    nested, representation = _config_fields(value[1:-1], dictionary=True), 'dict'
                else:
                    _require(value.startswith(constructor + '(') and value.endswith(')'),
                             'unsupported ' + name + ' representation')
                    nested = _config_fields(value[len(constructor) + 1:-1])
                    representation = constructor
                result[name] = {'representation': representation, 'fields': normalize(nested, name)}
        return result
    try:
        return normalize(_config_fields(text), 'engine')
    except (SyntaxError, TypeError) as exc:
        raise ValueError('unsupported engine config value') from exc


def nuisance_from_log(raw, config):
    clean = ANSI.sub('', raw)
    engines = re.findall(r'Initializing a V1 LLM engine .*? with config: (.*)', clean)
    api_args = re.findall(r'non-default args: (.*)', clean)
    _require(len(engines) == len(api_args) == 1 and 'speculative_config=None' in engines[0],
             'unique resolved engine/API config required; speculative decoding excluded')
    try:
        args = ast.literal_eval(api_args[0])
    except (SyntaxError, ValueError) as exc:
        raise ValueError('unsupported API args in startup log') from exc
    _require(isinstance(args, dict), 'API args must be a mapping')
    required = {'enable_prompt_tokens_details': True, 'enable_logging_iteration_details': True,
                'enable_chunked_prefill': True, 'enable_prefix_caching': True,
                'max_num_seqs': 16, 'seed': 20260711, 'max_model_len': 256,
                'max_num_batched_tokens': config['budget'], 'num_gpu_blocks_override': config['blocks']}
    _require(all(args.get(k) == v for k, v in required.items()), 'effective API configuration differs')
    # Only explicit manipulation may differ. Preserve all other
    # effective API/engine settings, including scheduler and compilation options.
    normalized = {k: v for k, v in args.items() if k not in (
        'max_num_batched_tokens', 'num_gpu_blocks_override')}
    return {'engine_config': normalize_engine_config(engines[0], config), 'api_nondefault_args': normalized,
            'max_model_len': 256, 'cpu_offload_gb': 0, 'block_size': 16,
            'sampling': {'temperature': 0, 'ignore_eos': True},
            'metrics_interval_s': config['metrics_interval_s']}


def make_plan(tokens, condition, episode_id, nonce, config):
    from cache_eviction_control import build_prompts
    if condition not in CONDITIONS:
        raise ValueError('unknown overlap condition')
    variants = build_prompts(tokens)
    d = list(tokens) + list(tokens[:64])
    b = variants['B1'] if condition.startswith('no-prefix-') else list(tokens)
    salt_d = 'icissp-overlap-' + nonce + '-D'
    salt_b = salt_d if condition.endswith('shared') and not condition.endswith('isolated') else salt_d + '-B'
    return {'schema_version': 'dos-overlap-pilot-v1', 'episode_id': episode_id,
            'condition': condition, 'nonce': nonce, 'cohort_indices': [0],
            'trigger_token': 8, 'endpoint_token': 28, 'seed': SEED,
            'prompts': {'D': d, 'B': b}, 'outputs': {'D': 32, 'B': 64},
            'prompt_sha256': {'D': prompt_hash(d), 'B': prompt_hash(b)},
            'salts': {'D': salt_d, 'B': salt_b},
            'request_ids': {role: episode_id + ':' + role for role in ('D', 'B')},
            'config': config,
            'victim_identity': {'prompt_sha256': prompt_hash(d),
                                'prompt_token_ids_sent': d, 'generation_seed': SEED,
                                'expected_prompt_tokens': 192, 'expected_completion_tokens': 32}}


def expected_hits(plan):
    return 112 if plan['condition'] == 'shared' else 0


def _require(test, message):
    if not test:
        raise ValueError(message)


def _progress(record):
    count, result = 0, []
    for event in record['raw_sse_events']:
        if event['token_ids']:
            count += len(event['token_ids'])
            result.append({'count': count, 'receive_perf_counter_ns': event['receive_perf_counter_ns']})
    return result


def classify_report(report):
    """An unresolved preemption candidate is neither an accepted nor a falsified H1."""
    gates, obs = report['gates'], report['observations']
    complete = gates['technical_complete'] and gates['trigger_exact']
    preemptions = obs.get('preemptions')
    evidence_bound = gates['server_bound'] and gates['metrics_complete']
    candidate = evidence_bound and gates['traffic_attributed'] and preemptions is not None and preemptions > 0
    report.update(technical_sse_measurement_complete=bool(complete), mechanism_candidate=bool(candidate),
                  causal_comparison_eligible=bool(report['valid_episode']), h1_status='not_evaluated')
    if preemptions is not None and preemptions > 0:
        report['mechanism_status'] = (
            'preemption_candidate_insufficient_attribution' if candidate else
            'unattributed_preemption_foreign_or_ambiguous_traffic' if evidence_bound and not gates['traffic_attributed']
            else 'unattributed_preemption_counter_change')
    else:
        report['mechanism_status'] = 'no_preemption_observed' if evidence_bound else 'unavailable'
    report['causal_status'] = ('eligible_for_controlled_comparison' if report['valid_episode'] else
                               'insufficient_mechanism_attribution' if candidate else 'ineligible_control_or_data_gates')
    if not complete:
        report['raw_endpoint_ms'] = None
    return report


def assessment_exit_code(reports):
    """0 accepted episodes; 3 retained candidates; 1 other incomplete/invalid data."""
    if any(not r['technical_sse_measurement_complete']
           or (not r['valid_episode'] and not r['mechanism_candidate']) for r in reports):
        return 1
    return 3 if any(r['mechanism_candidate'] for r in reports) else 0


def assess_episode(ep):
    """Recompute independent gates from raw records, retaining every failure."""
    gates = dict.fromkeys(('technical_complete', 'trigger_exact', 'server_bound',
                          'metrics_complete', 'traffic_attributed', 'cache_manipulation',
                          'iteration_accounting', 'effective_budget', 'budget_binding_observed',
                          'overlap_confirmed', 'victim_only_capacity_valid'), False)
    report = {'gates': gates, 'issues': [], 'observations': {}, 'endpoint_ms': None, 'raw_endpoint_ms': None,
              'valid_episode': False, 'h1_validated': False, 'novelty_validated': False}
    obs = report['observations']

    def gate(name, fn):
        try:
            fn()
            gates[name] = True
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            report['issues'].append(name + ': ' + str(exc))

    try:
        plan, requests, samples = ep['plan'], ep['requests'], ep['metrics_full_samples']
        config = plan['config']
        baseline = plan['condition'] == 'victim-only'
        roles = ('D',) if baseline else ('D', 'B')
        d = requests['D']
        report.update(episode_id=plan['episode_id'], condition=plan['condition'], config=config)
    except (ValueError, KeyError, TypeError) as exc:
        report['issues'].append('missing_or_invalid_raw_data: ' + str(exc))
        return classify_report(report)

    def complete():
        _require(plan['condition'] in CONDITIONS, 'unknown condition')
        _require(ep['historical_executor_sha256'] == BASE_SHA256, 'historical executor provenance mismatch')
        _require(plan['outputs'] == {'D': 32, 'B': 64}, 'output workload changed')
        _require(plan['cohort_indices'] == [0] and plan['trigger_token'] == 8
                 and plan['endpoint_token'] == 28, 'fixed cohort/endpoint changed')
        _require(plan['seed'] == SEED, 'seed changed')
        _require(len(plan['prompts']['D']) == 192 and len(plan['prompts']['B']) == 128,
                 'prompt lengths changed')
        common = plan['prompts']['D'][:128] == plan['prompts']['B']
        _require(common == (not plan['condition'].startswith('no-prefix-')), 'prefix condition changed')
        if not common:
            _require(plan['prompts']['D'][:16] != plan['prompts']['B'][:16], 'first block not different')
        for role in roles:
            r = requests[role]
            _require(not check_stream(r), role + ': ' + ','.join(check_stream(r)))
            _require(not r.get('validation_errors') and r.get('done_received') is True
                     and r.get('finish_reason') == 'length', role + ': incomplete SSE protocol')
            _require(r['request_id'] == plan['request_ids'][role], 'request ID mismatch')
            _require(r['prompt_token_ids_sent'] == plan['prompts'][role]
                     and r['prompt_token_ids_returned'] == plan['prompts'][role]
                     and r['prompt_sha256'] == plan['prompt_sha256'][role]
                     == prompt_hash(plan['prompts'][role]), 'prompt identity mismatch')
            _require(r['expected_completion_tokens'] == plan['outputs'][role]
                     and r['expected_prompt_tokens'] == len(plan['prompts'][role])
                     and r['generation_seed'] == plan['seed'], 'workload changed')
            _require(r['usage']['prompt_tokens'] == len(plan['prompts'][role])
                     and r['usage']['completion_tokens'] == plan['outputs'][role], 'SSE usage mismatch')
            _require(r['wire_payload']['cache_salt'] == plan['salts'][role]
                     and r['wire_payload']['prompt'] == plan['prompts'][role], 'wrong wire salt/prompt')
            _require(r['wire_payload']['max_tokens'] == r['wire_payload']['min_tokens'] == plan['outputs'][role]
                     and r['wire_payload']['seed'] == plan['seed'], 'wrong wire generation settings')
            _require(r['wire_payload']['temperature'] == 0 and r['wire_payload']['ignore_eos'] is True
                     and r['wire_payload']['stream'] is True
                     and r['wire_payload']['add_special_tokens'] is False
                     and r['wire_payload']['model'] == config['model'], 'wrong wire sampling/model')
        same_salt = plan['salts']['D'] == plan['salts']['B']
        _require(same_salt == (plan['condition'] in ('shared', 'no-prefix-shared')), 'salt assignment changed')
        ids = request_identities({'victim_requests': [d]})
        _require(ids == {0: plan['victim_identity']}, 'victim identity differs from predeclared cohort')
        _require(ep['cohort_declared_perf_ns'] < d['request_dispatch_ns'], 'cohort declared too late')
        _require(not ep.get('errors'), 'episode execution error')
    gate('technical_complete', complete)

    def trigger():
        t8, t28 = receipt_at_token(d, 8), receipt_at_token(d, 28)
        _require(type(t8) is int and type(t28) is int and t8 < t28, 'missing/degenerate token interval')
        progress = _progress(d)
        _require(ep['token_progress'] == progress, 'early/late/duplicate progress callbacks')
        crossings = [p for p in progress if p['count'] >= 8]
        _require(crossings and crossings[0]['count'] == 8, 'token 8 crossed inside a larger SSE batch')
        _require(len(ep['triggers']) == 1, 'missing or multiple triggers')
        tr = ep['triggers'][0]
        _require(tr['count'] == 8 and tr['trigger_perf_ns'] == t8
                 and tr['observed_perf_ns'] >= t8, 'trigger does not match raw token 8')
        if not baseline:
            _require(t8 <= requests['B']['task_created_ns'] <= requests['B']['request_dispatch_ns'] < t28,
                     'B dispatched outside D8..28')
        obs.update(token8_ns=t8, token28_ns=t28,
                   b_dispatch_delay_ms=(requests['B']['request_dispatch_ns'] - t8) / 1e6 if not baseline else None)
        report['endpoint_ms'] = (t28 - t8) / 1e6
        report['raw_endpoint_ms'] = report['endpoint_ms']
    gate('trigger_exact', trigger)

    try:
        states = [snapshot(s) for s in samples]
        _require(bool(states), 'no metric samples')
    except (ValueError, KeyError, TypeError) as exc:
        report['issues'].append('metrics_complete: ' + str(exc))
        report['endpoint_ms'] = None
        return classify_report(report)

    def bound():
        raw = ep['startup_log']
        _require(hashlib.sha256(raw.encode()).hexdigest() == ep['startup_log_sha256'], 'startup log hash mismatch')
        pids = set(re.findall(r'\(APIServer pid=(\d+)\)', raw))
        _require(pids == {str(ep['server_identity']['api_pid'])}, 'startup PID mismatch')
        _require(ep['server_identity']['boot_id'] == ep['clock_domain']['boot_id'], 'host boot changed')
        _require(ep['server_identity']['log_start_seconds'] - 5 <= ep['server_identity']['process_start_seconds']
                 <= ep['server_identity']['log_start_seconds'] + 300, 'startup time mismatch')
        _require("'enable_prompt_tokens_details': True" in raw, 'effective request cache details missing')
        _require('speculative_config=None' in raw, 'speculative decoding not excluded')
        nuisance = nuisance_from_log(raw, config)
        _require(ep['nuisance_config'] == nuisance and config['nuisance_sha256'] == hashlib.sha256(
            json.dumps(nuisance, sort_keys=True).encode()).hexdigest(), 'nuisance config fingerprint mismatch')
        _require(f'Max model len requested: {config["max_model_len"]}' in raw
                 and f'NUM_GPU_BLOCKS_OVERRIDE requested: {config["blocks"]}' in raw
                 and 'vLLM-Version: 0.17.1' in raw, 'startup/config provenance mismatch')
        for state in states:
            _require(state['cache_config'] == states[0]['cache_config'], 'cache config changed')
            verify_server_identity(state, ep['server_identity']['process_start_seconds'])
        verify_config(states[0], config['blocks'], 16)
        verify_config(states[-1], config['blocks'], 16)
    gate('server_bound', bound)

    def metrics():
        _require(ep['metrics_capture']['stop_error'] is None and ep['metrics_capture']['sampler_available'],
                 'metrics capture not closed')
        _require(len(samples) >= 3 and all(not s.get('error') for s in samples), 'missing/failed metrics')
        previous = -1
        for sample in samples:
            start, end = sample['scrape_start_perf_ns'], sample['response_received_perf_ns']
            _require(type(start) is int and type(end) is int and previous <= start <= end,
                     'unordered or invalid scrape bounds')
            previous = end
        _require(samples[0]['response_received_perf_ns'] <= d['request_dispatch_ns'], 'no pre-request metrics')
        last_end = max(requests[role]['stream_end_ns'] for role in roles)
        _require(samples[-1]['scrape_start_perf_ns'] >= last_end, 'missing full post-request metrics')
        t8, t28 = obs['token8_ns'], obs['token28_ns']
        _require(any(t8 <= s['scrape_start_perf_ns'] <= s['response_received_perf_ns'] <= t28
                     for s in samples), 'no complete scrape inside D8..28')
        for old, new in zip(states, states[1:]):
            _require(all(new['counters'][k] >= v for k, v in old['counters'].items()), 'counter reset')
        _require(all(all(v >= 0 and int(v) == v for v in s['counters'].values())
                     and all(v >= 0 and int(v) == v for v in s['gauges'].values())
                     for s in states), 'negative/fractional metrics')
        gaps = [b['response_received_perf_ns'] - a['response_received_perf_ns']
                for a, b in zip(samples, samples[1:])]
        obs['max_metrics_gap_ms'] = max(gaps) / 1e6
        _require(max(gaps) <= max(.25, ep['metrics_interval_s'] * 4) * 1e9, 'metrics coverage gap')
    gate('metrics_complete', metrics)

    delta = {k: states[-1]['counters'][k] - v for k, v in states[0]['counters'].items()}
    obs['counter_deltas'] = delta
    values = [parse_metrics(s['raw_body']) for s in samples]
    try:
        usage = [_one_series(v, 'vllm:kv_cache_usage_perc') for v in values]
        _require(all(0 <= v <= 1 for v in usage), 'invalid KV usage')
        obs['max_kv_usage_fraction'] = max(usage)
    except ValueError as exc:
        obs['max_kv_usage_fraction'] = None
        report['issues'].append('kv_usage_unavailable: ' + str(exc))
    obs['max_active_kv_blocks'] = (None if obs['max_kv_usage_fraction'] is None else
                                    obs['max_kv_usage_fraction'] * (config['blocks'] - 1))
    obs['max_waiting'] = max(s['gauges']['waiting'] for s in states)
    obs['preemptions'] = delta['preemptions']
    obs['capacity_pressure_observed'] = delta['preemptions'] > 0 or (
        obs['max_kv_usage_fraction'] is not None and obs['max_kv_usage_fraction'] >= .95)

    def traffic():
        prompt = 192 if baseline else 320
        _require(delta['queries'] == prompt and delta['prompt_tokens'] == prompt
                 and delta['prefill_requests'] == len(roles), 'foreign traffic or ambiguous recomputation counters')
        _require(all(s['gauges']['running'] + s['gauges']['waiting'] <= len(roles)
                     for s in states), 'unexpected additional requests')
    gate('traffic_attributed', traffic)

    def cache():
        _require(gates['technical_complete'] and gates['server_bound'] and gates['traffic_attributed'],
                 'no request-scoped metric attribution')
        hits = {}
        for role in roles:
            usage = requests[role]['usage']
            details = usage.get('prompt_tokens_details')
            # vLLM 0.17.1 omits details for zero hits even when enabled.
            value = 0 if details is None else details.get('cached_tokens')
            _require(type(value) is int and 0 <= value <= len(plan['prompts'][role]), 'invalid cached_tokens')
            hits[role] = value
        _require(hits['D'] == 0, 'D was not cold')
        if not baseline:
            _require(hits['B'] == expected_hits(plan), 'unexpected B cache hits')
        _require(delta['hits'] == sum(hits.values()), 'request/global cache counters disagree')
        computed = sum(len(plan['prompts'][r]) - hits[r] for r in roles)
        _require(delta['computed_context_tokens'] == computed, 'computed prefill counters disagree')
        obs['cache_hits_by_request'] = hits
        obs['new_prefill_tokens_by_request'] = {r: len(plan['prompts'][r]) - hits[r] for r in roles}
        _require(delta['preemptions'] == 0, 'cache attribution under preemption is unresolved')
    gate('cache_manipulation', cache)

    parsed = parse_text(ep.get('iteration_log', ''))
    rows = parsed['records']
    obs['scheduler_trajectory'] = rows
    obs['scheduler_trajectory_sha256'] = parsed['trajectory_sha256']
    def iterations():
        _require(not parsed['errors'], 'missing, malformed, gapped or multiple-engine iteration logs')
        _require(all(r['context_requests'] + r['generation_requests'] <= len(roles)
                     and r['generation_requests'] == r['generation_tokens'] for r in rows),
                 'ambiguous request count or non-unit decode')
        computed = sum(r['context_tokens'] for r in rows)
        obs['iteration_context_tokens'] = computed
        _require(delta['preemptions'] == 0, 'iteration attribution under preemption is unresolved')
        _require(computed == delta['computed_context_tokens'], 'iteration/metric prefill mismatch')
        _require(sum(r['generation_tokens'] for r in rows) == sum(plan['outputs'][r] - 1 for r in roles),
                 'missing/extra scheduler decode work')
        remaining, d_chunks, b_chunks, b_rows = 192, [], [], []
        for row in rows:
            if row['context_tokens']:
                if remaining:
                    remaining -= row['context_tokens']
                    _require(remaining >= 0, 'D/B context phases cannot be separated')
                    d_chunks.append(row['context_tokens'])
                else:
                    b_chunks.append(row['context_tokens'])
                    b_rows.append(row)
        _require(remaining == 0, 'D prefill incomplete')
        obs.update(d_prefill_chunks=d_chunks, b_prefill_chunks=b_chunks)
        obs['b_prefill_budget_binding'] = any(
            r['context_tokens'] + r['generation_tokens'] == config['budget'] for r in b_rows)
    gate('iteration_accounting', iterations)

    def budget():
        raw = ep['startup_log']
        _require(f'Chunked prefill is enabled with max_num_batched_tokens={config["budget"]}.' in raw,
                 'effective budget absent from startup log')
        _require(gates['iteration_accounting'] and all(
            r['context_tokens'] + r['generation_tokens'] <= config['budget'] for r in rows),
            'iteration exceeds budget or work cannot be attributed')
    gate('effective_budget', budget)
    gates['budget_binding_observed'] = gates['effective_budget'] and (
        any(r['context_tokens'] == config['budget'] for r in rows) if baseline
        else obs.get('b_prefill_budget_binding', False))
    obs['prefill_chunks'] = [r['context_tokens'] for r in rows if r['context_tokens']]

    if baseline:
        gates['overlap_confirmed'] = None
        gates['victim_only_capacity_valid'] = (gates['technical_complete'] and gates['iteration_accounting']
                                              and delta['preemptions'] == 0 and obs['max_waiting'] == 0)
    else:
        def overlap():
            _require(gates['trigger_exact'] and gates['iteration_accounting']
                     and gates['traffic_attributed'], 'no unambiguous scheduler attribution')
            t8, t28 = obs['token8_ns'], obs['token28_ns']
            b = requests['B']
            _require(any(e['token_ids'] and t8 < e['receive_perf_counter_ns'] < t28
                         for e in b['raw_sse_events']), 'no B token receipt within D8..28')
            concurrent = sum((r['context_requests'] == 1 and r['generation_requests'] == 1)
                             or r['generation_requests'] == 2 for r in rows)
            obs['concurrent_scheduler_steps'] = concurrent
            # B only starts after D8. With no preemption/speculation, only four
            # D decode steps can be after D28. >4 witnesses therefore place at
            # least one real co-scheduled step in the target interval, without
            # inventing nanosecond timestamps for second-resolution log lines.
            _require(concurrent > plan['outputs']['D'] - plan['endpoint_token'],
                     'scheduler overlap could occur entirely after D28')
        gate('overlap_confirmed', overlap)

    required = ('technical_complete', 'trigger_exact', 'server_bound', 'metrics_complete',
                'traffic_attributed', 'cache_manipulation', 'iteration_accounting', 'effective_budget')
    report['valid_episode'] = all(gates[k] for k in required) and (
        gates['victim_only_capacity_valid'] if baseline else gates['overlap_confirmed'])
    if not report['valid_episode']:
        report['endpoint_ms'] = None
    return classify_report(report)


def comparison_gates(episodes):
    """Technical contrasts only. No latency effect estimates or causal claim."""
    assessed = [assess_episode(ep) for ep in episodes]
    ids = [ep['plan']['episode_id'] for ep in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate episode identity')
    result = {'episodes': assessed, 'pairs': [], 'h1_validated': False, 'h1_status': 'not_evaluated',
              'mechanism_candidates': [r['episode_id'] for r in assessed if r['mechanism_candidate']]}
    for i, left in enumerate(episodes):
        lp, la = left['plan'], assessed[i]
        for j in range(i + 1, len(episodes)):
            rp, ra = episodes[j]['plan'], assessed[j]
            lc, rc = lp['config'], rp['config']
            if lp['condition'] == 'victim-only' or rp['condition'] == 'victim-only':
                continue
            same_victim = lp['victim_identity'] == rp['victim_identity']
            config_matched = ({k: v for k, v in lc.items() if k not in ('blocks', 'budget')}
                              == {k: v for k, v in rc.items() if k not in ('blocks', 'budget')})
            common = (same_victim and lp['prompts'] == rp['prompts'] and lp['seed'] == rp['seed']
                      and config_matched
                      and left['clock_domain'] == episodes[j]['clock_domain']
                      and lp['nonce'] != rp['nonce'] and lp['episode_id'] != rp['episode_id'])
            baseline_ok = all(any(
                ep['plan']['condition'] == 'victim-only'
                and ep['plan']['config'] == p['config']
                and ep['plan']['victim_identity'] == p['victim_identity']
                and a['valid_episode'] for ep, a in zip(episodes, assessed)) for p in (lp, rp))
            base_valid = common and la['valid_episode'] and ra['valid_episode'] and baseline_ok
            pair = {'left': lp['episode_id'], 'right': rp['episode_id'],
                    'victim_identity_matched': same_victim, 'baseline_confirmed': baseline_ok,
                    'configuration_matched_except_factors': config_matched,
                    'mechanism_candidate_present': la['mechanism_candidate'] or ra['mechanism_candidate']}
            if lc == rc and {lp['condition'], rp['condition']} in (
                    {'shared', 'isolated'}, {'no-prefix-shared', 'no-prefix-isolated'}):
                pair.update(kind='salt', valid_manipulation=bool(base_valid))
                pair['negative_control_validated'] = all(any(
                    ep['plan']['condition'] == condition and ep['plan']['config'] == lc
                    and ep['plan']['victim_identity'] == lp['victim_identity'] and a['valid_episode']
                    for ep, a in zip(episodes, assessed)) for condition in ('no-prefix-shared', 'no-prefix-isolated'))
            elif lp['condition'] == rp['condition'] and lc['blocks'] == rc['blocks'] and {lc['budget'], rc['budget']} == {64, 512}:
                small, large = (la, ra) if lc['budget'] == 64 else (ra, la)
                pair.update(kind='budget', budget_difference_validated=bool(
                    base_valid and small['gates']['budget_binding_observed']
                    and small['observations']['b_prefill_chunks'] != large['observations']['b_prefill_chunks']
                    and max(large['observations']['b_prefill_chunks'], default=0) > 64))
            elif lp['condition'] == rp['condition'] and lc['budget'] == rc['budget'] and {lc['blocks'], rc['blocks']} == {24, 64}:
                near, far = (la, ra) if lc['blocks'] == 24 else (ra, la)
                pair.update(kind='capacity', allocation_contrast_observed=bool(
                    base_valid and near['observations']['capacity_pressure_observed']
                    and not far['observations']['capacity_pressure_observed']
                    and near['observations']['max_active_kv_blocks'] is not None
                    and far['observations']['max_active_kv_blocks'] is not None
                    and far['observations']['max_active_kv_blocks']
                    > near['observations']['max_active_kv_blocks'] + .5))
            else:
                continue
            result['pairs'].append(pair)
    return result
