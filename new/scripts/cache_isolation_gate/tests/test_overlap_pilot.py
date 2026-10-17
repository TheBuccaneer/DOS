"""Adversarial offline fixtures and real frozen-executor/fake-transport tests."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parents[2]
sys.path.insert(0, str(SCRIPTS))
from cache_eviction_control import JsonlRecord, prompt_hash
from cache_smoke import BASE_SHA256, load_base
from overlap_contract import (CONDITIONS, SEED, assess_episode, assessment_exit_code,
                               comparison_gates, make_plan, nuisance_from_log, normalize_engine_config)
from overlap_pilot import read_episode, run_episode, startup_evidence

A = [10, 11, 12, 13] * 32
MS = 1_000_000


def metric_body(counters, running=0, waiting=0, usage=0, blocks=64, process_start=12345.25):
    names = {'queries': 'prefix_cache_queries_total', 'hits': 'prefix_cache_hits_total',
             'prompt_tokens': 'prompt_tokens_total', 'computed_context_tokens': 'request_prefill_kv_computed_tokens_sum',
             'prefill_requests': 'request_prefill_kv_computed_tokens_count', 'preemptions': 'num_preemptions_total'}
    info = (f'block_size="16",cpu_offload_gb="0",enable_prefix_caching="True",'
            f'kv_cache_memory_bytes="None",num_gpu_blocks="{blocks}",num_gpu_blocks_override="{blocks}"')
    lines = [f'vllm:cache_config_info{{{info}}} 1', f'process_start_time_seconds {process_start}',
             f'vllm:num_requests_running{{engine="0"}} {running}',
             f'vllm:num_requests_waiting{{engine="0"}} {waiting}',
             f'vllm:kv_cache_usage_perc{{engine="0"}} {usage}']
    return '\n'.join(lines + [f'vllm:{name}{{engine="0"}} {counters.get(key, 0)}' for key, name in names.items()])


def startup(config):
    args = {'enable_prompt_tokens_details': True, 'enable_logging_iteration_details': True,
            'enable_chunked_prefill': True, 'enable_prefix_caching': True,
            'max_num_seqs': 16, 'seed': 20260711, 'max_model_len': 256,
            'max_num_batched_tokens': config['budget'], 'num_gpu_blocks_override': config['blocks']}
    return ('vLLM-Version: 0.17.1\nMax model len requested: 256\n'
            f'NUM_GPU_BLOCKS_OVERRIDE requested: {config["blocks"]}\n'
            f'Chunked prefill is enabled with max_num_batched_tokens={config["budget"]}.\n'
            f'(APIServer pid=1234) non-default args: {args!r}\n'
            "Initializing a V1 LLM engine (v0.17.1) with config: model='public-fixture', "
            'speculative_config=None, dtype=torch.bfloat16, max_seq_len=256, seed=20260711, '
            f'scheduler_config=SchedulerConfig(max_num_batched_tokens={config["budget"]}, max_num_seqs=16), '
            f'cache_config=CacheConfig(num_gpu_blocks_override={config["blocks"]}, '
            'num_gpu_blocks=None, block_size=16), '
            "compilation_config={'mode': <CompilationMode.VLLM_COMPILE: 3>, "
            f"'compile_ranges_split_points': [{config['budget']}], "
            "'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, "
            "'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32], "
            "'inductor_compile_config': {'combo_kernels': True, 'benchmark_combo_kernel': True}}\n")


def scheduler_log(condition, budget, before_b=7):
    rows = []
    def row(c, ct, g, gt):
        rows.append(f'Iteration({len(rows)}): {c} context requests, {ct} context tokens, '
                    f'{g} generation requests, {gt} generation tokens, iteration elapsed time: 1.00 ms')
    for chunk in ([64] * 3 if budget == 64 else [192]):
        row(1, chunk, 0, 0)
    if condition == 'victim-only':
        for _ in range(31):
            row(0, 0, 1, 1)
    else:
        for _ in range(before_b):
            row(0, 0, 1, 1)
        chunks = [16] if condition == 'shared' else ([63, 63, 2] if budget == 64 else [128])
        for chunk in chunks:
            row(1, chunk, 1, 1)
        joint = 31 - before_b - len(chunks)
        for _ in range(joint):
            row(0, 0, 2, 2)
        for _ in range(63 - joint):
            row(0, 0, 1, 1)
    return '\n'.join(rows) + '\n'


def record(plan, role):
    tokens, count = plan['prompts'][role], plan['outputs'][role]
    first = 20 * MS if role == 'D' else 110 * MS
    times = [first + i * 10 * MS for i in range(count)]
    events = [{'event_index': i, 'receive_perf_counter_ns': ns, 'token_ids': [100 + i],
               'raw_data': f'token-{i}'} for i, ns in enumerate(times)]
    hits = 112 if role == 'B' and plan['condition'] == 'shared' else 0
    usage = {'prompt_tokens': len(tokens), 'completion_tokens': count}
    if hits:
        usage['prompt_tokens_details'] = {'cached_tokens': hits}
    return {'request_id': plan['request_ids'][role], 'request_index': 0,
            'status': 'complete', 'raw_sse_events': events,
            'output_token_ids': [100 + i for i in range(count)],
            'expected_prompt_tokens': len(tokens), 'expected_completion_tokens': count,
            'prompt_token_ids_sent': tokens, 'prompt_token_ids_returned': tokens,
            'prompt_sha256': prompt_hash(tokens), 'generation_seed': SEED,
            'request_start_ns': 10 * MS if role == 'D' else 91 * MS,
            'task_created_ns': 9 * MS if role == 'D' else 90 * MS,
            'request_dispatch_ns': 10 * MS if role == 'D' else 91 * MS,
            'stream_end_ns': times[-1] + MS,
            'first_token_receive_ns': times[0], 'last_token_receive_ns': times[-1],
            'usage': usage, 'validation_errors': [], 'done_received': True, 'finish_reason': 'length',
            'wire_payload': {'prompt': tokens, 'cache_salt': plan['salts'][role],
                             'max_tokens': count, 'min_tokens': count, 'seed': SEED,
                             'temperature': 0, 'ignore_eos': True, 'stream': True,
                             'add_special_tokens': False, 'model': plan['config']['model']}}


def fixture(condition='shared', budget=512, blocks=64, label='a'):
    config = {'blocks': blocks, 'budget': budget, 'max_model_len': 256,
              'model': 'meta-llama/Llama-3.1-8B-Instruct', 'metrics_interval_s': .02,
              'nuisance_sha256': 'f' * 64}
    plan = make_plan(A, condition, 'ep-' + label, label * 32, config)
    raw = startup(config)
    nuisance = nuisance_from_log(raw, config)
    config['nuisance_sha256'] = hashlib.sha256(json.dumps(nuisance, sort_keys=True).encode()).hexdigest()
    domain = {'hostname': 'offline', 'boot_id': 'boot', 'clock': 'time.perf_counter_ns'}
    ep = {'plan': plan, 'startup_log': raw, 'startup_log_sha256': hashlib.sha256(raw.encode()).hexdigest(),
          'nuisance_config': nuisance,
          'server_identity': {'api_pid': 1234, 'process_start_seconds': 12345.25,
                              'log_start_seconds': 12345, 'boot_id': 'boot'},
          'clock_domain': domain, 'cohort_declared_perf_ns': MS, 'errors': [],
          'historical_executor_sha256': BASE_SHA256,
          'metrics_interval_s': .02, 'metrics_capture': {'stop_error': None, 'sampler_available': True},
          'requests': {'D': record(plan, 'D')}, 'token_progress': [],
          'triggers': [{'count': 8, 'trigger_perf_ns': 90 * MS, 'observed_perf_ns': 90 * MS}],
          'iteration_log': scheduler_log(condition, budget), 'metrics_full_samples': []}
    if condition != 'victim-only':
        ep['requests']['B'] = record(plan, 'B')
    ep['token_progress'] = [{'count': i + 1, 'receive_perf_counter_ns': e['receive_perf_counter_ns']}
                            for i, e in enumerate(ep['requests']['D']['raw_sse_events'])]
    baseline = condition == 'victim-only'
    prompt = 192 if baseline else 320
    hits = 112 if condition == 'shared' else 0
    for tick in range(0, 821, 20):
        final = tick >= (360 if baseline else 800)
        counters = {'queries': prompt, 'hits': hits, 'prompt_tokens': prompt,
                    'computed_context_tokens': prompt - hits, 'prefill_requests': 1 if baseline else 2} if final else {}
        running = 0 if final or tick == 0 else (1 if baseline or tick > 340 or tick < 100 else 2)
        body = metric_body(counters, running=running, usage=14 / (blocks - 1) if running else 0, blocks=blocks)
        ep['metrics_full_samples'].append({'scrape_start_perf_ns': tick * MS,
                                           'response_received_perf_ns': tick * MS + 100,
                                           'http_status': 200, 'raw_body': body, 'error': None})
    return ep


def replace_startup(ep, old, new):
    ep['startup_log'] = ep['startup_log'].replace(old, new)
    ep['startup_log_sha256'] = hashlib.sha256(ep['startup_log'].encode()).hexdigest()
    ep['nuisance_config'] = nuisance_from_log(ep['startup_log'], ep['plan']['config'])
    ep['plan']['config']['nuisance_sha256'] = hashlib.sha256(
        json.dumps(ep['nuisance_config'], sort_keys=True).encode()).hexdigest()


class OverlapContractTests(unittest.TestCase):
    def test_all_five_episode_types_and_endpoint(self):
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                report = assess_episode(fixture(condition))
                self.assertTrue(report['valid_episode'], report['issues'])
                self.assertEqual(report['endpoint_ms'], 200)
                self.assertFalse(report['h1_validated'])

    def test_trigger_early_late_duplicate_and_batch_crossing(self):
        for kind in ('early', 'late', 'multiple', 'progress', 'batch'):
            ep = fixture()
            if kind in ('early', 'late'):
                ep['triggers'][0]['trigger_perf_ns'] += (-1 if kind == 'early' else 1) * MS
            elif kind == 'multiple':
                ep['triggers'].append(copy.deepcopy(ep['triggers'][0]))
            elif kind == 'progress':
                ep['token_progress'].append(copy.deepcopy(ep['token_progress'][7]))
            else:
                ep['token_progress'][7]['count'] = 9
            with self.subTest(kind=kind):
                report = assess_episode(ep)
                self.assertFalse(report['gates']['trigger_exact'])
                self.assertFalse(report['valid_episode'])

    def test_missing_late_sse_and_failed_requests_not_dropped(self):
        for kind in ('missing', 'late', 'timeout', 'cancelled'):
            ep = fixture()
            if kind == 'missing':
                ep['requests']['D']['raw_sse_events'].pop(20)
            elif kind == 'late':
                ep['requests']['D']['raw_sse_events'][20]['receive_perf_counter_ns'] = 9999 * MS
            else:
                ep['requests']['D']['status'] = kind
            report = assess_episode(ep)
            self.assertFalse(report['gates']['technical_complete'])
            self.assertIsNone(report['endpoint_ms'])
            self.assertEqual(ep['plan']['cohort_indices'], [0])

    def test_cache_salt_and_cold_victim_gates(self):
        for role, kind in [('D', 'cache'), ('B', 'cache'), ('B', 'salt')]:
            ep = fixture()
            if kind == 'cache':
                ep['requests'][role]['usage']['prompt_tokens_details'] = {'cached_tokens': 16}
            else:
                ep['requests'][role]['wire_payload']['cache_salt'] = 'wrong-tenant'
            report = assess_episode(ep)
            self.assertFalse(report['valid_episode'])
            self.assertFalse(report['gates']['cache_manipulation'] if kind == 'cache'
                             else report['gates']['technical_complete'])

    def test_changed_sampling_usage_or_effective_scheduler_rejected(self):
        for kind in ('sampling', 'usage', 'scheduler'):
            ep = fixture('isolated')
            if kind == 'sampling':
                ep['requests']['B']['wire_payload']['temperature'] = 1
            elif kind == 'usage':
                ep['requests']['B']['usage']['prompt_tokens'] = 0
            else:
                ep['startup_log'] = ep['startup_log'].replace("'max_num_seqs': 16", "'max_num_seqs': 32")
                ep['startup_log_sha256'] = hashlib.sha256(ep['startup_log'].encode()).hexdigest()
            self.assertFalse(assess_episode(ep)['valid_episode'])

    def test_client_lifecycle_does_not_prove_scheduler_overlap(self):
        ep = fixture()
        ep['iteration_log'] = scheduler_log('shared', 512, before_b=27)
        report = assess_episode(ep)
        self.assertTrue(report['gates']['iteration_accounting'])
        self.assertFalse(report['gates']['overlap_confirmed'])
        ep = fixture()
        for e in ep['requests']['B']['raw_sse_events']:
            e['receive_perf_counter_ns'] += 1000 * MS
        self.assertFalse(assess_episode(ep)['gates']['overlap_confirmed'])

    def test_post_trigger_metrics_and_counter_resets(self):
        ep = fixture()
        ep['metrics_full_samples'] = ep['metrics_full_samples'][:10]
        self.assertFalse(assess_episode(ep)['gates']['metrics_complete'])
        ep = fixture()
        ep['metrics_full_samples'][8]['error'] = 'http_error'
        self.assertFalse(assess_episode(ep)['valid_episode'])
        ep = fixture()
        ep['metrics_full_samples'][-1]['raw_body'] = metric_body({}, blocks=64)
        self.assertFalse(assess_episode(ep)['gates']['metrics_complete'])

    def test_foreign_traffic_and_preemption_preserved(self):
        ep = fixture()
        ep['metrics_full_samples'][-1]['raw_body'] = metric_body(
            {'queries': 448, 'hits': 112, 'prompt_tokens': 448,
             'computed_context_tokens': 336, 'prefill_requests': 3})
        self.assertFalse(assess_episode(ep)['gates']['traffic_attributed'])
        ep = fixture()
        ep['metrics_full_samples'][-1]['raw_body'] = metric_body(
            {'queries': 320, 'hits': 112, 'prompt_tokens': 320,
             'computed_context_tokens': 208, 'prefill_requests': 2, 'preemptions': 1}, waiting=0)
        report = assess_episode(ep)
        self.assertTrue(report['gates']['technical_complete'])
        self.assertEqual(report['observations']['preemptions'], 1)
        self.assertFalse(report['gates']['iteration_accounting'])
        self.assertFalse(report['valid_episode'])
        self.assertTrue(report['technical_sse_measurement_complete'])
        self.assertTrue(report['mechanism_candidate'])
        self.assertEqual(report['mechanism_status'], 'preemption_candidate_insufficient_attribution')
        self.assertFalse(report['causal_comparison_eligible'])
        self.assertEqual(report['causal_status'], 'insufficient_mechanism_attribution')
        self.assertEqual(report['h1_status'], 'not_evaluated')
        self.assertFalse(report['h1_validated'])
        self.assertEqual(report['raw_endpoint_ms'], 200)
        self.assertIsNone(report['endpoint_ms'])
        self.assertEqual(assessment_exit_code([report]), 3)
        comparison = comparison_gates([ep])
        self.assertEqual(comparison['mechanism_candidates'], [ep['plan']['episode_id']])

    def test_preemption_with_unbound_server_or_incomplete_SSE_is_not_accepted(self):
        ep = fixture()
        ep['metrics_full_samples'][-1]['raw_body'] = metric_body(
            {'queries': 320, 'hits': 112, 'prompt_tokens': 320,
             'computed_context_tokens': 208, 'prefill_requests': 2, 'preemptions': 1})
        ep['requests']['B']['status'] = 'incomplete'
        report = assess_episode(ep)
        self.assertFalse(report['technical_sse_measurement_complete'])
        self.assertTrue(report['mechanism_candidate'])
        self.assertIsNone(report['raw_endpoint_ms'])
        self.assertEqual(assessment_exit_code([report]), 1)
        ep['server_identity']['process_start_seconds'] += 10
        report = assess_episode(ep)
        self.assertFalse(report['mechanism_candidate'])
        self.assertEqual(report['mechanism_status'], 'unattributed_preemption_counter_change')
        self.assertEqual(report['h1_status'], 'not_evaluated')

    def test_preemption_with_foreign_traffic_is_not_a_mechanism_candidate(self):
        for foreign in ('completed_request', 'additional_running_request'):
            ep = fixture()
            counters = {'queries': 320, 'hits': 112, 'prompt_tokens': 320,
                        'computed_context_tokens': 208, 'prefill_requests': 2, 'preemptions': 1}
            if foreign == 'completed_request':
                counters.update(queries=448, prompt_tokens=448,
                                computed_context_tokens=336, prefill_requests=3)
            else:
                ep['metrics_full_samples'][10]['raw_body'] = metric_body({}, running=3)
            ep['metrics_full_samples'][-1]['raw_body'] = metric_body(counters)
            with self.subTest(foreign=foreign):
                report = assess_episode(ep)
                self.assertTrue(report['technical_sse_measurement_complete'])
                self.assertTrue(report['gates']['server_bound'])
                self.assertTrue(report['gates']['metrics_complete'])
                self.assertFalse(report['gates']['traffic_attributed'])
                self.assertFalse(report['mechanism_candidate'])
                self.assertEqual(report['observations']['preemptions'], 1)
                self.assertEqual(report['mechanism_status'],
                                 'unattributed_preemption_foreign_or_ambiguous_traffic')
                self.assertFalse(report['causal_comparison_eligible'])
                self.assertEqual(report['h1_status'], 'not_evaluated')
                self.assertEqual(report['raw_endpoint_ms'], 200)
                self.assertEqual(assessment_exit_code([report]), 1)
                self.assertEqual(comparison_gates([ep])['mechanism_candidates'], [])

    def test_wrong_server_log_and_effective_budget(self):
        ep = fixture()
        ep['server_identity']['process_start_seconds'] += 10
        self.assertFalse(assess_episode(ep)['gates']['server_bound'])
        ep = fixture()
        ep['startup_log'] = ep['startup_log'].replace('tokens=512.', 'tokens=64.')
        ep['startup_log_sha256'] = hashlib.sha256(ep['startup_log'].encode()).hexdigest()
        self.assertFalse(assess_episode(ep)['gates']['effective_budget'])

    def test_missing_metrics_do_not_hide_complete_request_status(self):
        ep = fixture()
        ep['metrics_full_samples'][-1]['raw_body'] = '# no cache metrics\n'
        report = assess_episode(ep)
        self.assertTrue(report['gates']['technical_complete'])
        self.assertFalse(report['gates']['metrics_complete'])
        self.assertIsNone(report['endpoint_ms'])

    def test_salt_contrast_requires_negative_controls_and_unique_episodes(self):
        shared, isolated = fixture('shared', label='s'), fixture('isolated', label='i')
        baseline = fixture('victim-only', label='v')
        report = comparison_gates([shared, isolated, baseline])['pairs'][0]
        self.assertTrue(report['valid_manipulation'])
        self.assertFalse(report['negative_control_validated'])
        controls = [fixture('no-prefix-shared', label='c'), fixture('no-prefix-isolated', label='d')]
        self.assertTrue(comparison_gates([shared, isolated, baseline, *controls])['pairs'][0]['negative_control_validated'])
        with self.assertRaisesRegex(ValueError, 'duplicate episode'):
            comparison_gates([shared, shared])

    def test_unbound_budget_cannot_pass_due_to_D_prefill_only(self):
        small, large = fixture('shared', 64, label='s'), fixture('shared', 512, label='l')
        report = assess_episode(small)
        self.assertTrue(report['valid_episode'], report['issues'])
        self.assertFalse(report['gates']['budget_binding_observed'])
        baselines = [fixture('victim-only', 64, label='v'), fixture('victim-only', 512, label='w')]
        pair = comparison_gates([small, large, *baselines])['pairs'][0]
        self.assertFalse(pair['budget_difference_validated'])

    def test_budget_difference_needs_matched_victim_and_baseline(self):
        small, large = fixture('isolated', 64, label='s'), fixture('isolated', 512, label='l')
        self.assertFalse(comparison_gates([small, large])['pairs'][0]['budget_difference_validated'])
        baselines = [fixture('victim-only', 64, label='v'), fixture('victim-only', 512, label='w')]
        pair = comparison_gates([small, large, *baselines])['pairs'][0]
        self.assertTrue(pair['budget_difference_validated'])
        self.assertNotEqual(small['startup_log'], large['startup_log'])
        self.assertEqual(small['nuisance_config'], large['nuisance_config'])
        large['plan']['victim_identity']['generation_seed'] += 1
        pair = comparison_gates([small, large, *baselines])['pairs'][0]
        self.assertFalse(pair['victim_identity_matched'])
        self.assertFalse(pair['budget_difference_validated'])

    def test_capacity_contrast_needs_observed_pressure(self):
        near, far = fixture('isolated', blocks=24, label='n'), fixture('isolated', blocks=64, label='f')
        baselines = [fixture('victim-only', blocks=24, label='v'), fixture('victim-only', blocks=64, label='w')]
        self.assertFalse(comparison_gates([near, far, *baselines])['pairs'][0]['allocation_contrast_observed'])
        near['metrics_full_samples'][10]['raw_body'] = metric_body({}, running=2, usage=1, blocks=24)
        far['metrics_full_samples'][10]['raw_body'] = metric_body({}, running=2, usage=23 / 63, blocks=64)
        # A fraction change alone, with identical absolute active blocks, fails.
        self.assertFalse(comparison_gates([near, far, *baselines])['pairs'][0]['allocation_contrast_observed'])
        far['metrics_full_samples'][10]['raw_body'] = metric_body({}, running=2, usage=26 / 63, blocks=64)
        self.assertTrue(comparison_gates([near, far, *baselines])['pairs'][0]['allocation_contrast_observed'])
        self.assertNotEqual(near['startup_log'], far['startup_log'])
        self.assertEqual(near['nuisance_config'], far['nuisance_config'])

    def test_factor_normalization_preserves_all_unrelated_settings(self):
        for old, new in [('dtype=torch.bfloat16', 'dtype=torch.float16'),
                         ("'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 32]",
                          "'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 24, 64]"),
                         ("'combo_kernels': True", "'combo_kernels': False")]:
            with self.subTest(changed=old):
                small, large = fixture('isolated', 64, label='s'), fixture('isolated', 512, label='l')
                baselines = [fixture('victim-only', 64, label='v'), fixture('victim-only', 512, label='w')]
                replace_startup(large, old, new)
                replace_startup(baselines[1], old, new)
                report = comparison_gates([small, large, *baselines])
                self.assertTrue(all(r['valid_episode'] for r in report['episodes']))
                self.assertTrue(report['pairs'][0]['baseline_confirmed'])
                self.assertFalse(report['pairs'][0]['configuration_matched_except_factors'])
                self.assertFalse(report['pairs'][0]['budget_difference_validated'])

    def test_wrong_engine_factor_or_compile_endpoint_rejected(self):
        config = fixture(budget=64)['plan']['config']
        for text in ('max_num_batched_tokens=512',
                     "compilation_config={'compile_ranges_split_points': [512]}",
                     'cache_config=CacheConfig(num_gpu_blocks=24)',
                     'dtype=torch.bfloat16, dtype=torch.float16'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                normalize_engine_config(text, config)
        normalized = normalize_engine_config(
            "seed=64, compilation_config={'compile_ranges_split_points': [32, 64], 'compile_sizes': [64]}", config)
        self.assertEqual(normalized['seed'], '64')
        self.assertEqual(normalized['compilation_config']['fields']['compile_sizes'], '[64]')
        self.assertEqual(normalized['compilation_config']['fields']['compile_ranges_split_points'],
                         [32, {'factor': 'budget'}])

    def test_early_uninitialized_blocks_accepted_but_later_capacity_is_strict(self):
        for blocks in (24, 64):
            ep = fixture('isolated', blocks=blocks)
            config = ep['plan']['config']
            with self.subTest(blocks=blocks):
                self.assertIn('num_gpu_blocks=None', ep['startup_log'])
                self.assertTrue(assess_episode(ep)['valid_episode'])
                early = normalize_engine_config(
                    f'cache_config=CacheConfig(num_gpu_blocks_override={blocks}, num_gpu_blocks=None)', config)
                initialized = normalize_engine_config(
                    f'cache_config=CacheConfig(num_gpu_blocks_override={blocks}, num_gpu_blocks={blocks})', config)
                self.assertEqual(early, initialized)
                for invalid in ('num_gpu_blocks_override=None', f'num_gpu_blocks={blocks + 1}'):
                    with self.assertRaises(ValueError):
                        normalize_engine_config('cache_config=CacheConfig(' + invalid + ')', config)
                # None is permitted only in the early repr, never in live metrics.
                ep['metrics_full_samples'][-1]['raw_body'] = ep['metrics_full_samples'][-1]['raw_body'].replace(
                    f'num_gpu_blocks="{blocks}"', 'num_gpu_blocks="None"')
                self.assertFalse(assess_episode(ep)['gates']['server_bound'])


class FrozenExecutorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_base(ROOT)

    def execute_fake(self, *, condition='shared', incomplete_b=False, timeout_b=False, batch_crossing=False):
        test = self
        class Transport:
            active = 0
            max_active = 0
            completed = 0
            payloads = []

            async def get_json(self, url, headers, timeout_s):
                await asyncio.sleep(0)
                baseline = condition == 'victim-only'
                done = self.completed == (1 if baseline else 2)
                prompt, hits = (192 if baseline else 320), (112 if condition == 'shared' else 0)
                counters = {'queries': prompt, 'hits': hits, 'prompt_tokens': prompt,
                            'computed_context_tokens': prompt - hits,
                            'prefill_requests': 1 if baseline else 2} if done else {}
                return 200, metric_body(counters, running=self.active,
                                        usage=(14 if self.active == 1 else 19) / 63 if self.active else 0)

            async def stream_completion(self, url, headers, payload, timeout_s, *, request_id=None):
                role = request_id.rsplit(':', 1)[-1]
                self.payloads.append((role, copy.deepcopy(payload)))
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                try:
                    yield json.dumps({'choices': [{'index': 0, 'token_ids': [], 'text': ''}],
                                      'prompt_token_ids': payload['prompt']})
                    if role == 'B' and timeout_b:
                        await asyncio.Event().wait()
                    for i in range(payload['max_tokens']):
                        if role == 'D' and batch_crossing and i == 8:
                            continue
                        await asyncio.sleep(.002)
                        ids = [107, 108] if role == 'D' and batch_crossing and i == 7 else [100 + i]
                        yield json.dumps({'choices': [{'index': 0, 'text': '', 'token_ids': ids}]})
                    usage = {'prompt_tokens': len(payload['prompt']), 'completion_tokens': payload['max_tokens']}
                    if role == 'B' and condition == 'shared':
                        usage['prompt_tokens_details'] = {'cached_tokens': 112}
                    yield json.dumps({'choices': [{'index': 0, 'token_ids': [], 'finish_reason': 'length'}], 'usage': usage})
                    if not (role == 'B' and incomplete_b):
                        yield '[DONE]'
                finally:
                    self.active -= 1
                    self.completed += 1
                    if self.completed == (1 if condition == 'victim-only' else 2):
                        with log.open('a') as stream:
                            stream.write(scheduler_log(condition, 512))

        with tempfile.TemporaryDirectory() as temp:
            output, log = Path(temp) / 'episode.jsonl', Path(temp) / 'server.log'
            log.write_text('startup already verified\n')
            synthetic = fixture(condition)
            evidence = {k: synthetic[k] for k in ('startup_log', 'startup_log_sha256', 'server_identity',
                                                 'clock_domain', 'nuisance_config')}
            plan = synthetic['plan']
            writer = JsonlRecord(output)
            transport = Transport()
            try:
                with patch('socket.socket.connect', side_effect=AssertionError('network forbidden in offline tests')):
                    ep, assessment = asyncio.run(run_episode(
                        test.base, transport, plan, writer, evidence,
                        base_url='http://127.0.0.1:8000', api_key='test-key', server_log=log,
                        interval=.01, settle_s=.2, request_timeout_s=.15 if timeout_b else 2))
            finally:
                writer.close()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            reread = read_episode(output)
            with self.assertRaises(FileExistsError):
                JsonlRecord(output)
            return ep, assessment, rows, transport, reread

    def test_real_executor_triggers_once_and_retains_complete_streams(self):
        ep, report, rows, transport, reread = self.execute_fake()
        self.assertTrue(report['valid_episode'], report['issues'])
        self.assertEqual(transport.max_active, 2)
        self.assertEqual([p[0] for p in transport.payloads], ['D', 'B'])
        self.assertEqual(len(ep['triggers']), 1)
        self.assertEqual(ep['triggers'][0]['count'], 8)
        self.assertEqual(len(ep['requests']['D']['output_token_ids']), 32)
        self.assertEqual(len(ep['requests']['B']['output_token_ids']), 64)
        self.assertEqual(ep, reread)
        self.assertTrue(any(r['record_type'] == 'sse_wire' for r in rows))
        self.assertEqual(rows[-1]['record_type'], 'episode_result')

    def test_incomplete_B_keeps_D_and_all_partial_data(self):
        ep, report, rows, _, _ = self.execute_fake(incomplete_b=True)
        self.assertFalse(report['valid_episode'])
        self.assertEqual(ep['requests']['D']['status'], 'complete')
        self.assertEqual(ep['requests']['B']['status'], 'incomplete')
        self.assertEqual(len(ep['requests']['D']['output_token_ids']), 32)
        self.assertEqual(len([r for r in rows if r['record_type'] == 'request_result']), 2)

    def test_timeout_keeps_fixed_cohort(self):
        ep, report, _, _, _ = self.execute_fake(timeout_b=True)
        self.assertFalse(report['valid_episode'])
        self.assertEqual(ep['plan']['cohort_indices'], [0])
        self.assertTrue(ep['requests']['B']['timed_out'])
        self.assertEqual(ep['requests']['D']['status'], 'complete')

    def test_victim_only_never_dispatches_B(self):
        ep, report, _, transport, _ = self.execute_fake(condition='victim-only')
        self.assertTrue(report['valid_episode'], report['issues'])
        self.assertEqual([p[0] for p in transport.payloads], ['D'])
        self.assertEqual(set(ep['requests']), {'D'})

    def test_actual_transports_for_isolated_and_negative_controls(self):
        for condition in ('isolated', 'no-prefix-shared', 'no-prefix-isolated'):
            with self.subTest(condition=condition):
                ep, report, _, transport, _ = self.execute_fake(condition=condition)
                self.assertTrue(report['valid_episode'], report['issues'])
                d, b = [row[1] for row in transport.payloads]
                self.assertEqual(d['cache_salt'] == b['cache_salt'], condition == 'no-prefix-shared')
                self.assertEqual(d['prompt'][:128] == b['prompt'], condition == 'isolated')

    def test_batched_trigger_is_invalid_and_does_not_launch_B(self):
        ep, report, _, transport, _ = self.execute_fake(batch_crossing=True)
        self.assertFalse(report['valid_episode'])
        self.assertEqual(ep['triggers'][0]['count'], 9)
        self.assertEqual([p[0] for p in transport.payloads], ['D'])
        self.assertEqual(ep['requests']['D']['status'], 'complete')
        self.assertEqual(ep['requests']['B']['status'], 'not_dispatched')

    def test_offline_reader_rejects_changed_result_and_raw_journal(self):
        _, _, rows, _, _ = self.execute_fake()
        for change in ('startup', 'metrics', 'trigger', 'salt'):
            changed = copy.deepcopy(rows)
            if change == 'startup':
                changed[-1]['episode']['server_identity']['api_pid'] += 1
            elif change == 'metrics':
                changed.pop(next(i for i, r in enumerate(changed) if r['record_type'] == 'metrics_sample'))
            elif change == 'trigger':
                changed[-1]['episode']['triggers'][0]['count'] = 7
            else:
                next(r for r in changed if r['record_type'] == 'request_wire')['payload']['cache_salt'] = 'wrong'
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'changed.jsonl'
                path.write_text('\n'.join(json.dumps(r) for r in changed))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    read_episode(path)

    def test_startup_evidence_uses_existing_vllm_log_contract(self):
        from datetime import datetime, timezone
        config = fixture()['plan']['config']
        body = startup(config) + (
            'UTC-Startzeit: 20261017T100000Z\n'
            'Max num batched tokens (chunked-prefill budget): 512\n'
            'Prefix-Caching: enabled\nOffload-Wert: 0\n'
            'Overriding num_gpu_blocks=2823 with num_gpu_blocks_override=64\n'
            'GPU KV cache size: 1,024 tokens\n'
            'Maximum concurrency for 256 tokens per request: 4.00x\n')
        epoch = datetime(2026, 10, 17, 10, tzinfo=timezone.utc).timestamp()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'startup.log'
            path.write_text(body)
            with patch('overlap_pilot.api_process_start_seconds', return_value=epoch):
                evidence = startup_evidence(path, config)
                self.assertEqual(evidence['server_identity']['api_pid'], 1234)
                self.assertEqual(evidence['startup_log'], body)
                self.assertIn('num_gpu_blocks=None', evidence['startup_log'])
                path.write_text(body.replace('GPU KV cache size: 1,024 tokens', 'GPU KV cache size: 384 tokens'))
                with self.assertRaisesRegex(ValueError, 'capacity'):
                    startup_evidence(path, config)
                path.write_text(body.replace("'enable_prompt_tokens_details': True", ''))
                with self.assertRaisesRegex(ValueError, 'prompt token details'):
                    startup_evidence(path, config)

    def test_historical_executor_hash(self):
        path = ROOT / 'new/scripts/prefill_confirmation/run_prefill_confirmation.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), BASE_SHA256)


if __name__ == '__main__':
    unittest.main()
