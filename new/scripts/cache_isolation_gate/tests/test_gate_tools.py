import asyncio
import copy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from audit_existing import audit_campaign
from trace_contract import check_stream, request_identities
from extract_primary_endpoint import extract_episode
from metrics_contract import finalize_existing_sampler
from parse_iteration_logs import parse_text
from record_metrics import validate_url, capture
from cache_smoke import SaltTransport
from cache_eviction_control import (ORDER, SALT_A, SALT_B, JsonlRecord,
                                    build_prompts, metric_delta, run_control,
                                    snapshot, verify_config, verify_server_identity,
                                    verify_server_log, verify_startup_time)
from summarize_metrics import parse_metrics, summarize
from analyze_pairs import quantify
from validate_contract import validate
from build_endpoint_table import build


def request(index=0):
    return {'request_index': index, 'status': 'complete', 'prompt_sha256': 'a'*64,
            'prompt_token_ids_sent': [7, 8], 'generation_seed': 4,
            'expected_prompt_tokens': 2, 'expected_completion_tokens': 3,
            'request_start_ns': 0, 'stream_end_ns': 9_000_000,
            'first_token_receive_ns': 1_000_000, 'last_token_receive_ns': 7_000_000,
            'output_token_ids': [10, 11, 12], 'raw_sse_events': [
                {'event_index': 0, 'receive_perf_counter_ns': 1_000_000, 'token_ids': [10]},
                {'event_index': 1, 'receive_perf_counter_ns': 7_000_000, 'token_ids': [11, 12]},
                {'event_index': 2, 'receive_perf_counter_ns': 8_000_000, 'token_ids': []}]}


def episode(condition='no_burst', order=1):
    return {'episode_id': condition, 'block_id': 'b1', 'status': 'complete',
            'schedule_row': {'condition': condition, 'within_block_order': order},
            'trigger': {'trigger_perf_ns': 2_000_000}, 'victim_requests': [request()],
            'burst_requests': []}


class TraceTests(unittest.TestCase):
    def test_valid_stream(self):
        self.assertEqual(check_stream(request()), [])

    def test_failed_request_rejected(self):
        row = request(); row['status'] = 'failed'
        self.assertIn('request_not_complete', check_stream(row))

    def test_empty_event_outside_bounds(self):
        row = request(); row['raw_sse_events'][-1]['receive_perf_counter_ns'] = 10_000_000
        self.assertIn('SSE_outside_request_bounds', check_stream(row))

    def test_empty_cohort_rejected(self):
        e = episode(); e['victim_requests'] = []
        with self.assertRaisesRegex(ValueError, 'empty'):
            request_identities(e)

    def test_missing_identity_each_field(self):
        for key in ('prompt_sha256', 'prompt_token_ids_sent', 'generation_seed', 'expected_prompt_tokens', 'expected_completion_tokens'):
            with self.subTest(key=key):
                e = episode(); del e['victim_requests'][0][key]
                with self.assertRaises(ValueError):
                    request_identities(e)

    def test_duplicate_id(self):
        e = episode(); e['victim_requests'].append(request())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            request_identities(e)

    def test_endpoint_grouped_tokens(self):
        r = extract_episode(episode(), [0], 3, 'trigger.trigger_perf_ns')
        self.assertEqual(r['cohort_elapsed_ms'], 5.0)

    def test_failed_request_without_sse_has_semantic_status(self):
        e = episode(); e['victim_requests'][0] = {'request_index': 0, 'status': 'failed'}
        r = extract_episode(e, [0], 3, 'trigger.trigger_perf_ns')
        self.assertEqual(r['status'], 'incomplete_or_invalid')
        self.assertIsNone(r['cohort_elapsed_ms'])

    def test_missing_member_not_dropped(self):
        r = extract_episode(episode(), [0, 1], 3, 'trigger.trigger_perf_ns')
        self.assertIsNone(r['cohort_elapsed_ms'])

    def test_unreached_target(self):
        self.assertIsNone(extract_episode(episode(), [0], 4, 'trigger.trigger_perf_ns')['cohort_elapsed_ms'])

    def test_preorigin_target(self):
        self.assertIsNone(extract_episode(episode(), [0], 1, 'trigger.trigger_perf_ns')['cohort_elapsed_ms'])


class AuditTests(unittest.TestCase):
    def audit(self, changes=None, extra=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dest = root/'new/runs/test/episodes'; dest.mkdir(parents=True)
            a, b = episode(), episode('prefill_burst', 2)
            if changes:
                changes(a, b)
            for name, e in [('a', a), ('b', b)]:
                (dest/(name+'.json')).write_text(json.dumps(e))
            if extra:
                (dest/'extra.json').write_text(extra)
            return audit_campaign(root, 'test')

    def test_valid_pair(self):
        r = self.audit(); self.assertEqual(r['integrity_errors'], []); self.assertEqual(r['matched_request_identity_pairs'], 1)

    def test_both_empty_not_matched(self):
        def change(a, b):
            a['victim_requests'] = []; b['victim_requests'] = []
        r = self.audit(change); self.assertEqual(r['matched_request_identity_pairs'], 0); self.assertTrue(r['integrity_errors'])

    def test_both_missing_identity_not_matched(self):
        def change(a, b):
            del a['victim_requests'][0]['generation_seed']; del b['victim_requests'][0]['generation_seed']
        self.assertEqual(self.audit(change)['matched_request_identity_pairs'], 0)

    def test_failed_request_fails_audit(self):
        r = self.audit(lambda a, b: a['victim_requests'][0].update(status='failed'))
        self.assertTrue(r['integrity_errors'])

    def test_bad_json_does_not_abort_other_files(self):
        r = self.audit(extra='{'); self.assertEqual(r['episode_count'], 3); self.assertTrue(r['integrity_errors']); self.assertEqual(r['matched_request_identity_pairs'], 1)

    def test_duplicate_request_does_not_crash(self):
        r = self.audit(lambda a, b: a['victim_requests'].append(request()))
        self.assertTrue(r['integrity_errors'])

    def test_missing_schema_does_not_crash(self):
        r = self.audit(lambda a, b: a.pop('schedule_row'))
        self.assertTrue(r['integrity_errors'])


class ObservationTests(unittest.TestCase):
    def test_recorder_lifecycle_preserves_http_errors_with_mock_transport(self):
        class Opener:
            def open(self, request, timeout):
                raise HTTPError(request.full_url, 503, 'unavailable', {}, None)
        with patch('record_metrics.build_opener', return_value=Opener()):
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp)/'metrics.jsonl'
                r = capture('http://127.0.0.1:8000/metrics', output, .12, .1)
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(rows[0]['record_type'], 'capture_start')
                self.assertEqual(rows[-1]['record_type'], 'capture_end')
                self.assertGreater(r['error_count'], 0)
                self.assertTrue(all(row['http_status'] == 503 for row in rows[1:-1]))
                with self.assertRaises(FileExistsError):
                    capture('http://127.0.0.1:8000/metrics', output, .1, .1)

    def test_cache_smoke_injects_salt_and_preserves_original_payload(self):
        class Delegate:
            async def stream_completion(self, url, headers, payload, timeout_s, *, request_id=None):
                self.payload = payload
                yield 'event'
        async def probe():
            delegate = Delegate(); original = {'prompt': [10], 'max_tokens': 2, 'min_tokens': 2}
            events = [e async for e in SaltTransport(delegate, 'tenant-A').stream_completion('local', {}, original, 2)]
            self.assertEqual(events, ['event']); self.assertEqual(delegate.payload['cache_salt'], 'tenant-A')
            self.assertNotIn('cache_salt', original)
            with self.assertRaises(ValueError):
                _ = [e async for e in SaltTransport(delegate, 'tenant-A').stream_completion('local', {}, {'prompt': [10]*129, 'max_tokens': 2, 'min_tokens': 2}, 2)]
        asyncio.run(probe())

    def test_full_sampler_preserves_posttrigger_samples_and_snapshot(self):
        class Sampler:
            samples = [{'response_received_perf_ns': 1}]
            async def stop(self):
                self.samples.append({'response_received_perf_ns': 3})
        e = {'trigger': {'metrics_raw_samples': [{'response_received_perf_ns': 1}]}}
        asyncio.run(finalize_existing_sampler(e, Sampler(), clock_domain={'boot_id': 'x'}))
        self.assertEqual(len(e['metrics_full_samples']), 2)
        self.assertEqual(len(e['trigger']['metrics_raw_samples']), 1)

    def test_sampler_stop_failure_still_persists(self):
        class Sampler:
            samples = [{'response_received_perf_ns': 1}]
            async def stop(self):
                raise RuntimeError()
        e = {}
        with self.assertRaises(RuntimeError):
            asyncio.run(finalize_existing_sampler(e, Sampler(), clock_domain={}))
        self.assertEqual(len(e['metrics_full_samples']), 1)
        self.assertEqual(e['metrics_capture']['stop_error'], 'RuntimeError')

    def test_metrics_url_local_only(self):
        for url in ('https://example.org/metrics', 'http://127.0.0.1:80/other', 'http://user:pw@127.0.0.1:8000/metrics'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_url(url)

    def test_metric_labels_and_counter_reset(self):
        rows = [{'http_status': 200, 'response_received_perf_ns': i+1,
                 'raw_body': 'vllm:num_preemptions_total{engine="0"} '+str(v)} for i, v in enumerate([10, 11, 1])]
        r = summarize(rows); s = next(iter(r['series'].values()))
        self.assertIsNone(s['counter_delta']); self.assertTrue(s['counter_reset_detected'])

    def test_missing_counter_not_zero(self):
        rows = [{'http_status': 200, 'response_received_perf_ns': 1, 'raw_body': 'vllm:x_total 2'},
                {'http_status': 200, 'response_received_perf_ns': 2, 'raw_body': 'vllm:y 1'}]
        self.assertIsNone(summarize(rows)['series']['vllm:x_total']['counter_delta'])

    def test_counter_delta(self):
        rows = [{'http_status': 200, 'response_received_perf_ns': i+1, 'raw_body': 'vllm:x_total '+str(v)} for i, v in enumerate([2, 5])]
        self.assertEqual(summarize(rows)['series']['vllm:x_total']['counter_delta'], 3)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            parse_metrics('vllm:x NaN')

    def test_iteration_counts_not_elapsed_fingerprint(self):
        text = 'Iteration(0): 1 context requests, 3 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 2.00 ms'
        r = parse_text(text)
        self.assertEqual(r['records'][0]['context_tokens'], 3)
        self.assertEqual(r['trajectory_sha256'], parse_text(text.replace('2.00', '5.00'))['trajectory_sha256'])

    def test_missing_iteration_logging_fails(self):
        self.assertTrue(parse_text('ordinary log')['errors'])

    def test_iteration_gap_detected(self):
        line = 'Iteration(0): 1 context requests, 3 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 2.00 ms'
        self.assertTrue(parse_text(line+'\n'+line.replace('(0)', '(2)'))['errors'])


class EvictionControlTests(unittest.TestCase):
    A = ([10, 11, 12, 13] * 32)

    def test_prompt_derivation_and_six_request_order(self):
        prompts = build_prompts(self.A)
        self.assertEqual(tuple(prompts), ORDER)
        self.assertEqual(prompts['A1'], prompts['A2'])
        self.assertEqual(prompts['A1'], prompts['A3'])
        self.assertEqual(len({tuple(prompts[name][:16]) for name in ('B1', 'B2', 'B3')}), 3)
        self.assertTrue(all(prompts[name][:16] != prompts['A1'][:16] for name in ('B1', 'B2', 'B3')))
        with self.assertRaises(ValueError):
            build_prompts([10] * 128)

    @staticmethod
    def metrics(counters, *, blocks=24, omit=None):
        labels = (f'block_size="16",cpu_offload_gb="0",enable_prefix_caching="True",'
                  f'kv_cache_memory_bytes="None",num_gpu_blocks="{blocks}",'
                  f'num_gpu_blocks_override="{blocks}"')
        lines = [f'vllm:cache_config_info{{{labels}}} 1',
                 'process_start_time_seconds 12345.25',
                 'vllm:num_requests_running{engine="0"} 0',
                 'vllm:num_requests_waiting{engine="0"} 0']
        names = {'queries': 'prefix_cache_queries_total', 'hits': 'prefix_cache_hits_total',
                 'prompt_tokens': 'prompt_tokens_total',
                 'computed_context_tokens': 'request_prefill_kv_computed_tokens_sum',
                 'prefill_requests': 'request_prefill_kv_computed_tokens_count',
                 'preemptions': 'num_preemptions_total'}
        lines += [f'vllm:{metric}{{engine="0"}} {counters[key]}'
                  for key, metric in names.items() if key != omit]
        return '\n'.join(lines) + '\n'

    def fake_case(self, *, fail_on=None, missing_after=False, delayed=False,
                  hits_by_request=None, initial_queries=0):
        test = self

        class Transport:
            def __init__(self):
                self.counters = dict.fromkeys(('queries', 'hits', 'prompt_tokens',
                                               'computed_context_tokens', 'prefill_requests',
                                               'preemptions'), 0)
                self.counters['queries'] = initial_queries
                self.pending = None
                self.polls = 0
                self.calls = []

            async def get_json(self, url, headers, timeout_s):
                if self.pending is not None:
                    self.polls += 1
                    if not delayed or self.polls >= 2:
                        name = self.pending
                        hits = (hits_by_request or {}).get(name, 112 if name == 'A2' else 0)
                        self.counters['queries'] += 128
                        self.counters['hits'] += hits
                        self.counters['prompt_tokens'] += 128
                        self.counters['computed_context_tokens'] += 128 - hits
                        self.counters['prefill_requests'] += 1
                        self.pending = None
                omit = 'hits' if missing_after and self.calls else None
                return 200, test.metrics(self.counters, omit=omit)

            async def stream_completion(self, url, headers, payload, timeout_s, *, request_id=None):
                name = request_id.rsplit('-', 1)[-1]
                self.calls.append((name, payload['cache_salt'], payload['prompt']))
                self.pending = name
                self.polls = 0
                yield 'event'

        transport = Transport()

        async def execute_completion_request(**kw):
            name = kw['request_id'].rsplit('-', 1)[-1]
            payload = {'prompt': kw['prompt_token_ids'], 'max_tokens': 2, 'min_tokens': 2}
            async for _ in kw['transport'].stream_completion(kw['url'], {}, payload, 1,
                                                              request_id=kw['request_id']):
                pass
            return {'status': 'incomplete' if name == fail_on else 'complete',
                    'output_token_ids': [20, 21], 'raw_sse_events': [{'raw_data': 'synthetic'}],
                    'prompt_token_ids_sent': kw['prompt_token_ids']}

        base = SimpleNamespace(execute_completion_request=execute_completion_request,
                               RealClock=lambda: object())
        return base, transport

    def run_case(self, *, expected_process_start=None, **kwargs):
        base, transport = self.fake_case(**kwargs)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'control.jsonl'
            writer = JsonlRecord(path)
            async def no_sleep(_):
                pass
            try:
                valid, assessments = asyncio.run(run_control(
                    base, transport, build_prompts(self.A), writer,
                    base_url='http://127.0.0.1:8000', api_key='test', model='model',
                    expected_blocks=24, sleep=no_sleep, max_metric_polls=3,
                    expected_process_start=expected_process_start))
            finally:
                writer.close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            with self.assertRaises(FileExistsError):
                JsonlRecord(path)
            return valid, assessments, rows, transport.calls

    def test_sequential_complete_records_and_delayed_metrics(self):
        valid, assessments, rows, calls = self.run_case(delayed=True)
        self.assertTrue(valid)
        self.assertEqual([call[0] for call in calls], list(ORDER))
        self.assertEqual([call[1] for call in calls], [SALT_A, SALT_A, SALT_B,
                                                       SALT_B, SALT_B, SALT_A])
        self.assertEqual(assessments['A2']['hits'], 112)
        self.assertEqual(assessments['A3']['computed_context_tokens'], 128)
        self.assertEqual(len([row for row in rows if row['record_type'] == 'request_result']), 6)
        self.assertEqual(len([row for row in rows if row['record_type'] == 'metrics_after_poll']), 12)

    def test_failed_sse_keeps_result_and_stops(self):
        valid, _, rows, calls = self.run_case(fail_on='B2')
        self.assertFalse(valid)
        self.assertEqual([call[0] for call in calls], ['A1', 'A2', 'B1', 'B2'])
        self.assertEqual(rows[-1]['record_type'], 'control_error')
        self.assertEqual([row['request'] for row in rows if row['record_type'] == 'request_result'][-1], 'B2')

    def test_missing_metrics_keeps_partial_data(self):
        valid, _, rows, calls = self.run_case(missing_after=True)
        self.assertFalse(valid)
        self.assertEqual(len(calls), 1)
        self.assertTrue(any(row['record_type'] == 'request_result' for row in rows))
        self.assertTrue(any(row['record_type'] == 'control_error' for row in rows))

    def test_cold_start_and_warmup_are_required(self):
        for hits, stopped_after in [({'A1': 112}, ['A1']),
                                    ({'A2': 0}, ['A1', 'A2']),
                                    ({'B1': 16}, ['A1', 'A2', 'B1']),
                                    ({'B2': 16}, ['A1', 'A2', 'B1', 'B2']),
                                    ({'B3': 16}, ['A1', 'A2', 'B1', 'B2', 'B3'])]:
            with self.subTest(hits=hits):
                valid, _, rows, calls = self.run_case(hits_by_request=hits)
                self.assertFalse(valid)
                self.assertEqual([call[0] for call in calls], stopped_after)
                self.assertIn('cache-hit prerequisite failed', rows[-1]['error'])

    def test_fresh_server_counters_required(self):
        valid, _, rows, calls = self.run_case(initial_queries=128)
        self.assertFalse(valid)
        self.assertEqual(calls, [])
        self.assertIn('fresh server', rows[-1]['error'])

    def test_metrics_process_identity_must_match_log_process(self):
        zero = dict.fromkeys(('queries', 'hits', 'prompt_tokens', 'computed_context_tokens',
                              'prefill_requests', 'preemptions'), 0)
        state = snapshot({'http_status': 200, 'raw_body': self.metrics(zero)})
        verify_server_identity(state, 12345.25)
        with self.assertRaisesRegex(ValueError, 'different API server'):
            verify_server_identity(state, 12350.25)
        valid, _, rows, calls = self.run_case(expected_process_start=12350.25)
        self.assertFalse(valid)
        self.assertEqual(calls, [])
        self.assertIn('different API server', rows[-1]['error'])
        verify_startup_time(12345, 12345.25)
        with self.assertRaisesRegex(ValueError, 'does not match startup log'):
            verify_startup_time(10000, 12345.25)

    def test_bad_cache_config_and_counter_reset_rejected(self):
        zero = dict.fromkeys(('queries', 'hits', 'prompt_tokens', 'computed_context_tokens',
                              'prefill_requests', 'preemptions'), 0)
        state = snapshot({'http_status': 200, 'raw_body': self.metrics(zero, blocks=64)})
        with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
            verify_config(state, 24, 16)
        later = snapshot({'http_status': 200, 'raw_body': self.metrics(zero)})
        earlier = snapshot({'http_status': 200, 'raw_body': self.metrics(dict(zero, hits=1))})
        with self.assertRaisesRegex(ValueError, 'counter reset'):
            metric_delta(earlier, later)

    def test_startup_log_requires_effective_capacity_and_length(self):
        body = ('UTC-Startzeit: 20261016T161712Z\n'
                'vLLM-Version: 0.17.1\nMax model len requested: 256\n'
                'Max num batched tokens (chunked-prefill budget): 512\n'
                'NUM_GPU_BLOCKS_OVERRIDE requested: 24\nPrefix-Caching: enabled\n'
                'Offload-Wert: 0\nChunked prefill is enabled with max_num_batched_tokens=512.\n'
                'Overriding num_gpu_blocks=2823 with '
                'num_gpu_blocks_override=24\nGPU KV cache size: 384 tokens\n'
                'Maximum concurrency for 256 tokens per request: 1.50x\n'
                '(APIServer pid=1234) INFO startup\n')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'server.log'
            path.write_text(body)
            log_hash, pid, log_start = verify_server_log(path, 24, 16, 256, 512)
            self.assertEqual(len(log_hash), 64)
            self.assertEqual(pid, 1234)
            self.assertGreater(log_start, 0)
            with self.assertRaisesRegex(ValueError, 'prefill budget'):
                verify_server_log(path, 24, 16, 256, 256)
            path.write_text(body.replace('(APIServer pid=1234)', '(EngineCore_DP0 pid=1234)'))
            with self.assertRaisesRegex(ValueError, 'unique API server PID'):
                verify_server_log(path, 24, 16, 256, 512)
            path.write_text(body.replace('384 tokens', '1,024 tokens'))
            with self.assertRaisesRegex(ValueError, 'capacity'):
                verify_server_log(path, 24, 16, 256, 512)

def endpoint_rows():
    rows = []
    for replicate, capacity, policy, budget, condition in itertools.product(('r1', 'r2', 'r3'), ('sufficient', 'near'), ('shared', 'isolated'), ('small', 'large'), ('reference', 'additional')):
        extra = {('shared', 'small'): 2, ('shared', 'large'): 4, ('isolated', 'small'): 7, ('isolated', 'large'): 3}[policy, budget]
        rows.append({'replicate_id': replicate, 'capacity_label': capacity, 'cache_policy': policy,
                     'budget_label': budget, 'condition': condition, 'endpoint_ms': 10+(extra if condition == 'additional' else 0),
                     'status': 'observed', 'cohort_contract_sha256': 'a'*64, 'workload_sha256': 'b'*64,
                     'nuisance_config_sha256': 'c'*64, 'model_revision': 'fixture-only',
                     'effective_kv_capacity_tokens': 100 if capacity == 'near' else 200,
                     'effective_max_num_batched_tokens': 4 if budget == 'small' else 8})
    return rows


class StatisticsTests(unittest.TestCase):
    def test_endpoint_table_adapter_retains_failed_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = {'predeclared_utc': '2026-10-10T00:00:00Z', 'model_revision': 'fixture',
                        'cohort_indices': [0], 'token_index_1based': 3, 'origin_field': 'trigger.trigger_perf_ns',
                        'cpu_offload_gb': 0, 'failure_policy': 'retain_invalid_no_numeric',
                        'workload_sha256': 'b'*64, 'budget_tokens': {'small': 4, 'large': 8},
                        'kv_capacity_tokens': {'near': 100, 'sufficient': 200},
                        'warmup_and_reset_protocol': 'fixture', 'independent_replicate_definition': 'fixture',
                        'precision_target': 'fixture'}
            cp = root/'contract.json'; cp.write_text(json.dumps(contract))
            e = episode(); e['victim_requests'][0] = {'request_index': 0, 'status': 'failed'}
            (root/'episode.json').write_text(json.dumps(e))
            mp = root/'map.json'; mp.write_text(json.dumps([{'episode_file': 'episode.json', 'condition': 'reference'}]))
            rows = build(cp, mp)
            self.assertEqual(rows[0]['status'], 'incomplete_or_invalid')
            self.assertIsNone(rows[0]['endpoint_ms'])
            self.assertEqual(len(rows[0]['cohort_contract_sha256']), 64)

    def test_positive_gamma_does_not_imply_reversal(self):
        rows = endpoint_rows()
        for row in rows:
            if row['cache_policy'] == 'isolated' and row['budget_label'] == 'small' and row['condition'] == 'additional':
                row['endpoint_ms'] = 12
        r = quantify(rows, bootstrap=100)['capacity_results']['near']
        self.assertEqual(r['effects']['Gamma']['mean_ms'], 1)
        self.assertFalse(r['point_estimate_reversal'])

    def test_effects_and_reversal_conditions(self):
        r = quantify(endpoint_rows(), bootstrap=100)['capacity_results']['near']
        self.assertEqual(r['effects']['E_shared']['mean_ms'], -2)
        self.assertEqual(r['effects']['E_isolated']['mean_ms'], 4)
        self.assertEqual(r['effects']['Gamma']['mean_ms'], 6)
        self.assertTrue(r['point_estimate_reversal'])
        self.assertFalse(r['protection_reversal_established'])

    def test_missing_cell_fails_without_exclusion(self):
        rows = endpoint_rows(); rows.pop()
        with self.assertRaises(ValueError):
            quantify(rows, bootstrap=100)

    def test_duplicate_condition_fails(self):
        rows = endpoint_rows(); rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            quantify(rows, bootstrap=100)

    def test_invalid_endpoint_not_excluded(self):
        rows = endpoint_rows(); rows[0].update(status='failed', endpoint_ms=None)
        with self.assertRaisesRegex(ValueError, 'invalid_endpoint'):
            quantify(rows, bootstrap=100)

    def test_capacity_mismatch(self):
        rows = endpoint_rows(); rows[0]['effective_kv_capacity_tokens'] += 1
        with self.assertRaisesRegex(ValueError, 'capacity_not_constant'):
            quantify(rows, bootstrap=100)

    def test_missing_common_field(self):
        rows = endpoint_rows(); del rows[0]['workload_sha256']
        with self.assertRaises(ValueError):
            quantify(rows, bootstrap=100)

    def test_template_cannot_pass_gate(self):
        obj = json.loads((SCRIPTS/'contracts/study_contract.template.json').read_text())
        r = validate(obj); self.assertTrue(r['errors']); self.assertFalse(r['runtime_gate_passed'])


class StarterTests(unittest.TestCase):
    def test_actual_shell_preserves_start_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); dest = root/'new/scripts/cache_isolation_gate'; dest.mkdir(parents=True)
            (dest/'run_server.sh').write_bytes((SCRIPTS/'run_server.sh').read_bytes())
            binary = root/'bin'; binary.mkdir()
            fake = binary/'vllm'
            fake.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo 0.17.1; else python3 -c "import sys,json;print(json.dumps(sys.argv[1:]))" "$@"; fi\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ['PATH'], MAX_NUM_BATCHED_TOKENS='1024')
            env.pop('KV_CACHE_MEMORY_BYTES', None)
            env.pop('NUM_GPU_BLOCKS_OVERRIDE', None)
            env.pop('MAX_MODEL_LEN', None)
            proc = subprocess.run(['bash', str(dest/'run_server.sh'), 'llama', '0'], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            command = json.loads(proc.stdout.splitlines()[-1])
            self.assertEqual(command[:2], ['serve', 'meta-llama/Llama-3.1-8B-Instruct'])
            for flag, value in [('--seed', '20260711'), ('--max-num-seqs', '16'), ('--max-model-len', '8192'), ('--cpu-offload-gb', '0'), ('--max-num-batched-tokens', '1024')]:
                self.assertEqual(command[command.index(flag)+1], value)
            self.assertIn('--enable-prefix-caching', command)
            self.assertIn('--enable-logging-iteration-details', command)
            self.assertIn('--enable-prompt-tokens-details', command)
            self.assertNotIn('--no-enable-prefix-caching', command)
            proc = subprocess.run(['bash', str(dest/'run_server.sh'), 'llama', '8'], env=env, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)

            control_env = dict(env, MAX_MODEL_LEN='256', NUM_GPU_BLOCKS_OVERRIDE='24',
                               MAX_NUM_BATCHED_TOKENS='512')
            proc = subprocess.run(['bash', str(dest/'run_server.sh'), 'llama', '0'],
                                  env=control_env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            command = json.loads(proc.stdout.splitlines()[-1])
            self.assertEqual(command[command.index('--max-model-len') + 1], '256')
            self.assertEqual(command[command.index('--num-gpu-blocks-override') + 1], '24')
            self.assertEqual(command[command.index('--max-num-batched-tokens') + 1], '512')
            for additions in ({'KV_CACHE_MEMORY_BYTES': '1234'},
                              {'NUM_GPU_BLOCKS_OVERRIDE': '0'},
                              {'NUM_GPU_BLOCKS_OVERRIDE': 'abc'},
                              {'MAX_MODEL_LEN': '0'},
                              {'MAX_MODEL_LEN': 'abc'}):
                with self.subTest(additions=additions):
                    bad_env = dict(control_env, **additions)
                    proc = subprocess.run(['bash', str(dest/'run_server.sh'), 'llama', '0'],
                                          env=bad_env, capture_output=True, text=True)
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertNotIn('"serve"', proc.stdout)
            # A frozen date forces a collision with a real existing log.
            date = binary/'date'
            date.write_text('#!/bin/sh\necho 20261017T100000Z\n')
            date.chmod(0o755)
            existing = root/'new/logs/cache_isolation_gate/server/server_llama_offload0_budget512_20261017T100000Z.log'
            existing.write_text('preserve existing evidence\n')
            proc = subprocess.run(['bash', str(dest/'run_server.sh'), 'llama', '0'],
                                  env=control_env, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(existing.read_text(), 'preserve existing evidence\n')
            self.assertNotIn('"serve"', proc.stdout)


if __name__ == '__main__':
    unittest.main()
