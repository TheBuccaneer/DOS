"""Adversarial plumbing checks, not a simulated serving-success study."""
import copy
import json
from pathlib import Path
import random
import unittest

from h7b import (SCHEMA, FEATURES, analyze, prepare, public_timing,
                 checked_stream, causal_gates, kde_overlap)


def events(times, prompt, cached=0):
    result = []
    for i, t in enumerate(times):
        payload = dict(choices=[dict(token_ids=[i + 10])])
        if i == len(times) - 1:
            payload['usage'] = dict(prompt_tokens=prompt, completion_tokens=len(times),
                                    prompt_tokens_details=dict(cached_tokens=cached))
        result.append(dict(raw_data=json.dumps(payload), receive_perf_counter_ns=int(t)))
    result.append(dict(raw_data='[DONE]', receive_perf_counter_ns=int(times[-1]) + 1))
    return result


def fixture(n_cal=32, n_shared=64, n_isolated=32):
    rng = random.Random(20261017)
    labels = []
    for _ in range((n_cal + n_shared + n_isolated + 15) // 16):
        block = [0, 1] * 8
        rng.shuffle(block)
        labels.extend(block)
    result = []
    for i in range(n_cal + n_shared + n_isolated):
        bit = labels[i]
        mode = 'isolated' if i >= n_cal + n_shared else 'shared'
        load = (i // 4) % 2
        reference_ms = 5 + load * 3 + rng.random() * .2
        hit = bool(bit and mode == 'shared')
        ttft_ms = 25 + load * 28 - int(hit) * 8 + rng.random() * 5
        base = (i + 1) * 1_000_000_000
        d_times = [base + int((1 + j * reference_ms) * 1e6) for j in range(32)]
        dispatch = d_times[7] + 1
        result.append(dict(schema=SCHEMA, trial_id=f'constructed-{i}',
                           time_block=f'block-{i // 16}',
                           source_kind='synthetic', namespace=mode,
                           split='calibration' if i < n_cal else 'test',
                           query_dispatch_ns=dispatch, trigger_receipt_ns=d_times[7],
                           reference_events=events(d_times, 192),
                           query_events=events([dispatch + int(ttft_ms * 1e6)],
                                               512, 496 if hit else 0),
                           evaluator=dict(secret_bit=bit)))
    return result


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.r = fixture()[0]

    def project(self, r):
        return public_timing(r['reference_events'], r['query_events'],
                             r['query_dispatch_ns'], r['trigger_receipt_ns'])

    def test_public_fields_are_an_explicit_allowlist(self):
        self.assertEqual(set(self.project(self.r)), {'ttft_ms', 'reference_pre_ms'})

    def test_cache_counters_and_labels_cannot_change_public_features(self):
        r = copy.deepcopy(self.r)
        r['evaluator']['secret_bit'] ^= 1
        q = json.loads(r['query_events'][0]['raw_data'])
        q['usage']['prompt_tokens_details']['cached_tokens'] = 496
        r['query_events'][0]['raw_data'] = json.dumps(q)
        self.assertEqual(self.project(self.r), self.project(r))

    def test_post_dispatch_reference_does_not_change_features(self):
        r = copy.deepcopy(self.r)
        for e in r['reference_events'][8:]:
            e['receive_perf_counter_ns'] += 1_000_000_000
        self.assertEqual(self.project(self.r), self.project(r))

    def test_missing_done_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing_done'):
            checked_stream(self.r['query_events'][:-1])

    def test_nonmonotone_sse_rejected(self):
        r = copy.deepcopy(self.r)
        r['reference_events'][2]['receive_perf_counter_ns'] = 0
        with self.assertRaisesRegex(ValueError, 'nonmonotone'):
            self.project(r)

    def test_coalesced_reference_endpoint_rejected(self):
        r = copy.deepcopy(self.r)
        e = r['reference_events']
        a, b = json.loads(e[6]['raw_data']), json.loads(e[7]['raw_data'])
        a['choices'][0]['token_ids'] += b['choices'][0]['token_ids']
        e[6]['raw_data'] = json.dumps(a)
        del e[7]
        with self.assertRaisesRegex(ValueError, 'individually timed'):
            self.project(r)

    def test_early_query_output_rejected(self):
        r = copy.deepcopy(self.r)
        r['query_dispatch_ns'] = r['reference_events'][7]['receive_perf_counter_ns']
        r['query_events'][0]['receive_perf_counter_ns'] = r['query_dispatch_ns'] - 1
        with self.assertRaisesRegex(ValueError, 'precedes dispatch'):
            self.project(r)

    def test_wrong_trigger_rejected(self):
        self.r['trigger_receipt_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'not reference token 8'):
            self.project(self.r)


class DesignTests(unittest.TestCase):
    def test_legacy_cannot_be_rebranded(self):
        rows = fixture()
        rows[0]['schema'] = 'dos-overlap-pilot-v1'
        with self.assertRaisesRegex(ValueError, 'legacy'):
            prepare(rows, allow_synthetic=True)

    def test_synthetic_requires_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            prepare(fixture())

    def test_duplicate_trial_rejected(self):
        rows = fixture()
        rows[1]['trial_id'] = rows[0]['trial_id']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prepare(rows, allow_synthetic=True)

    def test_block_leakage_rejected(self):
        rows = fixture()
        rows[32]['time_block'] = rows[0]['time_block']
        with self.assertRaisesRegex(ValueError, 'crosses train/test'):
            prepare(rows, allow_synthetic=True)

    def test_calibration_future_rejected(self):
        rows = fixture()
        rows[-1]['split'] = 'calibration'
        rows[-1]['namespace'] = 'shared'
        rows[-1]['time_block'] = 'late-calibration'
        with self.assertRaisesRegex(ValueError, 'before test'):
            prepare(rows, allow_synthetic=True)

    def test_missing_cache_truth_is_not_zero(self):
        rows = fixture()
        e = rows[0]['query_events'][0]
        d = json.loads(e['raw_data'])
        del d['usage']['prompt_tokens_details']
        e['raw_data'] = json.dumps(d)
        with self.assertRaisesRegex(ValueError, 'missing evaluator'):
            prepare(rows, allow_synthetic=True)

    def test_manipulation_failure_retained(self):
        rows = fixture()
        rows[40]['evaluator']['secret_bit'] ^= 1
        result = prepare(rows, allow_synthetic=True)
        self.assertEqual(len(result), 128)
        self.assertFalse(result[40]['manipulation_ok'])


class CausalityTests(unittest.TestCase):
    def test_future_and_current_samples_do_not_change_gate(self):
        rows = prepare(fixture(), allow_synthetic=True)
        base = causal_gates(rows, 10)['constructed-40']
        mutated = copy.deepcopy(rows)
        for r in mutated[40:]:
            r['public']['ttft_ms'] += 10000
            r['reuse_hit'] ^= 1
        self.assertEqual(base, causal_gates(mutated, 10)['constructed-40'])

    def test_unfinished_past_query_is_excluded(self):
        rows = prepare(fixture(), allow_synthetic=True)
        rows[39]['complete_ns'] = rows[41]['dispatch_ns'] + 1
        a = causal_gates(rows, 10)['constructed-40']
        rows[39]['public']['ttft_ms'] += 10000
        self.assertEqual(a, causal_gates(rows, 10)['constructed-40'])

    def test_single_class_gate_is_unknown(self):
        self.assertIsNone(kde_overlap([1, 2, 3, 4], [], 1))

    def test_kde_calibration_same_distribution_and_separated(self):
        a = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(kde_overlap(a, a, 1), 1, places=6)
        self.assertLess(kde_overlap(a, [100, 101, 102, 103], 1), 1e-9)

    def test_kde_remote_tail_does_not_erase_common_modes(self):
        overlap = kde_overlap([0, 1, 2, 1_000_000],
                              [0, 1, 2, 2_000_000], 1)
        self.assertGreater(overlap, .74)
        self.assertLess(overlap, .76)

    def test_attacker_history_ignores_unfinished_queries(self):
        raw = fixture()
        original = prepare(raw, allow_synthetic=True)[40]['public']['past_ttft_median_ms']
        raw[39]['query_events'][-1]['receive_perf_counter_ns'] = raw[41]['query_dispatch_ns'] + 1
        adjusted = prepare(raw, allow_synthetic=True)
        expected = sorted(adjusted[:39], key=lambda r: r['complete_ns'])[-4:]
        import statistics
        self.assertEqual(adjusted[40]['public']['past_ttft_median_ms'],
                         statistics.median(r['public']['ttft_ms'] for r in expected))
        self.assertNotEqual(original, adjusted[40]['public']['past_ttft_median_ms'])


class ModelBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fixture()
        cls.result = analyze(cls.rows, allow_synthetic=True)

    def test_learned_baselines_present_without_privileged_features(self):
        all_fields = {f for fields in FEATURES.values() for f in fields}
        self.assertEqual(all_fields, {'ttft_ms', 'past_ttft_median_ms', 'reference_pre_ms'})
        scores = self.result['per_trial'][0]['predictions']
        self.assertIn('ttft_learned', scores)
        self.assertIn('temporal_learned', scores)
        self.assertIn('joint_learned', scores)

    def test_test_labels_do_not_train_or_select(self):
        rows = copy.deepcopy(self.rows)
        for r in rows:
            if r['split'] == 'test':
                r['evaluator']['secret_bit'] ^= 1
        changed = analyze(rows, allow_synthetic=True)
        self.assertEqual(self.result['per_trial'], changed['per_trial'] if
                         not changed['manipulation_failures'] else
                         [dict(r, manipulation_ok=True) for r in changed['per_trial']])

    def test_constructed_data_cannot_be_reported_as_confirmation(self):
        self.assertEqual(self.result['evidence_kind'], 'synthetic_plumbing_only')
        self.assertFalse(self.result['hypothesis_demonstrated'])
        self.assertFalse(self.result['submission_ready'])
        self.assertEqual(self.result['n_test'], 96)

    def test_permutation_changes_selection_without_changing_block_coverage(self):
        trials = self.result['per_trial']
        changed = sum(r['preselected'] != r['preselected_reference_permuted']
                      for r in trials)
        self.assertGreater(changed, 0)
        for namespace in ('shared', 'isolated'):
            view = self.result['views'][namespace]
            self.assertEqual(view['preselected']['joint_learned']['n'],
                             view['preselected_reference_permuted'][
                                 'joint_reference_permuted']['n'])
            self.assertGreater(view['preselected_reference_permuted'][
                'selection_changed_n'], 0)
        for block in {r['time_block'] for r in self.rows if r['split'] == 'test'}:
            ids = {r['trial_id'] for r in self.rows if r['time_block'] == block}
            subset = [r for r in trials if r['trial_id'] in ids]
            self.assertEqual(sum(r['preselected'] for r in subset),
                             sum(r['preselected_reference_permuted'] for r in subset))

    def test_future_test_timing_cannot_change_prior_prediction(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]['query_events'][0]['receive_perf_counter_ns'] += 10_000_000
        rows[-1]['query_events'][1]['receive_perf_counter_ns'] += 10_000_000
        changed = analyze(rows, allow_synthetic=True)
        self.assertEqual(self.result['per_trial'][0], changed['per_trial'][0])


if __name__ == '__main__':
    unittest.main()
