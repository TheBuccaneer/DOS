"""Adversarial checks of the new offline reconstruction, no GPU or network."""
import copy
import json
import unittest
from pathlib import Path
from reanalyze import analyze_episode, metrics, stream
from counter_semantics import run_cases
from conditional_observer import gaussian, uniform

REPO = Path(__file__).resolve().parents[4]
RAW = REPO / 'new/runs/cache_isolation_gate'


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(x) for x in (RAW / 'overlap_shared_b64_20261017T114419Z.jsonl').read_text().splitlines()]

    def test_actual_stream(self):
        a = analyze_episode(self.rows)
        self.assertEqual(a['independent_integrity_errors'], [])
        self.assertAlmostEqual(a['endpoint_ms'], 422.003697)

    def test_missing_done(self):
        x = [r for r in self.rows if r['record_type'] == 'sse_wire' and r['role'] == 'D']
        self.assertIn('missing_done', stream(x[:-1])['errors'])

    def test_missing_token(self):
        x = [r for r in self.rows if r['record_type'] == 'sse_wire' and r['role'] == 'D']
        self.assertIn('usage_token_mismatch', stream(x[1:])['errors'])

    def test_malformed(self):
        x = [dict(raw_data='{', receive_perf_counter_ns=1)]
        self.assertIn('malformed_json', stream(x)['errors'])

    def test_missing_metric_is_not_zero(self):
        with self.assertRaises(ValueError):
            metrics('', 'vllm:num_preemptions_total')

    def test_counter_labels(self):
        self.assertEqual(metrics('vllm:num_preemptions_total{engine="0"} 2\n',
                                 'vllm:num_preemptions_total'), 2)

    def test_changed_stored_endpoint(self):
        x = copy.deepcopy(self.rows)
        x[-1]['assessment']['raw_endpoint_ms'] = 0
        self.assertIn('endpoint_copy_mismatch', analyze_episode(x)['independent_integrity_errors'])

    def test_warm_victim_rejected(self):
        x = copy.deepcopy(self.rows)
        for r in x:
            if r['record_type'] == 'sse_wire' and r['role'] == 'D' and r['raw_data'] != '[DONE]':
                d = json.loads(r['raw_data'])
                if d.get('usage'):
                    d['usage']['prompt_tokens_details'] = {'cached_tokens': 112}
                    r['raw_data'] = json.dumps(d)
        self.assertIn('D_not_cold', analyze_episode(x)['independent_integrity_errors'])

    def test_preemption_preserves_measurement_and_abstains(self):
        p = RAW / 'overlap_isolated_n24_b64_20261017T121844Z.jsonl'
        a = analyze_episode([json.loads(x) for x in p.read_text().splitlines()])
        self.assertTrue(a['measurement_complete'])
        self.assertFalse(a['runner_valid'])
        self.assertEqual(a['preemptions'], 1)
        self.assertEqual(a['mechanism_attribution'], 'unresolved')
        self.assertFalse(a['causal_effect_estimated'])

    def test_scheduled_generation_counter_is_not_new_output(self):
        cases = {x['case']: x['result'] for x in run_cases()}
        self.assertEqual(cases['positive_output_request_17_scheduled']['num_generation_tokens'], 17)
        self.assertEqual(cases['positive_output_request_17_scheduled']['num_ctx_tokens'], 0)
        self.assertEqual(cases['new_request_17_scheduled']['num_ctx_tokens'], 17)
        self.assertEqual(cases['zero_output_request_17_scheduled']['num_ctx_tokens'], 17)

    def test_raw_generation_residual_retained(self):
        p = RAW / 'overlap_isolated_n24_b64_20261017T121844Z.jsonl'
        a = analyze_episode([json.loads(x) for x in p.read_text().splitlines()])
        self.assertEqual(a['iteration_generation_tokens'], 110)
        self.assertEqual(a['new_output_tokens_after_first'], 94)
        self.assertEqual(a['generation_accounting_residual'], 16)

    def test_conditional_observer_negative_control(self):
        a = gaussian(20, 200, 2, 2, correlated=False)
        self.assertEqual(a['marginal_balanced_bayes_accuracy'],
                         a['conditional_balanced_bayes_accuracy'])

    def test_uniform_analytical_witness(self):
        a = uniform(20, 1000, 2)
        self.assertEqual(a['marginal_overlap'], 0.98)
        self.assertEqual(a['marginal_balanced_bayes_accuracy'], 0.51)
        self.assertTrue(a['residual_supports_disjoint'])

    def test_weak_reference_does_not_prove_separation(self):
        self.assertFalse(uniform(20, 1000, 20)['residual_supports_disjoint'])


if __name__ == '__main__':
    unittest.main()
