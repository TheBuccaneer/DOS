import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_primary_endpoint import extract_episode


def episode():
    rows = []
    for index, final in [(0, 5_000_000), (1, 7_000_000)]:
        rows.append({"request_index": index, "status": "complete",
                     "request_start_ns": 0, "stream_end_ns": 8_000_000,
                     "output_token_ids": [10, 11, 12, 13],
                     "raw_sse_events": [
                         {"event_index": 0, "receive_perf_counter_ns": 1_000_000, "token_ids": [10]},
                         {"event_index": 1, "receive_perf_counter_ns": final, "token_ids": [11, 12, 13]}]})
    return {"episode_id": "fixture", "block_id": "block", "status": "complete",
            "schedule_row": {"condition": "reference"}, "trigger": {"trigger_perf_ns": 2_000_000},
            "victim_requests": rows}


class EndpointTests(unittest.TestCase):
    def run_extract(self, data=None, target=3):
        return extract_episode(data if data is not None else episode(), [0, 1], target, "trigger.trigger_perf_ns")

    def test_grouped_event_and_slowest_member(self):
        result = self.run_extract()
        self.assertEqual(result["cohort_elapsed_ms"], 5.0)
        self.assertEqual(result["members"][0]["target_receipt_ns"], 5_000_000)
        self.assertFalse(result["design_validity_checked"])

    def test_unreached_target_is_retained(self):
        result = self.run_extract(target=5)
        self.assertEqual(result["status"], "incomplete_or_invalid")
        self.assertIsNone(result["cohort_elapsed_ms"])
        self.assertEqual(len(result["members"]), 2)

    def test_missing_member_does_not_shrink_cohort(self):
        data = episode()
        data["victim_requests"].pop()
        result = self.run_extract(data)
        self.assertIn("request_1:missing_request", result["issues"])
        self.assertIsNone(result["cohort_elapsed_ms"])

    def test_target_before_origin_is_invalid(self):
        result = self.run_extract(target=1)
        self.assertEqual(result["status"], "incomplete_or_invalid")

    def test_nonmonotonic_trace_is_rejected(self):
        data = episode()
        data["victim_requests"][0]["raw_sse_events"][1]["receive_perf_counter_ns"] = 1
        with self.assertRaises(ValueError):
            self.run_extract(data)

    def test_token_reconstruction_mismatch_is_rejected(self):
        data = episode()
        data["victim_requests"][0]["output_token_ids"] = [999]
        with self.assertRaises(ValueError):
            self.run_extract(data)

    def test_duplicate_request_identity_is_rejected(self):
        data = episode()
        data["victim_requests"].append(copy.deepcopy(data["victim_requests"][0]))
        with self.assertRaises(ValueError):
            self.run_extract(data)

    def test_failed_episode_has_no_primary_value(self):
        data = episode()
        data["status"] = "failed"
        result = self.run_extract(data)
        self.assertIsNone(result["cohort_elapsed_ms"])


if __name__ == "__main__":
    unittest.main()
