import unittest

from status_freshness import FeedbackFreshness


class FeedbackFreshnessTests(unittest.TestCase):
    def test_first_sequence_starts_fresh_and_age_advances_without_new_feedback(self):
        tracker = FeedbackFreshness(timeout_s=2.0)
        self.assertEqual(tracker.observe(41, 10.0), (True, 0))
        self.assertEqual(tracker.observe(41, 11.25), (True, 1250))
        self.assertEqual(tracker.observe(41, 12.01), (False, 2010))

    def test_new_sequence_refreshes_stale_feedback(self):
        tracker = FeedbackFreshness(timeout_s=1.0)
        tracker.observe(1, 2.0)
        self.assertEqual(tracker.observe(1, 3.1), (False, 1100))
        self.assertEqual(tracker.observe(2, 3.2), (True, 0))

    def test_missing_sequence_never_claims_live_data(self):
        tracker = FeedbackFreshness()
        self.assertEqual(tracker.observe(None, 5.0), (False, None))


if __name__ == "__main__":
    unittest.main()
