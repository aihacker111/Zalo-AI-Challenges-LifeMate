import unittest

from tests.helpers import requires_data, split

from lifemate.eval.scorer import Scorer, f1_score, match_calls, summarize


class MatchingTest(unittest.TestCase):
    fuzzy = {"sms.send": {"content"}}

    def test_fuzzy_and_exact_arguments(self):
        gold = [{"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "mai họp lúc 9 giờ nhé"}}]
        close = [{"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "mai họp lúc 9 giờ"}}]
        wrong_id = [{"name": "sms.send", "arguments": {"contact_id": "c_02", "content": "mai họp lúc 9 giờ nhé"}}]
        extra_arg = [{"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "mai họp lúc 9 giờ nhé", "x": 1}}]
        self.assertEqual(len(match_calls(close, gold, self.fuzzy)), 1)
        self.assertEqual(match_calls(wrong_id, gold, self.fuzzy), [])
        self.assertEqual(match_calls(extra_arg, gold, self.fuzzy), [])

    def test_types_matter(self):
        gold = [{"name": "a.b", "arguments": {"t": 26}}]
        self.assertEqual(match_calls([{"name": "a.b", "arguments": {"t": "26"}}], gold, {}), [])
        self.assertEqual(match_calls([{"name": "a.b", "arguments": {"t": True}}], [{"name": "a.b", "arguments": {"t": 1}}], {}), [])

    def test_one_to_one_maximum_matching(self):
        g = {"name": "a.read", "arguments": {}}
        self.assertEqual(len(match_calls([g, g, g], [g, g], {})), 2)

    def test_f1(self):
        self.assertAlmostEqual(f1_score(3, 4, 3), 6 / 7)
        self.assertAlmostEqual(f1_score(2, 2, 3), 0.8)
        self.assertEqual(f1_score(0, 1, 1), 0.0)


@requires_data
class ScorerOnTrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = split("train")
        cls.scorer = Scorer(cls.split)

    def test_gold_scores_100(self):
        gold = {s.id: s.gold_calls for s in self.split}
        self.assertEqual(summarize(self.scorer.score_predictions(gold))["score"], 100.0)

    def test_empty_submission_scores_out_of_scope_share(self):
        n_empty = sum(not s.gold_calls for s in self.split)
        score = summarize(self.scorer.score_predictions({}))["score"]
        self.assertAlmostEqual(score, 100 * n_empty / len(self.split))

    def test_extra_and_missing_read(self):
        s = next(x for x in self.split if len(x.gold_calls) == 3 and x.gold_calls[0]["name"] == "smart_home.get_status")
        extra = self.scorer.score_sample(s.id, s.gold_calls + [s.gold_calls[0]])
        missing = self.scorer.score_sample(s.id, s.gold_calls[1:])
        self.assertTrue(extra.hash_ok)
        self.assertAlmostEqual(extra.score, 6 / 7)
        self.assertAlmostEqual(missing.score, 0.8)

    def test_wrong_write_scores_zero(self):
        s = next(x for x in self.split if any(c["name"] == "smart_home.light_set" for c in x.gold_calls))
        calls = [dict(c) for c in s.gold_calls]
        for c in calls:
            if c["name"] == "smart_home.light_set":
                c["arguments"] = {**c["arguments"], "power": "on" if c["arguments"].get("power") == "off" else "off"}
        r = self.scorer.score_sample(s.id, calls)
        self.assertFalse(r.hash_ok)
        self.assertEqual(r.score, 0.0)

    def test_two_splits_in_one_process(self):
        public = split("public_test")
        self.assertNotEqual(public.zaic.__name__, self.split.zaic.__name__)
        self.assertEqual(len(public.open_session(public.samples[0].id).get_state_hash()), 64)


if __name__ == "__main__":
    unittest.main()
