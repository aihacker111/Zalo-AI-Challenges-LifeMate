import unittest

from tests.helpers import requires_data, split

from lifemate.agent.prompt import PromptBuilder
from lifemate.train.augment import rename_sample


@requires_data
class AugmentTest(unittest.TestCase):
    def test_renamed_calls_fit_renamed_specs(self):
        sp = split("train")
        builder = PromptBuilder(sp)
        for s in sp.samples[:400]:
            new, specs, _ = rename_sample(s, builder.specs_for(s), sp.apps, seed=1)
            by_name = {x.name: x for x in specs}
            for c in new.gold_calls:
                spec = by_name[c["name"]]
                self.assertLessEqual(set(c["arguments"]), set(spec.properties))
                if c["name"] == "sys.ask_missing_info":
                    target = by_name[c["arguments"]["tool_name"]]
                    self.assertLessEqual(set(c["arguments"]["params"]), set(target.properties))

    def test_renaming_is_deterministic(self):
        sp = split("train")
        s = sp.samples[0]
        specs = PromptBuilder(sp).specs_for(s)
        self.assertEqual(rename_sample(s, specs, sp.apps, 3)[0], rename_sample(s, specs, sp.apps, 3)[0])


if __name__ == "__main__":
    unittest.main()
