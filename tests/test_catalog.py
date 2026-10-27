import unittest

from tests.helpers import requires_data, split

from lifemate.catalog import action_schema, load_specs, render_tool


@requires_data
class CatalogTest(unittest.TestCase):
    def test_every_tool_renders(self):
        for name in ("train", "public_test"):
            for spec in load_specs(split(name).tools).values():
                text = render_tool(spec)
                self.assertTrue(text.startswith(f"- {spec.name}("), text)

    def test_gold_calls_fit_the_decoding_schema(self):
        sp = split("train")
        specs = load_specs(sp.tools)
        for s in sp:
            for c in s.gold_calls:
                spec = specs[c["name"]]
                self.assertLessEqual(set(c["arguments"]), set(spec.properties), (s.id, c))
                self.assertLessEqual(set(spec.required), set(c["arguments"]), (s.id, c))

    def test_action_schema_has_finish(self):
        specs = list(load_specs(split("train").tools).values())[:3]
        variants = action_schema(specs)["anyOf"]
        self.assertEqual(len(variants), 4)
        self.assertEqual(variants[-1]["properties"]["name"]["const"], "finish")


if __name__ == "__main__":
    unittest.main()
