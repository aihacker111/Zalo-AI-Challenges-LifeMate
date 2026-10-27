import unittest

from tests.helpers import REPO  # noqa: F401

from lifemate.agent.parser import ParseError, parse_action


class ParserTest(unittest.TestCase):
    def test_plain_json(self):
        a = parse_action('{"name": "sms.send", "arguments": {"contact_id": "c_01", "content": "hi"}}')
        self.assertEqual((a.name, a.arguments["contact_id"]), ("sms.send", "c_01"))

    def test_wrapped_outputs(self):
        self.assertEqual(parse_action('<think>\n\n</think>\n```json\n{"name": "alarm.list", "arguments": {}}\n```').name, "alarm.list")
        self.assertEqual(parse_action('<tool_call>{"name": "tv.guide", "arguments": "{}"}</tool_call>').name, "tv.guide")
        self.assertEqual(parse_action('Gọi: {"name": "x.y", "arguments": {"t": "a}b"}} xong').arguments, {"t": "a}b"})

    def test_finish(self):
        self.assertTrue(parse_action('{"name": "finish"}').is_finish)
        self.assertTrue(parse_action("FINISH").is_finish)

    def test_errors(self):
        for bad in ["", "không có json", '{"arguments": {}}', '{"name": "a.b", "arguments": [1]}']:
            with self.assertRaises(ParseError):
                parse_action(bad)


if __name__ == "__main__":
    unittest.main()
