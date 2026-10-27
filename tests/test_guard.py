import unittest

from tests.helpers import FakeSession

from lifemate.agent.guard import Guard, coerce_value
from lifemate.agent.parser import Action
from lifemate.catalog import ToolSpec
from lifemate.config import GuardConfig

READ = ToolSpec("contacts.search", "contacts", "read", "", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
WRITE = ToolSpec(
    "sms.send",
    "sms",
    "write",
    "",
    {
        "type": "object",
        "properties": {
            "contact_id": {"type": "string", "x-ref": "contacts.contacts", "description": "lấy từ contacts.search"},
            "content": {"type": "string", "x-match": "fuzzy"},
            "when": {"type": "string", "format": "date-time"},
            "count": {"type": "integer"},
        },
        "required": ["contact_id", "content"],
    },
)


def make_guard(valid=True, seen=()):
    return Guard(session=FakeSession(valid), specs={s.name: s for s in (READ, WRITE)}, allowed={READ.name, WRITE.name}, offset="+07:00", cfg=GuardConfig(), seen=set(seen))


class CoerceTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(coerce_value("2.000.000", {"type": "integer"}, "+07:00"), 2000000)
        self.assertEqual(coerce_value(26.0, {"type": "integer"}, "+07:00"), 26)
        self.assertEqual(coerce_value(912345678, {"type": "string"}, "+07:00"), "912345678")
        self.assertEqual(coerce_value("2026-11-20T07:00", {"type": "string", "format": "date-time"}, "+07:00"), "2026-11-20T07:00:00+07:00")
        self.assertEqual(coerce_value("x", {"type": "array", "items": {"type": "string"}}, "+07:00"), ["x"])


class GuardTest(unittest.TestCase):
    def test_unknown_tool_rejected(self):
        self.assertFalse(make_guard().review(Action("bank.transfer", {})).execute)

    def test_unseen_reference_rejected_then_allowed(self):
        g = make_guard()
        send = Action("sms.send", {"contact_id": "c_08", "content": "hi"})
        self.assertFalse(g.review(send).execute)
        g.record(Action("contacts.search", {"query": "Dũng"}), {"contacts": [{"id": "c_08"}]})
        self.assertTrue(g.review(send).execute)

    def test_duplicate_read_returns_cached_result(self):
        g = make_guard()
        a = Action("contacts.search", {"query": "Lan"})
        self.assertTrue(g.review(a).execute)
        g.record(a, {"contacts": []})
        d = g.review(Action("contacts.search", {"query": "Lan"}))
        self.assertFalse(d.execute)
        self.assertEqual(d.cached, {"contacts": []})

    def test_repair_drops_unknown_and_empty_optional_args(self):
        d = make_guard(seen={"c_01"}).review(Action("sms.send", {"contact_id": "c_01", "content": "hi", "when": "", "bogus": 1, "count": "3"}))
        self.assertTrue(d.execute)
        self.assertEqual(d.action.arguments, {"contact_id": "c_01", "content": "hi", "count": 3})

    def test_schema_error_not_executed(self):
        self.assertFalse(make_guard(valid=False, seen={"c_01"}).review(Action("sms.send", {"contact_id": "c_01", "content": "hi"})).execute)


if __name__ == "__main__":
    unittest.main()
