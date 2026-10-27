import unittest
from datetime import datetime

from tests.helpers import REPO  # noqa: F401

from lifemate.timeutil import day_table, normalize_datetime, utc_offset


class TimeTest(unittest.TestCase):
    def test_normalize_datetime(self):
        self.assertEqual(normalize_datetime("2026-11-20T07:00", "+07:00"), "2026-11-20T07:00:00+07:00")
        self.assertEqual(normalize_datetime("2026-11-20 7:05:30", "+07:00"), "2026-11-20T07:05:30+07:00")
        self.assertEqual(normalize_datetime("2026-11-20T07:00:00+0700", "+07:00"), "2026-11-20T07:00:00+07:00")
        self.assertEqual(normalize_datetime("ngày mai", "+07:00"), "ngày mai")

    def test_day_table(self):
        now = datetime.fromisoformat("2026-11-28T07:10:00+07:00")  # a Saturday
        table = day_table(now, 2).splitlines()
        self.assertEqual(table[0], "hôm nay (thứ Bảy) = 2026-11-28")
        self.assertEqual(table[1], "ngày mai (Chủ nhật) = 2026-11-29")
        self.assertEqual(utc_offset(now), "+07:00")


if __name__ == "__main__":
    unittest.main()
