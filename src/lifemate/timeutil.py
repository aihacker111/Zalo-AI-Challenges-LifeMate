"""Dates and times: a relative-day table for the prompt and ISO-8601 normalisation for arguments."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

WEEKDAYS_VI = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật"]
RELATIVE_VI = {0: "hôm nay", 1: "ngày mai", 2: "ngày kia"}

_DT_LOOSE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[T ](?P<h>\d{1,2}):(?P<m>\d{2})(?::(?P<s>\d{2}))?(?:\.\d+)?(?P<tz>Z|[+-]\d{2}:?\d{2})?$"
)


def parse_now(current_time: str) -> datetime:
    return datetime.fromisoformat(current_time)


def utc_offset(now: datetime) -> str:
    off = now.utcoffset() or timedelta(0)
    total = int(off.total_seconds() // 60)
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{sign}{total // 60:02d}:{total % 60:02d}"


def day_table(now: datetime, days: int = 8) -> str:
    """'hôm nay (thứ Ba) = 2026-11-24' lines so the model never has to do date arithmetic."""
    rows = []
    for i in range(days):
        d: date = (now + timedelta(days=i)).date()
        label = RELATIVE_VI.get(i, f"+{i} ngày")
        rows.append(f"{label} ({WEEKDAYS_VI[d.weekday()]}) = {d.isoformat()}")
    return "\n".join(rows)


def normalize_datetime(value: str, offset: str) -> str:
    """Complete a date-time to 'YYYY-MM-DDTHH:MM:SS+HH:MM'; returns the input unchanged when it is not one."""
    m = _DT_LOOSE.match(value.strip())
    if not m:
        return value
    tz = m.group("tz") or offset
    if tz == "Z":
        tz = "+00:00"
    elif len(tz) == 5:  # +0700
        tz = f"{tz[:3]}:{tz[3:]}"
    return f"{m.group('date')}T{int(m.group('h')):02d}:{m.group('m')}:{m.group('s') or '00'}{tz}"
