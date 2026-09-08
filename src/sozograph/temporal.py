"""Conservative date intervals with explicit precision and reference time."""
from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta


def event_interval(value: str, reference=None):
    text = str(value or "").strip()
    if re.fullmatch(r"\d{4}", text):
        year = int(text)
        if 1 <= year <= 9999:
            return f"{year:04d}-01-01", f"{year:04d}-12-31", "year"
    if re.fullmatch(r"\d{4}-\d{2}", text):
        year, month = map(int, text.split("-"))
        if 1 <= year <= 9999 and 1 <= month <= 12:
            return f"{text}-01", f"{text}-{calendar.monthrange(year, month)[1]}", "month"
    if reference:
        if isinstance(reference, str):
            reference = datetime.fromisoformat(reference.replace("Z", "+00:00"))
        anchor = reference.date() if isinstance(reference, datetime) else reference
        days = {"yesterday": -1, "today": 0, "tomorrow": 1}
        if text.casefold() in days:
            return (anchor + timedelta(days=days[text.casefold()])).isoformat(), "", "day"
        match = re.fullmatch(r"(\d+) (day|week)s? ago", text.casefold())
        if match:
            offset = int(match[1]) * (7 if match[2] == "week" else 1)
            return (anchor - timedelta(days=offset)).isoformat(), "", "day"
        if text.casefold() == "last month":
            prev = anchor.replace(day=1) - timedelta(days=1)
            return prev.replace(day=1).isoformat(), prev.isoformat(), "month"
    return None


def query_interval(query: str, reference=None):
    text = (query or "").casefold()
    explicit = re.search(r"\b(?:in|during) (\d{4}(?:-\d{2}(?:-\d{2})?)?)\b", text)
    if explicit:
        value = explicit[1]
        interval = event_interval(value)
        if interval:
            return interval[:2]
        try:
            day = date.fromisoformat(value).isoformat()
            return day, day
        except ValueError:
            return None
    for month, name in enumerate(calendar.month_name[1:], 1):
        match = re.search(rf"\b(?:in|during) {name.casefold()} (\d{{4}})\b", text)
        if match:
            interval = event_interval(f"{match[1]}-{month:02d}")
            return interval[:2] if interval else None
    if reference and "last month" in text:
        return event_interval("last month", reference)[:2]
    return None
