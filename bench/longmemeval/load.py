from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bench.locomo.load import QA, Conversation


def load_conversations(path, *, limit=None, offset=0):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("LongMemEval expects a JSON array")
    out = []
    for sample in raw[offset:None if limit is None else offset + limit]:
        sessions, ids, dates = (sample[k] for k in ("haystack_sessions", "haystack_session_ids", "haystack_dates"))
        if not len(sessions) == len(ids) == len(dates):
            raise ValueError("Session contents, IDs, and dates must align")
        turns, evidence = [], []
        for session, sid, stamp in zip(sessions, ids, dates, strict=True):
            # Official files use YYYY/MM/DD (weekday) HH:MM or ISO dates.
            parsed = None
            for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d (%A) %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
                try:
                    parsed = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    pass
            if parsed is None:
                parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                parsed = parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)
            for i, turn in enumerate(session):
                tid = f"{sid}:{i}"
                # Gold annotations never enter ingestion or runtime retrieval.
                if turn.get("has_answer"):
                    evidence.append(tid)
                turns.append({"id": tid, "session": str(sid), "speaker": str(turn["role"]),
                              "text": str(turn["content"]), "ts": (parsed + timedelta(microseconds=i)).isoformat(),
                              "date": stamp})
        question_id = str(sample["question_id"])
        category = "abstention" if question_id.endswith("_abs") else sample["question_type"]
        qa = QA(str(sample["question"]), str(sample["answer"]), category, evidence,
                question_date=str(sample.get("question_date", "")),
                evidence_sessions=[str(s) for s in sample.get("answer_session_ids", [])])
        out.append(Conversation(question_id, turns, [qa], ["user", "assistant"]))
    return out
