from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ..interaction import Interaction
from ..utils import parse_ts, safe_stringify, sha256_json, pick_first


_TEXT_FIELDS = (
    "text",
    "message",
    "content",
    "description",
    "notes",
    "summary",
    "title",
    "name",
    "status",
    "action",
    "event",
)

_TS_FIELDS = (
    "updated_at",
    "created_at",
    "timestamp",
    "date",
    "updatedAt",
    "createdAt",
)


def supabase_row_to_interaction(
    row: Dict[str, Any],
    *,
    table: Optional[str] = None,
    source: Optional[str] = None,
    row_id: Optional[str] = None,
) -> Interaction:
    """
    Convert a Supabase table row into an Interaction.
    """

    ts = parse_ts(pick_first(row, _TS_FIELDS))

    text_val = pick_first(row, _TEXT_FIELDS)
    if text_val is None:
        text_val = safe_stringify(row)

    _id = row_id or row.get("id") or row.get("_id")
    if not _id:
        _id = sha256_json(row)[:16]

    src = source
    if not src and table:
        src = f"supabase:{table}"

    return Interaction(
        id=str(_id),
        ts=ts or None,
        type="supabase",
        text=str(text_val),
        source=src,
        data=row,
        meta={"table": table} if table else {},
    )


def supabase_batch_to_interactions(
    rows: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    *,
    table: Optional[str] = None,
) -> List[Interaction]:
    """
    Convert many rows to Interactions.

    Accepts:
    - list of rows
    - dict mapping {row_id: row_dict}
    """
    interactions: List[Interaction] = []

    if isinstance(rows, dict):
        for row_id, row in rows.items():
            interactions.append(
                supabase_row_to_interaction(
                    row,
                    table=table,
                    source=f"supabase:{table}:{row_id}" if table else None,
                    row_id=str(row_id),
                )
            )
        return interactions

    for row in rows:
        interactions.append(
            supabase_row_to_interaction(
                row,
                table=table,
                source=f"supabase:{table}" if table else None,
            )
        )
    return interactions
