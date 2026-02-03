from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ..interaction import Interaction
from ..utils import parse_ts, safe_stringify, sha256_json, pick_first


# Common timestamp-like fields in RTDB nodes
_TS_FIELDS = (
    "updatedAt",
    "updated_at",
    "createdAt",
    "created_at",
    "timestamp",
    "date",
)


def rtdb_to_interaction(
    value: Any,
    *,
    path: Optional[str] = None,
    node_id: Optional[str] = None,
) -> Interaction:
    """
    Convert a RTDB node value into an Interaction.

    Accepts:
    - value: node payload (any JSON-serializable object)
    - path: RTDB path, e.g. "/users/u1/profile"
    """

    # Timestamp (best-effort)
    ts = None
    if isinstance(value, dict):
        ts = parse_ts(pick_first(value, _TS_FIELDS))

    # Text representation
    text_val = safe_stringify(value)

    # Stable id
    _id = node_id or (path.replace("/", "_") if path else None)
    if not _id:
        _id = sha256_json({"path": path, "value": value})[:16]

    return Interaction(
        id=str(_id),
        ts=ts or None,
        type="rtdb",
        text=text_val,
        source=f"rtdb:{path}" if path else None,
        data=value if isinstance(value, dict) else {"value": value},
    )


def rtdb_batch_to_interactions(
    snapshot: Union[Dict[str, Any], List[Any]],
    *,
    base_path: Optional[str] = None,
) -> List[Interaction]:
    """
    Convert a RTDB snapshot into a list of Interactions.

    - Dict snapshots are expanded one level deep
    - Lists are enumerated by index
    """

    interactions: List[Interaction] = []

    if isinstance(snapshot, list):
        for idx, value in enumerate(snapshot):
            path = f"{base_path}/{idx}" if base_path else str(idx)
            interactions.append(
                rtdb_to_interaction(value, path=path)
            )
        return interactions

    if isinstance(snapshot, dict):
        for key, value in snapshot.items():
            path = f"{base_path}/{key}" if base_path else str(key)
            interactions.append(
                rtdb_to_interaction(value, path=path, node_id=str(key))
            )
        return interactions

    # Fallback: single scalar value
    interactions.append(
        rtdb_to_interaction(snapshot, path=base_path)
    )
    return interactions
