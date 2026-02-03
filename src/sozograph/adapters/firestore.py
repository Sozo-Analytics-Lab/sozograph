from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from ..interaction import Interaction
from ..utils import parse_ts, safe_stringify, sha256_json, pick_first


# Common Firestore field names we try first for text & timestamps
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
)

_TS_FIELDS = (
    "updatedAt",
    "updated_at",
    "createdAt",
    "created_at",
    "timestamp",
    "date",
)


def firestore_to_interaction(
    doc: Dict[str, Any],
    *,
    source: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> Interaction:
    """
    Convert a Firestore document dict into an Interaction.

    - Tries to extract meaningful text deterministically
    - Falls back to compact stringify if needed
    - Does NOT call Gemini (fallback summarization handled upstream)
    """

    # Determine timestamp
    ts = parse_ts(pick_first(doc, _TS_FIELDS))

    # Determine text
    text_val = pick_first(doc, _TEXT_FIELDS)
    if text_val is None:
        text_val = safe_stringify(doc)

    # Determine id
    _id = doc_id or doc.get("id") or doc.get("_id")
    if not _id:
        _id = sha256_json(doc)[:16]

    return Interaction(
        id=str(_id),
        ts=ts or None,
        type="firestore",
        text=str(text_val),
        source=source,
        data=doc,
    )


def firestore_batch_to_interactions(
    docs: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
    *,
    collection_path: Optional[str] = None,
) -> List[Interaction]:
    """
    Convert a batch of Firestore docs to Interactions.

    Accepts:
    - list of document dicts
    - dict mapping {doc_id: doc_dict}
    """
    interactions: List[Interaction] = []

    if isinstance(docs, dict):
        for doc_id, doc in docs.items():
            interactions.append(
                firestore_to_interaction(
                    doc,
                    source=f"firestore:{collection_path}/{doc_id}"
                    if collection_path
                    else None,
                    doc_id=str(doc_id),
                )
            )
        return interactions

    for doc in docs:
        interactions.append(
            firestore_to_interaction(
                doc,
                source=f"firestore:{collection_path}" if collection_path else None,
            )
        )

    return interactions
