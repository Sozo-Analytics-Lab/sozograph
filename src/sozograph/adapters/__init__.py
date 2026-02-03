"""
Adapters for ingesting external data sources into SozoGraph.

Each adapter is responsible for normalizing an external object
(Firestore, RTDB, Supabase, etc.) into Interaction objects.
"""

__all__ = [
    "firestore",
    "rtdb",
    "supabase",
]