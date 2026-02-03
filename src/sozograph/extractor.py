from __future__ import annotations

import json
from typing import Dict, List

from google import genai
from google.genai import types
from pydantic import ValidationError

from .interaction import Interaction
from .schema import Fact, Preference, Entity, OpenLoop
from .prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    EXTRACTOR_JSON_SCHEMA,
    EXTRACTOR_USER_PROMPT_TEMPLATE,
)
from .utils import normalize_key, parse_ts


class Extractor:
    """
    Gemini-backed extractor that converts Interactions into candidate memory updates.
    """

    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def extract(self, interaction: Interaction, source_id: str) -> Dict[str, List]:
        """
        Extract candidate facts/prefs/entities/open_loops from a single Interaction.
        Returns dict with keys: facts, prefs, entities, open_loops.
        """
        prompt = EXTRACTOR_USER_PROMPT_TEMPLATE.format(
            schema=EXTRACTOR_JSON_SCHEMA.strip(),
            source_id=source_id,
            interaction_type=interaction.type,
            ts_iso=interaction.ts.isoformat(),
            interaction_text=interaction.short_text(),
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(role="system", parts=[types.Part(text=EXTRACTOR_SYSTEM_PROMPT)]),
                types.Content(role="user", parts=[types.Part(text=prompt)]),
            ],
            generation_config=types.GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        try:
            payload = json.loads(response.text)
        except Exception as e:
            raise RuntimeError(f"Extractor returned invalid JSON: {e}\n{response.text}")

        return self._validate_and_normalize(payload, source_id)

    def _validate_and_normalize(self, data: Dict, source_id: str) -> Dict[str, List]:
        """
        Validate model output and normalize keys/timestamps.
        """
        out = {
            "facts": [],
            "prefs": [],
            "entities": [],
            "open_loops": [],
        }

        for item in data.get("facts", []):
            try:
                f = Fact(
                    key=normalize_key(item["key"]),
                    value=item.get("value"),
                    confidence=float(item.get("confidence", 0.7)),
                    source=source_id,
                    ts=parse_ts(item.get("ts")) or None,
                )
                out["facts"].append(f)
            except ValidationError:
                continue

        for item in data.get("prefs", []):
            try:
                p = Preference(
                    key=normalize_key(item["key"]),
                    value=item.get("value"),
                    confidence=float(item.get("confidence", 0.7)),
                    source=source_id,
                    ts=parse_ts(item.get("ts")) or None,
                )
                out["prefs"].append(p)
            except ValidationError:
                continue

        for item in data.get("entities", []):
            try:
                e = Entity(
                    name=item["name"],
                    type=item.get("type", "other"),
                    aliases=item.get("aliases") or [],
                )
                out["entities"].append(e)
            except ValidationError:
                continue

        for item in data.get("open_loops", []):
            try:
                o = OpenLoop(
                    item=item["item"],
                    source=source_id,
                    ts=parse_ts(item.get("ts")) or None,
                )
                out["open_loops"].append(o)
            except ValidationError:
                continue

        return out
