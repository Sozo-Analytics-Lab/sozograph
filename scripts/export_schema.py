"""Regenerate the portable schema from the storage models."""
import json
from pathlib import Path

from sozograph.schema import Passport

root = Path(__file__).resolve().parents[1]
schema = Passport.model_json_schema(mode="serialization")
schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
schema["$id"] = "https://github.com/Sozo-Analytics-Lab/sozograph/blob/main/schemas/passport-2.2.schema.json"
schema["properties"]["version"] = {"type": "string", "const": "2.2", "default": "2.2"}
for target in [root / "schemas/passport-2.2.schema.json", root / "src/sozograph/passport.schema.json"]:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
