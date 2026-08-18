# Migrating from 0.1.x to 0.2.0

0.2.0 is a clean break. The library is no longer Gemini-only, and several
defects in 0.1.1 were losing data silently.

## Read this first

**0.1.1 discarded most of what it extracted.** `Fact`, `Preference`, and
`OpenLoop` declare `ts` with a `default_factory`. The extractor passed
`ts=parse_ts(...) or None`, and pydantic raises on an explicit `None` rather
than falling back to the default. That error was caught and skipped. Since the
prompt marked `ts` as optional, most extracted items hit this path and vanished.

The same line in the Firestore, RTDB, and Supabase adapters was not caught and
crashed ingestion outright, which is why two of the three README examples in
0.1.1 raised `ValidationError`.

After upgrading, expect noticeably fuller passports from the same input. Your
0.1.1 passports are not wrong, only incomplete. Re-ingest to fill them in.

## API changes

### `ingest()` returns a Passport

```python
# 0.1.x
passport, stats = sg.ingest(data)

# 0.2.0
passport = sg.ingest(data)
passport.stats        # the per-segment merge statistics
```

The old tuple form is available for one release as `sozograph.core.ingest()`
and raises a `DeprecationWarning`.

### Provider selection replaces the Gemini defaults

```python
# 0.1.x
sg = SozoGraph(api_key=GEMINI_KEY, extractor_model="gemini-3-flash-preview")

# 0.2.0
sg = SozoGraph()                            # resolved from the environment
sg = SozoGraph("gemini:gemini-2.5-flash")   # explicit
sg = SozoGraph("anthropic")
sg = SozoGraph("ollama:llama3.2")
```

`extractor_model` and `fallback_model` are gone. Use the provider spec or
`model=`.

`GEMINI_API_KEY` still works when you select Gemini. It no longer selects the
provider by itself unless nothing else is configured.

### Reading a passport needs no engine

```python
# 0.1.x
sg = SozoGraph(api_key=...)          # required a key even to render
text = sg.export_context(passport)

# 0.2.0
text = passport.context()            # no key, no SDK, no network
text = passport.context(query="where does she live?")
```

`SozoGraph()` no longer builds a client at construction. The provider is created
on first use.

### New persistence

```python
passport.save("user.json")
passport = Passport.load("user.json")
passport.to_compact_dict() / Passport.from_dict(...)
passport.to_json() / Passport.from_json(...)
```

0.1.x had `to_compact_dict()` with no inverse, so the state was not portable in
both directions.

## Install changes

Provider SDKs are optional extras. The core install is pydantic alone.

```bash
# 0.1.x
pip install sozograph                 # pulled google-genai

# 0.2.0
pip install sozograph                 # pydantic only
pip install "sozograph[gemini]"       # if you want Gemini
pip install "sozograph[all]"          # the four native providers
```

`python-dotenv` was declared and never imported. It is gone. Load your own
`.env` if you use one.

## Passport format 2.0

`Passport.load()` and `Passport.from_dict()` read a 1.0 file and migrate it. The
version stamp becomes `"2.0"` and `episodes` starts empty. Nothing is lost, but
existing passports gain no episodes retroactively. Re-ingest the source history
to get them.

New field:

```json
"episodes": [
  {"id": "seg_a1b2", "ts": "...", "summary": "...", "salience": 0.8,
   "source": "seg_a1b2", "participants": ["Melanie"], "keywords": ["kwekwe"]}
]
```

`sources` now records one entry per extraction segment rather than one per
interaction. On a long conversation the per-turn version produced an evidence
log larger than the memory it documented, and nothing referenced those entries:
facts cite the segment that produced them.

## Behavioural changes worth knowing

**Extraction is batched.** Interactions are grouped into ~1500-token segments,
one call each. On a 600-turn conversation that is roughly 27 calls instead of
600. Pass `batch=False` for the old per-interaction behaviour.

**Contradictions are deduplicated.** They were append-only, so re-ingesting the
same conversation re-appended the same entry every time. Identical changes are
now recorded once.

**Case differences are no longer contradictions.** `"Direct"` and `"direct"`
were compared with `str.strip()` equality and recorded as a change, so the value
flip-flopped on every ingest. The same applied to `7` and `"7"`.

**Keys go through the deduplication tiers.** Variants such as `"Detail Level"`,
`"detail level"`, and `"detail_level"` collapse to one key. Opposing pairs such
as `budget_min` and `budget_max` are explicitly blocked from merging.

**Identifiers are deterministic.** `SourceRef.id` used `abs(hash(str))`, which
Python salts per process, so ids differed between runs on identical input. They
are now SHA-256 based.

**Short conversational turns are no longer summarized.** The fallback summarizer
fired on any text under 30 characters, which meant a model call to be told that
"What colour did you go with?" is already readable. It now runs only on database
objects.

**The default context header changed** from `"SOZOGRAPH PASSPORT v1"` to
`"SOZOGRAPH PASSPORT"`. Pass `header=` if you were matching on it.

## Removed

| Removed | Replacement |
|---|---|
| `sozograph.ingest.ingest()` | `SozoGraph().ingest()` (the old one never called the extractor) |
| `FallbackSummarizer(api_key=..., model=...)` | `Summarizer(provider)`; the old name still resolves |
| `Extractor(api_key=..., model=...)` | `Extractor(provider)` |
| `Extractor._validate_and_normalize()` | `Extractor.validate(data, source_id=..., ts=...)` |
| `extractor_model`, `fallback_model` kwargs | provider spec or `model=` |
| `EXTRACTOR_JSON_SCHEMA` (a string in the prompt) | `EXTRACTION_SCHEMA`, a real JSON Schema sent to the engine |
