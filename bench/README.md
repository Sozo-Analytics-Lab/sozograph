# Resumable evaluation in 0.3.2

Prefer python -m bench.evaluate for new runs. Supply --dataset locomo or
longmemeval, --data, --extractor, --answerer, --judge, and optionally --out.
Use --dry-run before model calls. Credentials use provider environment settings;
the Python provider_factory argument supports custom transports.

--stage build constructs memory; recall requires completed construction and runs
offline; answer reuses construction; judge requires cached predictions; all runs
missing stages. Reuse the same output directory. Changing read options reuses
construction, changing the answerer reuses recall, and changing the judge reuses
predictions. Code changes deliberately invalidate caches.

--system lexical retrieves source chunks without extraction. full_context uses
all history and is not constrained to the memory budget. sozograph applies the
budget. --retention full --include-sources and --graph are optional ablations.
Report their actual storage, calls, and tokens alongside accuracy.

Artifacts retain normalized history, passports, stage costs/timing, selected
contexts, answers before judging, judgments, and code/dataset hashes. Incomplete
construction cannot produce a final complete score. Coverage is source membership,
not semantic answer presence; absent annotations yield null. Gold has_answer
labels never enter LongMemEval history. Exported hypotheses JSONL works with its
official evaluator; the included equivalence judge is a different labeled protocol.

bench.replay.paired_bootstrap resamples matched conversation groups.
python -m bench.profile measures synthetic offline latency and storage, not QA.

The legacy LoCoMo CLI documentation follows. It now preserves unjudged predictions
on partial grading; the new CLI provides stage replay and evidence diagnostics.

---

# Benchmarks

## LoCoMo

LoCoMo is the benchmark LightMem headlines, which is why it is the one used
here. The dataset is not vendored: download `locomo10.json` from
[snap-research/locomo](https://github.com/snap-research/locomo) into `data/`.

```bash
python -m bench.locomo.run --data data/locomo10.json
```

Defaults match LightMem's published setup exactly: GPT-4o-mini as backbone and
judge, the four non-adversarial question categories (single-hop, multi-hop,
temporal, open-domain), and per-conversation figures.

Check the dataset parsed correctly before spending anything:

```bash
python -m bench.locomo.run --data data/locomo10.json --dry-run
```

Add the full-context baseline for an upper bound:

```bash
python -m bench.locomo.run --data data/locomo10.json --systems sozograph,full_context
```

### On the comparison

The published rows in the results table are LightMem's own Table 3
(arXiv:2510.18866). They are **cited, not re-run**: reproducing them locally
requires conda, LLMLingua-2, sentence-transformers, Qdrant, and several
gigabytes of model weights. Only rows labelled `(measured)` come from this
harness.

Token and API-call counts are measured, not estimated. Every provider
accumulates a `Usage` record per call, and the runner keeps memory-construction
and question-answering on separate provider instances so the two columns can
never contaminate each other.
