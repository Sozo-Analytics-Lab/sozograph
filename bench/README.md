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
