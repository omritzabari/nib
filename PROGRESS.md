# Progress

Live task state. Updated at the end of every task. A fresh session reads this to resume.

**Phase 1 — data and evaluation infrastructure. No generative model yet.**

## Next action

> **T2 — charset.** Not started. Nothing is blocking it.

## Status

| ID | Task | Status | Verified by |
|----|------|--------|-------------|
| T0 | Repo skeleton, packaging, lint, git init | **done** | ruff clean · ruff format clean · pytest 2 passed |
| T1 | Config system (typed schema, single path root) | **done** | ruff clean · 16 tests passing |
| T2 | charset — char/index mapping | not started | |
| T3 | Synthetic IAM fixture generator | not started | |
| T4 | IAM parser | not started | blocked on real data for the full check only |
| T5 | Writer-disjoint split | not started | |
| T6 | Image normalisation (domain gap) | not started | needs Amri's phone photos |
| T7 | Pack to a single LMDB file | not started | |
| T8 | Dataset + collate | not started | |
| T9 | Checkpoint save/resume | not started | |
| T10 | Metrics: FID, CER, writer retrieval | not started | |
| T11 | Experiment tracking + visual sample log | not started | needs W&B username |
| T12 | Colab end-to-end smoke run | not started | **phase 1 exit criterion**, Amri runs |

## Waiting on Amri

- [ ] IAM registration + download (`words.tgz`, `xml.tgz`, `ascii.tgz`) into `data/raw/iam/`
- [ ] 4-5 phone photos of his own handwriting, deliberately imperfect, into `data/raw/personal/`
- [ ] Weights & Biases username (the identifier only — never the API key)
- [x] Commit permission granted (messages record what was done and what changed)

## Open questions

- **Python version.** Local is 3.13.2. Colab's version is unknown; run `!python --version`
  in the first Colab session and pin to match. Does not bite until torch enters at T9/T12.
- **Architecture decided (2026-08-29): option C.** Start from a released zero-shot
  checkpoint (Emuru line) rather than training a generator from scratch. Amri approved.
  Consequences: no per-writer training; the generator produces variable-length lines, so
  word-to-line assembly leaves the layout engine's scope; IAM becomes evaluation data, not
  training data; generation resolution is 64px, upscaled afterwards. VATr++ stays as a
  documented fallback and comparison baseline. Surveys and their unverified-claims lists are
  in `docs/`.
- **IAM licence is non-commercial research use.** Fine for a portfolio project; a blocker if
  this ever ships as a product. Flagged early on purpose.

## Log

- **2026-08-28 — T0 done.** Package `nib-synth` installs editable as `nib`. Deviated from the
  brief's proposed layout: `src/nib/...` instead of `src/...`, so imports work from any working
  directory. torch deliberately excluded from base dependencies. Added `.gitattributes` for
  LF/CRLF, which would otherwise show every file as modified between Windows and Colab.
  Not committed — awaiting Amri.
