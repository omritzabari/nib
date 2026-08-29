# Progress

Live task state. Updated at the end of every task. A fresh session reads this to resume.

**Phase 1 — data and evaluation infrastructure. No generative model yet.**

## Next action

> **T6 — image normalisation (domain gap).** Unblocked: 5 photos have arrived.
>
> **Do not start T7, T8, FID or CER.** Amri asked to wait for the full CVL release
> (`cvl-database-1-1.zip`, 4.2 GB) rather than build them on page-level data and redo
> the work. He will say when it is uploaded.

## Status

| ID | Task | Status | Verified by |
|----|------|--------|-------------|
| T0 | Repo skeleton, packaging, lint, git init | **done** | ruff clean · ruff format clean · pytest 2 passed |
| T1 | Config system (typed schema, single path root) | **done** | ruff clean · 16 tests passing |
| T2 | charset — char/index mapping | **done** | ruff clean · 31 tests passing |
| T3 | Synthetic IAM fixture generator | **done** | ruff clean · 49 tests · 20 writers / 1000 words in 5.7s |
| T4 | IAM parser | **done** | ruff clean · 72 tests (5 skipped, awaiting real IAM) |
| T5 | Writer-disjoint split | **done** | ruff clean · 111 tests · real CVL: 216/94 writers, 70.0/30.0% samples, 0 overlap |
| T6 | Image normalisation (domain gap) | **next** | 5 photos received; note the squared paper |
| T7 | Pack to a single LMDB file | **on hold** | waiting for full CVL, by Amri's request |
| T8 | Dataset + collate | **on hold** | waiting for full CVL, by Amri's request |
| T9 | Checkpoint save/resume | **done** | ruff clean · 132 tests · resume is bit-identical to an uninterrupted run |
| T10 | Metrics: FID, CER, writer retrieval | partly on hold | writer retrieval is doable now; FID and CER wait for full CVL |
| T11 | Experiment tracking + visual sample log | **done** | ruff clean · 157 tests · entity omri334jb configured |
| T12 | Colab end-to-end smoke run | not started | **phase 1 exit criterion**, Amri runs |

## Waiting on Amri

- [ ] IAM download into `data/raw/iam/`. **The FKI site is currently down** -- links do not
      load from either side. Not blocking: `tests/test_iam_real.py` activates by itself the
      moment `data/raw/iam/xml/*.xml` exists, and validates the reconstructed schema then.
      Priority order once reachable: `xml.tgz` (small, answers the schema question),
      then `ascii.tgz`, then `words.tgz` (~1.2GB).
- [x] 5 phone photos received. **They are on squared/graph paper** -- grid lines will be
      read as ink by naive binarisation. Ruled-line removal is now part of T6's scope.
- [ ] `cvl-database-1-1.zip` (4.2 GB, same Zenodo page) into `data/raw/cvl/` -- the cropped
      release has images only. Without the full one there is no CER baseline and no exact
      word cropping for FID.
- [ ] Weights & Biases username (the identifier only — never the API key)
- [x] Commit permission granted (messages record what was done and what changed)

## Open questions

- **Python version.** Local is 3.13.2 with torch 2.13.0+cpu (the CPU-only wheel, 122 MB).
  Colab's Python and torch versions are unknown; run `!python --version` and
  `import torch; torch.__version__` in the first Colab session and pin to match.
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
