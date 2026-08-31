# Progress

Live task state. Updated at the end of every task. A fresh session reads this to resume.

**Phase 1 — data and evaluation infrastructure. No generative model yet.**

## Next action

> **Phase 1 is complete (13/13). Phase 2 has begun and the model generates.**
>
> On 2026-08-31 the project produced handwriting for the first time: held-out CVL
> writer 0057, one real line as style, three lines that writer never wrote, in a
> visibly matching hand. See `docs/phase2-first-generation.md` and
> `outputs/probe_lines/lines_contrast.png`.
>
> ### The immediate next task
>
> **T14 — pack the clean lines into `data/processed/cvl_lines_64.lmdb`.**
> Then compact it and put a copy under `data/processed/upload/` for Drive.
> `scripts/build_index.py` is the template; the difference is that it reads
> `nib.data.cvl_lines.scan_lines` and normalises with `normalise_line`.
>
> Phase 2 runs T13 -> T16. T13 is done; T14 is local, T15 and T16 are Colab.
>
> ### Why the plan grew from two steps to four (2026-08-31)
>
> Measuring the line data before writing the pack turned up three things. All
> three were invisible while the working unit was a word, and all three become
> load-bearing the moment it is a line.
>
> **1,157 lines carry a transcription that is missing a word.** CVL dropped word
> crops whose segmentation failed, but the line image still holds that word's
> ink. Pairing the two charges the recogniser a deletion error for reading
> correctly. Confirmed by counting ink blobs: gap lines carry 0.58 more per line
> than complete ones, relative to their own word count. They are dropped and
> counted -- 10,862 clean lines remain, every held-out writer keeps at least 16.
>
> **The token budget was set for words and never revisited.** Emuru's VAE
> compresses width by 8, so one token is 8px (`lengths / 8` in its
> `modeling_emuru.py`). `EmuruGenerator` passes `max_new_tokens=96` = 768px. A
> real CVL line at 64px averages **886px**, p90 1198px. *It could not finish an
> average line.* The two 756px outputs recorded as "runaway generation" were
> 94-95 tokens: they hit our cap. The model's own default is 256. T16 replaces
> the fixed cap with a per-request budget from the text length, and counts every
> truncation.
>
> **The three reference numbers were measured on words.** FID 33.72 and
> retrieval 66.9% both came from `cvl_words_64.lmdb`; CER 12.33% came from 40
> unfiltered lines of a single writer. Scoring generated *lines* against a
> word-level FID floor compares different things. T15 re-measures all three on
> the line pack -- one run of `check_metrics.py --pack ...`, no new code.
>
> A fourth item from `PROGRESS.md` turned out to need no work: it asked for
> `normalise_ink` on the line path, but `normalise_word` already called it
> internally. Raw lines sit at 2nd-percentile brightness ~130; after
> normalisation ~21. The faint probe output came from loading raw images.
>
> ### Known and open
>
> - Emuru needs the style sample's *transcription*. A user photographing a page
>   has transcribed nothing, so the product must read it first -- TrOCR is already
>   here for that, at TrOCR's accuracy. This constrains the architecture.
> - Emuru also ships `generate_batch`, which `EmuruGenerator` does not use. It
>   takes per-sample `lengths` and would cut Colab generation time. Not needed
>   for correctness; worth doing if T16 is slow.
> - **Run evaluation on Colab, not locally.** 220s per line on CPU; minutes on a T4.
> - `data/processed/cvl_words_64.lmdb` shows 8 GB but holds 469 MB; it is sparse
>   and locked by a stale handle. The compacted copy for uploading is at
>   `data/processed/upload/cvl_words_64.lmdb`. A reboot clears the lock.
> - transformers is pinned to 4.x. 5.x cannot load Emuru and breaks TrOCR's
>   tokenizer -- the same version gap causes both.

## Status

| ID | Task | Status | Verified by |
|----|------|--------|-------------|
| T0 | Repo skeleton, packaging, lint, git init | **done** | ruff clean · ruff format clean · pytest 2 passed |
| T1 | Config system (typed schema, single path root) | **done** | ruff clean · 16 tests passing |
| T2 | charset — char/index mapping | **done** | ruff clean · 31 tests passing |
| T3 | Synthetic IAM fixture generator | **done** | ruff clean · 49 tests · 20 writers / 1000 words in 5.7s |
| T4 | IAM parser | **done** | ruff clean · 72 tests (5 skipped, awaiting real IAM) |
| T5 | Writer-disjoint split | **done** | ruff clean · 111 tests · real CVL: 216/94 writers, 70.0/30.0% samples, 0 overlap |
| T6 | Image normalisation (domain gap) | **done** | ruff clean · 200 tests · paper/ink/contrast spread 0 across conditions; ink% 38.0 -> 14.2 |
| T7 | Pack to a single LMDB file | **done** | ruff clean · batched writes · compaction 8.00 GB -> 24.9 MB |
| T8 | Dataset + collate | **done** | ruff clean · 19 tests · same-writer / different-word verified via style_keys |
| T9 | Checkpoint save/resume | **done** | ruff clean · 132 tests · resume is bit-identical to an uninterrupted run |
| T10 | Metrics: FID, CER, writer retrieval | **done** | FID 0.0000 self-check, floor 33.72 · CER 12.33% on real lines · retrieval **66.9% top-1** on 94 unseen writers |
| T11 | Experiment tracking + visual sample log | **done** | ruff clean · 157 tests · entity omri334jb configured |
| T12 | Colab end-to-end smoke run | **done** | T4, 2026-08-30: bit-identical resume, 6.1s Drive copy, 614 samples/s |

**Phase 2 — generation and its first real numbers.**

| ID | Task | Status | Verified by |
|----|------|--------|-------------|
| T13 | CVL line reader, with counted drops | **done** | ruff clean · 306 passed, 5 skipped · 10,862 of 13,473 lines kept, total_seen matches the disk exactly |
| T14 | Line pack -> `cvl_lines_64.lmdb` | next | |
| T15 | Re-measure FID / retrieval / CER on lines | todo | |
| T16 | Per-request token budget, then evaluate the generator | todo | |

## Waiting on Amri

- [ ] IAM download into `data/raw/iam/`. **The FKI site is currently down** -- links do not
      load from either side. Not blocking: `tests/test_iam_real.py` activates by itself the
      moment `data/raw/iam/xml/*.xml` exists, and validates the reconstructed schema then.
      Priority order once reachable: `xml.tgz` (small, answers the schema question),
      then `ascii.tgz`, then `words.tgz` (~1.2GB).
- [x] 5 phone photos received. **They are on squared/graph paper** -- grid lines will be
      read as ink by naive binarisation. Ruled-line removal is now part of T6's scope.
- [x] `cvl-database-1-1.zip` downloaded and extracted. 99,904 cropped word images with
      transcriptions in their filenames; 98,179 usable after filtering.
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

- **2026-08-31 — T13 done.** `nib.data.cvl_lines` reads CVL's 13,473 line crops and
  reassembles each transcription from the word filenames. 10,862 kept; the rest are
  counted by reason — 1,421 German, 1,132 with word-index gaps, 33 with no word files,
  25 from the writer CVL's own readme excludes. `total_seen` equals the file count on
  disk exactly. Word filenames are indexed in one unfiltered pass, keyed by
  (writer, text, line): filtering the index first would hide the very gaps it exists to
  find. Drop reasons are attributed in a fixed order — charset before incompleteness —
  so "incomplete" reports only English lines lost to data quality, and there is a test
  that pins that order. Added `normalise_line` beside `normalise_word`: the same
  mechanism, deliberately two names, because fixing the height destroys a word's
  relative scale and is exactly what makes lines comparable. Deleted `_cvl_lines` from
  `scripts/check_metrics.py`, which had reassembled transcriptions inline, filtered
  nothing, and drawn all 40 samples from one writer.

- **2026-08-28 — T0 done.** Package `nib-synth` installs editable as `nib`. Deviated from the
  brief's proposed layout: `src/nib/...` instead of `src/...`, so imports work from any working
  directory. torch deliberately excluded from base dependencies. Added `.gitattributes` for
  LF/CRLF, which would otherwise show every file as modified between Windows and Colab.
  Not committed — awaiting Amri.
