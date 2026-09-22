# Working agreement — nib

Few-shot handwriting synthesis. Learn a writer's hand from 1-2 sample pages,
generate new text in it. **The goal is that it works on a new person's own pages.**
Adapting to that person may include a short per-writer fine-tune, provided it fits
a reasonable time per user (Amri, 2026-09-14) -- it is no longer ruled out.

**It is an app for anyone, not Amri's hand** (Amri, 2026-09-21, a hard requirement).
Any person, with any handwriting -- cursive, print, messy -- uploads one page, gets
the model adapted to them, and typed text comes back in their hand. Amri's page is
one real-world test case, never the target: judge every change on the held-out CVL
writers first. Never reason from the properties of his hand to a design choice.
Cutting a user's own letters out of their page and stitching new text from them is
**rejected** -- it fits only print hands and is not a model adapted to the user.

**Read `PROGRESS.md` first.** It holds the live task state and the exact next action.

## How success is measured (Amri, 2026-09-22 — binding)

Set because experiments kept circling: a story, an hour of GPU, one number, a new story.

**Done** is decided by people, not by a metric: a blind test. A judge sees three real
lines of a writer, then two more -- one real, one generated, full lines -- and picks the
real one. 50% means they cannot tell. **nib works when judges pick the real line in at
most 60% of pairs**, over at least three writers who are not all Amri, 40 pairs each,
and no generated line leaves a word out. Judges need not know the writer.

**Until then every change is judged on one scorecard,** always on the same benchmark:
the 150 held-out CVL requests of cell 7s (`evaluate_generator.py --samples 150
--style-refs 4 --candidates 4 --keep hand`), or `rescore_run.py` over saved lines when
nothing needs generating again.

| | what | baseline (7s) |
|---|---|---|
| **primary** | HWD identity, paired by writer against the current baseline | 65.2% |
| guardrail | missing a word (TrOCR-base, teacher-forced) | 10.7% |
| guardrail | CER (TrOCR-base) | 9.81% |

**A change is kept only if** its paired identity interval is not wholly below zero,
missing-a-word does not rise, CER rises by at most one point, **and** the one thing it
was built to improve improves with its interval clear of zero -- named before the run.

Rules that stop the circling:

- The criterion is written in `PROGRESS.md` before the run, from the rule above. Nothing
  decides after the fact.
- Proxies diagnose and never decide: detectability, writer retrieval, FID, training loss.
- A GPU run over 30 minutes needs a cheap screen that passed first.
- A route that fails its criterion twice is closed. Reopening it is Amri's decision, on a
  different kind of evidence.

## Who does what

Amri (עמרי) is the architect: he decides, runs, and debugs. Claude writes the code.

- **Answer in Hebrew.** Technical terms, model names, file names and library names stay in English.
- **Code is in English** — identifiers, docstrings, comments. The repo must read well to an interviewer.
- **Explain before code.** Before each file: what it does, why it is built this way, what was decided inside it.
- **Define jargon.** One line, first time a DL/CV term appears. He is a third-year data engineering
  student — stats, linear algebra, databases and Python are solid; DL/CV vocabulary is not.
- **One component at a time.** Never generate ten files at once.
- **Never touch files outside this repo** without explicit permission for that specific action.

## Scope

**English only.** Hebrew is on hold and is his decision to reopen — do not plan Hebrew work,
do not chase the HHD dataset, do not propose collecting a Hebrew corpus. Cheap structural hooks
that keep the door open (charset abstraction, a direction parameter on the layout engine) stay.

## Hard rules

1. **No hardcoded paths** anywhere under `src/`. Everything comes from the config.
2. **The Colab notebook contains no logic.** It clones, installs, mounts Drive, calls `scripts/train.py`.
3. **src layout.** The package is `src/nib/`; import as `nib`, never `src.nib`.
4. **torch is an optional extra.** Colab ships its own CUDA-matched build; never force ours over it.

## Definition of done, for every task

A task is not complete until all four pass, in this order:

```bash
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
.venv/Scripts/python.exe -m pytest
```

...and `PROGRESS.md` is updated. Never leave the repo in a state where these fail.

## Session safety

Work may stop abruptly (usage limits). So:

- Work in increments that are individually verifiable and committable.
- Update `PROGRESS.md` as the last step of every task, before moving on.
- A fresh session with zero memory of prior conversation must be able to read
  `CLAUDE.md` + `PROGRESS.md` and resume without redoing anything.
