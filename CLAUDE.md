# Working agreement — nib

Few-shot handwriting synthesis. Learn a writer's hand from 1-2 sample pages,
generate new text in it. No per-user training.

**Read `PROGRESS.md` first.** It holds the live task state and the exact next action.

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
