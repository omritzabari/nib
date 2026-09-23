# Is Emuru the right model? — a survey, 2026-09-23

Written because the plasters have run out. Quality control, choosing by hand, ink
calibration and the stroke repair each bought something real; what is left —
mangled letters at the start and end of a line, a word missing in one line in
eight — comes from the model's own drawing, and five attempts to train it all
failed (PROGRESS.md: 7d, 7f, 7o, 7t, and the 7v screen that closed the route).

Everything this project built is model-agnostic: the scorecard, the quality
control, the ink calibration, the stroke repair and the blind test all take a
line image from anything. A new model can be judged in an afternoon.

## What exists, at line level, with weights

Only three line-level models since 2025 ship code or weights, by the field's own
curated list (`koninik/awesome-handwritten-text-generation`).

| model | venue | weights | trained on | our own reading |
|---|---|---|---|---|
| **Emuru** | CVPR 2025 | Apache-2.0 | synthetic fonts only | what nib runs today |
| **Eruku** | WACV 2026 | Apache-2.0 | **synthetic fonts only** | tried once, on the broken metric — see below |
| DiffBrush | ICCV 2025 | MIT code, IAM weights | IAM | closed: identity 40.6% against Emuru's 53.7% on the same writers (T37) |

Word-level models (One-DM, DiffusionPen, VATr++) would need the layout engine to
assemble lines and are trained on IAM, whose licence is research-only. Stroke
models (DiffInk, ICLR 2026, weights released) generate pen trajectories, not
images, and take an *online* style reference — a photographed page would first
have to be turned into strokes (InkSight). That is a different pipeline, not a
model swap.

## Eruku is the one candidate, and it was judged on the wrong evidence

Eruku is Emuru's successor from the same group, and it was built against the same
failures this project measured independently:

- **a learned end-of-generation token** instead of Emuru's padding-similarity
  heuristic — the heuristic cost this project 8.7% of one run to a stop that never
  came, and whole requests to a stop that fired inside the style image;
- **no style transcription needed**, which removes TrOCR from the enrolment path;
- **classifier-free guidance** over the text, a knob Emuru does not have.

Its paper reports, on **CVL lines with unseen styles** — the same benchmark this
project measures on:

| | HWD (lower is better) | ΔCER | FID |
|---|---|---|---|
| Emuru | 1.82 | 0.13 | 14.39 |
| **Eruku** | **1.72** | **0.04** | 12.32 |

ΔCER — how much worse the generated line reads than the real one — is a third of
Emuru's. That is exactly the defect left on our page: garbled words and dropped
ones.

**The benchmark stays valid.** The paper states the released model is trained on
synthetic fonts alone ("over 100k typewritten and calligraphic fonts"), never on
IAM, CVL or RIMES, so the held-out CVL writers are as unseen for Eruku as for
Emuru and the identity figure is comparable. (The Hugging Face card lists those
datasets as tags; the paper's training section is what this rests on, and the
first run should check that a CVL writer is not reproduced suspiciously well.)

**Why our own 2026-09-10 run said otherwise.** T18 and T19 ran Eruku over 300
lines and reported FID 80.74 and writer retrieval 5.3% against Emuru's 66.72 and
22.1%, and it was set aside. Writer retrieval was found a day later to measure
sharpness, not identity — a real line blurred by 0.8 px scores 12.2% where the
sharp one scores 96.8% — and a soft output is precisely what it punishes. HWD
identity did not exist yet, the quality control did not exist yet, and the run had
one draw and no choice by hand. The one figure that did survive pointed the other
way: Eruku's CER gap was +13.0 where Emuru's was +20.4.

`nib.models.eruku.ErukuGenerator` is already written, passes the style
transcription when there is one, checks that the style prefix is not returned in
the output, and exposes `cfg_scale`. Nothing needs building to try it.

## The run, and the criterion set before it

Two runs, one session, about two hours, both on the **same 150 held-out requests**
that every figure in this project uses, both with **one draw and no quality
control**, because a model swap must be judged on what the model draws, not on
what selection rescues:

1. **Emuru, one draw** — the matched control, about 27 minutes. None exists: every
   Emuru run on these requests had quality control.
2. **Eruku, one draw**, `cfg_scale` at its default 1.25 — about 88 minutes at the
   35 s a line measured in T18.

Judged on the scorecard of CLAUDE.md, paired by writer:

- **keep going with Eruku** if identity rises by 5 points or more with its interval
  above zero, or if missing-a-word falls by half or more while identity does not
  fall;
- **stop and stay on Emuru** otherwise.

If Eruku wins, the whole pipeline moves to it — quality control, choice by hand,
ink calibration, and the stroke repair retrained on *its* defects, which are not
Emuru's — and the blind test is run on its lines. If it loses, the model question
is closed for this project's data and the work is the product: the page engine,
the redraw loop, and the judges.
