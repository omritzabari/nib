# Independent review, answered — 2026-09-20

An answer to `review-request-2026-09-20.md`, written against the repository rather
than from the brief. Every figure below was computed from artefacts already on disk
(`outputs/*/analysis.npz`, `outputs/*/samples/`) with no GPU and no new generation.
The scripts are in `scripts/` where they have been kept; where a figure is quoted
without a script, the computation is stated in full so it can be repeated.

A second review of the same brief by Gemini is answered in §6.

---

## 1. The measurement, attacked

### 1.1 The obvious attack fails, and the metric survives it

If a set of images sits far from *every* writer's reference — because it is
generated, not because it is nobody's hand — then a common offset compresses
`gap = d(others) - d(own)` and deflates identity for reasons that have nothing to
do with the writer. Tested by recomputing identity from the relative gap,
`(d(others) - d(own)) / d(others)`, which is invariant to that offset:

| run | identity as reported | identity from the relative gap |
|---|---|---|
| zero-shot, one style line | 56.5% | 50.6% |
| + quality control (7c, 300 lines) | 65.0% | 55.9% |
| + closest draw by embedding (7e) | 69.8% | 60.8% |
| style line chosen by letters (7m) | 61.8% | 53.6% |
| two references (refs2) | 42.3% | 38.7% |

Every figure falls by 5 to 9 points and **the order does not change**. The
normalisation neither hides a gain nor invents one. The metric survives.

### 1.2 HWD was trained to tell fonts apart, and cannot see irregularity

HWD is a VGG-16 trained on 100M lines *rendered from fonts* to classify them.
Within one font every `a` is identical, so a feature space trained on that corpus
has no reason to encode variation *within* a line — there is none in its training
data to encode.

Variation within a line is the strongest human cue. In
`outputs/probe_passage_page1/comparison.png`, line 16, the real hand swings from a
large `Or` to a small `in the`; the generated line is even. That evenness is the
"still not my handwriting", not the missing i-dots.

HWD and Emuru come from the same lab and the same rendering pipeline. A
font-pretrained generator is being graded by a font-trained ruler, and anything
trained on real ink is graded down by it. **Emuru 53.7% against DiffBrush 40.6% is
not clean evidence**, and no conclusion about architecture should rest on it.

### 1.3 A second identity readout is already computed and is not reported

`Distance.nearest_is_right` — whether a writer's own reference is the closest of
all — is stored in every `analysis.npz` and printed by `hwd.describe`, but never
carried into a comparison.

| | identity | own writer nearest (chance ≈ 1.2%) |
|---|---|---|
| real lines | 100% | 97.3% |
| 7e (byhand) | 69.8% | **63.0%** |
| 7m (styleletters) | 61.8% | **46.6%** |
| typeface | 0% | 1.4% |

Both middle rows are the same 73 writers. Eight points of identity correspond to
**16.4 points of nearest-is-right**: the second readout separates configurations
about twice as sharply, says the same thing, and reads plainly. It is not the
broken writer-retrieval metric — it sits on HWD's features, not on this project's
embedder.

### 1.4 Real and generated do not receive the same post-processing

Every real pack line passes through `preprocessing.normalise_ink` — a percentile
contrast stretch. Every generated line passes through `generator.to_uint8`, which
rescales the decoder's raw min and max. These are not the same operation, and the
difference lands on pixel values that HWD reads.

Measured on 7e's 32 saved pairs: mean darkness 190.6 real against 177.5 generated,
and the standard deviation of ink darkness 48.0 against 38.1. Putting the generated
side through `normalise_ink` **overshoots** to 207.1 and 51.1. So the parity gap is
real, and the fix is to match *the writer's own style line*, not to apply the same
global stretch. It is a 15-minute ablation on Colab over images already on disk.

### 1.5 The only test that answers the product question has n = 5

`outputs/probe_passage_page1/blind/key.json` holds five pairs, and `A` is the real
line in four of the five. Around 80% accuracy the 95% interval at n = 5 is roughly
±44%. It is a demonstration, not a measurement.

### 1.6 What to measure instead: a classifier two-sample test

The goal is "someone who knows the hand cannot tell". The machine form of that is
the accuracy of a discriminator between the writer's real lines and the system's
output, where **50% is success**.

Ten hand-written statistics, no network, leave-one-out over 7e's 32 pairs:

```
discriminator accuracy   96.9%  ±4.3%     (50% = indistinguishable)
```

| | real | generated (7e) | that feature alone |
|---|---|---|---|
| spread of ink darkness (sd) | 48.00 | 38.14 | **93.8%** |
| ink height / 64 px band | 0.94 | 0.79 | **78.1%** |
| mean darkness | 190.63 | 177.51 | 73.4% |
| baseline wobble (sd) | 7.48 | 6.37 | 68.8% |
| components per 100 columns | 3.69 | 6.44 | 64.1% |
| ink per column | 5.94 | 4.94 | 57.8% |
| stroke width (2 × median distance transform) | 2.07 | 2.04 | noise |

Both facts are true at once: 69.8% identity, 96.9% detectability. Only the second
is the product's question. Applying `normalise_ink` to the generated side leaves
the total at 96.9% — the discriminator simply crosses to the other side of the same
feature.

### 1.7 Three corrections to the diagnosis in the brief

**"Small marks vanish" is not the mechanism.** Counting blobs of area ≤ 12 px with
both sides ≤ 5 px, the generated lines carry **2.3 to 3 times more** of them than
the real ones, on 35% less total ink, with the median component area falling from
173 px to 77. Stroke width is unchanged (2.07 against 2.04). That is not fading and
not thinning — it is **fragmentation**: continuous strokes break into pieces and
some pieces fall below threshold. A missing i-dot is one symptom of it.

The ink deficit holds at every threshold, so it is not an artefact of binarising:
ink at 200/160/128/96 is 6480/5470/4707/3926 real against 4221/3449/2862/2230
generated — a constant ratio near 0.65.

**The writing is drawn about 20% too small.** Ink height 0.94 of the band against
0.79; ink span 915 px against 743. This is *not* what row 12 tested: cropping tight
**erases** that information, which is why it cost 4.7 points. The untried operation
is the opposite one — **rescale the generated line so its ink extent matches the
style line's**.

**The fade test asked the wrong question.** "39% of stroke pixels are mid-grey in
both" asks whether the strokes are faded — they are not. The question that
separates them is whether the tone is as *varied*, and it is not: 20% less spread.

**On Amri's own page** (the five blind pairs): ink per column 4.96 → 3.77 (−24%),
ink height 0.74 → 0.63, darkness 199.5 → 190.3, darkness sd 50.5 → 45.1 — but
components per 100 columns 3.90 → **3.94**, no fragmentation at all. His hand is
disconnected print, so there are no joins to break. On his page the defect is
simply **less ink at the same number of marks**: the marks are present and too
weak. That is exactly what he reported.

---

## 2. The system drops words, and nothing in the pipeline catches it

`candidates.ACCEPT_CER` is 0.5, and `candidates.overrun` rejects a draw that writes
**beyond either end** of the target. There is no counterpart for a draw that writes
**less** than the target.

On Amri's page, from `comparison.png`:

| requested | written |
|---|---|
| `warm at noon` | `walm noon.` |
| `Order #378 at Lior's Cafe` | `Order 3 t Lior's Cafe:` |
| `If You find it, Please` | `I You find .. please` |

Roughly one short word a line. Each of those draws reads at about 7–14% CER, has
`overrun = 0`, and is accepted as the first readable draw. CER 10.6% against his own
8.7% looked like ordinary recogniser noise; it is not — it is content going missing.

A reader rejects a page with words missing before forming any opinion about the
handwriting. **This is the first thing to fix and it costs no GPU.** The check is
symmetric to `overrun`: align the reading to the target and reject a draw whose
longest run of deleted target characters reaches some *k*, calibrated the way
`ACCEPT_OVERRUN` was — on the 290 real CVL lines, where a real readable line should
almost never show a deletion run of 3 or more.

> **Correction, 2026-09-21: this check, built as `candidates.underrun`, does not
> work with the pipeline's reader.** The readings in the table above were taken by
> eye. TrOCR-small, the selector, drops short words from complete lines on its own
> — run on the fake generator, whose lines are complete by construction, it read
> "not the rapid calculation" as "not rapid calculation" and "it will be enough" as
> "it will", and at a threshold of 1 it set aside 16 of 24 correct draws. The check
> is therefore off by default. The width check (`candidates.WIDTH_BAND`) catches
> smears and truncations but not one missing word: on the page's five blind pairs
> every line sits inside the band. Catching a single dropped short word needs a
> reader asked *whether this text is in the image* — the recogniser scoring the
> target text teacher-forced — not *what text is in the image*. The failure stands;
> the fix proposed here does not.
>
> **Update, same day: built, and the failure is larger than this section says.**
> `TrOcrRecogniser.omissions()` does exactly that. On 7e's 150 kept lines it finds
> a word missing in 31 (20.7%) — all ten inspected by eye are real, and 13 of the
> 31 are the first word of the line — while flagging 1 of 120 complete real CVL
> lines. It finds "at" and "#378" on Amri's page and nothing on his complete
> lines. See `candidates.OMISSION_SUPPORT` and PROGRESS.md, T41.

---

## 3. What is wrong with the frame

**The conditioning channel is saturated.** Two style lines joined broke the model
(row 4). Choosing the style line by letter coverage did nothing (row 11). Per-writer
LoRA did nothing (rows 5, 8). When every intervention on the style channel returns
zero, 65–70% is what one prefix line is worth in Emuru, and it is not a training
problem.

**Row 6 is the positive control, and it is being read as a failure.** Training with
another writer's line as prefix and the loss on the target line cost 17.9 points —
because the prefix was uninformative about the target's style, so the cheapest
descent direction was to *suppress* the conditioning, and the model took it and
found 17.9 points there. That is proof the prefix pathway is trainable and is what
carries identity. It argues *for* the same-writer version, not against it.

**Every training failure is the same failure: the objective made the wrong descent
direction the cheapest one.** Row 10 (4,000 pooled lines, no prefix) had nothing but
the corpus mean to descend toward, so it descended toward it — loss −15%, identity
−18, FID 56 → 89. Row 6 had a prefix that predicted nothing, so it learned to ignore
prefixes. `--objective imitate` is the first setting in which the prefix is always
present and always informative, which makes copying it the cheapest direction
available.

**Two defects, and only one was ever attacked.** "Whose hand is it" is saturated at
70%. "Is this a photograph of ink" — 24–35% less ink, 1.7× the components, 20% too
small, flat tone — was never touched, needs no training, and is what the one real
user actually sees.

---

## 4. Candidates

**SD 1.5 / SDXL / Flux with a per-user LoRA: no.** At 64 px line height a general
diffusion model does not render legible text; the ControlNet on rendered text that
would force legibility does not exist off the shelf and would take days to train on
rendered-text ↔ handwriting pairs; 20–30 minutes a user breaks the enrolment budget;
and conditioning on rendered glyphs pushes the letterforms toward the font, which is
the opposite of the goal. The one version worth GPU is img2img at low strength as a
**refiner over Emuru's output**, restoring ink texture without touching letterform.

**One-DM** (ECCV 2024, `dailenson/One-DM`) — one-shot, released weights, and it
conditions explicitly on the **high-frequency components** of the reference, which
is the fine-detail problem stated directly. Word-level and IAM-trained, so it
inherits the licence problem and needs composition through the layout engine. Worth
an hour to wire zero-shot; the detail mechanism is the part to steal.

**HandwritingAgent** (arXiv 2606.18788, 2026-06) — synthesises SVG strokes from a
glyph bank segmented out of the reference, with a reasoning model planning stroke
order. No GPU. Reports HWD 1.33 against Emuru's 2.30 on IAM words. **The GitHub link
in the paper returned 404 when checked on 2026-09-20** — verify before planning
around it.

**InkSight** (Google, image → digital ink) makes the stroke route real: it recovers
trajectories from a static image, which is the step that was assumed impossible.
Still the most expensive of the three routes, but no longer research-only.

**Eruku** (arXiv 2510.23240) — worth reopening. It was closed for sharing Emuru's
regression, but the measurements say the dominant defect is ink deficit and
fragmentation, which its reliability work targets directly, and it is the cheapest
model swap available.

**Cut-and-paste — rejected by the project's owner, 2026-09-21.** An earlier draft
of this section recommended assembling new text from glyphs cut out of the
user's own page, reasoning from Amri's hand being disconnected print. That was
the wrong frame: the product is for *any* writer, cursive included, with the
model adapted to them, and stitching fits only print hands and adapts no model.
It is not a fallback. See CLAUDE.md.

---

## 5. The next experiments, in order

**P0 — reject draws that drop words. No GPU, one evening.** §2. Nothing else matters
while a page is missing a word a line.

**P1 — match the generated line's ink statistics to the writer's style line. No GPU,
one evening.** Vertical ink extent, ink per column, tone percentiles. Re-score HWD on
Colab over images already on disk (~15 minutes). Stop criterion: if identity does not
rise by ≥ 4 points and the discriminator does not fall below 85%, the post-processing
route is closed, for free.

**P2 — `--objective imitate`, as a general adapter. ~1 GPU-hour.** Judgement and
changes in §7.

**P3 — the blind test, properly.** 40 pairs, A/B balanced by coin flip, full lines,
three or more judges who know the hand, over several writers' pages. Conditions:
the system and real. Stop criterion: the lower bound of the 95% interval on judge accuracy
below 65%.

---

## 6. On the second review (Gemini, same brief)

Agrees on three points: HWD is font-trained and blind to human variability; MSE
averages small marks away. (A third point both reviews made -- that cut-and-paste
is the product answer -- is withdrawn: rejected by the owner, see §4.)

Four errors:

1. **"A generic, average output sits at the centre of the space, shrinks its distance
   to all writers, and so scores artificially high."** The opposite. Identity is
   `d(others) − d(own)`; an output equidistant from everyone scores zero. That is
   precisely what the typeface measures — `own 3.023, others 3.024, gap +0.001,
   identity 0.0%`. The metric is constructed to be immune to this, and the numbers
   show it is. Acting on this claim would mean discarding a sound metric.
2. **"Train a Siamese verifier on real handwriting instead."** Already done, already
   failed: the project's writer embedder drops from 96.8% to 12.2% on a real line
   blurred by 0.8 px. Any learned discriminator on this data latches onto generation
   artefacts. That is useful as a *detector* — §1.6 — but it is not identity.
3. **"ControlNet trained on rendered text, 20–30 minutes a user on a T4."** That
   ControlNet does not exist and would take days to train. The cost estimate covers
   only the LoRA and assumes the missing asset.
4. **"CER near zero for cut-and-paste."** The floor is the writer's own legibility,
   10.7%, not zero.

One design point to reject: a 2AFC on *a word or two* hides rhythm, size variation
and baseline drift — the defects that give the system away. Test the artefact that
ships: full lines, a full page.

One thing it added that this review missed: **InkSight**, which makes image →
trajectory a real step rather than an assumed impossibility.

One disagreement that matters: it proposes `imitate` **per-writer, 200–300 steps**.
That is row 5's recipe, which returned a precise zero and pushed CER from 12.7% to
19.7%. Sixteen lines will not move this pathway. It must be a **general adapter over
the 215 training writers** — row 10's scale with row 6's mechanism, in the right
direction — which also ships with the system and costs the user nothing at
enrolment.

---

## 7. `--objective imitate`: run it, third

**Worth running: yes.** It is the only objective that matches what the model does at
inference, row 6 proves the pathway it trains is the pathway that carries identity,
and as a general adapter it is the only training route compatible with "the user
uploads one page and nothing is trained at enrolment". Expect +3 to +8 identity
points, and know that identity is not what the user is judging.

**One correction to the brief's own risk list.** The paired canvas reaching ~1,800 px
against 768 px of pre-training is listed as a risk. At inference the model already
runs there: the style image is 500–1,100 px and the continuation adds ~900 more, so
generation has always operated at 2–2.6× the pre-training length. Training at 1,800 px
**brings training into the regime inference already occupies** for the first time.
It is an argument for the experiment, not against it.

**What to change.**

- **Learning rate 1e-4 → 3e-5, steps 2,000 → 3,000, and probe at 0 / 300 / 800 /
  1,500 / 3,000.** The objective decides which direction is cheapest; the learning
  rate decides how far the model travels before anything can be observed. With a
  10-minute probe available, the run should be a search for the peak, not one number
  at the end. Keep the adapter from the best probe, not the last step.
- **Keep rank 8.** Row 6 moved 17.9 points at rank 8. Capacity is not the bottleneck.
- **Keep the loss mask exactly as it is.** Any weight on the prefix reintroduces
  row 10.
- **Constrain the prefix line to the width band used at inference** (500–1,100 px).
  The model should learn to read style from the length it will actually be given.
- **Measure the gap rather than assuming 16 px.** At inference `join_style` returns a
  single style image untouched and the model writes `style_text + " " + gen_text`, so
  the separation is whatever word-space the model chooses. Measure it on twenty
  existing generations and train with that.
- **Probe more than identity.** Record `nearest_is_right` and the four ink statistics
  at every checkpoint. An adapter that gains identity while losing ink is trading what
  the user sees for what the metric sees.

**Detecting the 7o failure early.** Row 10's signature is a *domain* move — training
loss down, FID 56 → 89, HWD distance 2.76 — not an identity move. Three probes, in
increasing cost:

1. **Ink statistics, ~90 seconds.** Generate 8 lines at the checkpoint and compare
   ink per column, ink extent, components per 100 columns and tone spread against the
   writer's own style line. If they are moving *away* from the style line, the adapter
   is pulling toward the corpus mean. Visible within the first 300 steps.
2. **Identity on 40 held-out lines at step 0 and step 300, ~20 minutes total.** Step 0
   must reproduce the untrained figure, which also proves the probe works. If step 300
   is already below step 0, stop.
3. **The control that makes the run interpretable, ~5 minutes: 200 steps with
   `context="other"` at the same learning rate.** If `same` and `other` move identity
   in the *same* direction, the objective is doing nothing and the run is watching
   pure domain drift. If they diverge, the objective is working. This turns a
   one-armed run into a controlled one and is the single best addition to the plan.

**Stop criterion.** A paired gain of ≥ 5 points with an interval excluding zero on
the same 150 requests. Otherwise stop training Emuru: six nulls through one mechanism
is enough, and the remaining work is P0 and P1.
