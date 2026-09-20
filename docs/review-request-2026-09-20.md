# Independent review request — few-shot handwriting synthesis (2026-09-20)

Paste everything below the line into a fresh conversation with another model.

---

You are reviewing a working machine-learning project with fresh eyes. **Answer in
Hebrew**, but keep model names, metric names and file names in English.

I do not want reassurance and I do not want a summary of what I already know. I
want (a) to be told where my method or my measurements are wrong, and (b) concrete
proposals that can plausibly get this system out of the mediocre plateau it is on.
Be specific enough to act on: named models, named datasets, training objectives,
and an experiment I can run in about an hour on one GPU.

## The goal

Photograph one or two pages of a person's handwriting, type any English text, get
that text back **in that person's hand**, well enough that someone who knows their
handwriting cannot tell. Not a paper, not a portfolio piece — a system that works.
The person will always supply the transcription of their own page at enrolment
(either they copy a passage we give them, or they type what their page says), so
the text of every enrolment line is known exactly.

## Hard constraints

- One Colab **T4 (16 GB)**; experiments should fit in roughly an hour. Training a
  generator from scratch on tens of millions of images is out of reach.
- English only. 79-character charset including digits and punctuation.
- Enrolment must cost the user minutes at most, and they give one or two pages —
  about 20-40 lines.
- Released weights and a licence that a product could eventually live with matter;
  IAM-derived weights are non-commercial.
- The repository is public and includes a very detailed `PROGRESS.md`:
  https://github.com/omritzabari/nib

## What the system is now

- **Generator: Emuru** (Pippi et al., CVPR 2025, autoregressive: a T5 over a VAE's
  latent slices, 8 px per slice, 64 px line height). It is shown one line of the
  writer's handwriting plus its transcription, and continues in that style. It was
  pre-trained on millions of lines **rendered from fonts** and has never been
  trained on real handwriting.
- **Quality control:** for each line to write, up to four draws, each from one of
  four of the writer's lines chosen by width (500-1100 px generated best); a draw
  is kept if a recogniser reads it at ≤50% CER and it writes nothing beyond the
  requested text. Optionally all four draws are made and the one whose writer
  embedding is closest to the writer's page is kept.
- **Pipeline around it:** a photographed page is split into lines (22 of 22 on the
  real page), each line paired with its known text; output lines are produced one
  at a time.

## How it is measured

- **HWD identity (primary).** HWD is a VGG-16 trained on 100M rendered lines
  (aimagelab/HWD); per writer we take the mean feature of their lines. For a set of
  images we measure the distance to *their own* writer's reference and the mean
  distance to *every other* writer's reference; the **gap** between the two is how
  much the set says "this writer in particular". Identity is the generated gap as a
  share of the real lines' gap, so 0% = nobody's hand, 100% = what real lines of
  that writer carry. Every figure carries a 95% interval from resampling writers,
  and comparisons between conditions are paired by writer.
- Also FID, CER by TrOCR-base (a different recogniser than the one selecting
  draws), HWD distance, and writer retrieval — the last is known to be broken: a
  real line blurred by 0.8 px drops from 96.8% to 12.2%, so it measures sharpness.
- Data: CVL, 9,142 English lines, 309 writers, writer-disjoint split (216 train /
  94 test). IAM was unavailable (the FKI site was down). Every figure below is on
  writers the model has never seen.

## Everything that has been tried, with numbers

| # | What | Identity | Notes |
|---|---|---|---|
| 1 | Emuru zero-shot, one style line, 300 lines | **56.5%** [50.9, 61.9] | CER 30.4% vs 11.3% real; 10% of lines collapse into a smear |
| 2 | + quality control (4 style lines, up to 4 draws, first readable) | **65.0%** [60.3, 69.6] | paired gain +8.3 [2.4, 14.8]; CER 12.5% vs 10.7%; FID 67.7 -> 55.9 |
| 3 | + keep the readable draw closest to the writer's embedding | **69.8%** [64.5, 75.2] | paired +5.2 [-0.5, 10.1] over the same 150 lines; 4 draws always, ~42 s a line on a T4 |
| 4 | Two style lines joined side by side | 42.3% | CER 59.6%: the model loses the boundary between style text and target text. Closed |
| 5 | Per-writer LoRA on Emuru (16 lines, 150 steps, rank 8, 56 s a writer) | +0.1 [-6.8, 6.7] | no change at all; CER 12.7% -> 19.7% |
| 6 | Same, but with another writer's line as prefix and the loss on the writer's line only (to force the hand into the weights) | **-17.9** [-24.9, -11.1] | clearly worse |
| 7 | DiffBrush (ICCV 2025, latent diffusion, line-level, IAM-trained) zero-shot | 40.6% | against Emuru's 53.7% on the same 24 writers |
| 8 | DiffBrush per-writer LoRA, gentle (60 steps, 2e-5, self-attention only) | -0.2 [-2.2, +2.1] | a precise zero |
| 9 | DiffBrush per-writer LoRA, strong (150-300 steps, 1e-4, all attention) | text degrades | letters melt; the hand does not approach the writer |
| 10 | **General** LoRA adaptation of Emuru on 4,000 real CVL training-split lines (2,000 steps, 12 min) | **47.0%** [42.0, 51.7] | against 64.6% untrained on the same 150 requests. Training loss fell 15% while identity fell 18 points |
| 11 | Choosing the style line by which of the writer's lines shows most of the target's characters | -2.8 [-7.4, +1.7] | nothing |
| 12 | Cropping the generated line tight to its ink (it sits in a 64 px band with 7 px of white above and below, where real lines are cropped tight) | -4.7 [-7.2, -2.3] | worse, so this framing is not the problem |

Reference points: real lines score HWD 1.01 and find their own writer 97.3% of the
time; the same texts in a typeface score identity 0.0%.

## On the one real user (the page this is actually for)

One page, pen on lined paper, photographed with a phone; 22 of 22 lines segmented.
The system wrote all 22 lines of a text he had never written. CER 10.6% against
8.7% for his own lines — it reads about as well as he writes. His verdict: "not bad
at all, but still not my handwriting." Specifically **small marks vanish**: dots on
i, colons, the hook on r, "#", short digit strokes.

That last failure was investigated and three explanations were ruled out:
- **Not the VAE:** his real lines encoded and decoded by Emuru's VAE come back with
  every dot and mark intact.
- **Not a global fade:** the share of stroke pixels that are mid-grey is 39% in his
  real lines and 39% in the generated ones.
- **Not seams between the 8 px slices:** breaks in strokes do not align with the
  slice grid.

The remaining explanation, inferred and not proven: Emuru predicts each latent
slice by **mean squared error regression**, and the average of "a dot might be here,
or here, or not at all" is faint or nothing.

## The pattern I have been unable to break

Every attempt to *train* anything has either done nothing or made imitation worse
(rows 5, 6, 8, 9, 10). In all of them the task the model practised was
*reconstruct this line*, while what the system actually needs at generation time is
*imitate the hand in front of you*. The one variant not yet run: the prefix is
**another line by the same writer**, with the loss counted only on the second line —
row 6 with "another writer" replaced by "the same writer". It is written and ready.

A bug found on the way, which invalidated two runs before it was fixed: DiffBrush's
style encoder is a ResNet-18 with 40 batch norms, and training with the backbone in
train mode moved all of them; their statistics are buffers, not parameters, so
resetting the LoRA adapter could not undo it and every writer inherited the drift.

## What I want from you

1. **Attack the measurement first.** Is HWD identity, as defined above, actually
   measuring what I think? Is there a defensible reason a real gain could be hidden
   by it, or a fake gain created by it? What would you measure instead, given the
   goal is "a person who knows the hand cannot tell"?
2. **Tell me what is wrong with the approach**, not with the details. If the whole
   frame — a pre-trained line generator conditioned on one style line — is the
   ceiling, say so and say what replaces it.
3. **Give me concrete candidates**, with released weights where possible, and say
   for each: what it would take to adapt it here, what it would cost on a T4, and
   what you expect it to score. Please consider at least:
   - per-user LoRA / DreamBooth on a **general** image diffusion model (SD 1.5,
     SDXL, Flux) conditioned on rendered text, rather than on a handwriting-specific
     model;
   - **online / stroke-based** generation (predicting pen trajectories) plus a
     renderer, given that small marks are exactly what a stroke model would not
     drop;
   - a **non-generative** baseline I have never tried: cutting the user's own
     letters and words out of their page and assembling new text from them, with
     spacing and slant handled by layout. What would that score here, and is it the
     right product answer even if it is not research?
   - anything published in 2025-2026 that beats Emuru on unseen-writer imitation.
4. **Design the next experiment**, with a stop criterion and its cost in GPU hours,
   assuming I will run exactly one or two before deciding whether to stop improving
   the model and ship what I have.

Ask me for any number you need from the repository rather than guessing.
