# First generation (2026-08-31)

The project produced handwriting for the first time. Held-out writer 0057, one
real line as style, three lines of text that writer never wrote.

## What works

Emuru loads and generates legible, stylistically consistent cursive. Given one
real line of a writer's hand plus its transcription, it continues in that hand.
The slant, letterforms and stroke weight visibly match the reference.

## Four things that had to be found by running it

**transformers 5.x cannot load the model.** Its custom code defines
`_tied_weights_keys`, and 5.x expects `all_tied_weights_keys`. Pinned to 4.x,
which also removed the TrOCR tokenizer workaround -- both failures were the same
version gap.

**It wants [-1, 1] with ink dark.** Measured rather than assumed: the same style
image fed four ways gave darkest pixels of 183 ([0,1] ink-dark), 233 (inverted
[0,1]), a blob (inverted [-1,1]), and one legible letter ([-1,1] ink-dark). That
matches its own output convention of `(x + 1) / 2`.

**It needs the style sample's transcription.** `style_text + " " + gen_text` goes
through T5 together. A user photographing a page has transcribed nothing, so the
product must read the sample first -- which is what TrOCR is already here for, at
the accuracy TrOCR happens to have. This constrains the architecture and is worth
carrying forward.

**Feed it lines, not words.** This was a flaw in our pipeline, not the model.
`normalise_word` stretches every crop to exactly 64px, so a one-letter word
becomes as tall as a whole line and a long word shrinks -- relative scale, which
is part of how a hand looks, is destroyed. Word-level generation produced tiny
faint marks and two runaway 756px outputs. Line-level produced the result above.

## Still open

**Contrast.** Raw CVL line images are low-contrast and the model faithfully
reproduces what it is given, so the output is faint. `normalise_ink` fixes it
completely -- the fix belongs in the line loading path, not after generation.

**Speed.** 30s per word and about 220s per line on CPU. A 300-sample evaluation
is hours here and minutes on a T4. Evaluation runs belong on Colab.

**Runaway generation.** Two of five word-level requests hit the token limit
instead of stopping. This is the known Emuru failure that its successor, Eruku,
was built to fix. Worth watching at line level.

## Next

A line-level pack, then `scripts/evaluate_generator.py` on Colab for the first
real numbers against the phase-1 references: FID 33.72, CER 12.33%, retrieval
66.9%.
