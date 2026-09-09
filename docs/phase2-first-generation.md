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

## Correction, 2026-09-01

**The runaway was ours.** Emuru's VAE compresses width by a factor of eight, so
one generated token is eight pixels -- `lengths = (lengths / 8).ceil()` in its own
`_generate`. This wrapper passed a flat `max_new_tokens=96`, which is 768 pixels.
The two outputs recorded above as runaway were 756 pixels wide: 94 and 95 tokens,
stopping because they had run out of canvas.

96 was chosen when the working unit was a word, where it was generous by a factor
of five. At line level it could not finish an *average* line, which is 886 pixels.
The model's own default is 256.

The lesson is not about Emuru. A number that was right for one unit was carried
into another without being re-derived, and a plausible story from the literature
was available to explain the symptom -- so nobody did the arithmetic. The budget
now scales with the target's length, and every output that still reaches its cap
is counted and reported rather than absorbed.

## What the first Colab run then found, 2026-09-01

**One request in seventy-two produces nothing at all.** Emuru returns
`imgs[style_width : stop * 8]`, and it looks for the stop by scanning the whole
canvas -- style prefix included -- for ten consecutive latent slices resembling
its padding token. The last slices of a real style line already sit close to that
token: 0.86 to 0.97 cosine similarity against a threshold of 0.485. When the
model opens by emitting padding, the window straddles the boundary, the stop
lands at or before the style image's own width, and the slice is empty.

It is intermittent because `_img_encode` calls `latent_dist.sample()`: the style
latents are drawn afresh each time, so this is a bad draw rather than a bad
request. Three re-draws, each counted; a request that fails all four is excluded
together with its ground truth, so the pairing of generated to real never shifts.

## Next

`scripts/evaluate_generator.py` on Colab, against the line-level references in
`references/`.
