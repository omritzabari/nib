# First evaluation (2026-09-09)

The project's first measured result. Emuru, zero-shot, generating 300 lines in
the hands of 94 CVL writers it has never seen, each one a line that writer really
wrote so a real image of exactly that text in exactly that hand exists to compare
against. T4, 53 minutes, 0.09 lines per second.

## The numbers

| | generated | real handwriting | no style at all |
|---|---|---|---|
| FID | **63.92** | 19.15 | 254.29 |
| writer retrieval, top-1 | **20.5%** | 85.8% | 1.7% |
| writer retrieval, top-5 | **43.6%** | 94.4% | — |
| CER | **33.14%** | 12.06% | — |

The third column is the `fake` generator, which draws the target text in a
typeface. It exists to bound the scale from below: without it, 63.92 and 20.5%
are numbers with no units.

## What they say

**It produces handwriting.** On the FID scale running from printed text (254) to
real handwriting (19), the output sits at 64 — 81% of the way to real. This is
not a model emitting something that merely resembles writing.

**The identity carries only partly.** Writer retrieval is 20.5% against a chance
rate of 1.1%, so the style input is genuinely being used — eighteen times chance,
and an unstyled generator scores 1.7%. But real handwriting scores 85.8%, and on
that scale the output has travelled 23% of the way.

Those two figures together are the result: **Emuru writes convincing handwriting
that is only partly the right person's.** It picks up slant, stroke weight and
letterforms; it does not carry enough of a hand for a recogniser to name the
writer with confidence.

**The text is readable, at about three times the error rate of real handwriting.**
The gap is +21.1 points. Part of that is ours rather than the model's: 10.7% of
the output was truncated, and a line cut short loses its ending to deletion
errors. See below.

## What the infrastructure proved

The run completed, which the previous attempt did not.

At request 41 the model declined to write — the failure that killed the first
evaluation at 72 of 300. Three re-draws did not recover it, so the request was
excluded together with its ground truth and the run continued for another 259.
Four other requests were recovered by a retry that would not have existed a week
ago.

```
excluded        2 of 300
empty outputs   4 needed a retry (9 extra draws), 2 never produced anything
gallery         1127 images over 94 writers
```

The three metrics also reproduced exactly across devices: a T4 gave FID 19.1521
against 19.15215 measured on CPU, and writer retrieval 85.8% / 94.4% over 197
queries against 126 writers on both.

## The one thing worth fixing

```
truncated       32 of 298  (10.7%)
  longest affected: 82 chars, 2038px at a budget of 256 tokens
```

The token budget was 4.0 tokens per character, capped at 256. Measured over 1,524
lines of the pack, a character needs 2.79 tokens on average, 4.16 at the 95th
percentile and 5.04 at the 99th — so 4.0 covers about 92% of the data, and 5.9%
of lines would not fit their budget at all. The observed 10.7% is higher than
that, which says the model writes somewhat wider than the reference it is shown.

Raised to 5.5 tokens per character, above the 99th percentile, with the cap at
384 so it does not become the binding constraint for ordinary text: 47
characters would have clamped at 256, and more than half the pack is longer than
that. The share of real lines that do not fit falls from 5.9% to 0.7%.

A second evaluation follows. The CER gap should narrow, since some of it is
currently charged for endings the budget cut off.

## Next

Beyond the re-run, the open question is the one retrieval raises: 20.5% against a
ceiling of 85.8% is real style transfer and not close to a solved problem. Emuru
was trained on synthetic fonts and has never seen CVL, which is exactly the claim
being tested, and this is what zero-shot buys. Improving it means either a
different checkpoint or per-writer adaptation, and the second is what this
project set out to avoid.
