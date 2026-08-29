# nib

Few-shot handwriting synthesis. Give it one or two pages of someone's handwriting;
it produces a full page of *new* text in that hand.

No per-user training. One model is trained across hundreds of writers so that it learns
what "writing style" is in general; a new writer is handled at inference time from their
sample pages alone.

> **Status:** early. Phase 1 — data and evaluation infrastructure. No generative model yet.

## Why this is harder than it looks

The model generates a single word at a fixed height. Turning that into a page a person
would mistake for their own requires solving three things the papers mostly skip:

- **Domain gap.** Public handwriting datasets are flatbed scans. Real users photograph a
  page with a phone, at an angle, in room light. A style encoder trained naively learns to
  encode the lighting.
- **Layout.** Word spacing, baseline wobble, per-word slant, line wrap, ink consistency.
  None of it comes from the model. It is a separate engine, and it is most of what makes
  a page look real.
- **Evaluation.** "Does it look like my handwriting" is a human question. Pixel metrics
  do not answer it.

## Evaluation

Three automatic metrics, because no single one is sufficient — a model can produce
beautiful handwriting that is illegible, or legible handwriting in the wrong style:

| Metric | Question it answers |
|---|---|
| FID | Do generated images look like real handwriting in aggregate? |
| CER | Is the generated text actually readable as the intended text? |
| Writer retrieval | Is it in the *right person's* style? |

The headline number is none of these. It is the **deception rate**: the share of people
who look at a generated page and judge it to be genuine.

## Layout

```
configs/       Every parameter and path. No source file hardcodes either.
src/nib/       The package. Import as `nib`.
  data/        Parsing, preprocessing, dataset, packing
  models/      Style encoder, generator, discriminator
  losses/
  engine/      Training loop, checkpointing, metrics
  inference/   Style extraction, synthesis, page layout
  api/
  utils/
scripts/       Entry points: build_index.py, train.py, evaluate.py
notebooks/     Colab launcher only. Contains no logic.
tests/
```

Two rules that keep local development and Colab in sync:

1. No file under `src/` contains a hardcoded path. Paths come from the config.
2. The Colab notebook contains no logic. It clones, installs, mounts Drive, and calls
   `scripts/train.py`.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[torch,dev]"
pytest
```

On Colab, omit the `torch` extra — Colab's preinstalled build is matched to its GPU:

```bash
pip install -e ".[dev]"
```
