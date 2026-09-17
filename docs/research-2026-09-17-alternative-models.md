# Alternatives to Emuru (2026-09-17)

Why now: on Amri's own page Emuru writes readable lines that are not yet his hand,
and small marks vanish -- i-dots, colons, "#", short strokes. Three pipeline causes
were ruled out (the VAE, a global fade, slice seams); the likely cause is Emuru
regressing latents by mean squared error, which averages uncertain small marks away.
Per-writer LoRA fine-tuning is closed (7d no change, 7f -17.9 identity points).
So the question is which released model to put behind the `Generator` interface
next, where the same harness -- HWD identity, CER, cell 7g -- judges it.

Updates `research-2026-08-28-diffusion-vs-gan.md` and `research-2026-08-28-vatr-line.md`.

## Candidates

| | Eruku | DiffBrush | Paragraph LDM | HandwritingAgent |
|---|---|---|---|---|
| venue | WACV 2026 | ICCV 2025 | IJCV 2025 | arXiv 2606.18788, preprint |
| family | autoregressive, like Emuru | latent diffusion (SD1.5 VAE), 50 DDIM steps, + line and word discriminators | latent diffusion | vector strokes (SVG) |
| unit | line | **line** | **paragraph** (768x768) | word / line |
| style input | style line(s) | a style line of an unseen writer ("one-shot" per paper; config says `NUM_IMGS: 15`, not resolved) | a paragraph by the writer | images or online strokes |
| weights | on Hugging Face; **adapter already built (T18)** | IAM only (Google Drive); CVL weights asked for in issue #9, open | Google Drive | code link returns 404 |
| custom style and text | yes | **no** -- `generate.py` reads the dataset; content goes in as glyph tensors from `dataset.get_content`. A wrapper has to be written | **yes**, `demo.py` | not verified |
| trained on | synthetic, no IAM/CVL | IAM | IAM | not verified |
| licence | not verified | MIT | **not stated** | not verified |
| hardware | ran on our T4, 128 min / 300 lines | trained 8x4090, 4 days; inference not reported | inference > 8 GB VRAM; 9.06 s a paragraph on an A40 (Mayr et al.) | -- |

## What each reports (not comparable across papers)

- **Eruku paper**, one protocol for all, IAM lines: Eruku HWD **1.70**, Emuru 1.87,
  DiffusionPen 2.13, One-DM 2.83, VATr++ 2.38. CVL lines: Eruku **1.72**, Emuru 1.82,
  DiffusionPen 2.99. Emuru and Eruku never trained on IAM or CVL; the diffusion models
  did train on IAM and still lose on IAM lines here.
- **DiffBrush paper**, its own protocol, IAM lines: DiffBrush HWD **1.41**, DiffusionPen
  1.72, One-DM 1.80, VATr 1.87; delta-CER 8.59 against One-DM's 20.91. Emuru and Eruku
  are **not** in its tables, and no independent comparison with them was found.
- **Paragraph LDM paper**, its own protocol: IAM line-level HWD **0.86** with reranking,
  VATr 1.50. On CVL test (out of distribution) HWD 1.04 against VATr 1.62, but writer
  identification of its outputs falls to 11.2% top-1, from 50-56% on IAM -- its hand
  imitation weakens sharply off IAM. CER on paragraphs 4.77%, about 30% for lines
  over 75 characters. Reranks samples by writer identification and HTR, as we do.
- **HandwritingAgent**, its own tables, IAM lines: HWD 1.50 against Emuru 1.33 --
  worse on style -- FID 70.9 against 19.7.

## Our own history with Eruku

Measured on 2026-09-10 only by writer retrieval (5.3% against Emuru's 22.1%), the
metric shown on 2026-09-11 to measure sharpness rather than hand. **Eruku has never
been scored by HWD identity**, and its paper puts it ahead of Emuru on HWD on both
IAM and CVL lines. Its learned end token already removed Emuru's empty outputs.

## Reading

- The **cheapest real test** is Eruku through the current harness: no new code
  beyond the notebook cell. Its limit: the same autoregressive family, so it may
  share the vanishing small marks.
- The **strongest line-level diffusion candidate** is DiffBrush: a sampler rather
  than a regression, MIT, 64px lines like ours. Its risks: IAM-only training
  (generalisation to CVL and to Amri's pen unmeasured), no custom-input path, and no
  comparison with Emuru anywhere.
- The **most product-shaped** is the paragraph model: page of style in, page of
  text out, layout included. Its risks: licence unstated, a steep fall in imitation
  off IAM, and 768px for a whole paragraph leaves little resolution per letter.
- Word-level models (DiffusionPen, One-DM) would need word stitching and are
  superseded at line level by DiffBrush (One-DM's authors).

## DiffBrush, checked further

From the paper: **a separate model is trained per dataset** (IAM: 496 writers train,
161 test; CVL: 283 train, 27 test) and **there is no cross-dataset experiment** --
only the IAM model is released. One whole line is the style reference. Every image
is set on a **64 x 1024 canvas**: narrower lines are padded, wider ones resized to
fit -- CVL lines reach 1,762px and Amri's run 847-1,198px. Text goes in as Unifont
glyph images. Failure cases are in its Appendix M, not read.

**The generalisation risk is the main one.** Every model trained on IAM that has
been measured off IAM lost its imitation there: DiffusionPen on CVL lines HWD 2.99
against Emuru's 1.82 (Eruku paper, one protocol); the paragraph LDM's writer
identification fell from 50-56% on IAM to 11% on CVL. Emuru and Eruku trained on
millions of synthetic fonts, which is what generalises. A new writer photographed
in pen is further from IAM than CVL is.

**What changes the odds is fine-tuning.** For Emuru it failed for a structural
reason: the hand is copied from the style prefix, so the weights never needed it
(7d), and withholding it undid the copying (7f). A diffusion model has no prefix
to copy from -- the hand has to live in the weights and the style embedding -- which
is the setting where few-image personalisation of image diffusion models is
routine. **Whether that carries to handwriting from 10-22 lines is untested.**

## Not verified

DiffBrush's style-input count at test time and its inference time on a T4; whether
its weights generalise off IAM; Paragraph LDM's licence and T4 inference time;
HandwritingAgent's code and dependencies (repository 404); Eruku's licence;
DiffInk (arXiv 2509.23624, online trajectories) -- no code found.

## Sources

Eruku arXiv:2510.23240 · DiffBrush arXiv:2508.03256, github.com/dailenson/DiffBrush ·
Paragraph LDM arXiv:2409.00786, github.com/M4rt1nM4yr/paragraph_handwriting_imitation_ldm ·
HandwritingAgent arXiv:2606.18788 · github.com/koninik/awesome-handwritten-text-generation ·
DiffInk arXiv:2509.23624
