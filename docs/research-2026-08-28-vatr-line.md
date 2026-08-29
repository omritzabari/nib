# Literature survey — the VATr line (2026-08-28)

Verified against arXiv, official repos and proceedings. Items marked "not verified"
were not confirmable and must not be relied on.

## Headline

**VATr has been superseded twice by its own authors.** The planned architecture is a
faithful VATr reimplementation and is roughly two generations behind.

| Model | Venue | Output | Content conditioning | Res |
|---|---|---|---|---|
| VATr | CVPR 2023 | raster, word | Unifont glyph images | 32px |
| VATr++ | TPAMI 2024 | raster, word | Unifont glyph images | 32px |
| Emuru | CVPR 2025 | raster, **variable-length line** | char tokens (ByT5) + T5 encoder | 64px |
| Eruku | WACV 2026 | raster, line | multimodal prompt tokens | not verified |

Reported on IAM, unseen writers, line level: Emuru FID 13.89 / HWD 1.87 vs
VATr++ FID 34.00 / HWD 2.38. An independent comparison ("Quo Vadis HTG for HTR?",
ICCV Workshops 2025) found Emuru best on generation quality, and specifically best
when very little real target data exists — which is exactly the 1-2 page regime.

## Four findings that contradict decisions in the project brief

1. **The style encoder was never ImageNet-pretrained.** VATr's ResNet-18 is pretrained on
   ~100M synthetic font-rendered word images ("Font Square"), and its unseen-writer
   robustness is credited to that. ImageNet pretraining is a materially weaker starting point.
2. **Word-level generation plus an external layout engine is now a self-imposed handicap.**
   Emuru and Eruku generate variable-length lines natively; a 2025 IJCV model does full
   paragraphs. Line assembly, word spacing and baseline wobble are partly solved inside
   the model. Page-level concerns (margins, paper texture, line wrap) remain ours.
3. **96-128px is off-benchmark.** Published work is 32px (VATr/VATr++) or 64px
   (Emuru, One-DM). Emuru's VAE latent is tuned for 64px; 128px means retraining it.
   Alternative: generate at 64px and super-resolve.
4. **Zero-shot from synthetic-only training has displaced training on IAM writers.**
   Emuru never trains on IAM and still wins on IAM. This weakens the rationale for the
   writer-disjoint split (T5) as a *training* concern — it remains necessary for honest
   *evaluation*.

## Recommended base to build on

`https://github.com/aimagelab/Emuru-autoregressive-text-img` — MIT, ships both training and
inference, dataset streams from HuggingFace, checkpoint loads via
`AutoModel.from_pretrained("blowing-up-groundhogs/emuru", trust_remote_code=True)`.
Reported tested on PyTorch 2.7.1 / CUDA 12.8 on a single 4090, so an L4 is plausible.

Fallback if we want the glyph-archetype GAN specifically: `https://github.com/EDM-Research/VATr-pp`
(PyTorch 1.13 / Python 3.9 era, low activity).
Do **not** build on the Eruku repo yet — 6 commits, no README.

Retraining Emuru from scratch (T5-Large, 2.2M images) is out of reach at ~9 h/week on an L4.
The viable path is: start from the released checkpoint, fine-tune, treat VATr++ as a baseline.

## Not verified

Eruku's actual metric values and training resolution; DiffusionPen's conditioning mechanism
and resolution; Emuru's word-level IAM numbers; whether Emuru/Eruku fine-tune cleanly at
96-128px; recency of last commits for One-DM, DiffusionPen, VATr-pp; ScriptViT and
HandwritingAgent preprints (unvetted).

## Sources

VATr arXiv:2303.15269 · VATr++ arXiv:2402.10798 (TPAMI 2024) ·
Emuru arXiv:2503.17074 (CVPR 2025) · Eruku arXiv:2510.23240 (WACV 2026) ·
Quo Vadis arXiv:2508.09936 · One-DM ECCV 2024 · DiffusionPen arXiv:2409.06065
