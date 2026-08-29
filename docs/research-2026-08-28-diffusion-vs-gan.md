# Diffusion vs GAN for few-shot handwriting synthesis (2026-08-28)

## Verified landscape

| Work | Venue | Family | Style conditioning | Few-shot to unseen writers |
|---|---|---|---|---|
| VATr | CVPR 2023 | GAN | 15 style images | yes |
| WordStylist | ICDAR 2023 | latent diffusion | **writer class index** | **no — disqualified** |
| One-DM | ECCV 2024 | pixel diffusion | 1 style image | yes |
| DiffusionPen | ECCV 2024 | latent diffusion (SD1.5 VAE) | 5 style images | yes |
| DiffBrush | ICCV 2025 | diffusion **+ 2 discriminators**, line-level | style line image | yes |
| Paragraph-LDM | IJCV 2025 | latent diffusion, paragraph-level | style image | yes, zero-shot |
| Emuru | CVPR 2025 | autoregressive VAE + transformer | style image | yes, zero-shot |
| Eruku | WACV 2026 | autoregressive | style image | yes, zero-shot |

## The numbers that actually decide

**Training cost.** No paper reports wall-clock. What is reported: VATr trained entirely on a
**single RTX 2080 Ti** — weaker than our L4. DiffusionPen on a **single A100**. One-DM on
4x 3090 plus a second fine-tune stage. DiffBrush on 8x 4090 for ~4 days. Emuru pretrained on
20M synthetic images (not reproducible here; fine-tune from checkpoint only).

**Inference latency** — the only published comparison, Mayr et al. IJCV 2025 on an A40:

| System | Time per paragraph |
|---|---|
| HiGAN+ | 0.13 s |
| VATr | **0.28 s** |
| Paragraph-LDM | 9.06 s |
| WordStylist | **13.44 min** |

One-DM and DiffusionPen use 50 DDIM steps per word; no per-page latency is published for them.
This matters because the project deliverable includes an interactive demo.

**Quality.** Not comparable across papers — protocols differ. Consistent signal: diffusion beats
GAN on FID/HWD by roughly a factor of 2, not an order of magnitude. Within Emuru's protocol on
IAM words: DiffusionPen 15.54 FID / 1.78 HWD, One-DM 27.54 / 2.28, VATr 30.26 / 2.19.
**VATr has the best delta-CER (0.00) in that same table** — it preserves text accuracy best.

**Stability.** Diffusion removes the min-max game; the loss is MSE and monotone. But the honest
read is that the diffusion systems which actually beat VATr are **not adversarial-free**:
One-DM adds a CTC fine-tune stage, DiffBrush adds two discriminators. The stability win applies
to the base model, not to the content-accuracy machinery.

## Integration risk

DiffusionPen's *released inference script is closed-set* — it expects a training-set writer ID
and a JSON lookup of that writer's crops. A 2026 University at Buffalo graduate tech report
(**not peer-reviewed**) documents patching it to open-set successfully, and also documents three
real bugs in the released code. Treat as an integration risk list, not as a published result.

## Not verified

Wall-clock training time or GPU-hours for any diffusion method; per-page latency for One-DM,
DiffusionPen, DiffBrush, Emuru, Eruku; **any result at 96-128px** — every surveyed method runs at
32-64px, so our resolution target is off-distribution for every released checkpoint; whether
DiffusionPen fine-tunes on 24 GB (published batch 320 on an A100 — needs gradient accumulation);
L4-vs-4090/A100 throughput ratios for these workloads.
