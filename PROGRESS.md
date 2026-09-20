# Progress

Live task state. Updated at the end of every task. A fresh session reads this to resume.

**Phase 1 — data and evaluation infrastructure. No generative model yet.**

## Next action

> **Where things stand, 2026-09-17.** The system runs end to end on Amri's own page:
> 22 of 22 lines of new text, reading about as well as his hand, and to his eye "the
> base looks not bad at all" but it is not yet his hand -- small parts vanish. On
> CVL, Emuru carries 65% of a writer's identity (70% keeping the draw closest to the
> hand). Closed: per-writer LoRA fine-tuning (7d no change, 7f -17.9 points), two
> style lines joined, cropping the output tight (-4.7 points). Fixed: the style cut
> (T34), writing beyond the text (T35). The vanishing detail is not the VAE, not a
> fade and not slice seams; most likely the model's regression averaging small
> marks away.
>
> **DiffBrush, stage 1 (T36), 2026-09-17: it runs, on CPU, from our style lines.**
> Amri chose DiffBrush over Eruku -- Eruku shares Emuru's autoregressive regression
> and so probably its vanishing marks -- with stop points: (1) wrapper and CPU check,
> (2) zero-shot on 150 CVL lines, (3) per-writer fine-tune on 7d's 24 writers, which
> must beat Emuru's 53.6% there, paired, (4) his page. Cloned at `da9addc`
> (2025-11-24) into `third_party/DiffBrush/`, gitignored; checkpoint 1.17 GB in
> `checkpoints/diffbrush/`; Stable Diffusion 1.5's VAE from Hugging Face. Scratch
> run, no repo code yet: the UNet loads with 0 missing and 0 unexpected keys, 163M
> parameters, **42 s a line on CPU** for 50 DDIM steps (Emuru: 220 s). The released
> code calls `.cuda()` while building training-only proxies, bypassed on CPU; it also
> fetches ImageNet ResNet-18 weights (45 MB) at construction, then overwritten by the
> checkpoint. Every character in CVL's 9,142 lines and in both passages is in its
> charset; 28.5% of CVL lines are wider than its fixed 1,024px canvas. It does not
> need the style line's transcription -- Emuru does.
>
> Three lines, `outputs/diffbrush_smoke_cpu.png`: real cursive handwriting with
> colons, semicolons and i-dots present; words garbled in places ("shitics w ithat",
> "ods old", "abbut theo", "11:30" -> "1:30"). From CVL writer 0368 the hand follows
> the style's slant and cursive. From Amri's line 12, the two draws differ a lot: one
> a thin generic cursive unlike his, one a bold upright print much nearer. Three
> samples say it works, not how well.
>
> **Stages 1-3 built, 2026-09-17, while Amri was away** (he approved building the
> new model and touching project files only, not Emuru's code):
>
> - **T36, the adapter** `nib.models.diffbrush.DiffBrushGenerator`, behind the same
>   interface as Emuru: loads in 13 s, 38 s a line on CPU. `--generator diffbrush`
>   in `evaluate_generator.py` and `probe_writer.py`; Emuru stays the default and
>   writes where it did.
> - **T37, the fine-tune** `nib.models.diffbrush_finetune` and
>   `scripts/evaluate_finetune_diffbrush.py`. Run tiny on CPU end to end on the real
>   checkpoint (2 writers, 2 steps): 0.39M trainable of 163M, 5.5 s a training step
>   at batch 2, generation before and after, reset, Emuru's 7d images scored as a
>   third condition, HWD and CER for all three. The first attempt died on DiffBrush's
>   own gradient checkpointing, which asks for gradients on frozen weights; the
>   transformer blocks now run without it while an adapter trains.
> - **Notebook:** 7h sets DiffBrush up (clone at `da9addc`, checkpoint kept on
>   Drive), 7i is stage 2 on 150 CVL lines, 7j is 7g with DiffBrush, 7k is stage 3.
>
> ### Both ran, both negative -- and the pattern is now unmistakable, 2026-09-20
>
> | same 150 requests, one reference, 73 writers | identity | FID | CER gap |
> |---|---|---|---|
> | 7c, the system as it stands | **64.6% [58.7, 70.0]** | 55.9 | +1.8 |
> | 7m, style line chosen by letters | 61.8% [56.3, 67.0] | 61.0 | +1.3 |
> | 7o, after the general adaptation | **47.0% [42.0, 51.7]** | 88.5 | +3.3 |
>
> **T38, by letters: no gain.** Paired by writer against 7c's own 150 lines,
> **-2.8 points [-7.4, +1.7]** -- inseparable from zero, tending slightly the wrong
> way. Likely because a request carries only four style lines, so their letter
> coverage barely differs and the choice mostly threw away the width preference
> that was measured to matter. Worth one more look only with a larger pool.
>
> **T39, the general adaptation: a clear loss.** Identity fell 18 points, FID rose
> from 56 to 89, HWD distance to 2.76 -- 87% of the way from real handwriting to no
> style at all. Training loss had fallen 15% over the 2,000 steps, which is the
> whole lesson: **it got better at reconstructing CVL lines and worse at imitating a
> hand it had never seen.**
>
> **Four attempts at training this model, and one thing common to all of them.**
> 7d (a writer's own lines, no prefix) moved nothing; 7f (another writer's line as
> prefix) cost 17.9 points; 7o (4,000 pooled real lines, no prefix) cost 18; and
> DiffBrush's per-writer runs did nothing or broke the text. In every one of them
> the task the model practised was *reconstruct this line*, never *imitate the hand
> in front of you*. Emuru's identity comes entirely from the second, and we have
> been training the first.
>
> **The one variant never tried:** the prefix is **another line by the same writer**,
> and the loss counts only the writer's second line -- exactly what the system does
> at generation time, and exactly 7f with "another writer" replaced by "the same
> writer". `prepare_pair` and `masked_mse` already exist from 7f. About 12 minutes to
> train and 35 to measure. **Awaiting Amri:** run it, or stop model work and build
> the page engine, the interface and the blind test on 7c/7e as they stand.
>
> ### Two experiments built, both aimed at the copying itself -- 2026-09-18
>
> Amri's objection, taken: the plan was always to train broadly on many hands and
> fine-tune per user at the end, and only the second half was ever tried. The order
> was mine and it was wrong -- a general adaptation by LoRA costs about 12 minutes of
> GPU, against the 88 minutes the three per-writer runs took. Both of these attack
> the same thing, the model's ability to copy *this* writer from one line of theirs:
>
> - **T38, choose the style line by its letters** (`--style-by letters`, cell 7m).
>   The model is handed one line per draw, so a page of twenty reaches it a line at
>   a time, and until now that line was chosen by width alone. Now, among lines of a
>   workable width, the one showing most of the characters the target needs comes
>   first. No training. 4 tests; the default is unchanged, so earlier runs stay
>   comparable.
> - **T39, teach Emuru real handwriting once** (`scripts/adapt_emuru.py`, cells 7n
>   and 7o). Emuru was pre-trained on millions of lines rendered from **fonts** and
>   has never seen a pen: its 65% is what imitation is worth on a domain it never
>   saw. One LoRA adapter is trained on 4,000 real lines by the 216 training-split
>   writers -- never the 94 measured on -- and ships with the system; nothing is
>   trained at enrolment. `nib.models.adapters` saves and reloads it (19 MB), and
>   `evaluate_generator.py --adapter` / `probe_writer.py --adapter` put it back on.
>   5 tests. Smoke on CPU with the real checkpoint: trains, saves, and reloads 576
>   tensors onto a fresh model.
>
> Both are measured against **7c's first 150 lines, identity 64.6%**, line for line
> against one reference, and both must beat it clearly. 495 tests pass.
>
> ### DiffBrush is closed -- 2026-09-18
>
> With the drift fixed, both ends of the dose range were tried:
>
> - **Gentle (60 steps at 2e-5 on `attn1`), 24 writers, measured:** identity moves
>   -0.2 points [-2.2, +2.1]. A tight zero.
> - **Strong (150 and 300 steps at 1e-4 on every attention), one writer, on CPU,
>   by eye:** the text falls apart -- "fontume", "bammoad qparetl swhgling" -- and the
>   hand does not approach the writer. The mean difference from the released drawing
>   stops at 19.8 grey levels and stays there when the dose is doubled (19.6): it
>   moves without converging on anything.
>
> And without any fine-tune DiffBrush carries **40.6%** of a writer's identity
> against Emuru's 53.7% on those writers, and 65.0% for Emuru with quality control
> on the wider 7c run. So it is closed on both counts: as a replacement for Emuru
> and as a model that can be taught one person's hand. What the round bought: the
> batch-norm drift, which would have poisoned any future fine-tune of any model here.
>
> **What the three models say together.** A model generalises to a new hand about as
> well as its training data spans hands: Emuru, trained on millions of synthetic
> fonts, holds 65-70%; DiffBrush, trained on IAM alone, 40%; the paragraph LDM's own
> paper reports the same collapse off IAM. That points at training on more real
> handwriting rather than at another architecture.
>
> **Proposed next, awaiting Amri:** one general (not per-writer) LoRA adaptation of
> **Emuru** on CVL's 216 training writers -- real pen on paper, where it has only
> ever seen fonts -- then the standard harness on the held-out writers. The test
> writers stay out, so the measurement stays honest. Cost, from measured rates: about
> 12 minutes for 2,000 steps plus about 35 minutes to score 150 lines. Criterion:
> beat 65.0% clearly. The alternative Amri holds: stop looking for a better model and
> build the product on Emuru as it is.
>
> ### The clean stage 3 run: the gentle dose does exactly nothing -- 2026-09-18
>
> Cell 7l again with the drift fixed, 16 minutes, same 24 writers:
>
> | 96 targets, 24 writers | identity | HWD distance | CER |
> |---|---|---|---|
> | DiffBrush released | 40.6% [33.5, 47.3] | 1.97 | 12.3% |
> | DiffBrush fine-tuned, 60 steps at 2e-5 on `attn1` | 40.5% [33.1, 47.5] | 1.98 | 14.2% |
> | Emuru released | **53.7% [46.6, 59.7]** | 2.22 | 12.7% |
>
> **fine-tuned minus released: -0.2 points [-2.2, +2.1]** -- a tight zero, not an
> unclear result: the measurement would have seen two points and there are none.
> Emuru reproduced 53.7% again and the released figure landed where it had, so the
> harness is sound. The gentle recipe is simply too small to change the drawing --
> measured directly on CPU as a mean difference of one grey level.
>
> **Where that leaves it.** Both ends of the dose range have now been tried, and
> neither has been measured cleanly *and* had an effect: the strong end (7k) ran
> under the drift, the gentle end is valid and inert. Whether a dose in between
> moves the hand without breaking the text is still open, and is being previewed on
> CPU before any more GPU is spent.
>
> ### Both stage 3 runs were measuring a model that drifted -- found 2026-09-18
>
> **DiffBrush's style encoder is a ResNet-18 with 40 batch norms, and training moved
> every one of them.** Their running statistics are buffers, not parameters, so
> `reset_lora` -- which does return the adapter to exactly zero, checked -- could not
> undo them. Each writer inherited the drift of all the writers before, and after 24
> writers the style encoder had been averaged into a generic hand. That is precisely
> the symptom: in 7l every writer came out in the same clean, anonymous script.
>
> How it was found, all on CPU: a released draw, 60 training steps, a reset, another
> draw. The two draws should have been identical and were not -- 27 grey levels apart,
> and different widths. The adapter's weights were zero, so something else had moved;
> 40 of 40 batch norms had, after two steps.
>
> **The fix: train the adapter with the model in eval mode.** Gradients still flow --
> eval changes behaviour, not autograd -- while batch norms and dropout stay where
> generation will find them. Verified on the real checkpoint: 0 of 40 batch norms
> move, released and after-reset draws are **identical to the pixel** (mean difference
> 0.000), and the fine-tuned draw differs from released by a mean of 1.0 grey level,
> so the adapter does change the drawing. Two tests now fail without the fix: one
> pinning the batch norms, one demanding that a writer leave the model drawing exactly
> as it did before.
>
> **So 7k's and 7l's figures below say nothing about DiffBrush.** Both need re-running.
> What survives from them: the released-model numbers (identity 45.4%, CER ~13%) and
> Emuru's 53.7%, which are drawn before any training.
>
> Also confirmed while looking: the training objective itself is sound. The released
> model predicts the noise with an MSE of 0.34 at t=50 down to 0.004 at t=900, where
> 1.0 would mean the latent scale or the schedule was wrong.
>
> ### Stage 3 failed its criterion -- T37's run, 2026-09-18
>
> Cell 7k on a T4, 36 minutes: 24 writers, 16 train lines, 4 targets each, 300 LoRA
> steps at rank 8 and lr 1e-4, 52 s a writer. Saved to Drive
> (`results/finetune_diffbrush_w24_t16_s300_r8`).
>
> | 96 targets, 24 writers, one reference | identity | HWD distance | own writer nearest | CER |
> |---|---|---|---|---|
> | DiffBrush released | 45.4% [36.9, 53.3] | 2.30 | 50.0% | 12.1% |
> | DiffBrush fine-tuned | **36.5% [28.9, 44.4]** | **1.83** | 20.8% | **34.6%** |
> | Emuru released | **53.7% [46.6, 59.7]** | 2.22 | 83.3% | 12.7% |
> | real | | 0.77 | 100.0% | 11.0% |
>
> - fine-tuned minus released: **-8.9 [-17.5, +1.9]**, not separable from zero.
> - fine-tuned minus Emuru released: **-17.2 [-24.9, -9.0]**, real, and the wrong way.
> - **The harness checks out:** Emuru's 53.7% reproduces 7d exactly, so the reference
>   and the pairing are sound.
>
> **The criterion, set before the run, failed.** But the way it failed is not "the
> hand did not move": HWD distance fell from 2.30 to 1.83, the largest move of any
> condition -- the output became *more* like real handwriting -- while identity fell
> and CER nearly tripled, 12.1% to 34.6%. That is the signature of a recipe that
> overwhelms the text: 300 steps at lr 1e-4 over 16 lines is 37 passes through each
> line, and the adapter sits on the attention that carries the glyphs as well as on
> the rest. A second suspect: **67 of 384 training lines (17%) were squeezed** to the
> 1,024px canvas, which narrows letters, and writers differ a lot in how many
> (0 to 13).
>
> **Not yet looked at:** the 96 adapted images, which would say whether it wrote the
> wrong text or the right text badly.
>
> **Product decision, Amri, 2026-09-18: enrolment always comes with the text.**
> A new user either copies a passage the system gives them, or uploads a page of
> their own *and types its transcription line by line*. Either way the text of every
> enrolment line is known, so no recogniser stands in the enrolment path and its
> 11% error rate is not inherited. This is what makes a per-writer fine-tune possible
> at all: training needs a text for every line. DiffBrush needs no transcription to
> *write*, only to be trained; Emuru needs one either way.
>
> **The plan that follows, in order:** (1) 7k says whether a per-writer fine-tune
> buys the hand; (2) if it does, further training on more writers -- CVL's 216
> training-split writers are here, IAM's site was down and DiffBrush has it already,
> font-rendered text is cheap but is not a hand -- to move the starting point for
> every new user; (3) with a better starting point, the per-writer fine-tune needs
> fewer lines, which is what "one paragraph" would take. The GPU cost of (2) is not
> measured; it is a continuation, not the authors' 8x4090 four days.
>
> **A licence note for the product, not for the experiment.** DiffBrush's code is
> MIT, but its released weights were trained on IAM, whose licence is non-commercial
> research. Fine for this project; to be checked before anything ships, where a base
> trained on synthetic fonts -- as Emuru's was -- is the cleaner ground.
>
> **Stage 2, early and without a GPU: zero-shot DiffBrush carries less of the hand
> than Emuru.** The whole harness was run with DiffBrush on CPU -- 60 lines, four
> style lines each, two draws (`outputs/eval_diffbrush_lines_refs4_cand2`), 62.9
> minutes -- and then both models' images for those same 60 requests were scored
> against one reference locally:
>
> | same 60 lines, 43 writers | identity | HWD distance | own writer nearest |
> |---|---|---|---|
> | DiffBrush, 2 draws | 49.5% [42.5, 57.1] | **2.05** | 23.3% |
> | Emuru 7c, 4 draws | **61.7% [54.0, 69.4]** | 2.41 | 55.8% |
> | real | | 1.19 | 95.3% |
> | **difference, paired by writer** | **-12.3 [-21.5, -3.2]** | | |
>
> Not like for like -- two draws against four -- but the direction is clear and it is
> what the survey predicted for an IAM-trained model off IAM. Read together with the
> distance, DiffBrush writes handwriting that looks *more* like handwriting and
> *less* like this writer: realistic, and somebody else's. So stage 3 is the whole
> question, as planned. Also from that run: CER 16.9% against 9.7% real; 12 of 60
> lines reach the canvas edge, but the text is usually complete -- DiffBrush spreads
> a line to fill 1,024px -- while long texts make it repeat words ("notion notion",
> "Alas Als"), which T35's rule caught in 11 of 73 draws.
>
> **Next, Amri, on a T4:** cells 1-4, then **7h**, then **7k** -- the decisive one,
> with its criterion written in the cell: `fine-tuned minus Emuru released` wholly
> above zero, and Emuru's own figure close to 7d's 53.7% or the comparison is void.
> Then 7j (his page, zero-shot, by eye) and 7i (the stage 2 number) as time allows.
> No T4 timing exists for DiffBrush yet; the first progress line of each gives it.
>
> **Amri chose alternative models**, and to leave the page engine
> until a model writes lines properly. Survey: `docs/research-2026-09-17-alternative-models.md`.

> **Phase 1 is complete (13/13). Phase 2 has begun and the model generates.**
>
> On 2026-08-31 the project produced handwriting for the first time: held-out CVL
> writer 0057, one real line as style, three lines that writer never wrote, in a
> visibly matching hand. See `docs/phase2-first-generation.md` and
> `outputs/probe_lines/lines_contrast.png`.
>
> ### The references, measured on lines and on English only
>
> T15 did not need Colab -- `check_metrics.py` generates nothing, it only runs
> Inception, the embedding and TrOCR over real images. Measured on CPU and
> committed to `references/references_cvl_lines_64.json`:
>
> | | on **lines**, English only | on words (phase 1) |
> |---|---|---|
> | FID floor | **19.15** | 33.72 |
> | writer top-1 | **85.8%** | 66.9% |
> | writer top-5 | **94.4%** | 90.1% |
> | CER on real lines | **11.45%** (300 lines) | 12.33% (40 lines, one writer, unfiltered) |
>
> **The FID floor nearly halved against the word-level one.** A line holds a
> whole sentence, so lines vary less from one another in Inception's feature
> space than isolated words do. A generated set scoring 60 would have read as
> 1.8x the floor against words and is in fact 3.1x against lines. The word-level
> number would have flattered every result this project is about to produce.
>
> **Writer retrieval is much higher than it looked.** More handwriting per image
> means more evidence of the hand, so the bar the generator has to clear rose
> from 66.9% to 85.8%.
>
> **Excluding the German passage took 2.28 points off CER**, from 13.73% to
> 11.45%. That difference is the size of the penalty we were imposing on TrOCR
> for reading a language it was never trained on. 11.45% is its honest error rate
> on English handwriting.
>
> The two retrieval figures come from different samples (126 writers against 116)
> and are not directly comparable with the German-inclusive run. CER is.
>
> **CER cannot be measured on Colab.** It reads the *raw* CVL line images rather
> than the pack, and the 5 GB of sources are deliberately not copied to the VM.
> That is why `references.update` merges rather than replaces: a machine that
> measures two of the three must not delete the third.
>
> ### T16 ran. The project has numbers.
>
> 2026-09-09, T4, 53 minutes, 300 lines in the hands of 94 unseen writers.
> Full write-up in `docs/phase2-first-evaluation.md`.
>
> | | generated | real | no style at all |
> |---|---|---|---|
> | FID | **63.92** | 19.15 | 254.29 |
> | writer top-1 | **20.5%** | 85.8% | 1.7% |
> | writer top-5 | **43.6%** | 94.4% | — |
> | CER | **33.14%** | 12.06% | — |
>
> **It produces handwriting** — 81% of the way from printed text to real, on the
> FID scale. **The identity carries only partly** — retrieval is 18x chance, so
> the style input is genuinely used, but only 23% of the way from chance to what
> real handwriting scores. Emuru writes convincing handwriting that is only
> partly the right person's.
>
> The run completed where the previous one died: at request 41 the model declined
> to write, three re-draws failed, and it was excluded with its ground truth
> rather than taking the run down. Four others were saved by a retry.
>
> ### The first trustworthy style figure -- Emuru, T23's run
>
> `notebooks/colab_eval.ipynb` cells 1 to 6 on a T4: 300 requests, 295 kept, 53.9
> minutes of generation (0.09 lines/s, the rate of 2026-09-09).
>
> | | generated | real | typeface |
> |---|---|---|---|
> | **HWD** | **2.00 [1.88, 2.13]** | 0.86 [0.81, 0.91] | 2.99 [2.91, 3.06] |
>
> **Emuru's output sits 53% of the way from real handwriting to a typeface**, and
> its interval is nowhere near the real one. Reference: 1,104 real lines over 93
> writers, none a target or a style line.
>
> The rest: FID 66.72 [60.56, 72.88] · writer top-1 20.3% [15.9, 25.1] · CER 28.1%
> [23.9, 32.4] against 11.3% real, gap +16.8 · 13 of 295 truncated (4.4%) · 5
> excluded as empty, 1 saved by a retry. The earlier runs' FID (63.92, 69.46) and
> retrieval (20.5%, 22.1%) all fall inside this run's intervals. Truncation halved
> against the 8.7% of 2026-09-10 on the same requests; not explained.
>
> The harness check before it (cell 5, fake generator, 120 samples) printed
> generated = typeface = 3.02 at 100% and real 1.10 -- HWD's first run on a GPU.
>
> **What 53% does not say.** The typeface is the far end for *no handwriting at
> all*, not for *someone else's handwriting*. HWD 2.00 mixes two distances -- not
> looking like real handwriting, and not looking like this writer -- and the
> typeface anchor cannot separate them. Retrieval had a chance level for exactly
> this; HWD does not have one yet.
>
> ### Emuru carries more than half of a writer's identity -- T24's run, 2026-09-14
>
> Cell 6 again, same 300 requests, now with the identity block. 296 kept, 56.5
> minutes of generation, saved to Drive (`results/eval_emuru_lines`, 296 images).
>
> | | gap to the other 92 writers | own writer nearest |
> |---|---|---|
> | generated | 1.05 [0.94, 1.15] | 47.3% [37.6, 57.0] |
> | real | 1.85 [1.77, 1.94] | 97.8% [94.6, 100.0] |
> | typeface | -0.01 [-0.08, 0.07] | 1.1% [0.0, 3.2] -- chance 1.1% |
>
> **Identity: the generated lines carry 56.5% [50.9, 61.9] of what real lines do.**
> The typeface sits at zero as it must, so shared text does not create the gap.
> Distance repeated the first run: HWD 2.03 [1.90, 2.16] against 0.86 real and
> 2.99 typeface, 55% of the way. FID 67.70 [61.81, 73.59], retrieval 20.9%
> [16.6, 25.7], CER 30.4% against 11.3% (gap +19.1), 20 of 296 truncated (6.8%),
> 4 excluded.
>
> **This changes the reading of the project's central number.** Retrieval put
> Emuru a quarter of the way from chance to real; identity puts it past half.
> The gap is what T21 predicted -- retrieval collapses under blur and HWD does
> not. Nearest-writer (47.3%, per writer over their 1-7 lines) and retrieval
> (20.9%, per single line) are not the same question and are not compared.
>
> Two cautions. "Identity" is whatever sets one CVL writer's lines apart from
> another's, and that includes their pen -- which a real user's photographs would
> share too, but which is not the hand alone. And each run builds its own
> reference, so 7b's figure is compared through its own real anchor, not raw.
>
> ### Two style lines side by side make Emuru worse -- T27, 2026-09-14
>
> Cell 7b: the same 300 requests with `--style-refs 2`, the two lines joined into
> one wide image and their texts joined with a space (`nib.models.style.join_style`).
> 295 kept, 86.4 minutes of generation against 56.5, saved to Drive
> (`results/eval_emuru_lines_refs2`).
>
> | | one line (cell 6) | two lines (7b) | |
> |---|---|---|---|
> | HWD identity | 56.5% [50.9, 61.9] | **42.3% [36.2, 48.7]** | separate |
> | own writer nearest | 47.3% [37.6, 57.0] | 25.0% [16.3, 33.7] | separate |
> | HWD distance | 2.03 [1.90, 2.16] | 2.22 [2.07, 2.36] | overlap |
> | FID | 67.70 [61.81, 73.59] | 90.64 [80.97, 100.30] | separate |
> | CER | 30.4% [25.7, 35.2] | **59.6% [54.6, 64.4]** | separate |
> | writer top-1 | 20.9% [16.6, 25.7] | 18.6% [14.2, 23.1] | overlap |
> | truncated / excluded | 6.8% / 4 | 8.1% / 5 | |
>
> The criterion was fixed before the run -- clear 61.9% -- and it fell the other
> way: worse on identity, on FID, and CER nearly doubled. Real lines scored the
> same in both runs (0.86, identity gap 1.85 against 1.81), so the ruler did not
> move; the output did.
>
> **Closed for Emuru: more evidence as a wider image.** Not closed: more evidence.
> The collapse in CER says the model is not writing the target text, which points
> at how two lines reach it rather than at the idea. Unverified hypothesis: Emuru
> puts the style text and the target text through T5 together, and with two
> lines' worth of prefix it loses where the style ends -- writing style words, or
> skipping target words. The saved samples decide it without a GPU.
>
> ### The system on Amri's own page -- T33's run, 2026-09-16
>
> Cell 7g on a T4: `passage_page1.jpg` split into 22 of 22 lines, lines 2, 6, 18-22
> set aside, learned from lines 1, 4, 5, 8, 9, 11, 12, 14, 15, 17, and lines 3, 7, 10,
> 13, 16 written again without being shown. `--keep hand`. Saved to Drive,
> `results/probe_passage_page1`.
>
> | | |
> |---|---|
> | targets generated | 5 of 5 |
> | page 2, a text never written | **22 of 22 lines**, none missing |
> | CER on the 5 targets, TrOCR-base | real **8.7%** [6.7, 10.7] · generated **10.6%** [7.1, 14.1] · gap +1.9 |
> | selection | 108 draws for 27 requests (4.00 each) · 0 kept as the best of an unreadable set |
> | the hand changed the pick | 24 of 27 (89%) -- 100 of 150 (67%) on CVL in 7e |
>
> **It works end to end on a real user.** Every line of page 2 came out readable,
> and the generated targets read about as well as his own lines -- a gap of 1.9
> points, where CVL gave 1.8 under the same selection. Five lines, about 200
> characters: the intervals overlap and are wide. CER is partly flattered, as in
> 7c: TrOCR-small rejected the unreadable draws, TrOCR-base measured.
>
> **What these figures cannot say is whether it is his hand.** The run has no
> identity figure -- HWD identity needs many writers -- and 89% is not evidence
> that the embedding knows him: it was trained on CVL, and a rule picking at random
> among the readable draws would move the pick up to three times in four. Whether
> it looks like him is decided by eye on `comparison.png` and `written.png` (the
> CPU dry run did not carry his "y"), and then by the blind pairs.
>
> **Amri's verdict, having looked: "not bad, but still far from my handwriting --
> a lot of it is slightly erased at the edges."** The images, in
> `outputs/probe_passage_page1/`, show what that is:
>
> - **Carried:** his "y" and "g" as a z with a tail (windy, maybe, Thursday,
>   though), the crossed "I", the open "a", mostly unjoined letters. The dry run's
>   missing "y" is there this time.
> - **Fine detail lost:** dots on "i" ("ın", "ıs"), the hook on "r" (from -> "flom",
>   car -> "cal", bread, promise, whether; his own "r" is sometimes hookless --
>   "strange", "warm" on the real line 3 -- but not always), "#" dropped twice,
>   digits drawn small, a broken "B".
> - **Smaller letters, same stroke.** Measured on the five target pairs, same 64px
>   band: the main body of the letters is 21px against 30 (smaller on all five),
>   ink per character 74 px against 103, stroke width 2.49 against 2.63 (the
>   same). Slant 24° against 29°, mixed line by line -- not a finding. Unverified
>   link: detail that small falls below what 64px can hold.
> - **Line edges.** About 7 of 27 lines start with a stray mark (". quiz",
>   ". Lior", "t Kept", ") or"), and the marks match how style lines end ("9:45.",
>   "14.50.", "flat", "fox,") -- unverified: the last glyph of the style line
>   leaking past the cut. 2 of 27 end in junk after the text ("the t t t", "cafe:
>   te Te"): selection accepts CER <= 50%, so extra letters pass. Text errors
>   besides: "runnining", "in in", "17/8" dropped, "You" -> "Yf".
>
> Not worth a blind test yet: the author himself sees the difference.
>
> **The smaller letters are framing -- and cropping them tight makes identity worse.**
> On CVL too: real lines are cropped to the ink (0 white rows above and below,
> median), Emuru's lines carry 7 above and 7 below, an ink band of 48 of 64 rows. So
> in the same 64px band the letter body reads 17px against 22-23 (cell 6 and 7c,
> smaller on 83% and 89% of lines), ink per character 60 against 93. Tested without
> a GPU on 7c's saved 300 lines, locally -- where HWD reproduced the Colab figures
> exactly (identity 65.0% [60.3, 69.6], distance 1.939, nearest 58.8%) -- by cropping
> each generated line to its ink and rescaling it to 64px:
>
> | 286 lines, one reference | identity | HWD distance | own writer nearest |
> |---|---|---|---|
> | as generated | 65.0% [60.3, 69.6] | 1.939 | 58.8% |
> | cropped tight to the ink | 60.3% [55.5, 65.0] | 2.128 | 52.9% |
> | **difference, paired by writer** | **-4.7 [-7.2, -2.3]** | | |
>
> So the metric was not understating Emuru through framing, and a tight crop does
> not belong in the output on identity grounds. Unverified reading: Emuru's strokes
> are already as thick as the writer's at the smaller size (2.49 against 2.63 on
> Amri's page), so scaling the line up by a third thickens the pen along with the
> letters. Whether showing Emuru its style lines framed the way it writes changes
> anything is a separate question, which only a GPU run answers.
>
> Local HWD on Windows needs `num_workers=0`: the package's DataLoader starts a
> worker, the worker re-runs the calling script, and the first attempt hung for 13
> minutes at zero CPU. 164 s on CPU for 950 reference lines and three sets of 286.
>
> **Line starts: a cut in the wrong place, found and fixed (T34), not yet run.**
> Emuru's VAE encodes `floor(width / 8)` slices -- measured with the released
> `emuru_vae` on widths 800-809 -- and `generate` cuts its output at the style
> image's full width. On any width not divisible by 8 the last `width % 8` pixels
> of the style line never reach the model, and the cut lands up to 7 px inside the
> new line. On cell 6's saved 296 lines, rebuilt from the seed, outputs opening on
> a sliver (a first ink cluster of 12 px or less, before a gap, when the first word
> has 3+ letters) rose with `width % 8`: 4.9% at 0 (n 41), 5.8% at 1-3, 8.6% at 4-5,
> 11.3% at 6-7 (n 71). By eye the flagged ones are both kinds: the tail of the
> style line ("n species") and a clipped first letter ("( isdaining", "howd"). 23
> events -- a trend, not proven. On Amri's page the four style lines drawn from
> were 11, 12, 4, 1; line 11 ("14.50.") is 847 px wide, `% 8` = 7 against a 5 px
> margin, so its final full stop is shaved -- and three outputs open on a full
> stop. `EmuruGenerator._as_tensor` now widens every style line with white to a
> whole slice, as `finetune.prepare_line` already did for training. The check is
> the next 7g run: stray starts should fall, empty outputs must not rise.
>
> **7g again with T34 and T35, 2026-09-16.** 5 of 5 targets, 22 of 22 lines of page
> 2, nothing lost. CER on the targets: real 8.7%, generated 13.5% [7.5, 19.8]
> against 10.6% [7.1, 14.1] before -- five lines, overlapping. 108 draws; 1 set aside
> for writing beyond its text; 0 kept from an unreadable set; the hand moved the
> pick in 24 of 27 as before. The earlier images are kept in
> `outputs/probe_passage_page1_before_t34/`.
>
> By eye, against the earlier run (different draws, so indicative only): **junk
> after the text gone** (2 of 27 -> 0); **clear stray marks at line starts 7 -> 2**
> (". I spent", ". Kept" -- style line 11 ends in a full stop, so the model still
> writes the end of the style text first, not only because of the cut). Amri: "the
> base itself looks not bad at all", but many parts are still not written or
> vanish -- dots, pieces of letters, gaps inside letters. On this page: "11:30" ->
> faint marks, "5:45" -> "5 5", "#378 at" -> "3 T", "#1" -> "1", broken first
> letters in "Chloe", "Friday", "Shopping".
>
> **Why parts vanish -- three explanations ruled out, no GPU:**
>
> - **Not the VAE.** Amri's real lines encoded and decoded by `emuru_vae`, posterior
>   mean and sampled alike, come back with every dot, "#", quote and i-dot; 86-94%
>   of ink pixels stay dark, the rest are edges.
> - **Not a global fade.** The share of stroke pixels that are grey is 39% in his
>   real lines, 39% in both generated runs; CVL 39% real, 43% generated.
> - **Not seams between 8px slices.** Column-to-column change in ink, by x mod 8,
>   sits at 0.94-1.05 of its mean for generated lines, 0.97-1.01 for real.
>
> What is left is the prediction itself. Emuru's T5 is trained by mean squared
> error on the latents (per its `forward`), and a regression predicts the average
> of what could come next: where a small mark's position or presence is uncertain
> -- a dot, a colon, a crossbar, a digit's short stroke -- the average is faint or
> nothing. That fits what vanishes, and would be a property of the model rather
> than of our pipeline. **Inference, not measured.**
>
> **Line ends: writing beyond the text, calibrated and fixed (T35), not yet run.**
> Read locally with TrOCR-base, "about the t.t. 1/" scores 27.8% CER and "cafe :
> he" 15.4%, far under the 50% that counts as readable. TrOCR also puts a space
> before every punctuation mark ("dream ." -- target 13 is read word for word and
> still scores 4.9%, all of it those spaces; real lines pay it too), so a length
> check would reject good lines. Instead `candidates.overrun` aligns the whole
> target inside the reading, spaces removed, and counts the characters outside it
> at the larger end. Calibrated with TrOCR-small, the selector (downloaded with
> Amri's permission), on readable lines:
>
> | readable lines rejected | at 2+ | **at 3+** |
> |---|---|---|
> | real CVL, 290 | 8 (2.8%) | **2 (0.7%)** -- misread ends, "Framework" for "Zemanek" |
> | real Amri, 22 | 0 | **0** |
> | generated Amri, 27 | 3 | **2** -- exactly the two with junk after the text |
> | cell 7c as kept, 293 | 21 | **13 (4.4%)** -- mostly a repeated last word: "they they", "in ins", "on on on" |
>
> A draw now counts as readable only at CER <= 50% **and** at most 2 characters
> beyond the text; the ones set aside for it are counted in the selection log.
> In hand mode every draw is made anyway, so it costs nothing; in readable mode an
> occasional extra draw. The stray starts of one character (". Grandpa") are
> below the threshold on purpose -- T34 is the fix aimed at those.
>
> ### Keeping the draw closest to the hand -- T30's run, 2026-09-15
>
> Cell 7e: the first 150 of 7c's requests, all four draws made, the readable one
> nearest the style lines by the writer embedding kept. 600 draws, 105.6 minutes
> (10.6 s a draw); the hand changed the pick in 100 of 150 requests.
>
> | | 7c, first readable (300) | 7e, closest hand (150) |
> |---|---|---|
> | HWD identity | 65.0% [60.3, 69.6] | **69.8% [64.5, 75.2]** |
> | own writer nearest | 58.8% | 63.0% [52.1, 74.0] |
> | CER / real | 12.5% / 10.7% | 12.0% / 10.8% |
> | FID | 55.87 | 54.11 [50.02, 58.20] |
> | truncated | 0.3% | 2.0% |
>
> **Line for line, against one reference: close, not proven.** Both runs' first 150
> requests are identical by key. Scored locally against 7e's reference (861 lines,
> 73 writers, 147 lines kept):
>
> | same 150 lines | 7c first readable | 7e closest hand | difference, paired |
> |---|---|---|---|
> | HWD identity | 64.6% [58.7, 70.0] | 69.8% [64.5, 75.2] | **+5.2 [-0.5, 10.1]** by writer |
> | HWD distance | 2.210 | 2.004 | |
> | own writer nearest | 56.2% | 63.0% | chance 1.4% |
> | CER (TrOCR-base) | 9.9% | 12.0% | +2.0 [-0.3, 4.4] by line |
>
> Identity rose for 66% of writers and the interval's lower end is -0.5: very likely
> a real gain of a few points, but it does not clear zero at 150 lines. It also
> costs a little legibility, again not clearly. So: kept as an option for when
> likeness matters more than time, not made the default; 7c's rule stays default.
>
> The first reading of it, before the paired test: up by 4.8 points on identity, but **not yet a finding**: different sample counts,
> a different reference (73 writers against 85), and overlapping intervals. The
> sharp test is line for line over the same 150 requests -- 7c's first 150 are in
> `outputs/` already; 7e's need downloading. Writer retrieval rose to 35.3%, and
> means nothing here: the embedding that chose is the one that scores it.
>
> Cost, for the product: four draws always, about 42 s a line on a T4, against
> about 13 s a line for keeping the first readable draw.
>
> ### Withholding the hand makes the fine-tune worse -- T31's run, 2026-09-16
>
> Cell 7f: the same 24 writers, 16 train lines and 4 targets as 7d, each training
> line placed after a training-split writer's line, loss on the writer's line only,
> teacher noise 0.5. 96 image pairs, saved to Drive
> (`results/finetune_w24_t16_s150_r8_other_n0.5`). Ran on the code before T34/T35.
>
> | | released | fine-tuned | |
> |---|---|---|---|
> | training | -- | 75 s a writer (max 79), against 56 in 7d | |
> | train loss, first tenth -> last | -- | 0.70 -> 0.47 (-34%) | |
> | HWD identity | 53.6% [47.1, 59.5] | **35.7% [30.5, 41.4]** | |
> | **difference, paired by writer** | | **-17.9 [-24.7, -11.1]** | **real, and negative** |
> | own writer nearest | 79.2% | 54.2% | chance 4.2% |
> | HWD distance | 2.22 | **3.26** | real 0.77 |
> | CER | 12.5% [8.6, 17.7] | 18.5% [12.8, 24.9] | real 11.0% |
>
> **The criterion, set before the run -- a difference above zero, CER not clearly
> worse -- failed the other way.** The released side reproduced 7d (53.7% and
> 12.7% there) on fresh draws, so the harness is steady and the drop is the
> adapter's. Distance rose to 3.26, beyond the 2.99 a typeface scored against other
> references: the output moved away from handwriting in general, not only from
> this writer.
>
> **A likely reason, unverified.** Emuru's identity comes from copying the hand of
> the line before it. Training with another writer's line in that place teaches
> the model the opposite -- that the line before says nothing about the hand to
> write -- and 150 steps on 16 lines cannot store the hand in its place. Noise 0.5
> is confounded with it; separating them costs two more 70-minute runs.
>
> **Per-writer LoRA fine-tuning is closed at these settings**: no gain with the
> hand in context (7d), a clear loss with it withheld (7f).
>
> ### Fine-tuning, as built, does not help -- T29's run, 2026-09-15
>
> Cell 7d: 24 held-out writers, 16 train lines, 4 targets each, generated with
> Emuru as released and after 150 LoRA steps at rank 8, both through quality
> control. 96 image pairs, saved to Drive (`results/finetune_w24_t16_s150_r8`).
>
> | | released | fine-tuned | |
> |---|---|---|---|
> | **training time** | -- | **56 s a writer on a T4** (max 59) | |
> | train loss, first tenth -> last | -- | 0.41 -> 0.26 on average | |
> | HWD identity | 53.7% [46.6, 59.7] | 53.8% [47.8, 59.0] | |
> | **difference, paired by writer** | | **+0.1 [-6.8, 6.7]** | none |
> | own writer nearest | 83.3% | 79.2% | chance 4.2% |
> | HWD distance | 2.22 | 2.36 | worse |
> | CER | 12.7% [9.3, 17.1] | 19.7% [15.7, 23.9] | worse, overlapping |
>
> **Cost answered, benefit not there.** A minute a writer is a reasonable time per
> user. But identity did not move at all, and the text got harder to read. The
> fine-tune learned its training lines -- loss down 37% -- and none of it carried
> to new lines as identity.
>
> **A likely reason, unverified.** Training is teacher-forced: every slice is
> predicted from the text and *the writer's own preceding slices*. The hand is
> always in the context, so nothing pushes it into the adapter's weights -- the
> model only has to continue in the style it can see, which it already does
> zero-shot. What it can pick up instead is these particular lines, which fits
> CER rising. If so, style has to be *withheld* from the context during training
> for the weights to carry it: a prefix from another writer, or none.
>
> Not compared with 7c's 65%: 24 writers here against 85, and a style pool of 16
> lines against 4. The comparison that counts is the paired one within this run.
>
> ### Quality control works -- T28's run, 2026-09-14
>
> Cell 7c: 300 requests, a pool of four style lines each, up to four draws, the
> first one TrOCR-small reads at CER <= 50% kept. 406 draws for 300 requests
> (1.35 each), 234 accepted first time, 7 kept as the best of an unreadable set.
> 63.2 minutes of generation against 56.5 for cell 6. Saved to Drive,
> `results/eval_emuru_lines_refs4_cand4`. The fake check before it: 61 draws for
> 60 requests, generated = typeface, identity -0.5%.
>
> | | cell 6, one draw | 7c, best of 4 | |
> |---|---|---|---|
> | CER | 30.4% [25.7, 35.2] | **12.5% [10.5, 14.7]** | separate |
> | CER gap to real | +19.1 | **+1.8** | |
> | FID | 67.70 [61.81, 73.59] | **55.87 [52.60, 59.13]** | separate |
> | HWD identity | 56.5% [50.9, 61.9] | **65.0% [60.3, 69.6]** | overlap, barely |
> | own writer nearest | 47.3% [37.6, 57.0] | 58.8% [48.2, 69.4] | overlap |
> | HWD distance | 2.03 [1.90, 2.16] | 1.94 [1.84, 2.05] | overlap |
> | writer top-1 | 20.9% | 19.3% | overlap |
> | excluded / truncated | 4 / 6.8% | **0 / 0.3%** | |
>
> **The output now reads nearly as well as real handwriting**: a CER gap of 1.8
> points, where it was 19.1. CER is partly flattered by selection -- a different
> recogniser chose, but both are TrOCR -- so the clean evidence is **FID, which
> separates**, and which no recogniser touches. Identity rose by 8.5 points to
> 65.0%, where the removal bound had put 60-67%, and **the paired test makes it a finding**: over
> the 84 writers both runs scored, identity 56.7% -> 65.0%, difference **+8.3
> points [2.4, 14.8]** resampled per writer; it rose for 65% of writers. Own
> writer nearest 46.4% -> 58.3%, difference +11.9 [-1.2, 23.8], not separate.
>
> Not like for like in two ways, both stated in the run: `--style-refs 4` draws
> different targets from the same seed, and consuming four style lines a request
> left 85 writers with enough spare reference lines (14 samples withheld) against
> 93. Real anchors agree (identity gap 1.86 against 1.85). And two changes are
> confounded -- choosing the style line by width, and re-drawing -- which matters
> for understanding and not for the product, which gets both.
>
> ### What the samples show -- no GPU, 2026-09-14
>
> Both runs downloaded from Drive to `outputs/`. Each run's requests were rebuilt
> from the seed (296/296 and 295/295 matched the saved keys), so every sample's
> style text is known. Scratch script, sheets of samples looked at by eye.
>
> **Emuru's output is bimodal, not uniformly mediocre.** One style line: median
> per-sample CER **9.3%** against a mean of 30.8%. The mean is carried by a tail --
> 10% of lines above 90% CER -- and those lines are not bad handwriting, they are
> no handwriting: two or three words, then a smear of repeated vertical strokes to
> the end of the budget, or a blank. The median line reads well and looks like a
> hand.
>
> **Two lines break the text, not only the tail.** Median CER 54.4%, 21% above
> 90%. Typical two-line outputs repeat words ("has has", "trong trong"), drop
> words, and pull words in from the style text ("plant" from "plants and
> animals"). Worst cases are the same smear as above. CER rises with the combined
> style text: 33% under 60 characters, 79% over 100. The hypothesis that the model
> loses the boundary between style and target text fits what is on the page. Per
> writer, two lines beat one for 31% of the 91 writers in both runs.
>
> **The style line matters even at one.** Style lines under 500px: mean CER 69%,
> failures 22% (n 18). 500-1100px: 26-28%, failures 9-10% (n 242). Over 1100px:
> 37% (n 36). A page offers a choice of style line; the evaluation drew one at
> random.
>
> **Catching broken lines would raise identity too, not only CER.** Recomputed from
> the saved images and reference keys -- which reproduced cell 6 exactly, 56.5% /
> 47.3% / 30.4% -- with unreadable samples removed from generated and real alike:
>
> | kept | samples | writers | identity | own writer nearest | CER |
> |---|---|---|---|---|---|
> | all | 296 | 93 | 56.5% | 47.3% | 30.4% |
> | CER <= 90% | 267 | 91 | 60.0% | 47.3% | 21.1% |
> | CER <= 50% | 224 | 86 | 67.3% | 52.3% | 9.5% |
>
> A bound, not a measurement of a fix: removal is not a re-draw, it can remove
> hard writers along with bad draws (93 writers fall to 86), and these carry no
> intervals. It says the broken tail costs identity as well as legibility.
>
> **Fine-tuning is available.** The Emuru repository
> (`github.com/aimagelab/Emuru-autoregressive-text-img`, MIT) ships training code,
> per `docs/research-2026-08-28-vatr-line.md`; reported on one 4090. Whether it
> adapts to one writer from 22 lines without overfitting is untested.
>
> ### The goal, restated by Amri -- 2026-09-14
>
> **A system that works**: photograph one or two pages of your own handwriting,
> type any text, get it back in your hand. Not a portfolio piece and not a paper.
> New approaches are welcome; the measure is whether it works in the end. GPU is
> not a constraint -- Amri will buy more Colab units when needed.
>
> Two things follow. The product will be given a **page**, 20 to 30 lines, while
> every evaluation so far gave the model **one line** -- so the number of style
> lines is the main experimental axis, not a detail. And the VM that produced HWD
> 2.00 was reclaimed before cell 7 ran: the images are gone and only the printed
> figures above survive.
>
> ### The plan
>
> 1. **Know where we stand.** T24's identity anchor, then one GPU session:
>    notebook cells 1-6 (Emuru, one style line) and 7b (two lines). Every run
>    cell now saves to Drive itself. Pinning the Emuru and Eruku Hugging Face
>    revisions first would keep every experiment on the same model -- the Colab
>    run warned their remote code was downloaded fresh.
> 2. **Amri's own handwriting.** First pipeline component: split a photographed
>    page into lines. Then a probe -- his lines as style, new sentences -- and a
>    small blind test.
> 3. **Fidelity, one experiment at a time, cheapest first**, each closed if the
>    identity figure does not move outside its interval: pick the most
>    representative style line; more style lines (side by side made Emuru worse at two, T27;
>    Eruku takes any width and is untested at two or more); best-of-N against
>    the writer's other lines; per-writer fine-tuning, **which needs Amri to
>    reopen "no per-user training"**; another model only if those fail.
> 4. **Finish the pipeline**: transcribe the style lines or use a dictated
>    passage, generate line by line, lay out a page, output, interface.
>
> Deprioritised: retraining the writer-retrieval embedding. The identity anchor
> and a blind test cover what it was for.
>
> **Awaiting Amri:** (a) per-writer fine-tuning as an experiment, yes or no;
> (b) the definition of "works" -- proposed: people who know his handwriting
> cannot pick his real line out of a real/generated pair much better than 50%.
>
> ### The immediate next task
>
> **Done: generation with quality control (T28)** -- CER gap +19.1 -> +1.8, FID
> 67.70 -> 55.87 (separate), identity 56.5% -> 65.0% (intervals touch). It is the
> generation step from here on.
>
> Paired identity comparison with cell 6: done, +8.3 points [2.4, 14.8].
>
> **In progress, Claude: T29, per-writer fine-tuning.** Emuru's cached config:
> `slices_per_query` 1, `vae_channels` 1, T5 `google-t5/t5-large` (d_model 1024,
> 24+24 layers, 16 heads), tokenizer `google/byt5-small`. `_img_encode` samples
> the VAE posterior, rearranges it one slice per 8px, adds teacher noise, and
> prepends a learned start token. First component `nib.models.finetune`: line
> preparation with white after the text, LoRA attach and reset, a training loop.
> Then `scripts/evaluate_finetune.py`: same writers, same targets, with and without.
>
> **Ran 2026-09-15, negative -- see "Fine-tuning, as built, does not help"
> above.** Amri chose both directions, in this order: **T30**, keep the draw
> closest to the hand (cell 7e, 150 lines, then a line-for-line comparison with
> 7c's first 150 using the images already in `outputs/`); then his pages through
> the pipeline; and **T31**, the fine-tune with the writer's strokes withheld
> (cell 7f), because "we feel stuck". Both are coded and verified locally before
> the GPU runs. What cell 7d ran:
> notebook cell 7d, after
> cells 1-4: 24 held-out writers, 16 train lines, 4 targets, 150 steps at rank 8,
> quality control on both conditions. Read the paired `difference` under HWD
> identity, `training` seconds per writer, and CER for both. Training time on a
> T4 has not been measured -- 5.8 s a step was on CPU -- and the progress line
> after the first writer is the first real figure. 150 steps and lr 2e-4 are
> starting points, not tuned.
>
> The first 7d run died in `attach_lora` before training a step: Colab ships
> torchao 0.10.0, and peft raises on any torchao under 0.16.0 while merely checking
> whether a layer is torchao's. Emuru has no torchao weights, so `attach_lora` now
> answers that check itself when an old torchao is present. Re-run cell 1 to pull,
> then 7d.
>
> **In progress, Amri:** writing out the two passage pages.
> Use the page by *selection*, not concatenation. For each line to write: pick
> style lines from the page in the length range that works (500-1100px), draw
> several candidates, read each with TrOCR against the intended text, reject the
> broken ones, keep the best. Selecting by readability first, because it is
> independent of HWD; selecting by closeness to the writer comes second and must
> use page lines disjoint from the ones identity is measured against, or it
> measures itself. Needed in the product whatever else is done, fine-tuning
> included. Costs candidates x generation time: 11 s a line measured, so four
> candidates is about 45 s a line before any batching.
>
> **Next: per-writer fine-tuning, approved by Amri 2026-09-14** "if it fits a
> reasonable time per user"; `CLAUDE.md` updated to match. What reading Emuru's
> own code established, none of it run yet:
>
> - `forward(img, input_ids, attention_mask)` is teacher-forced MSE on VAE latents.
>   The VAE is frozen in `__init__`; T5 and two linear projections train.
>   `train_T5.py` optimises all parameters with AdamW, lr 1e-4, batch 2. No LoRA.
> - `generate()` joins `style_text + ' ' + gen_text` and continues the style
>   image. Training images are single lines, so a two-line prefix is out of what
>   it learned -- consistent with T27.
> - So a writer's training data is simply their lines with transcriptions, and
>   generation afterwards uses one of those lines as the prefix.
> - Memory, arithmetic only: 2.88 GB of fp32 weights is about 0.7B parameters;
>   full AdamW needs about 16 bytes a parameter, 11.5 GB before activations,
>   against a T4's 15 GB. LoRA on T5's attention keeps optimiser state small and
>   the per-writer result a few megabytes.
>
> - Read further (`modeling_emuru.py` on Hugging Face, `custom_datasets/load_hf_dataset.py`
>   in the repository): T5 is a standard `T5ForConditionalGeneration` at
>   `model.T5`, so PEFT's LoRA can target its attention projections. One training
>   sample is **one line image and its full text**, padded to a fixed **768px**
>   width -- no style-plus-target pairs.
> - So Emuru learned on canvases of at most 768px and is used here on about
>   1,740px (style 870 + target 870, medians). Checked against cell 6 whether
>   that length drives the broken tail: it does not, cleanly. Failures above 90%
>   CER by total canvas: 13.3% under 1,300px, 11.1%, 5.8% at 1,600-1,900, 13.7%,
>   5.6% over 2,400 -- a U, not a rise. And two lines are bad at the same length:
>   at 1,600-1,900px one line gives 25.0% mean CER, two give 50.5%. The tail looks
>   like sampling, which a re-draw addresses; two lines fail for another reason.
>
> Planned as T29, on CVL first so it does not wait for Amri's pages: for held-out
> writers, fine-tune on part of their lines, generate the rest, and compare
> identity with and without, same writers, same targets. Measure minutes per
> writer. Then Amri's page 1 against his page 2.
>
> Closed: joining style lines side by side for Emuru. Open but lower: Eruku with
> several lines (untested, 128 min a run).
>
> `analyse_run.py` does not read HWD yet; cell 9 needs that before it can compare
> two runs on style. Cells that should not be run sit below a "kept for
> reference" divider, commented out.
>
> **In parallel, Amri:** copy out `configs/passages/english_page1.txt` and
> `english_page2.txt` by hand, one sheet each, keeping every line break exactly, in
> pen on plain paper with margins, without crossing out. Photograph each from above
> in daylight and copy the original files -- not through WhatsApp, which
> recompresses -- to `data/raw/personal/` as `passage_page1` and `passage_page2`.
>
> **Then, Claude:** split those photographs into lines (22 per page expected), pair
> each line with its passage text, and run the first probe on Amri's own hand:
> page 1 as style, page 2's lines generated and set beside his real page 2.
>
> ### HWD's floor was not a floor -- 2026-09-11, T23
>
> HWD is the distance between each writer's *mean* feature on either side, and a
> mean over few lines is noisy -- noise that adds to the distance even when both
> sides are the same person's real hand. Real lines against other real lines by
> the same 90 held-out writers, k lines per writer per side:
>
> | k | 1 | 2 | 3 | 6 | 10 |
> |---|---|---|---|---|---|
> | HWD, real vs real | 1.760 | 1.255 | 1.005 | 0.704 | 0.539 |
>
> Under the exact sampling of a 300-sample run (93 writers, 1 to 7 lines each),
> real against real read **1.055 / 1.063 / 1.107** over three seeds. The 0.641
> quoted as the floor came from an uncommitted script at about seven or eight
> lines per writer, judging by the curve. Cell 6 as it stood would have reported
> even a perfect generator as nearly a fifth of the way to no hand at all.
>
> Content barely registers: a typeface scored 3.080 against the target's own text
> and 3.075 against a different one.
>
> **What changed.** Every run now scores three aligned sets against one shared
> reference -- up to 12 real lines per writer that are neither a target nor a
> style line: the generated lines, the real target lines (same writers, texts and
> counts), and the same texts in a typeface. Generated and real differ in nothing
> but being generated. HWD carries a 95% interval by resampling writers, its
> per-writer terms go into `analysis.npz`, and the per-writer computation matches
> the package's own `HWDScore` to 2e-8 on real lines. An HWD failure no longer
> takes CER or the saved bundle with it.
>
> Also found on the way. Cell 2 imported `nib` inside the Colab kernel, where
> `/content/nib` -- the clone -- reads as an empty namespace package, and failed
> with `No module named 'nib.engine'`; it now checks in a fresh interpreter, which
> is where the scripts run. And `hwd` has never been importable locally: its
> `editdistance` dependency has no Windows wheel for Python 3.13, which is why
> 0.641 was measured through a workaround nobody saved.
>
> ### The task before it, for reference
>
> **Re-run T16 with the corrected token budget.** 10.7% of the output was
> truncated, and a line cut short loses its ending to deletion errors, so part of
> the CER gap is ours rather than the model's. `TOKENS_PER_CHAR` moved from 4.0
> to 5.5 (above the 99th percentile of 5.04, measured over 1,524 lines) and
> `MAX_TOKENS` from 256 to 384, since at 5.5 a cap of 256 would clamp every line
> past 47 characters. The share of real lines that do not fit their budget falls
> from 5.9% to 0.7%.
>
> Nothing else changes: same pack, same references, same notebook. Re-run the
> install cell to pull the fix, skip cell 5, run cell 6.
>
> ### Setting a run up from scratch, for reference
>
> 1. Upload `data/processed/upload/cvl_lines_64.lmdb` (**127 MB, 9,142 records**)
>    to `MyDrive/nib/`. **Never** the copy in `data/processed/` -- LMDB reserves
>    its map size up front, so that one is 8 GB on the wire. The two packs this
>    project has built share a filename and differ by 1,720 records, so check the
>    size: 148 MB is the older one, which still contained the German passage.
> 2. Open `notebooks/colab_eval.ipynb` and run it top to bottom.
>
> It also needs `MyDrive/nib/checkpoints/writer_embedder.pt`, which should
> already be there from the T11 run.
>
> `evaluate_generator.py` refuses to start when the pack and the references
> disagree on record count, so uploading the wrong one costs seconds rather than
> an hour. Skip cell 5 -- the raw images it needs are not on the VM.
>
> Watch the truncation count in the output. It is the check on whether the new
> token budget is right; if it is high, `TOKENS_PER_CHAR` in
> `src/nib/models/emuru.py` is too low.
>
> Phase 2 runs T13 -> T16. Only T16 is left, and only it needs a GPU.
>
> ### Why the plan grew from two steps to four (2026-08-31)
>
> Measuring the line data before writing the pack turned up three things. All
> three were invisible while the working unit was a word, and all three become
> load-bearing the moment it is a line.
>
> **1,157 lines carry a transcription that is missing a word.** CVL dropped word
> crops whose segmentation failed, but the line image still holds that word's
> ink. Pairing the two charges the recogniser a deletion error for reading
> correctly. Confirmed by counting ink blobs: gap lines carry 0.58 more per line
> than complete ones, relative to their own word count. They are dropped and
> counted. 9,142 clean lines remain after the German passage comes out too.
>
> **The token budget was set for words and never revisited.** Emuru's VAE
> compresses width by 8, so one token is 8px (`lengths / 8` in its
> `modeling_emuru.py`). `EmuruGenerator` passes `max_new_tokens=96` = 768px. A
> real CVL line at 64px averages **886px**, p90 1198px. *It could not finish an
> average line.* The two 756px outputs recorded as "runaway generation" were
> 94-95 tokens: they hit our cap. The model's own default is 256. T16 replaces
> the fixed cap with a per-request budget from the text length, and counts every
> truncation.
>
> **The three reference numbers were measured on words.** FID 33.72 and
> retrieval 66.9% both came from `cvl_words_64.lmdb`; CER 12.33% came from 40
> unfiltered lines of a single writer. Scoring generated *lines* against a
> word-level FID floor compares different things. T15 re-measures all three on
> the line pack -- one run of `check_metrics.py --pack ...`, no new code.
>
> A fourth item from `PROGRESS.md` turned out to need no work: it asked for
> `normalise_ink` on the line path, but `normalise_word` already called it
> internally. Raw lines sit at 2nd-percentile brightness ~130; after
> normalisation ~21. The faint probe output came from loading raw images.
>
> ### Product idea, Amri, 2026-09-10: enrol from a dictated passage
>
> Instead of asking a new user to photograph a page they already have, hand them
> a printed passage and ask them to copy it. The transcription is then known
> exactly, because we chose it -- no recogniser in the loop and no error
> inherited from one.
>
> This is how CVL itself was built, which is why 99,904 of its words carry a
> transcription nobody typed.
>
> Two things it buys beyond the obvious. The passage can be **designed**: one
> that contains every character in the charset guarantees the system has seen
> how this person forms a `Q` and a semicolon, rather than hoping they turned
> up. And it stays worth doing even with Eruku, whose `style_text` is optional
> but documented as helping -- a dictated passage gives the best possible one for
> free rather than merely avoiding the worst.
>
> The cost is that it is a different product. "Copy this passage" works for
> enrolment and does not work on a page that already exists -- an old letter, a
> diary, a grandparent's hand. Both may be wanted; they are not the same feature.
>
> ### The style metric was broken, and is now replaced -- 2026-09-11
>
> **Read this before trusting any retrieval figure in this file.** A *real* line,
> by unquestionably the right writer, blurred by 0.8 pixels:
>
> | | ours (retrieval) | HWD |
> |---|---|---|
> | other real lines | 96.8% | 0.641 |
> | the same lines, blurred 0.8px | **12.2%** | 0.721 |
> | the same lines, resampled 1/2 | 10.1% | 0.636 |
> | the same text in a typeface | -- | 2.931 |
>
> Blur does not change whose handwriting something is, and every generative
> decoder produces exactly that softness. So writer retrieval has been reporting
> sharpness as much as style, and **Emuru's 22.1% is better than a real line by
> the correct writer put through a mild blur.** The comparisons of 2026-09-10 --
> Emuru against Eruku, the cfg sweep -- are all confounded by output sharpness.
>
> `nib.engine.metrics.hwd` replaces it as the primary style measure. HWD is a
> VGG16 trained on 100M rendered lines, it moves 12% under the damage that costs
> ours 87%, it separates handwriting from a typeface by 4.5x, and it is what
> Emuru's and Eruku's own papers report -- so our figures become comparable to
> published ones for the first time. Optional extra: `pip install -e ".[hwd]"`.
>
> Retrieval stays, because a percentage is far more legible than a distance and
> two agreeing measures beat one -- but it needs retraining with blur, resampling
> and noise in its augmentation before its number means what it says. About 20
> minutes on a T4, per the T11 notes.
>
> **Next run:** `--generator emuru --samples 300` with the hwd extra installed.
> That is the project's first honest style number.
>
> **The HWD column above is relative, not absolute.** Its figures came from a
> script that was never committed, at a lines-per-writer nobody recorded. Within
> that one protocol the reading holds: blur moved HWD 12% and a typeface sat 4.5x
> away. As a floor for a run, 0.641 does not; see T23 above.
>
> ### Where the project stood before that, 2026-09-10
>
> | | Emuru | Eruku cfg 1.25 | Eruku cfg 1.0 | real |
> |---|---|---|---|---|
> | FID | **69.5** | 80.7 [73.2, 88.3] | -- | 19.15 |
> | writer top-1 | **22.1%** | 5.3% [3.0, 8.0] | 10.0% [3.3, 18.3] | 85.8% |
> | CER gap | +20.4 | **+13.0** | -- | -- |
> | empty outputs | 6 | **0** | **0** | -- |
>
> Eruku writes the text better and the hand worse. Its learned end-of-generation
> token does fix one failure completely -- Emuru needed four retries and lost two
> requests outright, Eruku lost none -- and does nothing for truncation, which
> sits at 8% for both.
>
> `cfg_scale` is the dial: dropping it from 1.25 to 1.0 doubled retrieval. The
> intervals at 60 samples overlap, so this is a direction and not yet a finding;
> cfg 2.0 completes the line.
>
> **The number that has not moved is the project's whole claim.** Writer
> retrieval is 22.1% at best against a ceiling of 85.8% -- a quarter of the way
> from chance to what real handwriting scores. Everything else is now measured
> carefully; this is not solved.
>
> The largest untried lever on it: **more than one line of style**. The model is
> given a single line and asked to learn a hand. Emuru's interface takes one
> image, but a wider image holding several concatenated lines is still one image,
> and Eruku takes any width.
>
> ### Known and open
>
> - `find_page` returns the whole frame for `data/raw/personal/dim.jpeg`, so the
>   black cloth above the sheet reaches line segmentation as six false lines.
> - Line segmentation loses lightly written strokes: the first "The" on Amri's
>   sample has 63 pixels below the ink threshold. Hysteresis recovers it and brings
>   the grid back with it (16 lines on the WhatsApp copy), so it is built and off.
> - `data/raw/personal/page1_transcription.txt` is split by sentence, not by written
>   line, and differs from the page ("wanderd", "Project"). The dictated passage
>   replaces it.
> - Emuru needs the style sample's *transcription*. A user photographing a page
>   has transcribed nothing, so the product must read it first -- TrOCR is already
>   here for that, at TrOCR's accuracy. This constrains the architecture.
> - Emuru also ships `generate_batch`, which `EmuruGenerator` does not use. It
>   takes per-sample `lengths` and would cut Colab generation time. Not needed
>   for correctness; worth doing if T16 is slow.
> - **Run evaluation on Colab, not locally.** 220s per line on CPU; minutes on a T4.
> - `data/processed/cvl_words_64.lmdb` shows 8 GB but holds 469 MB; it is sparse
>   and locked by a stale handle. The compacted copy for uploading is at
>   `data/processed/upload/cvl_words_64.lmdb`. A reboot clears the lock.
> - transformers is pinned to 4.x. 5.x cannot load Emuru and breaks TrOCR's
>   tokenizer -- the same version gap causes both.

## Status

| ID | Task | Status | Verified by |
|----|------|--------|-------------|
| T0 | Repo skeleton, packaging, lint, git init | **done** | ruff clean · ruff format clean · pytest 2 passed |
| T1 | Config system (typed schema, single path root) | **done** | ruff clean · 16 tests passing |
| T2 | charset — char/index mapping | **done** | ruff clean · 31 tests passing |
| T3 | Synthetic IAM fixture generator | **done** | ruff clean · 49 tests · 20 writers / 1000 words in 5.7s |
| T4 | IAM parser | **done** | ruff clean · 72 tests (5 skipped, awaiting real IAM) |
| T5 | Writer-disjoint split | **done** | ruff clean · 111 tests · real CVL: 216/94 writers, 70.0/30.0% samples, 0 overlap |
| T6 | Image normalisation (domain gap) | **done** | ruff clean · 200 tests · paper/ink/contrast spread 0 across conditions; ink% 38.0 -> 14.2 |
| T7 | Pack to a single LMDB file | **done** | ruff clean · batched writes · compaction 8.00 GB -> 24.9 MB |
| T8 | Dataset + collate | **done** | ruff clean · 19 tests · same-writer / different-word verified via style_keys |
| T9 | Checkpoint save/resume | **done** | ruff clean · 132 tests · resume is bit-identical to an uninterrupted run |
| T10 | Metrics: FID, CER, writer retrieval | **done** | FID 0.0000 self-check, floor 33.72 · CER 12.33% on real lines · retrieval **66.9% top-1** on 94 unseen writers |
| T11 | Experiment tracking + visual sample log | **done** | ruff clean · 157 tests · entity omri334jb configured |
| T12 | Colab end-to-end smoke run | **done** | T4, 2026-08-30: bit-identical resume, 6.1s Drive copy, 614 samples/s |

**Phase 2 — generation and its first real numbers.**

| ID | Task | Status | Verified by |
|----|------|--------|-------------|
| T13 | CVL line reader, with counted drops | **done** | ruff clean · 9,142 of 13,473 lines kept, total_seen matches the disk exactly |
| T14 | Line pack -> `cvl_lines_64.lmdb` | **done** | 9,142 lines, 309 writers, 127 MB compacted · `check_data.py` all green · rebuilt 2026-09-02 without the German passage |
| T15 | Re-measure FID / retrieval / CER on lines | **done** | CPU, 2026-09-09: FID floor 19.15 · writer 85.8% top-1, 94.4% top-5 · CER 11.45% over 300 lines · FID(real, same real) 0.0000 |
| T16 | Per-request token budget, then evaluate the generator | **done** | T4, 2026-09-09, two runs of 300. Best: FID 69.46 · writer 22.1% top-1 · CER gap +20.4% · 8.7% truncated |
| T17 | Confidence intervals, full-sample CER, saved analysis | **done** | 364 tests · every figure now carries a 95% spread · `analysis.npz` reconstructs a run's metrics exactly |
| T18 | Eruku adapter, and Eruku measured | **done** | T4, 2026-09-10, 128 min over 300: FID 80.74 [73.2, 88.3] · writer 5.3% [3.0, 8.0] · CER gap +13.0% · **zero empty outputs** |
| T19 | Sweep Eruku's guidance scale | **done, negative** | cfg 1.0 / 1.25 / 2.0 -> 10.0% / 5.3% / 6.7%, all intervals overlapping. cfg 2.0 clearly worse (15% truncated). Even cfg 1.0's upper bound is below Emuru's 22.1% |
| T20 | Several style lines as one reference | **done, thresholded** | 1 and 2 lines generate normally (0.89x, 0.85x of real width); 4 breaks Emuru (0.25x). The prefix outgrows what its stopping heuristic tolerates |
| T21 | Calibrate the style metric | **done** | A real line blurred 0.8px scores 12.2% against 96.8% untouched. The metric measures sharpness |
| T22 | HWD as the style metric | **done, floor superseded by T23** | 0.641 real / 0.721 blurred / 2.931 typeface, measured here at an unrecorded lines-per-writer |
| T23 | HWD's floor and ceiling measured inside every run | **done** | T4: Emuru HWD **2.00 [1.88, 2.13]** against real 0.86 and typeface 2.99 -- 53% of the way to no hand · fake check generated = typeface = 3.02 on GPU · real vs real 1.76 -> 0.54 from 1 to 10 lines per writer · per-writer HWD matches `HWDScore` to 2e-8 · 384 passed |
| T24 | HWD identity anchor: own writer against every other | **done** | T4: Emuru identity **56.5% [50.9, 61.9]** of real, own writer nearest 47.3% against chance 1.1% · fake on GPU -0.5% [-5.5, 4.4] · typeface gap -0.01 · run cells save to Drive themselves; cell 7b adds two style lines · 388 passed |
| T25 | Split a photographed page into lines | **done, one photo open** | 13 lines on 4 of 5 photos of Amri's page (angle, normal, shadow, WhatsApp) · `dim` fails upstream, `find_page` returns the whole frame -- strict xfail · 11 tests + 1 xfail |
| T26 | Dictated passage, two pages | **done, awaiting Amri's handwriting** | each page holds all 79 charset characters, every lowercase letter at least 3 times, lines of 35-44 characters · 7 tests |
| T27 | Emuru with two style lines, measured | **done, negative** | T4: identity 42.3% [36.2, 48.7] against 56.5% [50.9, 61.9] for one line · FID 90.64 against 67.70 · CER 59.6% against 30.4% · all three separate |
| T28 | Generation with quality control | **done** | T4, 300 lines: CER 12.5% [10.5, 14.7] against 10.7% real (gap +1.8, was +19.1) · FID 55.87 [52.60, 59.13], was 67.70 -- separate · identity 65.0% [60.3, 69.6], was 56.5%; paired per writer +8.3 points [2.4, 14.8] · 1.35 draws a line, 63 min · 0 excluded · 13 tests |
| T37 | DiffBrush per-writer fine-tune (stage 3) | **done, negative: closed** | after the drift fix, 24 writers: identity 40.5% fine-tuned against 40.6% released, difference -0.2 [-2.2, 2.1] -- a tight zero · strong doses (150, 300 steps at 1e-4) break the text without approaching the hand · the bug found on the way: the backbone trained in train mode, so 40 batch norms in the style encoder drifted and no reset could undo them; fixed by training in eval mode, verified on the real checkpoint (0 of 40 move, released and after-reset draws identical to the pixel) · 19 tests | the backbone trained in train mode, so 40 batch norms in the style encoder moved every step and no reset could undo them -- each writer inherited the last one's drift · fixed by training in eval mode; verified on the real checkpoint: 0 of 40 move, released and after-reset draws identical to the pixel · what the two runs still say: released 45.4%, Emuru 53.7% · 19 tests | T4, 36 min, 24 writers, 52 s each: identity 45.4% released -> 36.5% fine-tuned, against Emuru 53.7% (which reproduced 7d exactly) · paired difference to Emuru -17.2 [-24.9, -9.0] · CER 12.1% -> 34.6% while HWD distance fell 2.30 -> 1.83 · 67 of 384 train lines squeezed · looks like too strong a recipe rather than a model that cannot learn a hand | CPU, real checkpoint, 2 writers, 2 steps: 0.39M trainable of 163M, 5.5 s a step at batch 2, three conditions scored end to end · DiffBrush's gradient checkpointing turned off in its transformer blocks while training (it raised on frozen weights) · 13 tests, 481 passed | `nib.models.diffbrush_finetune`: noise-prediction loss on the writer's line with another of their lines as style, LoRA on the UNet's `to_q/to_k/to_v/to_out.0`, lines wider than 1024 squeezed and counted · `scripts/evaluate_finetune_diffbrush.py`: 7d's writers and targets, released vs fine-tuned, and `--compare-with` 7d's run scores Emuru released as a third condition against the same reference · 13 tests |
| T39 | Teach Emuru real handwriting once, for everyone | **done, negative** | T4: 2,000 steps on 4,000 training-split lines in 12.5 min, training loss 0.42 -> 0.36 · identity 47.0% [42.0, 51.7] against 64.6% for the same 150 requests untrained · FID 56 -> 89, CER gap +1.8 -> +3.3 · better at reconstructing CVL lines, worse at imitating an unseen hand · build: `scripts/adapt_emuru.py`: one LoRA adapter on 4,000 lines by the 216 training-split writers, 2,000 steps at 1e-4 · `nib.models.adapters` saves 19 MB and reloads 576 tensors onto a fresh model · `--adapter` on both scripts, run directories marked · 5 tests |
| T38 | Choose the style line by its letters | **done, no effect** | T4, 150 lines: identity 61.8% against 64.6%, paired by writer -2.8 [-7.4, +1.7] · four style lines a request leave little to choose between · build: among style lines of a workable width, the one showing most of the target's characters comes first · no training · default unchanged · 4 tests |
| T36 | DiffBrush behind the Generator interface | **done; DiffBrush closed -- 40.6% against Emuru's 53.7% on the same writers** | `nib.models.diffbrush.DiffBrushGenerator`: loads in 13 s, 38 s a line on CPU, `check_output` passes, `$` refused by name · `--generator diffbrush` in `evaluate_generator.py` and `probe_writer.py`, Emuru the default and unchanged · `paths.third_party` · 11 tests, 468 passed · stage 1 scratch: checkpoint loads exactly, 163M params, charset covers all of CVL and the passages |
| T35 | A draw that writes beyond its text is not readable | **code done, GPU check pending** | `overrun`: target aligned inside the reading, spaces removed, characters outside at the larger end · rejected at 3+ · TrOCR-small calibration: real CVL 2 of 290 (0.7%), real Amri 0 of 22, generated Amri 2 of 27 (both junk), 7c 13 of 293 · counted as `rejected_for_overrun` · 8 new tests (26) |
| T34 | Style line widened to whole VAE slices | **code done, GPU check pending** | the released VAE encodes floor(w/8) slices on widths 800-809, so up to 7 px of style went unseen and the cut landed inside the new line · cell 6: sliver starts 4.9% at w%8=0 -> 11.3% at 6-7 · `_as_tensor` pads with white · 4 new tests · 457 passed with T35 · check: next 7g, stray starts down, empties not up |
| T33 | The system on Amri's own page | **run done: "not bad, still far from my hand"** | T4, 2026-09-16: 22 of 22 lines split, page 2 written 22 of 22 · CER on 5 targets real 8.7% [6.7, 10.7], generated 10.6% [7.1, 14.1] · 108 draws for 27 requests, hand moved the pick in 24 · no identity figure for one writer · build: | `scripts/probe_writer.py`: splits a dictated page, refuses a line count that does not match the passage, sets aside `--skip` lines, learns from 10 lines and writes 5 others again for comparison, then writes page 2's text in the hand · `comparison.png`, `written.png`, `blind/` pairs with `key.json` · notebook cell 7g; needs `MyDrive/nib/personal/passage_page1.jpg` |
| T32 | Segmentation of a real passage page | **done** | Amri's `passage_page1.jpg` (pen, lined paper, 2792px): 22 of 22 lines · two fixes found by looking at the crops: never shrink a photo by more than 20% (thin strokes fell below every threshold at 1600px -- "Uri" lost its U, "P.S." its P), and small marks join the nearest letter rather than the nearest line centre, which brings dots, commas and full stops back · the five squared-paper photos unchanged at 13 lines (dim still xfail) |
| T31 | Fine-tune with the writer's strokes withheld | **done, negative** | T4, 24 writers: identity difference paired by writer **-17.9 [-24.7, -11.1]** (53.6% -> 35.7%) · HWD distance 2.22 -> 3.26 · CER 12.5% -> 18.5% · 75 s a writer · per-writer LoRA closed at these settings · build: `--context other --noise 0.5`: each training line placed after a training-split writer's line, loss on the writer's line only (`masked_mse`), teacher noise 0.5 against latent ink std 1.17 (measured) · 5 new tests (17) · notebook cell 7f, the same 24 writers as 7d |
| T30 | Keep the draw closest to the hand | **done, gain not proven** | paired over the same 150 lines, one reference: identity +5.2 points [-0.5, 10.1], CER +2.0 [-0.3, 4.4] · kept as an option, not the default · run: T4, 150 lines: identity **69.8% [64.5, 75.2]** (7c over 300: 65.0% [60.3, 69.6]) · CER 12.0% against 10.8% real, gap +1.1 · FID 54.11 [50.02, 58.20] · the hand changed the pick in 100 of 150 · 600 draws in 105.6 min, 10.6 s a draw · writer retrieval 35.3% is NOT independent here · build: | `--keep hand`: every draw made, unreadable ones set aside, the readable draw nearest the style lines by this project's writer embedding kept; HWD judges · 4 new tests (17) · notebook cell 7e, 150 lines, the same first 150 requests as 7c |
| T29 | Per-writer fine-tuning on CVL | **done, negative at these settings** | T4, 24 writers: 56 s a writer · identity difference paired by writer +0.1 [-6.8, 6.7] · CER 12.7% -> 19.7% · build: | `nib.models.finetune` + `scripts/evaluate_finetune.py` · 11 tests on a tiny T5 shaped like Emuru · smoke on the real checkpoint, CPU: LoRA 4.72M of 719M (0.66%), adapter 19 MB, fresh adapter and reset both give the released loss exactly (0.48608), 5.8 s a training step, generation runs with the adapter attached · the whole experiment run tiny on CPU (2 writers, 1 step) end to end · notebook cell 7d |

## Waiting on Amri

- [ ] IAM download into `data/raw/iam/`. **The FKI site is currently down** -- links do not
      load from either side. Not blocking: `tests/test_iam_real.py` activates by itself the
      moment `data/raw/iam/xml/*.xml` exists, and validates the reconstructed schema then.
      Priority order once reachable: `xml.tgz` (small, answers the schema question),
      then `ascii.tgz`, then `words.tgz` (~1.2GB).
- [x] 5 phone photos received. **They are on squared/graph paper** -- grid lines will be
      read as ink by naive binarisation. Ruled-line removal is now part of T6's scope.
- [x] `cvl-database-1-1.zip` downloaded and extracted. 99,904 cropped word images with
      transcriptions in their filenames; 98,179 usable after filtering.
- [ ] Weights & Biases username (the identifier only — never the API key)
- [x] Commit permission granted (messages record what was done and what changed)

## Open questions

- **Python version.** Local is 3.13.2 with torch 2.13.0+cpu (the CPU-only wheel, 122 MB).
  Colab on 2026-09-11: Python 3.13.15, T4, torch 2.11.0+cu128, and transformers 4.57.6
  after the install -- which therefore completed, `editdistance` included. Whether `hwd`
  imports there is what cell 2 now checks.
- **Architecture decided (2026-08-29): option C.** Start from a released zero-shot
  checkpoint (Emuru line) rather than training a generator from scratch. Amri approved.
  Consequences: no per-writer training; the generator produces variable-length lines, so
  word-to-line assembly leaves the layout engine's scope; IAM becomes evaluation data, not
  training data; generation resolution is 64px, upscaled afterwards. VATr++ stays as a
  documented fallback and comparison baseline. Surveys and their unverified-claims lists are
  in `docs/`.
- **IAM licence is non-commercial research use.** Fine for a portfolio project; a blocker if
  this ever ships as a product. Flagged early on purpose.

## Log

- **2026-09-16 — T31 run: withholding the hand makes the fine-tune clearly worse.**
  Same 24 writers and targets as 7d, another writer's line before each training line,
  teacher noise 0.5: identity fell from 53.6% to 35.7%, a paired difference of -17.9
  points [-24.7, -11.1], and HWD distance rose past where a typeface sits. The released
  side reproduced 7d. Likely, training against a different hand in the prefix unlearns
  the in-context copying that Emuru's identity comes from. With 7d's null result, per-
  writer LoRA fine-tuning is closed at these settings.

- **2026-09-16 — T35: writing beyond the text no longer passes as readable.** Two of
  the 27 lines on Amri's page ran on past their text and passed selection at 50% CER.
  A new measure aligns the text inside the reading and counts what is left over at
  either end, ignoring TrOCR's habit of a space before punctuation. Calibrated with
  TrOCR-small over 290 real CVL lines, Amri's 22, and 620 generated: at three or more
  characters it sets aside 0.7% of real lines and both of Amri's junk lines, and in
  cell 7c's kept output mostly catches a repeated last word.

- **2026-09-16 — T34: the style line reaches Emuru on whole slices.** Following the
  stray marks at the start of Amri's lines, the released VAE turned out to encode
  `floor(width / 8)` slices while Emuru cuts its output at the full style width -- up
  to 7 px of style unseen, and the cut up to 7 px inside the new line. The rate of
  outputs opening on a sliver in cell 6 rose from 4.9% to 11.3% with `width % 8`.
  Style lines are now padded with white to a whole slice, as training lines already
  were. Every earlier run carried the misalignment; runs from here on do not, which
  is worth remembering when comparing across that line.

- **2026-09-16 — T33 run: the whole system on Amri's page, on a GPU.** Cell 7g learned
  from 10 of his lines, wrote 5 others again and all 22 lines of page 2, with nothing
  lost: 108 draws, none of the 27 requests left without a readable one. The generated
  targets read at 10.6% CER against 8.7% for his own lines. Amri, looking: not bad, still
  far from his hand, "erased at the edges". The images show fine detail lost (i dots,
  r hooks, #), letters a third smaller than his in the same band with the same stroke,
  stray marks at 7 of 27 line starts and junk after the text on 2. The Emuru remote
  code downloaded fresh again -- the revision is still not pinned.

- **2026-09-15 — the first line in Amri's own hand.** Amri wrote page 1 of the dictated
  passage in pen on lined paper; page 2 will not be written, so the probe works from
  one page. Segmentation split it into 22 of 22 lines, and looking at the crops found
  what the count hid -- thin strokes and small marks lost -- fixed in T32. A CPU dry
  run of `scripts/probe_writer.py`, one draw and no selection, wrote line 10 again
  without seeing it and a line of page 2's text: readable, recognisably handwriting,
  closer to a tidy generic hand than to his -- his "y" in particular is not carried.
  The run then died computing an interval over a single target, which a real run with
  five does not do; fixed so a one-target check reports the bare figure.

- **2026-09-15 — T30 and T31 built, both run end to end on CPU before any GPU.**
  T30 keeps, of every readable draw, the one whose writer embedding is nearest the
  style lines; the embedding chooses and HWD judges, so writer retrieval is not
  independent in that mode. Local fake run: 120 draws for 60 requests, identity -0.5%
  as a typeface must. T31 puts a training-split writer's line before each of the
  writer's lines, counts the loss on the writer's line only, and raises teacher noise
  to 0.5 -- after measuring that ink slices have a standard deviation of 1.17, so
  Emuru's own 0.1 was 9% of it. Local tiny run: training, generation, HWD and CER all
  complete; with two writers the script declines to judge, as it should.

- **2026-09-15 — T29 run: a minute a writer, and no gain.** 150 LoRA steps took 56 s
  per writer on a T4, which settles the cost. Over the same 24 writers and the same 96
  target lines, identity moved +0.1 points [-6.8, 6.7] and CER rose from 12.7% to
  19.7%. Training loss fell 37%, so the adapter learned -- the lines, apparently, not
  the hand. Working hypothesis: teacher forcing keeps the writer's own strokes in the
  context at every step, so nothing needs to live in the weights. The first attempt
  died on Colab's torchao 0.10.0, which peft rejects while checking a layer it would
  never have claimed.

- **2026-09-15 — T29 built: per-writer LoRA fine-tuning, and the experiment that
  judges it.** `nib.models.finetune` prepares a writer's lines as Emuru trained on
  them -- one line, its full text, white after the last stroke so the stopping
  rule survives -- attaches LoRA to T5's attention projections, trains, and resets
  between writers. On the real checkpoint, CPU: 4.72M trainable of 719M, an adapter
  of 19 MB, a fresh adapter and a reset both reproducing the released loss exactly,
  and generation running with the adapter attached. `scripts/evaluate_finetune.py`
  generates the same targets for the same writers with and without, and tests the
  difference paired by writer; run tiny on CPU end to end, which exposed that two
  writers give a tight, meaningless interval, so it now declines to judge below ten.
  The identity gain from quality control was confirmed the same way first: +8.3
  points [2.4, 14.8] over 84 shared writers.

- **2026-09-14 — T28: re-drawing the broken lines makes Emuru read like real
  handwriting.** Up to four draws a line, each from one style line chosen by width,
  the first readable one kept: 1.35 draws a line on average, 12% more generation time.
  CER 12.5% against 10.7% for the real lines, a gap of 1.8 points where one draw left
  19.1; FID 55.87 against 67.70, intervals separate; identity 65.0% against 56.5%,
  intervals touching. No request was lost, and truncation fell from 6.8% to 0.3% --
  a line that runs to its budget reads badly and gets re-drawn. The first attempt at
  the cell died on its own harness check, 40 samples against FID's minimum of 50.

- **2026-09-14 — T24 and T27 measured: one line carries 56.5% of the identity, two
  carry less.** Cell 6 gave Emuru's first identity figure, 56.5% [50.9%, 61.9%] of what
  real lines carry, with the typeface at zero -- past half, where retrieval had put it
  at a quarter. Cell 7b joined two style lines into one image and fell below it on
  every measure that separated: identity 42.3% [36.2%, 48.7%], FID 90.64 against 67.70,
  CER 59.6% against 30.4%. The pass mark was set before the run. The real anchors did
  not move between runs, so the model did. A CER that doubles means the target text is
  not being written, which points at the joining rather than at the value of more
  evidence; the saved samples will say which.

- **2026-09-14 — T25 and T26: from a photographed page to lines, and a page made to be
  copied.** Amri's existing sample is one page photographed five ways -- 13 lines on
  squared paper, with a transcription split by sentence rather than by written line.
  Enough to build a splitter on; not enough to enrol from or test against. The
  splitter assigns connected components whole to the line their centre sits in, so a
  descender travels with its letter. Reaching 13 lines on four of the five photographs
  took four measured fixes: a strict ink threshold (ink sits at 0 after normalisation,
  surviving grid at 34-150); cutting straight runs out of the ink, so letters touching
  grid lines are no longer dropped as ruling; a border band against desk strips and
  dark page edges; and three components minimum per line, against punch holes.
  Hysteresis, to recover a lightly written "The", was built, measured and left off --
  it brings the grid back. The fifth photograph fails upstream in `find_page`, and its
  test is a strict xfail that says so. The passage is two pages of ordinary prose,
  each holding all 79 characters, every lowercase letter at least three times, every
  line within 44 characters; page 1 enrols, page 2 is held back so the system's version
  of it can be set beside Amri's.

- **2026-09-14 — T24: HWD can tell "not this writer" from "not real handwriting".**
  Every set is also measured against every other writer's reference, at no extra
  network pass. The gap -- how much farther the others are than the right writer -- is
  zero for output that is nobody's hand however clean or blurred, and the generated
  gap as a share of the real lines' is the identity figure, with an interval over
  writers. Locally the fake generator scored -0.2% [-6.0%, 5.8%], and real lines found
  their own writer nearest 97.7% of the time against a chance of 2.3%, so the anchor
  has range to measure in. Amri restated the goal the same day -- a system that works
  on his own pages, not a portfolio piece -- which makes the number of style lines the
  main axis: the product brings a page, the evaluation gave one line. The notebook
  gained cell 7b (two lines) and saves each run to Drive itself, after the VM holding
  the first HWD run was reclaimed unsaved. Also learned: NotebookEdit addresses cells
  by position, so an insert shifts every later edit onto the wrong cell; the notebook
  was rebuilt from HEAD by a script that checks each cell's heading before replacing.

- **2026-09-14 — the project's first trustworthy style figure.** Emuru on 300 held-out
  lines, T4: HWD 2.00 [1.88, 2.13] against 0.86 for the real target lines and 2.99 for
  the same texts in a typeface, all three against one shared reference in the same
  run -- 53% of the way from real handwriting to none. Against the old 0.641 constant
  it would have read 59%: the in-run floor matters most for a model close to real,
  which Emuru is not. FID and retrieval land where the earlier runs did. What the
  figure cannot yet say is how much of the distance is "not this writer" rather than
  "not real handwriting"; that needs a wrong-writer anchor, which the saved images
  allow without a GPU.

- **2026-09-11 — T23: HWD's floor and ceiling are measured inside every run.** Before
  spending the hour on cell 6, checked whether its HWD would be comparable to the 0.641
  floor. It would not: HWD compares per-writer means, and real lines against real lines
  read 1.76 at one line per writer and 0.54 at ten -- 1.06 under a 300-sample run's own
  sampling. 0.641 was measured at about seven or eight lines per writer by a script
  nobody committed, and would have made a perfect generator look nearly a fifth of the
  way to no hand at all. Every run now scores generated, real and typeface against one
  shared reference of real lines that are neither targets nor style lines, with
  intervals over writers; the per-writer computation matches `HWDScore` to 2e-8.
  Generated images are saved before any metric runs, and an HWD failure no longer
  takes CER with it. Also fixed cell 2, which imported `nib` inside the Colab kernel
  and found the clone directory as a namespace package instead.

- **2026-09-09 — T16 ran, and the project has its first measured result.** 300 lines
  in the hands of 94 unseen writers, T4, 53 minutes. FID 63.92 against a floor of
  19.15 and 254.29 for printed text; writer retrieval 20.5% top-1 against 85.8% for
  real and 1.7% for a generator with no style at all; CER gap +21.1 points. Read
  together: it produces handwriting (81% of the way from print to real) that is only
  partly the right person's (23% of the way from chance to real). `docs/phase2-first-evaluation.md`
  carries the full write-up.

  The run completed where the previous attempt died at 72 of 300. At request 41 the
  model declined to write, three re-draws failed, and it was excluded together with
  its ground truth instead of ending the run; four other requests were recovered by a
  retry. Every fix from the past fortnight was exercised by a real run rather than by
  a test.

  10.7% of the output was truncated, which is our bug and not the model's: the budget
  of 4.0 tokens per character covers about 92% of the data. Raised to 5.5, above the
  measured 99th percentile of 5.04, with the cap at 384 so it does not clamp ordinary
  text — at 5.5 a cap of 256 binds past 47 characters, and more than half the pack is
  longer. Real lines that do not fit their budget: 5.9% before, 0.7% after. The test
  for this is now derived from the pack rather than asserted on synthetic strings,
  because length and pixels-per-character are anticorrelated in real handwriting and
  no made-up string exposes that.

- **2026-09-09 — the German passage comes out, and the references with it.** A sample
  image from the pipeline check read `'Dann bist du deines Dienstes frey'`. The charset
  filter works on *characters*, and roughly three quarters of Goethe's Faust is pure
  ASCII, so 584 German lines had walked straight through it. Identified by the
  offending characters rather than by assumption, which mattered: text 3 loses lines to
  the same filter and is **English** — an article about an Austrian computer, the
  Mailuefterl, where only the lines naming it carry an umlaut. Excluding it too would
  have thrown away 2,051 sound English lines. Only passage 6 goes.

  The pack fell from 10,862 lines to 9,142, and **CER fell from 13.73% to 11.45%** —
  that 2.28 points is the size of the penalty we had been imposing on TrOCR for reading
  a language it was never trained on.

  Rebuilding exposed a hole in the references module itself: it identifies a pack by
  *filename*, and a rebuilt pack keeps its name, so numbers measured on 10,862 lines sat
  beside a pack of 9,142 looking entirely current. The file now records `pack_records`,
  and `evaluate_generator.py` refuses to start on a mismatch — before the 2.9 GB
  checkpoint download, not after an hour of GPU.

  Also fixed, found while checking the thinner pack: the retrieval gallery silently
  skipped writers with too few images **while keeping their queries**, which cannot
  match and quietly depressed the score. Those queries are now withheld and counted.

- **2026-08-31 — T15 done, and T16's code with it.** The line-level references are
  measured and committed: FID floor 19.06, writer retrieval 83.7% top-1 / 97.8% top-5,
  CER 13.73% over 300 lines (13.36% over 40 -- the cheap estimate held). It did not need a GPU — `check_metrics.py` generates nothing, and the
  220s-per-line figure belongs to generation alone. Two of the three moved far enough
  to change how a result reads: the FID floor nearly halved, so the word-level number
  would have made any generated set look almost twice as good as it is. They live in
  `references/`, which is a committed directory rather than part of the ignored
  `outputs/` — same reasoning as the committed writer split, since the numbers must
  survive the trip to whichever machine generates. T16's code landed too: a token
  budget from the text length, truncation counting, `--unit lines`, and a retrieval
  gallery that excludes the target samples so a match cannot be credited to shared
  content. `pyproject.toml` gained a `models` extra — the `transformers<5` pin was a
  decision recorded in prose and enforced nowhere, so a fresh Colab session would have
  installed 5.x and failed thirty minutes in. `notebooks/colab_eval.ipynb` runs the
  whole thing and holds no logic.

- **2026-08-31 — T14 done.** `cvl_lines_64.lmdb`: 10,862 lines, 309 writers, 148 MB
  compacted, built in 245s. One script packs both units — `build_index.py --unit
  lines|words` — because the two differ in exactly four things (reader, normaliser,
  key, header source) and those now sit in a `UNITS` table with everything else
  shared. A second near-identical script was the easier thing to write and the worse
  thing to read. `PackedWord` is now `PackedSample`; the record holds lines too and a
  name that lies is worse than a mechanical rename. The compacted upload copy is
  produced by the build rather than left as a step to remember — the module already
  said the shipped artefact is always compacted, and forgetting it once already cost
  an 8 GB upload. `check_data.py` reports both packs; only the word pack is required,
  since a machine set up for one job should not be told it is broken for lacking the
  other. Two tests exist specifically to catch a line pack built with the word
  scanner: mean width over 400px, and spaces in the text.

- **2026-08-31 — T13 done.** `nib.data.cvl_lines` reads CVL's 13,473 line crops and
  reassembles each transcription from the word filenames. 10,862 kept; the rest are
  counted by reason — 1,421 German, 1,132 with word-index gaps, 33 with no word files,
  25 from the writer CVL's own readme excludes. `total_seen` equals the file count on
  disk exactly. Word filenames are indexed in one unfiltered pass, keyed by
  (writer, text, line): filtering the index first would hide the very gaps it exists to
  find. Drop reasons are attributed in a fixed order — charset before incompleteness —
  so "incomplete" reports only English lines lost to data quality, and there is a test
  that pins that order. Added `normalise_line` beside `normalise_word`: the same
  mechanism, deliberately two names, because fixing the height destroys a word's
  relative scale and is exactly what makes lines comparable. Deleted `_cvl_lines` from
  `scripts/check_metrics.py`, which had reassembled transcriptions inline, filtered
  nothing, and drawn all 40 samples from one writer.

- **2026-08-28 — T0 done.** Package `nib-synth` installs editable as `nib`. Deviated from the
  brief's proposed layout: `src/nib/...` instead of `src/...`, so imports work from any working
  directory. torch deliberately excluded from base dependencies. Added `.gitattributes` for
  LF/CRLF, which would otherwise show every file as modified between Windows and Colab.
  Not committed — awaiting Amri.
